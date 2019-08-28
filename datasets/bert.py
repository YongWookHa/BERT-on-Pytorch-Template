import numpy as numpy
import os
import sentencepiece as spm
import random
import pandas as pd
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils.tokenization import FullTokenizer
from utils.misc import truncate_tokens_pair, get_random_word

class SentencePairDataset(Dataset):  # tab separated setence pair dataset
    def __init__(self, config, tokenizer, mode):
        "mode : ['train', 'validate']"
        self.config = config 
        self.max_len = self.config.max_len
        self.mask_prob = self.config.mask_prob

        self.indexer = tokenizer.convert_tokens_to_ids
        self.tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))
        self.vocab = tokenizer.vocab

        self.total_lines = sum([1 for _ in open(self.config.data_path, "r", encoding="utf8")])
        with open(self.config.data_path, "r", encoding="utf-8", errors="ignore") as f:
            self.lines = f.readlines()  # load all on memory         
            num_train_data = int(self.config.train_data_ratio * self.total_lines)
            
            if mode == 'train':
                self.lines = self.lines[:num_train_data]
            elif mode == 'validate':
                self.lines = self.lines[num_train_data:-1]
            else:
                raise ValueError("Invalid Mode: '{}'".format(mode))

            iter_bar = tqdm(range(len(self.lines)), desc="{} data".format(mode))
            for i in iter_bar:
                self.lines[i] = self.lines[i].split('\t')[:2]

    def __len__(self):
        "Total number of data"
        return len(self.lines)
    
    def __getitem__(self, idx):
        sent_a, sent_b, is_next = self.random_sent(idx)
        tokens_a, tokens_b = sent_a.split(), sent_b.split()
        
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        tokens_a, tokens_a_label = self.random_word(tokens_a)
        tokens_b, tokens_b_label = self.random_word(tokens_b)
        
        # Add Special Tokens
        cls_idx = self.indexer('[CLS]')
        sep_idx = self.indexer('[SEP]')
    
        bert_input = [cls_idx] + tokens_a + [sep_idx] + tokens_b + [sep_idx]
        bert_label = [0] + tokens_a_label + [0] + tokens_b_label + [0]
        segment_label = [1]*(len(tokens_a)+2) + [2]*(len(tokens_b)+1)

        # Zero Padding
        n_pad = self.max_len - len(bert_input)
        bert_input.extend([0]*n_pad)
        bert_label.extend([0]*n_pad)
        segment_label.extend([0]*n_pad)

        batch = [bert_input, bert_label, segment_label, is_next]

        batch_tensors = [torch.tensor(b) for b in batch]
        return batch_tensors

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.indexer('[MASK]')

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = self.indexer(get_random_word(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.indexer(token)

                output_label.append(self.indexer(token))

            else:
                tokens[i] = self.indexer(token)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, idx):
        t1, t2 = self.get_corpus_line(idx)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, idx):
        return self.lines[idx][0], self.lines[idx][1]

    def get_random_line(self):
        x = random.choice(self.lines)[1]
        for a in x:
            if isinstance(a, int):
                print(x)
                raise ValueError
        return x


class MSRPDataset(Dataset):  # tab separated setence pair dataset
    def __init__(self, config, tokenizer, mode):
        "mode : ['train', 'validate']"
        self.config = config 
        self.max_len = self.config.max_len
        self.mask_prob = self.config.mask_prob

        self.indexer = tokenizer.convert_tokens_to_ids
        self.tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))
        self.vocab = tokenizer.vocab

        self.is_next_pair = []
        self.not_next_pair = []

        if mode == 'train':
            data_path = self.config.data_path
        elif mode == 'validate':
            data_path = os.path.dirname(self.config.data_path) + "/msr_paraphrase_test.txt"
        else:
            raise ValueError("Invalid Mode: '{}'".format(mode))

        self.total_lines = sum([1 for _ in open(data_path, "r", encoding="utf8")])
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            f.readline()
            for line in f:
                # Quality ID1 ID2 String1 String2
                line = line.split('\t')
                q, s1, s2 = line[0], line[3], line[4]
                if q == '1':
                    self.is_next_pair.append([s1, s2])
                else:
                    self.not_next_pair.append([s1, s2])
        
        self.is_next_pair = [list(map(self.tokenize, x)) for x in self.is_next_pair]
        self.not_next_pair = [list(map(self.tokenize, x)) for x in self.not_next_pair]

        self.random_sent_generator = self.random_sent()

    def __len__(self):
        "Total number of data"
        return len(self.is_next_pair) + len(self.not_next_pair)
    
    def __getitem__(self, _):
        tokens_a, tokens_b, is_next = next(self.random_sent_generator)

        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [1]*(len(tokens_a)+2) + [2]*(len(tokens_b)+1)
        input_mask = [1]*len(tokens)

        # For masked Language Models
        masked_tokens, masked_pos = [], []
        n_pred = max(1, int(round(len(tokens)*self.mask_prob)))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens)
                    if token != '[CLS]' and token != '[SEP]']

        random.shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if random.random() < 0.8: # 80%
                tokens[pos] = '[MASK]'
            elif random.random() < 0.5: # 10%
                tokens[pos] = get_random_word(self.vocab)
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        # Zero Padding for masked target
        n_pad = self.max_len - n_pred
        masked_ids.extend([0]*n_pad)
        masked_pos.extend([0]*n_pad)
        masked_weights.extend([0]*n_pad)

        batch = (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)
        
        batch_tensors = [torch.tensor(x, dtype=torch.long) for x in batch]
        return batch_tensors

    def random_sent(self):
        # output_text, label(isNotNext:0, isNext:1)
        is_next_idx = 0
        not_next_idx = 0
        while True:
            if random.random() > 0.5:
                sent_a, sent_b = random.choice(self.is_next_pair)
                yield sent_a, sent_b, 1
            else:
                sent_a, sent_b = random.choice(self.not_next_pair)
                yield sent_a, sent_b, 0

