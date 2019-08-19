import numpy as numpy
import sentencepiece as spm
import random
import pandas as pd
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils.tokenization import FullTokenizer
from utils.misc import truncate_tokens_pair, get_random_word

class SentencePairDataset(Dataset):  # msr_paraphrase dataset
    def __init__(self, config, tokenizer, mode):
        "mode : ['train', 'validate']"
        self.config = config 
        self.max_len = self.config.max_len
        self.max_pred = self.config.max_pred
        self.mask_prob = self.config.mask_prob

        self.indexer = tokenizer.convert_tokens_to_ids
        self.tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))
        self.vocab = tokenizer.vocab

        train_data, validate_data = [], []
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
                self.lines[i] = list(map(self.tokenize, self.lines[i].split('\t')[:2]))

    def __len__(self):
        "Total number of data"
        return len(self.lines)
    
    def __getitem__(self, idx):
        tokens_a, tokens_b, is_next = self.random_sent(idx)

        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [1]*(len(tokens_a)+2) + [2]*(len(tokens_b)+1)
        input_mask = [1]*len(tokens)

        # For masked Language Models
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens)*self.mask_prob))))
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
        # when n_pred < max_pred, we only calculate loss within n_pred
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
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        batch = (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)
        
        batch_tensors = [torch.tensor(x, dtype=torch.long) for x in batch]
        return batch_tensors

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
        return random.choice(self.lines)[1]


