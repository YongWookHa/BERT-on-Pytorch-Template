import numpy as numpy
import sentencepiece as spm
import random
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader, TensorDataset, Dataset

class SentencePairDataset(Dataset):
    def __init__(self, config, mode):
        "mode : ['train', 'validate']"
        self.config = config 
        if self.config.tokenizer is 'bpe':
            prefix = bpe_model
            cmd = '--input={} --vocab_size={} --model_prefix={}'
            cmd = cmd.format(self.config.data_dir, self.config.vocab_size, prefix)
            try:
                spm.SentencePieceTrainer.Train(cmd)
                sp = spm.SenetencePieceProcesseor()
                sp.Load('{}.model'.format(prefix))
            except Exception:
                raise
            self.tokenize = sp.EncodeAsPieces  
        elif self.config.tokenizer is None:  # split
            self.tokenize = lambda x: x.split()
        else:
            raise NotImplementedError

        train_data, validate_data = [], []
        self.total_lines = sum([1 for _ in open(self.config.data_dir, "r", encoding="utf8")])
        with open(self.config.data_dir, "r", encoding="utf-8", errors="ignore") as f:
            self.lines = f.readlines()  # load all on memory         
            num_train_data = int(self.config.train_data_ratio * self.total_lines)
            
            if mode == 'train':
                self.lines = self.lines[:num_train_data]
            elif mode == 'validate':
                self.lines = self.lines[num_train_data:]
            else:
                raise ValueError("Invalid Mode: '{}'".format(mode))

            for i in range(len(self.lines)):
                self.lines[i] = list(map(self.tokenize, self.lines[i].split('\t')[:2]))

    def __len__(self):
        "Total number of data"
        return len(self.lines)
    
    def __getitem__(self, idx):
        t1, t2, is_next_label = self.random_sent(idx)
        t1_random, t1_label = self.random_word(t1)  # mask random words
        t2_random, t2_label = self.random_word(t2)  # mask random words

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = ['[CLS]'] + t1_random + ['[SEP]']
        t2 = t2_random + ['[SEP]']

        t1_label = ['[CLS]'] + t1_label + ['[SEP]']
        t2_label = t2_label + ['[SEP]']

        segment_label = [1]*(len(t1)+2) + [2]*(len(t2)+1)
        bert_input = (t1 + t2)[:self.max_len]
        bert_label = (t1_label + t2_label)[:self.max_len]

        padding = ['[PAD]']*(self.max_len - len(bert_input))
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:  # 80% randomly change token to mask token
                    tokens[i] = "[MASK]"
                elif prob < 0.9:  # 10% randomly change token to random token
                    tokens[i] = random.choice(random.choice(self.lines)[random.randint(0,1)])
                else:  # 10% randomly change token to current token
                    pass

            output_label.append(token)
        return tokens, output_label

    def random_sent(self, idx):
        t1, t2 = self.get_corpus_line(idx)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self):
        return self.lines[idx][0], self.lines[idx][1]

    def get_random_line(self, ):
        return random.choice(self.lines)[1]
    

class BERTDataLoader:
    def __init__(self, config):
        self.config = config

        if self.config.mode == "pretrain":
            train_dataset = SentencePairDataset(self.config, 'train')
            validate_dataset = SentencePairDataset(self.config, 'validate')

            self.train_dataset_len = len(train_dataset)
            self.validate_dataset_len = len(validate_dataset)

            self.train_dataloader = DataLoader(train_dataset,
                                        batch_size = self.config.batch_size,
                                        num_workers = self.config.data_loader_workers,
                                        pin_memory = self.config.pin_memory)

            self.validate_dataloader = DataLoader(validate_dataset,
                                        batch_size = self.config.batch_size,
                                        num_workers = self.config.data_loader_workers,
                                        pin_memory = self.config.pin_memory)



            
            


