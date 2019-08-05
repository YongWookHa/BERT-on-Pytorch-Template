import numpy as numpy
import sentencepiece as spm

import torch

from torch.utils.data import DataLoader, TensorDataset, Dataset

class SentencePairDataLoader:
    def __init__(self, config):
        self.config = config

        if config.data_mode == "corpus":
            self.f = open(self.config.data_folder, "r", encoding="utf-8", errors='ignore')
            self.tokenizer = self.config.tokenizer
            if self.tokenizer is 'bpe':
                prefix = bpe_model
                cmd = '--input={} --vocab_size={} --model_prefix={}'
                cmd = cmd.format(self.config.data_folder, vocab_size, prefix)
                try:
                    spm.SentencePieceTrainer.Train(cmd)
                    self.sp = spm.SenetencePieceProcesseor()
                    sp.Load('{}.model'.format(prefix))
                except Exception:
                    raise
                # bpe = self.sp.EncodeAsPieces(line) # to list
            else:
                raise NotImplementedError
            self.max_len = max_len 
            self.batch_size = self.config.batch_size
        else:
            raise NotImplementedError

        

            