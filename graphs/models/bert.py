import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import json
from easydict import EasyDict as edict
from utils.tensor import split_last, merge_last

def gelu(x):
    "Implementation of the gelu activation fucntion by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, config, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.dim))
        self.beta = nn.Parameter(torch.zeros(config.dim))
        self.variance_epsilon = variance_epsilon
    
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, config):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size+3, config.dim)
        self.pos_embed = nn.Embedding(config.max_len, config.dim)
        self.seg_embed = nn.Embedding(config.n_segments + 1, config.dim)

        self.norm = LayerNorm(config)
        self.drop = nn.Dropout(config.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (S, ) -> (B, S)

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))

class MultiheadedSelfAttention(nn.Module):
    "Multi-Headed Dot Product Attention"
    def __init__(self, config):
        super().__init__()
        self.proj_q = nn.Linear(config.dim, config.dim)
        self.proj_k = nn.Linear(config.dim, config.dim)
        self.proj_v = nn.Linear(config.dim, config.dim)
        self.drop = nn.Dropout(config.p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = config.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask.float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) - > (B, H, S, W) -trans -> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

class PositionWiseFeedForward(nn.Module):
    "FeedForward Neural Networks for each position"
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, config.dim_ff)
        self.fc2 = nn.Linear(config.dim_ff, config.dim)
        
    def forward(sefl, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S ,D)
        return self.fc2(gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    "Transformer Block"
    def __init__(self, config):
        super().__init__()
        self.attn = MultiheadedSelfAttention(config)
        self.proj = nn.Linear(config.dim, config.dim)
        self.norm1 = LayerNorm(config)
        self.pwff = PositionWiseFeedForward(config)
        self.norm2 = LayerNorm(config)
        self.drop = nn.Dropout(config.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.proj(h)))
        return h

class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden = self.config.dim
        self.n_layers = self.config.n_layers
        self.attn_heads = self.config.n_heads

        self.feed_forward_hidden = self.config.dim_ff
        self.embedding = Embeddings(self.config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
    
    def forward(self, x, seg):
        # attention masking for padded token
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, seg)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BERT(self.config)
        self.next_sentence = NextSentencePrediction(self.config)
        self.mask_lm = MaskedLanguageModel(self.config)

    def forward(self, x, segment_ids):
        x = self.bert(x, segment_ids)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, config):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(config.dim, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, config):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(config.dim, config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))