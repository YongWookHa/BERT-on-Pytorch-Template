import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import json
from easydict import EasyDict as edict
from utils.tensor import split_last, merge_last

class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, config):
        super().__init__()
        self.transformer = models.Transformer(config)
        self.fc = nn.Linear(config.dim, config.dim)

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
            mask = mask[:, None, None, :].float()
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

class Block(nn.Module):
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

class Transformer(nn.Module):
    "Transformer with Self-Attentive Blocks"
    def __init__(self, config):
        super().__init__()
        self.embed = Embeddings(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
    
    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h

class BERTModel4Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = Transformer(self.config)
        self.fc = nn.Linear(self.config.dim, self.config.dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(self.config.dim, self.config.dim)
        self.activ2 = gelu
        self.norm = LayerNorm(self.config)
        self.classifier = nn.Linear(self.config.dim, 2)
        # decoder is shared with embedding layer
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf