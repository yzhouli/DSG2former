"""DSG equations (9)-(18), SDSformer (21)-(28), temporal gate (31)-(33)."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class SentimentGame(nn.Module):
    def __init__(self, mode="game"):
        super().__init__()
        self.mode = mode
        self.alpha = nn.Parameter(torch.tensor([0.0, 1.0, 1.0]))
        self.psi = nn.Parameter(torch.zeros(2))

    def forward(self, drivers):
        inner, outer = drivers.unbind(-2)
        if self.mode == "self":
            return inner
        if self.mode == "external":
            return outer
        payoff = self.alpha[0] + self.alpha[1] * inner + self.alpha[2] * outer
        if self.mode == "payoff":
            return payoff
        rumor, other, inactive = payoff.unbind(-1)
        return torch.stack([torch.sigmoid(rumor - other), torch.sigmoid(other - rumor),
                            torch.sigmoid(inactive + self.psi[0] * rumor + self.psi[1] * other)], -1)


class GraphAttention(nn.Module):
    def __init__(self, dim, heads, bias_size, mask_rate, encoding, dropout):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.mask_rate, self.encoding = mask_rate, encoding
        self.bias = nn.Embedding(bias_size, heads)
        nn.init.trunc_normal_(self.bias.weight, std=0.02)
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, positions):
        n, dim = x.shape
        q, k, v = self.qkv(x).view(n, 3, self.heads, dim // self.heads).permute(1, 2, 0, 3).unbind(0)
        bias = None
        if self.encoding in ("rpe", "rpe_mask"):
            bias = self.bias(positions).permute(2, 0, 1)
            if self.encoding == "rpe_mask":
                masked = positions >= self.bias.num_embeddings * self.mask_rate
                # Sentinel 0 remains globally visible, matching paper indexing.
                bias = bias + masked.to(x.dtype)[None] * -100.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.projection(y.transpose(0, 1).reshape(n, dim))


class AttentionUnit(nn.Module):
    def __init__(self, dim, heads, bias_size, rate, encoding, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = GraphAttention(dim, heads, bias_size, rate, encoding, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, positions):
        x = x + self.attention(self.norm1(x), positions)
        return x + self.mlp(self.norm2(x))


class DSG2former(nn.Module):
    def __init__(self, dim=48, heads=3, blocks=2, bias_size=50, encoding="rpe_mask",
                 temporal="gate", feature="game", dropout=0.1, history="recursive"):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.game = SentimentGame(feature)
        self.embedding = nn.Linear(3, dim)
        # Each paper block contains two MHA+MLP residual units.
        self.units = nn.ModuleList([AttentionUnit(dim, heads, bias_size, rate, encoding, dropout)
                                    for _ in range(blocks) for rate in (0.5, 1.0)])
        self.norm = nn.LayerNorm(dim)
        self.spatial_head = nn.Linear(dim, 3)
        self.temporal_head = nn.Linear(dim, 3)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))
        self.temporal, self.encoding, self.history = temporal, encoding, history
        if temporal == "gru":
            self.cell = nn.GRUCell(dim, dim)
        elif temporal == "rnn":
            self.cell = nn.RNNCell(dim, dim)
        elif temporal == "mha":
            self.temporal_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def spatial(self, drivers, positions):
        x = self.embedding(self.game(drivers))
        if self.encoding == "ape":
            position = torch.arange(len(x), device=x.device, dtype=x.dtype)[:, None]
            frequency = torch.exp(torch.arange(0, x.shape[-1], 2, device=x.device) * (-math.log(10000) / x.shape[-1]))
            pe = torch.zeros_like(x)
            pe[:, 0::2] = torch.sin(position * frequency)
            pe[:, 1::2] = torch.cos(position * frequency[:x.shape[-1] // 2])
            x = x + pe
        for unit in self.units:
            x = unit(x, positions)
        return self.norm(x)

    def forward(self, drivers, positions, previous=None, history=None, previous_mask=None):
        h = self.spatial(drivers, positions)
        spatial_logits = self.spatial_head(h)
        if previous is None:
            previous = h
        elif previous_mask is not None:
            # A changing candidate set needs identity-aligned states. New users
            # have no historical representation and start from current features.
            previous = torch.where(previous_mask[:, None], previous, h)
        if self.temporal == "none":
            evolved = h
            logits = spatial_logits
        elif self.temporal == "gate":
            alpha = torch.sigmoid(self.gate(h) - self.gate(previous))
            evolved = alpha * h + (1 - alpha) * previous
            logits = self.temporal_head(evolved)
        elif self.temporal in ("gru", "rnn"):
            evolved = self.cell(h, previous)
            logits = self.temporal_head(evolved)
        elif self.temporal == "mha":
            sequence = torch.stack((history or []) + [h], dim=1)
            evolved = self.temporal_attention(h[:, None], sequence, sequence, need_weights=False)[0][:, 0]
            logits = self.temporal_head(evolved)
        else:
            raise ValueError(self.temporal)
        next_state = h if self.history == "previous_spatial" else evolved
        return logits, spatial_logits, next_state, h
