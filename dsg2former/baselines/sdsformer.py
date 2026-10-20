"""SDSformer ablation: static sentiment-game spatial representation only.

Same four attention/MLP units as the paper, no temporal gate or temporal head.
The count head is the user's extension, not part of the original SDSformer.
"""
import torch
from torch.nn import functional as F
from dsg2former.multitask_model import MultiTaskDSG2former


class SDSformer(MultiTaskDSG2former):
    def __init__(self, **options):
        super().__init__(**options)
        del self.gate
        del self.temporal_head

    def action_cached(self, current, previous=None, previous_mask=None):
        return self.spatial_head(current)

    def evolve_cached(self, current, previous=None, previous_mask=None):
        return current

    def forward_batch(self, drivers, positions, valid, previous=None, previous_mask=None):
        x = self.embedding(self.game(drivers))
        batch, n, dim = x.shape
        for unit in self.units:
            attention = unit.attention
            q, k, v = attention.qkv(unit.norm1(x)).reshape(batch, n, 3, attention.heads, dim//attention.heads).permute(2, 0, 3, 1, 4).unbind(0)
            bias = attention.bias(positions).permute(0, 3, 1, 2)
            bias = bias + (positions >= attention.bias.num_embeddings*attention.mask_rate).to(x.dtype)[:, None]*-100.
            bias = bias.masked_fill(~valid[:, None, None, :], float('-inf'))
            z = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=attention.dropout if self.training else 0.)
            x = x + attention.projection(z.transpose(1, 2).reshape(batch, n, dim))
            x = x + unit.mlp(unit.norm2(x))
        h = self.norm(x)
        spatial = self.spatial_head(h)
        return spatial, spatial, h
