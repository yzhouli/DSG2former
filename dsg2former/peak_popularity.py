"""Small history/state neural head with a learned persistence reference.

The two-block micro encoder and existing GRU/state fusion dimensions stay fixed.
The extra scalar gate removes the frozen small-base ceiling on continuation
forecasts. Target counts are never inputs to forward().
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from .residual_popularity import ResidualPopularity


class PeakAwarePopularity(ResidualPopularity):
    def __init__(self, base_dim, state_dim, first_cap=1., continuation_cap=2.):
        super().__init__(base_dim, state_dim, first_cap, continuation_cap)
        self.persistence_gate = nn.Linear(25, 1)
        nn.init.zeros_(self.persistence_gate.weight)
        nn.init.zeros_(self.persistence_gate.bias)

    def forward(self, numeric, observed, sequence, state, initial):
        lengths = (sequence[:, :, 1] > .5).sum(1)
        ix = (torch.arange(8, device=sequence.device)[None]+8-lengths[:, None]).clamp(max=7)
        ordered = sequence.gather(1, ix[:, :, None].expand(-1, -1, 2))
        packed = pack_padded_sequence(ordered, lengths.clamp_min(1).cpu(),
                                     batch_first=True, enforce_sorted=False)
        _, h = self.history(packed)
        h = torch.where((lengths > 0)[:, None], h[0], torch.zeros_like(h[0]))
        base_h = self.base.features(numeric, observed)
        z = torch.cat((h, self.state(state), base_h, torch.log1p(observed)), -1)
        delta = torch.tanh(self.correction(z).squeeze(-1))
        cap = torch.where(initial, delta.new_tensor(self.first_cap),
                          delta.new_tensor(self.continuation_cap))
        base = F.softplus(self.base.output(base_h).squeeze(-1))
        previous = torch.expm1(sequence[:, -1, 0]).clamp_min(0)
        gate = torch.sigmoid(self.persistence_gate(z).squeeze(-1))
        gate = torch.where(initial | (lengths == 0), torch.zeros_like(gate), gate)
        anchor = (1-gate)*base + gate*previous
        return anchor*torch.exp(cap*delta), delta
