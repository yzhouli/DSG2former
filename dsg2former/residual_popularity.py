"""Low-capacity neural count base and bounded history/final-state correction."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class NumericCountBase(nn.Module):
    def __init__(self, dim, hidden=12):
        super().__init__()
        self.hidden = nn.Linear(dim+5, hidden)
        self.output = nn.Linear(hidden, 1)
        for layer in [self.hidden, self.output]:
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)

    def features(self, numeric, observed):
        return torch.tanh(self.hidden(torch.cat((numeric, torch.log1p(observed)), -1)))

    def forward(self, numeric, observed):
        return F.softplus(self.output(self.features(numeric, observed)).squeeze(-1))


class ResidualPopularity(nn.Module):
    def __init__(self, base_dim, state_dim, first_cap=.05, continuation_cap=.75):
        super().__init__()
        self.base = NumericCountBase(base_dim)
        self.history = nn.GRU(2, 4, batch_first=True)
        self.state = nn.Sequential(nn.Linear(state_dim, 4), nn.Tanh())
        self.correction = nn.Sequential(nn.Linear(4+4+12+5, 8), nn.Tanh(), nn.Linear(8, 1))
        nn.init.normal_(self.correction[-1].weight, std=.001); nn.init.zeros_(self.correction[-1].bias)
        self.first_cap = first_cap; self.continuation_cap = continuation_cap

    def forward(self, numeric, observed, sequence, state, initial):
        lengths = (sequence[:, :, 1] > .5).sum(1)
        ix = (torch.arange(8, device=sequence.device)[None]+8-lengths[:, None]).clamp(max=7)
        ordered = sequence.gather(1, ix[:, :, None].expand(-1, -1, 2))
        packed = pack_padded_sequence(ordered, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.history(packed)
        h = torch.where((lengths > 0)[:, None], h[0], torch.zeros_like(h[0]))
        base_h = self.base.features(numeric, observed)
        z = torch.cat((h, self.state(state), base_h, torch.log1p(observed)), -1)
        delta = torch.tanh(self.correction(z).squeeze(-1))
        cap = torch.where(initial, delta.new_tensor(self.first_cap), delta.new_tensor(self.continuation_cap))
        base_pred = F.softplus(self.base.output(base_h).squeeze(-1))
        return base_pred*torch.exp(cap*delta), delta
