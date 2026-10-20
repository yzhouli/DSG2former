"""Compact neural count fusion; retain the existing two-block micro encoders."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def count_priors(sequence, completed, initial, root_mean, use_context):
    """Causal baselines that a learned residual may adjust, not fitted trees."""
    counts = np.expm1(np.asarray(sequence)[:, :, 0]).clip(0)
    continuation = .02 + .25*counts[:, -1] + .05*counts[:, -4:].mean(1)
    first = np.full(len(counts), float(root_mean), dtype=np.float32)
    if use_context:
        c = np.asarray(completed)
        for i in range(len(c)):
            for minimum in [5., 1.]:
                available = [j for j in range(5) if np.expm1(c[i, 5*j]) >= minimum-1e-5]
                if available:
                    first[i] = np.expm1(c[i, 5*available[0]+1])
                    break
    return np.where(initial, np.maximum(first, .25), continuation).astype(np.float32)


class RegimeCountFusion(nn.Module):
    """GRU/history + bottleneck final state; separate observable-hour readouts."""
    def __init__(self, state_dim, numeric_dim, history_hidden=16, state_hidden=16,
                 hidden=32, dropout=.2, hurdle=False):
        super().__init__()
        self.hurdle = hurdle
        self.history = nn.GRU(2, history_hidden, batch_first=True)
        self.state_projection = nn.Sequential(nn.Linear(state_dim, state_hidden),
            nn.LayerNorm(state_hidden), nn.GELU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(history_hidden+state_hidden+numeric_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.readouts = nn.ModuleList([nn.Linear(hidden, 2 if hurdle else 1) for _ in range(2)])
        for layer in self.readouts:
            nn.init.normal_(layer.weight, std=.002); nn.init.zeros_(layer.bias)

    def forward(self, sequence, state, numeric, prior, initial):
        lengths = (sequence[:, :, 1] > .5).sum(1)
        idx = (torch.arange(8, device=sequence.device)[None]+8-lengths[:, None]).clamp(max=7)
        ordered = sequence.gather(1, idx[:, :, None].expand(-1, -1, 2))
        packed = pack_padded_sequence(ordered, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.history(packed)
        h = torch.where((lengths > 0)[:, None], h[0], torch.zeros_like(h[0]))
        z = self.fusion(torch.cat((h, self.state_projection(state), numeric), -1))
        outputs = torch.stack([layer(z) for layer in self.readouts], 1)
        out = outputs[torch.arange(len(z), device=z.device), initial.long()]
        magnitude = prior * torch.exp(2*torch.tanh(out[:, 0]))
        if self.hurdle:
            # Positive size >=1; separate chance of any archive-observed user.
            activity = out[:, 1] + torch.where(initial, out.new_tensor(2.2), out.new_tensor(-1.7))
            mean = torch.sigmoid(activity) * (1+magnitude)
            return mean, activity, 1+magnitude
        return magnitude, None, None


def count_loss(output, target, initial, anchors, raw_weight=1., balanced=True, hurdle_weight=.1):
    pred, activity, positive = output
    groups = [initial == False, initial == True] if balanced else [torch.ones_like(initial)]
    terms = []
    for j, mask in enumerate(groups):
        if not mask.any():
            continue
        # All scales are estimated from the head's training portion only.
        scale = anchors[j] if balanced else anchors[2]
        loss = F.mse_loss(torch.log1p(pred[mask]), torch.log1p(target[mask])) / scale[1]
        loss = loss + raw_weight*F.mse_loss(pred[mask], target[mask]) / scale[0]
        if activity is not None:
            active = target[mask] > 0
            loss = loss + hurdle_weight*F.binary_cross_entropy_with_logits(activity[mask], active.float())
            if active.any():
                loss = loss + hurdle_weight*F.mse_loss(torch.log1p(positive[mask][active]), torch.log1p(target[mask][active]))
        terms.append(loss)
    return torch.stack(terms).mean()
