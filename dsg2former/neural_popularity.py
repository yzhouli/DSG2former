"""Small neural popularity head; existing two-block backbones stay unchanged."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def final_micro_state(model, current, previous, mask, age, temporal):
    """The state used by the existing temporal residual, before classification."""
    if not temporal:
        return current
    old = torch.where(mask[:, None], previous, current)
    delta = torch.log1p(age.clamp_min(0))[:, None]
    alpha = torch.sigmoid(model.gate(torch.cat((current, torch.zeros_like(delta)), -1))
                          - model.gate(torch.cat((old, delta), -1)))
    evolved = alpha * current + (1 - alpha) * old
    return torch.where(mask[:, None], evolved, current)


def observed_history(snapshot, steps=8):
    """Use already observed counts, never the current popularity_target."""
    if steps != 8:
        raise ValueError('The prepared causal cache stores exactly eight hours')
    history = np.asarray(snapshot['history_summary'], dtype=np.float32)
    if len(history) < 13:
        raise ValueError('Expected the prepared causal count-history features')
    length = min(steps, int(snapshot['window']))
    counts = history[4:12].copy()
    valid = np.arange(steps) >= steps - length
    if np.any(counts < 0) or not np.isfinite(counts).all() or np.any(counts[~valid]):
        raise ValueError('Invalid observed history or nonzero padding')
    sequence = np.stack((np.log1p(counts), valid.astype(np.float32)), -1)
    # Known chain participants and elapsed hours are observable at the cutoff.
    context = np.log1p(np.maximum(history[[0, 3]], 0))
    return sequence.astype(np.float32), context.astype(np.float32)


def pool_micro_state(state, logits):
    """Pool all users of the candidate-independent observed macro graph."""
    if len(state) == 0 or len(state) != len(logits):
        raise ValueError('Expected a nonempty aligned observed user graph')
    return torch.cat((state.mean(0), logits.softmax(-1).mean(0),
                      state.new_tensor([np.log1p(len(state))])))


class HistoryStatePopularityHead(nn.Module):
    def __init__(self, state_dim, history_hidden=32, state_hidden=64, dropout=.1,
                 use_state=True):
        super().__init__()
        self.state_dim = state_dim
        self.use_state = use_state
        self.history = nn.GRU(2, history_hidden, batch_first=True)
        if use_state:
            self.state_projection = nn.Sequential(nn.Linear(state_dim, state_hidden),
                nn.LayerNorm(state_hidden), nn.GELU(), nn.Dropout(dropout))
        self.readout = nn.Sequential(
            nn.Linear(history_hidden + (state_hidden if use_state else 0) + 2, 64),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, sequence, state, context):
        if sequence.ndim != 3 or sequence.shape[1:] != (8, 2):
            raise ValueError('Expected B x8hours x(logcount,observed-mask)')
        mask = sequence[:, :, 1] > .5
        lengths = mask.sum(1)
        # Move valid left-padded observations to the front before packing.
        indices = (torch.arange(8, device=sequence.device)[None] + 8 - lengths[:, None]).clamp(max=7)
        ordered = sequence.gather(1, indices[:, :, None].expand(-1, -1, 2))
        packed = pack_padded_sequence(ordered, lengths.clamp_min(1).cpu(),
                                     batch_first=True, enforce_sorted=False)
        _, h = self.history(packed)
        h = torch.where((lengths > 0)[:, None], h[0], torch.zeros_like(h[0]))
        pieces = [h, context]
        if self.use_state:
            pieces.insert(1, self.state_projection(state))
        return F.softplus(self.readout(torch.cat(pieces, -1)).squeeze(-1))
