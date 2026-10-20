"""User action + next-window popularity heads over a shared DSG2former.

The popularity branch must receive a causal graph, without forecast candidate
sampling. Temporal caches are raw spatial states, so a frozen spatial encoder
can be evaluated once without changing the temporal-head optimization problem.
"""
import torch
from torch import nn

from .paper_model import PaperDSG2former


class MultiTaskDSG2former(PaperDSG2former):
    def __init__(self, **options):
        super().__init__(**options)
        # Four observed-history summaries plus log1p(causal graph node count).
        self.popularity_head = nn.Sequential(nn.Linear(self.dim+5, self.dim), nn.Tanh(),
                                             nn.Linear(self.dim, 1), nn.Softplus())
        for module in self.popularity_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def evolve_cached(self, current, previous, previous_mask):
        previous = torch.where(previous_mask[:, None], previous, current)
        alpha = torch.sigmoid(self.gate(current)-self.gate(previous))
        return alpha*current+(1-alpha)*previous

    def action_cached(self, current, previous, previous_mask):
        return self.temporal_head(self.evolve_cached(current, previous, previous_mask))

    def popularity_cached(self, current, previous, previous_mask, window_index, node_count, history_summary):
        evolved = self.evolve_cached(current, previous, previous_mask)
        pooled = current.new_zeros((len(node_count), self.dim))
        pooled.index_add_(0, window_index, evolved)
        pooled = pooled/node_count[:, None].clamp_min(1)
        # history_summary contains observed cumulative participants, previous
        # new participants, previous active participants, and elapsed hours.
        observed = torch.cat((history_summary, node_count[:, None]), dim=1)
        if torch.any(observed < 0):
            raise ValueError("Popularity history summaries must be nonnegative")
        return self.popularity_head(torch.cat((pooled, torch.log1p(observed)), dim=1)).squeeze(1)
