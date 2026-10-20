"""Equation-level DSG2former: (11)-(18), (21)-(28), (30)-(34).

Keep this separate from legacy experiments. Underspecified choices are explicit
constructor arguments; the default dimension is the paper's three game features.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .model import SentimentGame, AttentionUnit


class PaperDSG2former(nn.Module):
    def __init__(self, dim=3, heads=3, blocks=2, bias_size=50,
                 mask_rates=(1., .5), dropout=0., gate_activation="sigmoid",
                 initial_payoff_scale=.01, norm_eps=1e-6, initialization="ancestor"):
        super().__init__()
        if dim % heads or len(mask_rates) != 2:
            raise ValueError("Head dimensions or two mask rates are invalid")
        self.dim = dim
        self.game = SentimentGame()
        with torch.no_grad():
            self.game.alpha.copy_(torch.tensor([0., initial_payoff_scale, initial_payoff_scale]))
        self.embedding = nn.Identity() if dim == 3 else nn.Linear(3, dim)
        self.embedding.out_features = dim  # Shared evaluation helper interface.
        self.units = nn.ModuleList([AttentionUnit(dim, heads, bias_size, rate, "rpe_mask", dropout)
                                    for _ in range(blocks) for rate in mask_rates])
        for unit in self.units:
            unit.norm1.eps = norm_eps
            unit.norm2.eps = norm_eps
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.spatial_head = nn.Linear(dim, 3)
        # Eq. (31) specifies W1 and W2 without biases. Sigma is not identified
        # in the paper; expose the activation rather than silently hard-code it.
        activation = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}[gate_activation]
        self.gate = nn.Sequential(nn.Linear(dim, dim, bias=False), activation(), nn.Linear(dim, 1, bias=False))
        self.temporal_head = nn.Linear(dim, 3)
        if initialization != "ancestor":
            raise ValueError("Use the documented CG_LDGR spatial initialization")
        # Available early TensorFlow spatial source: GlorotUniform Dense
        # kernels; zero attention/head bias; MLP Normal(std=1e-6) bias;
        # relative-position TruncatedNormal(std=.02, truncated at two SD).
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for unit in self.units:
            for module in unit.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.bias, std=1e-6)
            nn.init.trunc_normal_(unit.attention.bias.weight, std=.02, a=-.04, b=.04)

    def forward_batch(self, drivers, positions, valid, previous, previous_mask):
        x = self.embedding(self.game(drivers))
        batch, n, dim = x.shape
        for unit in self.units:
            attention = unit.attention
            q, k, v = attention.qkv(unit.norm1(x)).reshape(batch, n, 3, attention.heads, dim // attention.heads).permute(2, 0, 3, 1, 4).unbind(0)
            bias = attention.bias(positions).permute(0, 3, 1, 2)
            bias = bias + (positions >= attention.bias.num_embeddings * attention.mask_rate).to(x.dtype)[:, None] * -100.
            bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
            z = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                              dropout_p=attention.dropout if self.training else 0.)
            x = x + attention.projection(z.transpose(1, 2).reshape(batch, n, dim))
            x = x + unit.mlp(unit.norm2(x))
        h = self.norm(x)
        previous = torch.where(previous_mask[:, :, None], previous, h)
        # Numerically stable equivalent of exp(g(h))/(exp(g(h))+exp(g(prev))).
        alpha = torch.sigmoid(self.gate(h) - self.gate(previous))
        evolved = alpha * h + (1 - alpha) * previous
        # Eq. (30)-(32): cache H_t (spatial), never H_t^evol (fused).
        return self.temporal_head(evolved), self.spatial_head(h), h

    def freeze_spatial(self):
        """Experimental Settings V.A.4: freeze the pretrained spatial encoder."""
        for module in [self.game, self.embedding, self.units, self.norm, self.spatial_head]:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def options_for_reproduction(self):
        return {"temporal_history": "previous spatial representation H_(t-1)",
                "unknown_user_initial_state": "current spatial representation when no prior H exists",
                "state_updates": "all genuinely observed users present in snapshot, including silent users",
                "input_projection": "identity at dim=3; learned projection is an explicit optional variant"}
