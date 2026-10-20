"""Keras/TensorFlow Adam convention used by the available CG_LDGR source.

Keras uses epsilon-hat after sqrt(v), before variance bias correction is folded
into the learning-rate factor. This differs from torch.optim.Adam's epsilon.
"""
import torch


class TensorFlowAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=.001, betas=(.9, .999), epsilon=1e-7):
        super().__init__(params, dict(lr=lr, betas=betas, epsilon=epsilon, iteration=0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["iteration"] += 1
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("The paper model uses dense gradients")
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                m.add_(grad - m, alpha=1-b1)
                v.add_(grad.square() - v, alpha=1-b2)
                step = p.new_tensor(group["iteration"])
                alpha = p.new_tensor(group["lr"]) * (1-p.new_tensor(b2).pow(step)).sqrt() / (1-p.new_tensor(b1).pow(step))
                p.sub_(m * alpha / (v.sqrt() + group["epsilon"]))
        return loss
