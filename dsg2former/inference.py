"""Opt-in static-input inference for the unchanged dense three-backbone model.

Validate inputs once, outside the timed/captured forward. Parameters, attention,
RPE, temporal readout and ensemble weights remain unchanged. No training path
is patched. CUDA replay returns a borrowed buffer overwritten by the next replay.
"""
import copy

import torch
from torch import nn

from .sds_aligned import PaperAlignedDSG2former


class _PrevalidatedEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, h, positions, valid):
        # Same operations as SDSformerEncoder.forward after its input checks.
        h = h * valid[:, :, None]
        for block in self.encoder.blocks:
            h = block(h, positions, valid)
        return self.encoder.norm(h) * valid[:, :, None]


class PreparedActionEnsemble(nn.Module):
    """Single static graph, all user actions, three original neural backbones.

    The source is the final node, matching the scale-study/deployment packing.
    Use update_ for new same-shaped inputs/history, including new graph relations.
    Recreate this object when graph size, precision or weights change. Frozen
    weights are borrowed; input buffers are owned to preserve their CUDA addresses.
    This is a graph/action kernel; it does not implement macro prediction.
    """
    def __init__(self, models, inputs, previous, mask, age):
        super().__init__()
        if len(models) != 3 or len(previous) != 3 or len(inputs) != 7:
            raise ValueError('Expected three backbones, three histories and seven graph tensors')
        if any(type(m) is not PaperAlignedDSG2former for m in models):
            raise ValueError('Only paper-aligned dense DSG2former backbones are supported')
        if any(m.training or any(u.training for u in m.modules()) for m in models):
            raise ValueError('Prepare evaluated models; training/dropout is not supported')
        x, drivers, positions, adjacency, role, valid, source = inputs
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 2:
            raise ValueError('Expected one graph with at least one user and a final source node')
        n = x.shape[1]
        if drivers.shape != (1, n, 2, 3) or positions.shape != (1, n, n):
            raise ValueError('Misaligned drivers or positions')
        if adjacency.shape != positions.shape or role.shape != (1, n) or valid.shape != (1, n):
            raise ValueError('Misaligned graph inputs')
        if positions.dtype != torch.long or role.dtype != torch.long or source.dtype != torch.long:
            raise ValueError('Indices must be int64')
        if valid.dtype != torch.bool or mask.dtype != torch.bool:
            raise ValueError('Validity and history masks must be boolean')
        # Tensor-to-host reads are intentional here, never inside forward/replay.
        if source.shape != (1,) or source.item() != n - 1 or not valid.all().item():
            raise ValueError('This static packing requires valid users followed by one source')
        if not ((role >= 0) & (role < 2)).all().item():
            raise ValueError('Role indices must be 0 or 1')
        if mask.shape != (n - 1,) or age.shape != mask.shape:
            raise ValueError('Misaligned user history mask or age')
        tensors = [*inputs, *previous, mask, age]
        if any(t.device != x.device for t in tensors):
            raise ValueError('All graph and history inputs must share a device')
        if any(t.is_floating_point() and not torch.isfinite(t).all().item() for t in tensors):
            raise ValueError('Nonfinite inputs are unsupported')
        prepared = []
        for model, old in zip(models, previous):
            if x.shape[-1] != model.feature_projection[0].in_features:
                raise ValueError('Feature dimension differs from the checkpoint')
            if old.shape != (n - 1, 4 * model.encoder.dim):
                raise ValueError('History representation differs from the checkpoint')
            if not ((positions >= 0) & (positions < model.encoder.bias_size)).all().item():
                raise ValueError('RPE lookup indices are outside the checkpoint table')
            if any(p.device != x.device or p.dtype != x.dtype for p in model.parameters()):
                raise ValueError('Checkpoint parameters must match the input device and dtype')
            if drivers.dtype != x.dtype or age.dtype != x.dtype or old.dtype != x.dtype:
                raise ValueError('Features, drivers and histories must have matching precision')
            # Shallow module copy shares frozen weights but leaves the original
            # model and its validation/training behavior entirely untouched.
            fast = copy.copy(model)
            fast._modules = model._modules.copy()
            fast.encoder = _PrevalidatedEncoder(model.encoder).eval()
            prepared.append(fast)
        self.models = nn.ModuleList(prepared)
        self.inputs = tuple(t.detach().clone() for t in inputs)
        self.previous = tuple(t.detach().clone() for t in previous)
        self.mask, self.age = mask.detach().clone(), age.detach().clone()
        self.bias_size = min(m.encoder.bias_size for m in models)
        self.user_nodes = n - 1
        self.cuda_graph = None
        self.graph_output = None
        self.eval()

    def train(self, mode=True):
        if mode:
            raise ValueError('PreparedActionEnsemble is inference-only')
        return super().train(False)

    @torch.inference_mode()
    def update_(self, inputs, previous, mask, age):
        """Validate new resident inputs and copy them into fixed-address buffers.

        Validation and copies are explicit per-request costs, outside graph replay.
        They may not be omitted from end-to-end deployment latency reports.
        """
        if len(inputs) != 7 or len(previous) != 3:
            raise ValueError('Expected seven graph tensors and three histories')
        incoming = [*inputs, *previous, mask, age]
        buffers = [*self.inputs, *self.previous, self.mask, self.age]
        for source, target in zip(incoming, buffers):
            if source.shape != target.shape or source.dtype != target.dtype or source.device != target.device:
                raise ValueError('New inputs must match prepared shapes, dtypes and device')
        _, _, positions, _, role, valid, source = inputs
        checks = [torch.isfinite(t).all() for t in incoming if t.is_floating_point()]
        checks += [(positions >= 0).all(), (positions < self.bias_size).all(),
                   (role >= 0).all(), (role < 2).all(), valid.all(),
                   (source == self.user_nodes).all()]
        # One host synchronization for all value checks, rather than checking
        # positions/validity separately during every one of three backbones.
        if not torch.stack(checks).all().item():
            raise ValueError('Invalid finite values, RPE indices, roles, source or validity')
        for target, value in zip(buffers, incoming):
            target.copy_(value)
        return self

    @torch.inference_mode()
    def forward(self):
        probabilities = []
        for model, old in zip(self.models, self.previous):
            logits, state = model.spatial(*self.inputs)
            logits = model.action_cached(state[0, :self.user_nodes], logits[0, :self.user_nodes],
                                         old, self.mask, self.age)
            probabilities.append(logits.softmax(-1))
        return torch.stack(probabilities).mean(0)

    @torch.inference_mode()
    def capture(self, warmup=10):
        if self.inputs[0].device.type != 'cuda':
            raise ValueError('CUDA Graph requires CUDA inputs')
        if self.cuda_graph is not None:
            raise ValueError('This instance already owns a captured graph')
        device = self.inputs[0].device
        with torch.cuda.device(device):
            current = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self()
            current.wait_stream(stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self()
            self.cuda_graph, self.graph_output = graph, output
        return self

    @torch.inference_mode()
    def replay(self):
        if self.cuda_graph is None:
            raise ValueError('Capture before replay')
        self.cuda_graph.replay()
        return self.graph_output
