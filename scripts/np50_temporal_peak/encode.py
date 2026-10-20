"""Chronological cross-information state retrieval, without future labels."""
from collections import defaultdict
import gc
import numpy as np
import torch

from dsg2former.causal_user_state import VisibleUserStates
from dsg2former.neural_popularity import final_micro_state
from scripts.weibo_comparison.neural import spatial_outputs

STAGE = {'train': 0, 'validation': 1, 'test': 2}
TENSORS = ['current', 'logits', 'previous', 'mask', 'age', 'bootstrap', 'has_history']
META = ['target', 'thread_id', 'window', 'cutoff', 'end']


def effective(a, policy, device='cpu'):
    current, logits, previous, mask, age = [a[k].to(device) for k in TENSORS[:5]]
    if policy == 'global_bootstrap':
        fallback = ~mask & a['has_history'].to(device)
        previous = torch.where(fallback[:, None], a['bootstrap'].to(device), previous)
        # A feature initializer has no measured previous-state timestamp.
        # Its age input is explicitly zero, not an invented observation time.
        age = torch.where(fallback, torch.zeros_like(age), age)
        mask = mask | fallback
    return current, logits, previous, mask, age


@torch.no_grad()
def probability(model, action, policy, device='cpu'):
    model = model.to(device).eval()
    values = effective(action, policy, device)
    return torch.cat([model.action_cached(*[x[i:i+8192] for x in values]).softmax(-1).cpu()
                      for i in range(0, len(values[0]), 8192)]).numpy()


@torch.no_grad()
def pool(model, macro, policy, temporal, device='cpu'):
    model = model.to(device).eval(); nodes = macro['nodes']
    vals = effective(nodes, policy, device); current, logits, previous, mask, age = vals
    d = model.encoder.dim; parts = []
    for i in range(0, len(current), 8192):
        z, score, old, seen, dt = [v[i:i+8192] for v in vals]
        if temporal:
            z = final_micro_state(model, z, old, seen, dt, True)
            score = model.action_cached(*[v[i:i+8192] for v in vals])
        parts.append(torch.cat((z[:, :d], score.softmax(-1)), -1).cpu().numpy())
    x = np.concatenate(parts); offsets = macro['offsets']; sizes = np.diff(offsets)
    means = np.add.reduceat(x.astype(np.float64), offsets[:-1], axis=0)/sizes[:, None]
    return np.c_[means, np.log1p(sizes)].astype(np.float32)


@torch.no_grad()
def encode(model, data, requested, device):
    """Earlier source stages may contribute observed states, later stages cannot.

    Only candidate-independent macro graphs write the cache. All reads at a
    timestamp precede all writes at that timestamp. Labels and state_update_users
    are not consulted for cache writes. Output ordering stays identical to parent.
    """
    maximum = max(STAGE[s] for s in requested)
    flat = [(ti, t, branch, si, s) for ti, t in enumerate(data) if STAGE[t['split']] <= maximum
            for branch in ['snapshots', 'macro_snapshots'] for si, s in enumerate(t[branch])
            if branch == 'macro_snapshots' or t['split'] in requested]
    flat.sort(key=lambda item: (item[4]['cutoff'], item[0], item[2], item[3]))
    store = VisibleUserStates(); records = defaultdict(list)
    model = model.to(device).eval(); checked = written = 0
    for start in range(0, len(flat), 64):
        batch = flat[start:start+64]
        logits, state = spatial_outputs(model, [(t, s) for _, t, _, _, s in batch], device)
        # Reuse the frozen learned feature projection, with only the user's
        # observed 32-feature profile. All graph/source/global columns are zero
        # in standardized coordinates. This is a declared proxy initializer,
        # never claimed to be an actually observed earlier hidden state.
        features = torch.zeros((sum(s['user_count'] for *_, s in batch), 184), device=device)
        cursor = 0
        for *_, s in batch:
            n = s['user_count']; features[cursor:cursor+n, :32] = torch.as_tensor(s['x'][:n, :32], device=device)
            cursor += n
        raw = model.feature_projection(features)
        boot = torch.cat((raw, raw, torch.zeros_like(raw), torch.zeros_like(raw)), -1).cpu()
        logits, state = logits.cpu(), state.cpu(); cursor = 0
        for j, (ti, t, branch, si, s) in enumerate(batch):
            cutoff = float(s['cutoff']); stage = STAGE[t['split']]; store.advance(cutoff)
            n = s['user_count']; current = state[j, :n].clone(); score = logits[j, :n].clone()
            previous = torch.zeros_like(current); mask = torch.zeros(n, dtype=torch.bool); age = torch.zeros(n)
            has = torch.as_tensor(s['raw_x'][:n, 0] > 0)
            for k, uid in enumerate(s['node_ids']):
                hit = store.read(uid, stage)
                if hit is not None:
                    when, value = hit; previous[k] = value; mask[k] = True; age[k] = (cutoff-when)/3600
                    checked += 1
            values = dict(current=current, logits=score, previous=previous, mask=mask, age=age,
                          bootstrap=boot[cursor:cursor+n].clone(), has_history=has)
            cursor += n
            if t['split'] in requested:
                if branch == 'snapshots':
                    ix = torch.as_tensor(s['sample_nodes'], dtype=torch.long)
                    values = {k: v[ix] for k, v in values.items()}
                values.update(target=np.asarray(s['target']) if branch == 'snapshots' else s['popularity_target'],
                              thread_id=t['thread_id'], window=s['window'], cutoff=cutoff, end=s['end'])
                records[t['split'], branch].append((ti, si, values))
            if branch == 'macro_snapshots':
                for k, uid in enumerate(s['node_ids']):
                    if has[k]:
                        store.observe(uid, stage, current[k].clone()); written += 1
        if start % 8192 == 0:
            print(dict(stage='encode', graphs=start, total=len(flat), global_hits=checked, observed_writes=written), flush=True)
    del store, state, logits, boot; gc.collect()
    result = {}
    for split in requested:
        result[split] = {}
        for branch, label in [('snapshots', 'action'), ('macro_snapshots', 'macro')]:
            rows = [v for _, _, v in sorted(records.pop((split, branch)), key=lambda v: (v[0], v[1]))]
            tensors = {k: torch.cat([v[k] for v in rows]) for k in TENSORS}
            if branch == 'snapshots':
                tensors['target'] = torch.as_tensor(np.concatenate([v['target'] for v in rows]), dtype=torch.long)
                tensors.update({k: np.concatenate([np.repeat(v[k], len(v['current'])) for v in rows]) for k in META[1:]})
                result[split][label] = tensors
            else:
                result[split][label] = dict(nodes=tensors, offsets=np.r_[0, np.cumsum([len(v['current']) for v in rows])],
                    **{k: np.asarray([v[k] for v in rows]) for k in META})
    result['causal_audit'] = dict(strict_earlier_reads=checked, observed_macro_writes=written,
        candidate_state_writes=0, future_label_state_writes=0, same_time_read_before_write=True,
        source_stage_restriction=True, historical_feature_bootstrap_declared=True)
    return result
