"""Causal numeric inputs, from the same unchanged184feature prepared graphs."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from dsg2former.neural_popularity import observed_history
R = Path('results/pheme_macro_residual_study')
CACHE = Path('results/pheme_neural_macro_optimized/cache')
DATA = Path('data/pheme_model_improvement/graphs.pt')
BACKBONES = ['d24_regular', 'd96_regular', 'd48_strong_reg']
SEEDS = [42, 2025, 2026]
META = ['target', 'thread_id', 'window', 'cutoff', 'end', 'previous_new_users']
read = lambda p: json.loads(Path(p).read_text())
def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def from_graphs(data, split, scaler):
    out = {k: [] for k in META+['ics', 'profile', 'observed', 'sequence']}
    for t in data:
        if t['split'] != split:
            continue
        for s in t['macro_snapshots']:
            n = s['user_count']
            out['ics'].append(scaler.transform(s['raw_x'][:n, 32:35]).mean(0))
            out['profile'].append(s['raw_x'][:n, :16].mean(0))
            out['observed'].append(np.r_[s['history_summary'][:4], n])
            out['sequence'].append(observed_history(s)[0])
            for k, v in [('target', s['popularity_target']), ('thread_id', t['thread_id']),
                         ('window', s['window']), ('cutoff', s['cutoff']), ('end', s['end']),
                         ('previous_new_users', s['history_summary'][1])]:
                out[k].append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def scaler_from_resources(resources):
    scaler = StandardScaler(); scaler.mean_ = resources['mean']; scaler.scale_ = resources['scale']
    return scaler


def prepare():
    path = R/'development_common.pt'
    if path.exists():
        return
    data = torch.load(DATA, weights_only=False, map_location='cpu')
    # Same prior-evidence coordinates already present in the184micro inputs.
    training = np.concatenate([s['raw_x'][s['sample_nodes'], 32:35]
        for t in data if t['split'] == 'train' for s in t['snapshots'] if len(s['target'])])
    scaler = StandardScaler().fit(training)
    out = {s: from_graphs(data, s, scaler) for s in ['train', 'validation']}
    out['scaler'] = dict(mean=scaler.mean_, scale=scaler.scale_)
    torch.save(out, path)


def partition(common, fold):
    tr = common['train']; ids = list(dict.fromkeys(tr['thread_id'].tolist()))
    if fold == 2:
        return tr, common['validation'], np.ones(len(tr['target']), bool), None
    left, right = [(0.6, .8), (.8, 1.)][fold]
    fit_ids = set(ids[:int(len(ids)*left)]); valid_ids = set(ids[int(len(ids)*left):int(len(ids)*right)])
    fit = np.array([t in fit_ids for t in tr['thread_id']]); valid = np.array([t in valid_ids for t in tr['thread_id']])
    return {k: v[fit] for k, v in tr.items()}, {k: v[valid] for k, v in tr.items()}, fit, valid


def micro_state(data, variant, fold):
    # Keep the final Transformer coordinates plus class probabilities/count;
    # discard duplicated raw/source/product coordinates from the old4d summary.
    a = data['train']['state'][variant]; b = data['validation']['state'][variant]
    dim = (a.shape[1]-4)//4
    a = np.c_[a[:, :dim], a[:, -4:]]; b = np.c_[b[:, :dim], b[:, -4:]]
    if fold == 2:
        return a, b
    ids = list(dict.fromkeys(data['train']['macro']['thread_id'].tolist()))
    left, right = [(0.6, .8), (.8, 1.)][fold]
    fit_ids = set(ids[:int(len(ids)*left)]); valid_ids = set(ids[int(len(ids)*left):int(len(ids)*right)])
    tids = data['train']['macro']['thread_id']
    return a[np.array([t in fit_ids for t in tids])], a[np.array([t in valid_ids for t in tids])]
