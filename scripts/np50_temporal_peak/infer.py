"""Frozen checkpoint inference; replay re-encodes all required observed graphs."""
import argparse
import gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from dsg2former.sds_aligned import align_graph_positions
from scripts.np50_temporal_peak.common import *
from scripts.np50_temporal_peak.encode import encode, probability, pool, META
from scripts.np50_temporal_peak.train import load_model, memory_predict
from scripts.weibo_comparison import neural


def freeze(name):
    work, root = paths(name); policy = read(root / 'micro_selection.json')['policy']
    macro = read(root / 'macro_selection.json')
    files = [root / 'micro_selection.json', root / 'macro_selection.json']
    for b in BACKBONES:
        for seed in SEEDS:
            parent = work / 'results/neural' / b / f'seed_{seed}'
            files.append(parent / 'spatial_model.pt')
            files.append(parent / 'temporal_model.pt' if policy == 'chain' else root / b / f'seed_{seed}' / policy / 'temporal_model.pt')
            for variant in ['spatial', 'temporal']:
                config = macro[variant]['config']
                folder = parent / variant if config == 'legacy' else root / 'counts' / config / variant / b / f'seed_{seed}'
                for n in ['model.pt', 'memory_gate.pt']:
                    files.append(folder / n)
    proof = dict(frozen_before_new_test=True, plan_sha256=sha(OUT / 'plan.json'),
        input_sha256=sha(work / 'data/pheme_model_improvement/graphs.pt'),
        checkpoint_sha256={str(p.relative_to(ROOT)): sha(p) for p in files})
    path = root / 'inference_plan.json'
    if path.exists(): assert read(path) == proof
    else: write(path, proof)


def infer(args):
    work, root = paths(args.dataset); proof = read(root / 'inference_plan.json')
    assert proof['plan_sha256'] == sha(OUT / 'plan.json')
    for p, h in proof['checkpoint_sha256'].items(): assert sha(ROOT / p) == h, p
    assert sha(work / 'data/pheme_model_improvement/graphs.pt') == proof['input_sha256']
    parent = work / 'results/neural' / args.backbone / f'seed_{args.seed}'
    policy = read(root / 'micro_selection.json')['policy']
    checkpoint = parent / 'temporal_model.pt' if policy == 'chain' else root / args.backbone / f'seed_{args.seed}' / policy / 'temporal_model.pt'
    model, _ = load_model(checkpoint)
    data = torch.load(work / 'data/pheme_model_improvement/graphs.pt', weights_only=False, map_location='cpu')
    align_graph_positions(data)
    if policy == 'chain':
        d = neural.cached(model, data, 'test', args.device)
        a = d['action']; states = d['state']; p = neural.action_probability(model, a, 'temporal')
    else:
        d = encode(model, data, ['test'], args.device)['test']
        a = d['action']; p = probability(model, a, policy, args.device)
        states = {v: pool(model, d['macro'], policy, v == 'temporal', args.device) for v in ['spatial', 'temporal']}
    del data; gc.collect()
    common = torch.load(work / 'results/neural/common.pt', weights_only=False, map_location='cpu')['test']
    for k in META: np.testing.assert_array_equal(d['macro'][k], common[k])
    selection = read(root / 'macro_selection.json')
    for v in ['spatial', 'temporal']:
        config = selection[v]['config']
        folder = parent / v if config == 'legacy' else root / 'counts' / config / v / args.backbone / f'seed_{args.seed}'
        forecast = memory_predict(folder, common, states[v], peak=config != 'legacy')
        dest = root / args.backbone / f'seed_{args.seed}' / v / ('test_predictions' if args.mode == 'test' else 'exact_replay')
        dest.mkdir(parents=True, exist_ok=True)
        if v == 'spatial':
            with np.load(parent / 'spatial/test_predictions/action_predictions.npz') as old:
                action = {k: old[k] for k in old.files}
        else:
            action = dict(probability=p, **{k: (a[k].numpy() if k == 'target' else a[k]) for k in META})
        np.savez_compressed(dest / 'action_predictions.npz', **action)
        np.savez_compressed(dest / 'popularity_predictions.npz', prediction=forecast,
                            **{k: common[k] for k in [*META, 'previous_new_users']})
        write(dest / 'completion.json', dict(complete=True, full_spatial_encoding_recomputed=True,
              micro_policy=policy, macro_config=config, inference_plan_sha256=sha(root / 'inference_plan.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['freeze', 'test', 'replay'])
    p.add_argument('--dataset', choices=list(SPEC), required=True)
    p.add_argument('--backbone', choices=BACKBONES); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu'); args = p.parse_args()
    torch.set_num_threads(1); verify_plan()
    if args.mode == 'freeze': freeze(args.dataset)
    else: infer(args)
