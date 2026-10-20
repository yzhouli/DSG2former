"""Frozen tree inference; neural inputs have separately passed full encoding replay."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import torch
from scripts.np50_tree_readout.common import *
from scripts.np50_tree_readout.prepare import forecast_features
from scripts.np50_tree_readout.train import count_prediction

NEURAL_NAMES = {'SDSformer': 'SDSformer（更新宏观头）', 'DSG2former': 'DSG2former+（时间与宏观优化）'}


def inputs(name):
    work, parent, out = paths(name); files = [out / 'selection.json', out / 'features.pt', work / 'results/neural/common.pt', parent / 'summary.json']
    selection = read(out / 'selection.json')
    for seed in SEEDS:
        for model in MODELS:
            folder = out / f'seed_{seed}'; config = selection['models'][model]
            files += [folder / 'action' / config['action']['config'] / 'model.joblib', folder / model / config['count']['config'] / 'model.joblib']
            for base in [parent / 'published' / NEURAL_NAMES[model], ROOT / 'results/np50_comparison' / SPEC[name]['slug'] / model]:
                files += [base / f'seed_{seed}' / n for n in ['action_predictions.npz', 'popularity_predictions.npz']]
    return sorted(set(files))


def freeze(name):
    _, _, out = paths(name)
    proof = dict(frozen_before_tree_test=True, plan_sha256=sha(OUT / 'plan.json'),
        source_and_input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs(name)},
        neural_test_encoding_replay='Previously independently verified for all branches and seeds',
        benchmark_previously_inspected=True)
    path = out / 'inference_plan.json'
    if path.exists(): assert read(path) == proof
    else: write(path, proof)


def infer(name, seed, replay=False):
    work, parent, out = paths(name); proof = read(out / 'inference_plan.json')
    assert sha(OUT / 'plan.json') == proof['plan_sha256']
    for p, h in proof['source_and_input_sha256'].items(): assert sha(ROOT / p) == h, p
    data = torch.load(out / 'features.pt', map_location='cpu', weights_only=False)['test']
    common = torch.load(work / 'results/neural/common.pt', map_location='cpu', weights_only=False)['test']
    selection = read(out / 'selection.json')['models']
    for model in MODELS:
        base = parent / 'published' / NEURAL_NAMES[model] / f'seed_{seed}'
        with np.load(base / 'action_predictions.npz') as z: action = dict(z)
        with np.load(base / 'popularity_predictions.npz') as z: macro = dict(z)
        np.testing.assert_array_equal(data['target'], action['target'])
        for k in ['target', 'thread_id', 'window', 'cutoff', 'end']: np.testing.assert_array_equal(macro[k], common[k])
        with np.load(ROOT / 'results/np50_comparison' / SPEC[name]['slug'] / model / f'seed_{seed}/popularity_predictions.npz') as z: original = z['prediction']
        a = selection[model]['action']; folder = out / f'seed_{seed}'
        tree = joblib.load(folder / 'action' / a['config'] / 'model.joblib')
        tree_probability = tree.predict_proba(data['x'])
        probability = (1-a['tree_weight'])*action['probability'] + a['tree_weight']*tree_probability
        c = selection[model]['count']; tree = joblib.load(folder / model / c['config'] / 'model.joblib')
        prediction = count_prediction(tree, forecast_features(common, original, macro['prediction']), COUNT_CONFIGS[c['config']]['loss'])
        reference = original if c['reference'] == 'original' else macro['prediction']
        prediction = (1-c['tree_weight'])*reference + c['tree_weight']*prediction
        variants = {
            model+' + tree': (probability, prediction),
            model+' + action-tree': (probability, macro['prediction']),
            model+' selected raw-tree': (tree_probability, macro['prediction']),
        }
        for title, (p, count) in variants.items():
            dest = out / title / f'seed_{seed}' / ('exact_replay' if replay else 'test_predictions'); dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'action_predictions.npz', **dict(action, probability=p))
            np.savez_compressed(dest / 'popularity_predictions.npz', **dict(macro, prediction=count))
            write(dest / 'completion.json', dict(complete=True, inference_plan_sha256=sha(out / 'inference_plan.json'),
                frozen_neural_inputs=True, tree_predictions_recomputed=True, raw_tree_is_not_a_neural_architecture=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['freeze', 'test', 'replay'])
    p.add_argument('--dataset', choices=list(SPEC), required=True); p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(); torch.set_num_threads(1); verify_plan()
    if a.mode == 'freeze': freeze(a.dataset)
    else: infer(a.dataset, a.seed, a.mode == 'replay')
