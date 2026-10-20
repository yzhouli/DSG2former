"""Freeze tree configurations using validation alone, including neural controls."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.np50_tree_readout.common import *


def action_metrics(y, p):
    return dict(accuracy=float(np.mean(p.argmax(1) == y)),
                ce=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()))


def count_metrics(y, p):
    return dict(mse=float(np.mean((y-p)**2)),
                msle=float(np.mean((np.log1p(y)-np.log1p(p))**2)))


def select(name):
    _, _, out = paths(name)
    caches = {s: torch.load(out / f'seed_{s}/forecast_inputs.pt', map_location='cpu', weights_only=False) for s in SEEDS}
    actions = {s: {c: dict(np.load(out / f'seed_{s}/action/{c}/validation.npz')) for c in ACTION_CONFIGS} for s in SEEDS}
    ids = [t['thread_id'] for t in map(__import__('json').loads, (ROOT / SPEC[name]['root'] / 'threads.jsonl').open()) if t['split'] == 'validation']
    # Source-file order is chronological under the fixed information-level split.
    assert len(ids) == len(set(ids)) and len(ids) >= 2
    first = set(ids[:len(ids)//2]); result = {}
    for model in MODELS:
        candidates = []
        for config in ACTION_CONFIGS:
            for weight in WEIGHTS:
                runs = []
                for s in SEEDS:
                    y = actions[s][config]['target']; p = caches[s][model]['validation']['action_probability']
                    assert p.shape == (len(y), 3)
                    runs.append(action_metrics(y, (1-weight)*p + weight*actions[s][config]['probability']))
                candidates.append(dict(config=config, tree_weight=weight, **{k: float(np.mean([r[k] for r in runs])) for k in runs[0]}))
        chosen = min(candidates, key=lambda r: (-r['accuracy'], r['ce'], r['tree_weight'], r['config']))
        count_candidates = []
        masks = {}
        baseline = {}
        for s in SEEDS:
            va = caches[s][model]['validation']; mask = np.isin(va['thread_id'], list(first))
            assert mask.any() and (~mask).any()
            masks[s] = [mask, ~mask]
            baseline[s] = [count_metrics(va['target'][m], va['original'][m]) for m in masks[s]]
        denominators = [{k: float(np.mean([baseline[s][i][k] for s in SEEDS])) for k in ['mse', 'msle']} for i in range(2)]
        for reference in ['original', 'current']:
            for config in COUNT_CONFIGS:
                trees = {s: np.load(out / f'seed_{s}' / model / config / 'validation.npz')['prediction'] for s in SEEDS}
                for weight in WEIGHTS:
                    cohorts = [[], []]; full = []
                    for s in SEEDS:
                        va = caches[s][model]['validation']; p = (1-weight)*va[reference] + weight*trees[s]
                        full.append(count_metrics(va['target'], p))
                        for i, mask in enumerate(masks[s]): cohorts[i].append(count_metrics(va['target'][mask], p[mask]))
                    means = [{k: float(np.mean([r[k] for r in cohort])) for k in ['mse', 'msle']} for cohort in cohorts]
                    ratios = [v[k]/max(denominators[i][k], 1e-12) for i, v in enumerate(means) for k in ['mse', 'msle']]
                    count_candidates.append(dict(reference=reference, config=config, tree_weight=weight,
                        worst_ratio=max(ratios), sum_ratios=sum(ratios), cohorts=means,
                        **{k: float(np.mean([r[k] for r in full])) for k in ['mse', 'msle']}))
        count_chosen = min(count_candidates, key=lambda r: (r['worst_ratio'], r['sum_ratios'], r['tree_weight'], r['reference'], r['config']))
        result[model] = dict(action=chosen, count=count_chosen,
            action_candidates=candidates, count_candidates=count_candidates,
            count_validation_control_by_chronological_half=denominators)
    write(out / 'selection.json', dict(complete=True, test_used=False, models=result,
          action_selection='Mean three-seed validation Accuracy, then CE; zero tree weight included',
          count_selection='Minimize worst normalized MSE/MSLE across early/late validation-source halves, then ratio sum; original/current neural controls included',
          seeds=SEEDS, class_averaging=None))
    print({m: {k: result[m][k] for k in ['action', 'count']} for m in MODELS}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', choices=list(SPEC), required=True)
    torch.set_num_threads(1); verify_plan(); select(p.parse_args().dataset)
