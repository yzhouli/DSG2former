"""Independent label-subset, rank-AUC and count-metric verification of final arrays."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.np50_comparison.common import ROOT, OUT, SPEC, SEEDS, METHODS, sha, write


def binary_auc(labels, scores):
    order = np.argsort(scores, kind='mergesort')
    values = scores[order]; y = labels[order].astype(np.int64)
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(values)])
    ranks = starts + (sizes + 1) / 2
    positives = np.add.reduceat(y, starts); n = int(y.sum()); m = len(y) - n
    return float((np.dot(ranks, positives) - n * (n + 1) / 2) / (n * m))


def references(name):
    spec = SPEC[name]
    if name == 'PHEME':
        folder = ROOT / 'results/pheme_macro_memory_study/temporal/seed_42/test_predictions'
    else:
        old = 'weibo_v1_v2' if name == 'Weibo-V1' else 'weibo_public_v2'
        folder = ROOT / 'results/weibo_comparison' / old / 'DSG2former/seed_42'
    old = np.load(folder / 'action_predictions.npz')
    selected = np.load(ROOT / spec['raw'] / 'selection.npz')
    lookup = {str(t): (selected['kept_parent_rows'][a:b], int(n)) for t, a, b, n in zip(
        selected['thread_ids'], selected['offsets'][:-1], selected['offsets'][1:], selected['parent_row_counts'])}
    labels, ids = [], []
    for tid in dict.fromkeys(old['thread_id'].tolist()):
        y = old['target'][old['thread_id'] == tid]; ix, n = lookup[tid]
        assert len(y) == n
        labels.extend(y[ix].tolist()); ids.extend([tid] * len(ix))
    macro = np.load(folder / 'popularity_predictions.npz')
    # Old ground-truth counts have already been checked against raw records;
    # changing sampled negatives must preserve every count and time boundary.
    return np.asarray(labels), np.asarray(ids), {k: macro[k] for k in ['target', 'thread_id', 'window', 'cutoff', 'end']}


def verify(name):
    folder = OUT / SPEC[name]['slug']; summary = json.loads((folder / 'summary.json').read_text())
    y, ids, count_truth = references(name); checked = 0
    for method in METHODS:
        entry = summary['rows'][method]; runs = []
        for seed in SEEDS:
            base = folder / method / f'seed_{seed}'
            a = np.load(base / 'action_predictions.npz'); m = np.load(base / 'popularity_predictions.npz')
            np.testing.assert_array_equal(a['target'], y); np.testing.assert_array_equal(a['thread_id'], ids)
            for k, v in count_truth.items(): np.testing.assert_array_equal(m[k], v)
            p = a['probability']; pred = p.argmax(1)
            matrix = np.zeros((3, 3), dtype=np.int64); np.add.at(matrix, (y, pred), 1)
            values = dict(accuracy=float(np.trace(matrix) / len(y)))
            for c in range(3):
                tp = int(matrix[c, c]); fn = int(matrix[c].sum()) - tp; fp = int(matrix[:, c].sum()) - tp
                values[f'recall_{c}'] = tp / (tp + fn)
                values[f'f1_{c}'] = 2 * tp / (2 * tp + fp + fn)
                values[f'auc_{c}'] = binary_auc(y == c, p[:, c])
            actual, estimate = m['target'].astype(float), m['prediction'].astype(float)
            values['mse'] = float(np.mean(np.square(estimate-actual)))
            values['rmse'] = float(np.sqrt(values['mse']))
            values['msle'] = float(np.mean(np.square(np.log1p(estimate)-np.log1p(actual))))
            expected = next(r for r in entry['per_seed'] if r['seed'] == seed)
            np.testing.assert_array_equal(matrix, expected['confusion_matrix'])
            for k in values: np.testing.assert_allclose(values[k], expected[k], rtol=0, atol=1e-12)
            runs.append(values); checked += 1
        for k in values:
            np.testing.assert_allclose(np.mean([r[k] for r in runs]), entry['mean'][k], rtol=0, atol=1e-12)
            np.testing.assert_allclose(np.std([r[k] for r in runs], ddof=1), entry['sample_sd'][k], rtol=0, atol=1e-12)
    result = dict(passed=True, dataset=name, published_model_seed_pairs_verified=checked,
                  action_rows=len(y), macro_windows=len(count_truth['target']),
                  labels_independently_subset_from_original_frozen_truth=True,
                  macro_truth_identical_to_independently_audited_original_counts=True,
                  class_averaging=None, auc_independent_tie_aware_ranks=True,
                  all_metrics_and_seed_mean_sample_sd_recomputed=True, summary_sha256=sha(folder / 'summary.json'))
    write(folder / 'independent_verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', choices=list(SPEC) + ['all'], default='all')
    args = p.parse_args()
    for name in SPEC if args.dataset == 'all' else [args.dataset]: verify(name)
