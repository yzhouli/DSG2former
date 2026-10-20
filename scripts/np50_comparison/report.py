"""Score new frozen predictions, verify exact replay and rebuild raw count truth."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scripts.np50_comparison.common import ROOT, OUT, SPEC, SEEDS, BACKBONES, METHODS, sha, write

CLASSES = ['传播谣言', '辟谣', '不参与']


def truth(name):
    spec = SPEC[name]; root = ROOT / spec['root']; raw = ROOT / spec['raw']
    actions = defaultdict(list); macro = defaultdict(list)
    flat = {t['thread_id']: t for t in map(json.loads, (raw / 'candidate_rows.jsonl').open())} if name == 'PHEME' else None
    with (root / 'threads.jsonl').open() as f:
        for line in f:
            t = json.loads(line)
            if t['split'] != 'test': continue
            tid = t['thread_id']; start = t['source']['timestamp']
            rows = flat[tid]['rows'] if flat is not None else json.loads((raw / 'rows' / (tid + '.json')).read_text())['rows']
            actions['target'].extend(r['label'] for r in rows)
            actions['thread_id'].extend([tid] * len(rows))
            groups = np.ceil((np.asarray([r['timestamp'] for r in t['reactions']]) - start)/3600).astype(int) - 1
            windows = max(1, int(groups.max()) + 1) if len(groups) else 1
            if name != 'PHEME': windows = 72
            by_window = defaultdict(set)
            for r, w in zip(t['reactions'], groups):
                if r['id'] != t['source']['id'] and r['timestamp'] > start:
                    by_window[int(w)].add(r['uid'])
            known = {t['source']['uid']}
            for w in range(windows):
                active = by_window[w]
                for k, v in [('target', len(active - known)), ('thread_id', tid), ('window', w),
                             ('cutoff', start + w * 3600), ('end', start + (w + 1) * 3600)]: macro[k].append(v)
                known |= active
    return dict(action={k: np.asarray(v) for k, v in actions.items()},
                macro={k: np.asarray(v) for k, v in macro.items()})


def checked(folder, expected):
    arrays = []
    for kind, key in [('action', 'action'), ('popularity', 'macro')]:
        path = folder / 'test_predictions' / (kind + '_predictions.npz')
        replay_path = folder / 'exact_replay' / path.name
        with np.load(path) as a, np.load(replay_path) as replay:
            assert a.files == replay.files
            for k in a.files: np.testing.assert_array_equal(a[k], replay[k])
            for k, v in expected[key].items(): np.testing.assert_array_equal(a[k], v)
            arrays.append({k: a[k] for k in a.files})
    return arrays


def score(action, macro):
    y, p = action['target'], action['probability']
    assert y.ndim == 1 and p.shape == (len(y), 3) and np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=2e-6)
    pred = p.argmax(1); cm = np.bincount(3*y + pred, minlength=9).reshape(3, 3)
    tp = cm.diagonal(); recall = tp/cm.sum(1); f1 = 2*tp/(cm.sum(1)+cm.sum(0))
    _, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    np.testing.assert_allclose(recall, r, atol=1e-12); np.testing.assert_allclose(f1, f, atol=1e-12)
    target, prediction = macro['target'].astype(float), macro['prediction'].astype(float)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    mse = float(np.mean((target-prediction)**2))
    values = dict(accuracy=float(np.mean(pred == y)), mse=mse, rmse=float(np.sqrt(mse)),
                  msle=float(np.mean((np.log1p(target)-np.log1p(prediction))**2)))
    for c in range(3):
        values[f'recall_{c}'] = float(recall[c]); values[f'f1_{c}'] = float(f1[c])
        values[f'auc_{c}'] = float(roc_auc_score(y == c, p[:, c]))
    return values, cm.tolist()


def report(name):
    spec = SPEC[name]; root = ROOT / spec['root']; work = root / 'runtime'; out = OUT / spec['slug']
    out.mkdir(parents=True, exist_ok=True); expected = truth(name)
    selection = json.loads((work / 'results/baselines/selection.json').read_text())['selected_variants']
    result = {}; replays = 0
    for method in METHODS:
        runs = []
        for seed in SEEDS:
            if method in ['SDSformer', 'DSG2former']:
                variant = 'spatial' if method == 'SDSformer' else 'temporal'
                pieces = [checked(work / 'results/neural' / b / f'seed_{seed}' / variant, expected) for b in BACKBONES]
                action = dict(pieces[0][0], probability=sum(a['probability'].astype(float)/3 for a, _ in pieces))
                macro = dict(pieces[0][1], prediction=sum(m['prediction'].astype(float)/3 for _, m in pieces))
                replays += 3
            else:
                folder = work / 'results/trgcn' / f'seed_{seed}' if method == 'TRGCN' else work / 'results/baselines' / method / selection[method] / f'seed_{seed}'
                action, macro = checked(folder, expected); replays += 1
            values, cm = score(action, macro)
            dest = out / method / f'seed_{seed}'; dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'action_predictions.npz', **action)
            np.savez_compressed(dest / 'popularity_predictions.npz', **macro)
            runs.append(dict(seed=seed, **values, confusion_matrix=cm))
        keys = list(values)
        result[method] = dict(per_seed=runs, mean={k: float(np.mean([r[k] for r in runs])) for k in keys},
                              sample_sd={k: float(np.std([r[k] for r in runs], ddof=1)) for k in keys})
    assert replays == 36  # 6baselines*3seeds + 2variants*3backbones*3seeds.
    # Each checked bundle contains an action and a popularity prediction file.
    document = dict(complete=True, dataset=name, sampling='NP50', seeds=SEEDS, rows=result,
                    class_order=CLASSES, class_averaging=None, test_action_rows=len(expected['action']['target']),
                    test_macro_windows=len(expected['macro']['target']), verified_replay_bundles=replays,
                    raw_macro_targets_independently_reconstructed=True, action_truth_from_new_selected_rows=True,
                    original_duplicates_preserved=True, source_script_sha256=sha(__file__),
                    dataset_manifest_sha256=sha(ROOT / spec['raw'] / 'dataset_manifest.json'))
    write(out / 'summary.json', document); merged()
    print(json.dumps(dict(complete=True, dataset=name, test_rows=document['test_action_rows'], replay_bundles=replays)), flush=True)


def merged():
    lines = ['# NP50：三个新数据集重新训练的性能', '',
             '每个种子先合并该数据集全部测试预测(N,3)与标签(N,)，逐行argmax后统计混淆矩阵。'
             'Recall/F1逐类计算；AUC也分别以每类概率对其余类别计算，不做类别平均。'
             '流行度误差合并全部测试窗口计分。表中为42、2025、2026种子的均值±样本标准差。', '',
             '沿用先前验证选定的模型超参数和184维匹配特征、两个论文对齐Block、三分支等权纯神经读出。'
             '所有任务权重及依赖标签的影响回归重新训练；历史上下文不变的无监督资源固定。'
             '立场为既有弱标签，负样本保留原数量的一半。']
    csv_rows = []; completed = []
    for name, spec in SPEC.items():
        path = OUT / spec['slug'] / 'summary.json'
        if not path.exists(): continue
        summary = json.loads(path.read_text()); completed.append(name)
        lines += ['', '## ' + name, '', f"测试行动样本{summary['test_action_rows']:,}条，流行度窗口{summary['test_macro_windows']:,}个。", '',
                  '|模型|Accuracy↑|谣言 Recall↑|辟谣 Recall↑|不参与 Recall↑|谣言 F1↑|辟谣 F1↑|不参与 F1↑|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method, d in summary['rows'].items():
            keys = ['accuracy', 'recall_0', 'recall_1', 'recall_2', 'f1_0', 'f1_1', 'f1_2']
            lines.append('|' + method + '|' + '|'.join(f"{d['mean'][k]:.6f} ± {d['sample_sd'][k]:.6f}" for k in keys) + '|')
            csv_rows.append(dict(dataset=name, model=method, **{k+'_'+suffix: d['mean' if suffix=='mean' else 'sample_sd'][k]
                            for k in d['mean'] for suffix in ['mean', 'sample_sd']}))
        lines += ['', '|模型|谣言 AUC↑|辟谣 AUC↑|不参与 AUC↑|MSE↓|RMSE↓|MSLE↓|', '|---|---:|---:|---:|---:|---:|---:|']
        for method, d in summary['rows'].items():
            keys = ['auc_0', 'auc_1', 'auc_2', 'mse', 'rmse', 'msle']
            lines.append('|' + method + '|' + '|'.join(f"{d['mean'][k]:.6f} ± {d['sample_sd'][k]:.6f}" for k in keys) + '|')
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if csv_rows:
        with (OUT / 'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    write(OUT / 'completion.json', dict(complete=len(completed)==len(SPEC), datasets=completed, models=METHODS, seeds=SEEDS))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', choices=list(SPEC), required=True)
    report(p.parse_args().dataset)
