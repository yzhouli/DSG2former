"""Pooled per-class metrics and independent rank-AUC verification."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.np50_temporal_peak.common import *
from scripts.np50_comparison.report import score, checked, CLASSES
from scripts.np50_comparison.verify import references, binary_auc

VARIANTS = {'SDSformer（更新宏观头）': 'spatial', 'DSG2former+（时间与宏观优化）': 'temporal'}


def report(name):
    _, root = paths(name); y, ids, macro_truth = references(name)
    expected = dict(action=dict(target=y, thread_id=ids), macro=macro_truth)
    original = read(ROOT / 'results/np50_comparison' / SPEC[name]['slug'] / 'summary.json')
    results = dict(original['rows']); checked_bundles = 0
    for method, variant in VARIANTS.items():
        runs = []
        for seed in SEEDS:
            pieces = [checked(root / b / f'seed_{seed}' / variant, expected) for b in BACKBONES]
            checked_bundles += len(pieces)
            action = dict(pieces[0][0], probability=sum(a['probability'].astype(float)/3 for a, _ in pieces))
            macro = dict(pieces[0][1], prediction=sum(m['prediction'].astype(float)/3 for _, m in pieces))
            values, matrix = score(action, macro)
            # Separate tie-aware rank implementation verifies every reported AUC.
            for c in range(3):
                np.testing.assert_allclose(values[f'auc_{c}'], binary_auc(y == c, action['probability'][:, c]), atol=1e-12, rtol=0)
            dest = root / 'published' / method / f'seed_{seed}'; dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'action_predictions.npz', **action)
            np.savez_compressed(dest / 'popularity_predictions.npz', **macro)
            runs.append(dict(seed=seed, **values, confusion_matrix=matrix))
        results[method] = dict(per_seed=runs,
            mean={k: float(np.mean([r[k] for r in runs])) for k in values},
            sample_sd={k: float(np.std([r[k] for r in runs], ddof=1)) for k in values})
    document = dict(complete=True, dataset=name, rows=results, seeds=SEEDS,
        previous_result_keys=['SDSformer', 'DSG2former'], optimized_result_keys=list(VARIANTS),
        test_action_rows=len(y), test_macro_windows=len(macro_truth['target']),
        independent_labels_counts_and_rank_auc_checked=True, exact_full_replay_bundles=checked_bundles,
        micro_selection=read(root / 'micro_selection.json'), macro_selection=read(root / 'macro_selection.json'),
        previous_summary_sha256=sha(ROOT / 'results/np50_comparison' / SPEC[name]['slug'] / 'summary.json'),
        class_averaging=None, prior_test_benchmark_already_inspected=True)
    assert checked_bundles == 18
    write(root / 'summary.json', document); merged()


def merged():
    lines = ['# NP50：时间状态与传播高峰优化实验', '',
        '保持三个数据版本、重复记录、信息级时间划分、184维特征及两个论文Block不变。'
        '复用本轮NP50训练的空间编码器，冻结空间参数，重新拟合时间读出及选中的宏观头。'
        '纯神经输出，无树模型。旧SDSformer/DSG2former作为保留对照，不覆盖旧成绩。', '',
        '逐种子拼接全部测试概率(N,3)与标签(N,)后计分；Recall/F1/AUC逐类呈现，'
        '无类别平均。宏观误差合并全部窗口。均值±三个种子的样本标准差。', '',
        '时间候选：原传播链缓存、跨信息严格早于时刻的用户缓存、以及缓存缺失时用既有历史特征初始化。'
        '初始化是历史特征代理，不冒充真实早期隐藏状态。仅无候选用户的宏观图写入缓存；'
        '相同时刻全部先读后写，禁止后续分区向前提供状态。', '',
        '宏观候选：原神经头；加入可学习上一小时人数参照、放宽修正幅度、解冻基础头及增加原人数误差项。'
        '微观选择用三个种子的验证表现；宏观先在48维分支、种子42验证筛选，再固定配置训练其他分支/种子。'
        '本基准测试集此前已被观察，不能视为独立盲测或外部SOTA认证。', '']
    flat = []; complete = []
    def fmt(r, k): return f"{r['mean'][k]:.6f} ± {r['sample_sd'][k]:.6f}"
    for name, spec in SPEC.items():
        path = OUT / spec['slug'] / 'summary.json'
        if not path.exists(): continue
        d = read(path); complete.append(name)
        lines.extend([f'## {name}', '',
            f"测试行动行 {d['test_action_rows']}；宏观窗口 {d['test_macro_windows']}。"
            f"所选时间策略 `{d['micro_selection']['policy']}`；宏观 SDS=`{d['macro_selection']['spatial']['config']}`，"
            f"DSG=`{d['macro_selection']['temporal']['config']}`。", '',
            '|模型|Accuracy↑|Recall 谣言/辟谣/不参与↑|F1 谣言/辟谣/不参与↑|', '|---|---:|---|---|'])
        for method, r in d['rows'].items():
            recall = '<br>'.join(fmt(r, f'recall_{c}') for c in range(3))
            f1 = '<br>'.join(fmt(r, f'f1_{c}') for c in range(3))
            lines.append(f"|{method}|{fmt(r, 'accuracy')}|{recall}|{f1}|")
            flat.append(dict(dataset=name, model=method, **{k+'_mean': v for k, v in r['mean'].items()},
                             **{k+'_sample_sd': v for k, v in r['sample_sd'].items()}))
        lines.extend(['', '|模型|AUC 谣言/辟谣/不参与↑|MSE↓|RMSE↓|MSLE↓|', '|---|---|---:|---:|---:|'])
        for method, r in d['rows'].items():
            auc = '<br>'.join(fmt(r, f'auc_{c}') for c in range(3))
            lines.append(f"|{method}|{auc}|{fmt(r, 'mse')}|{fmt(r, 'rmse')}|{fmt(r, 'msle')}|")
        lines.append('')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    if flat:
        with (OUT / 'summary.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    write(OUT / 'completion.json', dict(complete=len(complete) == 3, datasets=complete,
          new_model_seed_pairs=len(complete)*6, full_replay_bundles=len(complete)*18))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', choices=list(SPEC), required=True)
    report(p.parse_args().dataset)
