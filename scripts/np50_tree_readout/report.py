"""Publish pooled per-class test metrics with preserved neural and baseline controls."""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.np50_tree_readout.common import *
from scripts.np50_comparison.report import score, checked
from scripts.np50_comparison.verify import references, binary_auc


def report(name):
    _, parent, out = paths(name); y, ids, truth = references(name)
    expected = dict(action=dict(target=y, thread_id=ids), macro=truth)
    controls = read(parent / 'summary.json'); rows = dict(controls['rows']); new = []
    for model in MODELS:
        for suffix in [' + tree', ' + action-tree', ' selected raw-tree']:
            method = model+suffix; new.append(method); runs = []
            for seed in SEEDS:
                a, m = checked(out / method / f'seed_{seed}', expected); values, cm = score(a, m)
                for c in range(3): np.testing.assert_allclose(values[f'auc_{c}'], binary_auc(y == c, a['probability'][:, c]), atol=1e-12, rtol=0)
                dest = out / 'published' / method / f'seed_{seed}'; dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest / 'action_predictions.npz', **a)
                np.savez_compressed(dest / 'popularity_predictions.npz', **m)
                runs.append(dict(seed=seed, **values, confusion_matrix=cm))
            rows[method] = dict(per_seed=runs,
                mean={k: float(np.mean([r[k] for r in runs])) for k in values},
                sample_sd={k: float(np.std([r[k] for r in runs], ddof=1)) for k in values})
    write(out / 'summary.json', dict(complete=True, dataset=name, rows=rows, new_result_keys=new,
        test_action_rows=len(y), test_macro_windows=len(truth['target']), seeds=SEEDS,
        class_averaging=None, selection=read(out / 'selection.json'),
        neural_control_summary_sha256=sha(parent / 'summary.json'), exact_tree_replay_bundles=18,
        prior_test_benchmark_already_inspected=True))
    merged()


def merged():
    lines = ['# NP50：加入树模型的配对对照', '',
        '使用同一NP50数据、信息级时间划分、184维因果特征和已有两层论文Block。保留纯神经及六个基线成绩。', '',
        '微观：HistGradientBoosting分类器读取184维原始匹配特征，与神经三类概率加权；'
        '权重及树配置依据三个种子验证Accuracy/交叉熵选择。权重为1时属于纯树分类，不代表神经架构收益。', '',
        '宏观：树校准原版/新版本神经人数预测，另输入相同可观测历史、用户画像与时间特征；'
        '这是神经预测校准实验，与早期latent-PCA混合读出不同。原有神经预测已经整合微观最终状态与历史流行度。'
        '配置与混合权重按验证源信息前后两半的最差归一化MSE/MSLE选择，包含纯神经零权重控制。', '',
        '“+ action-tree”只替换微观；“+ tree”同时检验宏观校准；“selected raw-tree”是对应配置的纯树微观对照，宏观仍用新纯神经头。', '',
        '每种子拼接全部测试(N,3)概率和(N,)标签计分；Recall/F1/AUC逐类，无类别平均。'
        '宏观合并全部窗口。均值±三个种子的样本标准差。测试基准此前已被观察，不作为独立盲测或SOTA认证。', '']
    flat = []; done = []
    for name, spec in SPEC.items():
        p = OUT / spec['slug'] / 'summary.json'
        if not p.exists(): continue
        d = read(p); done.append(name)
        lines += ['## '+name, '', f"行动样本 {d['test_action_rows']:,}；宏观窗口 {d['test_macro_windows']:,}。", '']
        for model in MODELS:
            sel = d['selection']['models'][model]
            lines += [f"{model}：微观树权重 {sel['action']['tree_weight']}、`{sel['action']['config']}`；宏观树权重 {sel['count']['tree_weight']}、`{sel['count']['config']}`、神经参照 `{sel['count']['reference']}`。", '']
        for keys, labels in [(['accuracy', 'recall_0', 'recall_1', 'recall_2', 'f1_0', 'f1_1', 'f1_2'], ['Accuracy↑', '谣言Recall↑', '辟谣Recall↑', '不参与Recall↑', '谣言F1↑', '辟谣F1↑', '不参与F1↑']),
                             (['auc_0', 'auc_1', 'auc_2', 'mse', 'rmse', 'msle'], ['谣言AUC↑', '辟谣AUC↑', '不参与AUC↑', 'MSE↓', 'RMSE↓', 'MSLE↓'])]:
            lines += ['|模型|'+'|'.join(labels)+'|', '|---|'+'|'.join(['---:']*len(keys))+'|']
            for method, r in d['rows'].items():
                lines.append('|'+method+'|'+'|'.join(f"{r['mean'][k]:.6f} ± {r['sample_sd'][k]:.6f}" for k in keys)+'|')
            lines.append('')
        for method, r in d['rows'].items():
            flat.append(dict(dataset=name, model=method, **{k+'_mean': v for k, v in r['mean'].items()}, **{k+'_sample_sd': v for k, v in r['sample_sd'].items()}))
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    if flat:
        with (OUT / 'summary.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    write(OUT / 'completion.json', dict(complete=len(done) == len(SPEC), datasets=done, seeds=SEEDS))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', choices=list(SPEC), required=True)
    verify_plan(); report(p.parse_args().dataset)
