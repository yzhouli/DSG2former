"""Run the separate tree ablation without changing ongoing neural training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.np50_tree_readout.common import *


def sources():
    return sorted((ROOT / 'scripts/np50_tree_readout').glob('*.py')) + [
        ROOT / 'scripts/np50_temporal_peak/train.py', ROOT / 'scripts/np50_comparison/report.py',
        ROOT / 'scripts/np50_comparison/verify.py', ROOT / 'scripts/weibo_comparison/neural.py']


def run(names):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'pipeline.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = dict(created_before_tree_fits=True, datasets=list(SPEC), seeds=SEEDS,
        action_configs=ACTION_CONFIGS, count_configs=COUNT_CONFIGS, blend_weights=WEIGHTS,
        action_input='Same184 normalized causal features, duplicates retained; raw-feature ablation, not historical latent-PCA',
        count_input='Original/current neural count forecasts and observed causal history/profile/time',
        train_options=dict(max_iter=250, learning_rate=.05, early_stopping=False),
        selection='Validation only; classification Accuracy/CE; count worst normalized MSE/MSLE across chronological validation halves',
        prior_test_benchmark_already_inspected=True, class_averaging=None,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources()})
    path = OUT / 'plan.json'
    if path.exists(): assert read(path) == plan
    else: write(path, plan)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def launch(script, name, mode=None, seed=None):
        _, _, out = paths(name); logs = out / 'logs'; logs.mkdir(exist_ok=True)
        cmd = [sys.executable, str(ROOT / 'scripts/np50_tree_readout' / script)]
        if mode: cmd.append(mode)
        cmd += ['--dataset', name]
        if seed is not None: cmd += ['--seed', str(seed)]
        with (logs / f'{script}_{mode}_{seed}.log').open('a') as f:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(dict(dataset=name, script=script, mode=mode, seed=seed, complete=True), flush=True)
    # Fit all datasets first, so waiting for neural test replay never blocks CPU fits.
    for name in names:
        _, _, out = paths(name)
        if (out / 'selection.json').exists(): continue
        launch('prepare.py', name, 'features')
        with ThreadPoolExecutor(3) as pool: list(pool.map(lambda s: launch('prepare.py', name, 'forecasts', s), SEEDS))
        with ThreadPoolExecutor(3) as pool: list(pool.map(lambda s: launch('train.py', name, seed=s), SEEDS))
        launch('selection.py', name)
    for name in names:
        _, parent, out = paths(name)
        if (out / 'summary.json').exists(): continue
        while not (parent / 'summary.json').exists(): time.sleep(15)
        launch('infer.py', name, 'freeze')
        for mode in ['test', 'replay']:
            with ThreadPoolExecutor(3) as pool: list(pool.map(lambda s: launch('infer.py', name, mode, s), SEEDS))
        launch('report.py', name)
    print('All tree ablations completed.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--datasets', nargs='+', choices=list(SPEC), default=list(SPEC))
    run(p.parse_args().datasets)
