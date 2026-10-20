"""Execute unchanged task trainers inside an NP50 dataset workspace."""
import argparse
import inspect
import fcntl
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from scripts.np50_comparison.common import ROOT, identify, setup


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['fit', 'common', 'base', 'count', 'freeze', 'predict', 'test', 'replay'])
    p.add_argument('--kind', choices=['neural', 'baseline'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--method'); p.add_argument('--variant', default='paper')
    p.add_argument('--backbone'); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    args = p.parse_args(); name = identify(args.root)
    torch.set_num_threads(1)
    from scripts.weibo_comparison import prepare_features
    prepare_features.setup = setup
    _, work = setup(args.root)
    locks = work / 'results/task_locks'; locks.mkdir(parents=True, exist_ok=True)
    key = '_'.join(str(v) for v in [args.kind, args.mode, args.method or args.backbone or 'common', args.variant, args.seed])
    lock = (locks / (key + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if args.kind == 'neural':
        from scripts.weibo_comparison import neural
        if args.mode == 'fit' and (neural.R / args.backbone / f'seed_{args.seed}' / 'completion.json').exists():
            return
        # Preserve PHEME's original batch size; the two Weibo corpora used64.
        if name == 'PHEME':
            source = inspect.getsource(neural.fit_spatial)
            assert source.count('range(0,len(order),64)') == 1
            assert source.count('order[a:a+64]') == 1
            source = source.replace('range(0,len(order),64)', 'range(0,len(order),32)').replace('order[a:a+64]', 'order[a:a+32]')
            exec(compile(source, '<NP50-PHEME-original-batch32>', 'exec'), vars(neural))
        neural.R.mkdir(parents=True, exist_ok=True)
        if args.mode == 'fit': neural.fit(args)
        elif args.mode == 'common':
            if not (neural.R / 'common.pt').exists():
                data = torch.load('data/pheme_model_improvement/graphs.pt', weights_only=False, map_location='cpu')
                torch.save(neural.numeric_common(data), neural.R / 'common.pt')
        elif args.mode == 'base':
            c = torch.load(neural.R / 'common.pt', weights_only=False, map_location='cpu')
            neural.fit_base(neural.R / 'numeric_base' / f'seed_{args.seed}', c['train'], c['validation'], 'ics', args.seed)
        elif args.mode == 'count': neural.count_fit(args)
        elif args.mode == 'freeze':
            files = [p for p in neural.R.rglob('*.pt') if p.name not in ['common.pt', 'development.pt', 'development_final.pt']]
            neural.write(neural.R / 'inference_plan.json', dict(frozen_before_test=True,
                checkpoint_sha256={str(p): neural.sha(p) for p in files},
                source_sha256=neural.sha(neural.__file__), input_sha256=neural.sha('data/pheme_model_improvement/graphs.pt')))
        else: neural.predict(args)
        return
    from scripts.weibo_comparison import baselines
    if name == 'PHEME':
        from dsg2former.baselines.native import rbf_svm
        baselines.svm = rbf_svm
    if args.method == 'ICS-SVM':
        baselines.method(args)
        return
    argv = [args.mode, '--root', str(args.root), '--method', args.method,
            '--variant', args.variant, '--seed', str(args.seed)]
    if args.method == 'TRGCN':
        entry = 'deterministic_trgcn.py'
    elif args.method == 'ICS-SVM':
        entry = 'baselines.py'
    else:
        entry = 'gpu_baseline_worker.py'
    path = ROOT / 'scripts/weibo_comparison' / entry
    sys.argv = [str(path), *argv]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
