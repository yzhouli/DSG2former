"""Process-local adapters for unchanged audited experiment implementations."""
import argparse
import inspect
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.unified_action_comparison.common import ROOT, OUT, SPEC, configure


def redirect_base_paths(module, functions):
    for name in functions:
        source=inspect.getsource(getattr(module,name))
        assert 'results/np50_comparison' in source, name
        source=source.replace('results/np50_comparison','results/unified_action_comparison/base')
        exec(compile(source,'<Twitter-isolated-result-path>','exec'),vars(module))


def main():
    p=argparse.ArgumentParser();p.add_argument('entry',choices=['base','peak_train','peak_infer','tree_prepare','tree_train','tree_select','tree_infer','base_report','peak_report','tree_report'])
    a,remaining=p.parse_known_args();configure()
    import torch
    torch.set_num_threads(1)
    if a.entry=='base':
        if '--kind' in remaining and remaining[remaining.index('--kind')+1]=='baseline':
            from scripts.unified_action_comparison.common import setup
            q=argparse.ArgumentParser();q.add_argument('mode');q.add_argument('--kind');q.add_argument('--root',type=Path,required=True)
            q.add_argument('--method');q.add_argument('--variant',default='paper');q.add_argument('--seed',type=int,default=42);q.add_argument('--device',default='cpu')
            args=q.parse_args(remaining);setup(args.root)
            from scripts.weibo_comparison import baselines
            if args.method=='TRGCN':
                torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
                torch.set_flush_denormal(True)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if args.method!='ICS-SVM' and args.device.startswith('cuda'):
                from scripts.weibo_comparison.gpu_baseline_worker import enable
                enable(args.device,args.method)
            if args.method=='TRGCN':baselines.trgcn(args)
            else:baselines.method(args)
            return
        sys.argv=[str(ROOT/'scripts/np50_comparison/worker.py'),*remaining]
        runpy.run_path(sys.argv[0],run_name='__main__');return
    names=dict(peak_train='scripts/np50_temporal_peak/train.py',peak_infer='scripts/np50_temporal_peak/infer.py',
               tree_prepare='scripts/np50_tree_readout/prepare.py',tree_train='scripts/np50_tree_readout/train.py',tree_select='scripts/np50_tree_readout/selection.py')
    if a.entry in names:
        sys.argv=[str(ROOT/names[a.entry]),*remaining];runpy.run_path(sys.argv[0],run_name='__main__');return
    q=argparse.ArgumentParser();q.add_argument('--dataset',choices=list(SPEC),required=True)
    q.add_argument('--mode',default='test');q.add_argument('--seed',type=int,default=42);b=q.parse_args(remaining)
    if a.entry=='tree_infer':
        from scripts.np50_tree_readout import infer as m
        redirect_base_paths(m,['inputs','infer']);m.verify_plan()
        if b.mode=='freeze':m.freeze(b.dataset)
        else:m.infer(b.dataset,b.seed,b.mode=='replay')
        return
    from scripts.np50_comparison import report as base
    from scripts.np50_comparison import verify
    def references(name):
        truth=base.truth(name)
        return truth['action']['target'],truth['action']['thread_id'],truth['macro']
    verify.references=references
    if a.entry=='base_report': m=base
    elif a.entry=='peak_report':
        from scripts.np50_temporal_peak import report as m
        redirect_base_paths(m,['report'])
    else:
        from scripts.np50_tree_readout import report as m
    m.merged=lambda:None
    m.report(b.dataset)


if __name__=='__main__':main()
