"""Native-module baselines with declared Chinese/large-corpus data adapters."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.svm import SVC
from scripts.weibo_comparison.prepare_features import setup
from scripts.weibo_comparison.extract import sha

_SVM_CACHE={}


class WeightedRepeatedSVC(SVC):
    """Coalesce identical (feature,label) solver terms with exact count weights.

    This preserves the SVM training objective and every evaluation row; libsvm
    internal probability calibration folds differ from raw-row replication.
    """
    def fit(self,x,y,sample_weight=None):
        key=(id(x),id(y),x.shape)
        if key not in _SVM_CACHE:
            rows,count=np.unique(np.c_[x,y],axis=0,return_counts=True)
            _SVM_CACHE[key]=(rows[:,:-1],rows[:,-1].astype(y.dtype),count.astype(float))
        xx,yy,ww=_SVM_CACHE[key]
        if sample_weight is not None:raise ValueError('Only raw repeated-row weights supported')
        return super().fit(xx,yy,sample_weight=ww)


def svm(c,sigma,seed=42,probability=False):
    return WeightedRepeatedSVC(C=float(c),kernel='rbf',gamma=1/(2*float(sigma)**2),
        probability=probability,random_state=seed,cache_size=1024)


def plan():
    from scripts.baseline_comparison.run_comparison import CONFIGS
    out=Path('results/baselines');out.mkdir(parents=True,exist_ok=True)
    p=out/'plan.json'
    if not p.exists():
        files=[*Path(__file__).resolve().parent.glob('*.py'),*Path(__file__).resolve().parents[2].glob('dsg2former/baselines/*.py')]
        prepared=Path('data/pheme_paper_baseline_resources')
        p.write_text(json.dumps(dict(prepared=str(prepared),prepared_sha256=sha(prepared/'graphs.pt'),
            configurations=copy.deepcopy(CONFIGS),seeds=[42,2025,2026],source_sha256={str(f):sha(f) for f in files},
            native_module_scope='Same native kernels/modules and fixed paper configurations as existing PHEME adaptations; fresh fits on this corpus',
            svm_compute_adapter='Identical feature,label terms coalesced only inside SVM solver with multiplicity weights. Raw rows and evaluation unchanged. Native libsvm probability folds differ from replicated rows.',
            micro_task='Next-hour support/weak-refute/nonparticipation, not source veracity detection',
            macro_adapter='Same frozen-native-encoder CountAdapter as PHEME; no claim macro head belongs to native publication',
            selection='Validation only; DPM A/B/C seed42, best2 confirm across3seeds; choose by mean val accuracy,then CE,then AUC',
            horizon_hours=72),indent=2))


def method(args):
    from scripts.baseline_comparison import train_method as m
    m.rbf_svm=svm
    if args.mode!='fit':
        proof=json.loads(Path('results/baselines/selection.json').read_text())
        for p,h in proof['checkpoint_sha256'].items():assert sha(p)==h,p
    a=SimpleNamespace(root='results/baselines',method=args.method,variant=args.variant,seed=args.seed,mode=args.mode)
    m.run(a) if args.mode=='fit' else m.test(a)


def prepare_trgcn():
    from scripts.trgcn_baseline import prepare as m
    from sklearn.feature_extraction.text import TfidfVectorizer
    original=TfidfVectorizer
    def chinese_vectorizer(**kw):
        return original(analyzer='char',ngram_range=(1,2),**kw)
    m.TfidfVectorizer=chinese_vectorizer;m.main()
    out=Path('results/trgcn');out.mkdir(parents=True,exist_ok=True)
    previous=Path(__file__).resolve().parents[2]/'results/pheme_trgcn_baseline/plan.json'
    old=json.loads(previous.read_text());p=out/'plan.json'
    if not p.exists():
        options=dict(old['options']);options['input_dim']=json.loads(Path('data/pheme_trgcn/completion.json').read_text())['feature_dim']
        # The constructor uses input_dim; retain all other native options.
        p.write_text(json.dumps(dict(options=options,prepared='data/pheme_trgcn/graphs.pt',
            prepared_sha256=sha('data/pheme_trgcn/graphs.pt'),source_sha256={str(Path(__file__).resolve()):sha(__file__)},
            chinese_adapter='Train-only Chinese character1/2gram TF-IDF top5000, rather than English word tokenization',
            original_options=old['options'],test_used=False),indent=2))


def trgcn(args):
    from scripts.trgcn_baseline import train as m
    if args.mode!='fit':
        proof=json.loads(Path('results/trgcn/selection.json').read_text())
        for p,h in proof['checkpoint_sha256'].items():assert sha(p)==h,p
    m.R=Path('results/trgcn');m.main(SimpleNamespace(mode=args.mode,seed=args.seed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','fit','test','replay','prepare_trgcn']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--method');p.add_argument('--variant',default='paper');p.add_argument('--seed',type=int,default=42);a=p.parse_args();setup(a.root);torch.set_num_threads(1)
    if a.mode=='plan':plan()
    elif a.mode=='prepare_trgcn':prepare_trgcn()
    elif a.method=='TRGCN':trgcn(a)
    else:method(a)
