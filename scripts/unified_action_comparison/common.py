"""Versioned five-corpus action protocol; prior experiments stay immutable."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.np50_comparison.common import sha, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/unified_action_comparison'
SEEDS = [42, 2025, 2026]
BACKBONES = ['d24_regular', 'd96_regular', 'd48_strong_reg']
SPEC = {
    'PHEME': dict(slug='pheme', parent='data/pheme_information_semantic098_full', history_parent='data/pheme_history', language='en', horizon=None, refute=.5),
    'Weibo-V1': dict(slug='weibo_v1', parent='data/weibo_v1_v2_information', language='zh', horizon=72, refute=.5),
    'Weibo-public': dict(slug='weibo_public', parent='data/weibo_public_v2_information', language='zh', horizon=72, refute=.5),
    'Twitter15': dict(slug='twitter15', parent='data/twitter15_np50_information', language='en', horizon=72, refute=.5),
    'Twitter16': dict(slug='twitter16', parent='data/twitter16_np50_information', language='en', horizon=72, refute=2/3),
}
for _s in SPEC.values():
    _s['root'] = 'data/unified_action_v1/' + _s['slug']
    _s['raw'] = _s['root'] + '/hourly'
    _s['history'] = _s['root'] + '/history'


def link(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        assert destination.resolve() == source, destination
    elif not destination.exists():
        destination.symlink_to(source, target_is_directory=source.is_dir())


def identify(root):
    return next(n for n,s in SPEC.items() if (ROOT/s['root']).resolve() == Path(root).resolve())


def setup(root):
    root=Path(root).resolve(); spec=SPEC[identify(root)]; work=root/'runtime'
    work.mkdir(parents=True,exist_ok=True)
    encoder='pheme_bert' if spec['language']=='en' else 'weibo_bert_fp16'
    for src,dest in [(root/'hourly',work/'data/pheme_multitask_hourly'),
                     (root/'history',work/'data/pheme_history'),
                     (ROOT/'models'/encoder,work/'models/pheme_bert'),
                     (ROOT/'scripts',work/'scripts'),(ROOT/'dsg2former',work/'dsg2former')]:link(src,dest)
    # Chinese lexicon fitting must never mutate the shared English lexicon.
    external=work/'external'; external.mkdir(exist_ok=True)
    for p in (ROOT/'external').iterdir():
        if p.name!='vaderSentiment':link(p,external/p.name)
    lex=external/'vaderSentiment';lex.mkdir(exist_ok=True)
    if spec['language']=='en':link(ROOT/'external/vaderSentiment/vader_lexicon.txt',lex/'vader_lexicon.txt')
    for f in ['threads.jsonl','action_labels.npz','raw_audit.json']:link(root/f,root/'hourly'/f)
    os.chdir(work)
    return ROOT,work


def configure():
    from scripts.np50_comparison import common as base
    base.SPEC.clear();base.SPEC.update(SPEC);base.OUT=OUT/'base';base.setup=setup;base.identify=identify
    from scripts.weibo_comparison import prepare_features
    prepare_features.setup=setup
    from scripts.np50_temporal_peak import common as peak
    peak.OUT=OUT/'optimized'
    from scripts.np50_tree_readout import common as tree
    tree.OUT=OUT/'tree'
    def paths(name):
        s=SPEC[name];dest=tree.OUT/s['slug'];dest.mkdir(parents=True,exist_ok=True)
        return ROOT/s['root']/'runtime',peak.OUT/s['slug'],dest
    tree.paths=paths
    from scripts.np50_comparison import report
    from scripts.unified_action_comparison.truth import truth
    report.truth=truth
