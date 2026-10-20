"""New Twitter experiment namespaces; historical results remain immutable."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.twitter_comparison.extract import ROOT, sha, write

OUT = ROOT / 'results/twitter_comparison'
SEEDS = [42, 2025, 2026]
BACKBONES = ['d24_regular', 'd96_regular', 'd48_strong_reg']
SPEC = {f'Twitter{y}': dict(slug=f'twitter{y}', root=f'data/twitter{y}_np50_information',
                           raw=f'data/twitter{y}_np50_information/hourly',
                           history=f'data/twitter{y}_np50_information/history') for y in [15,16]}


def link(target, dest):
    target, dest = Path(target).resolve(), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink(): assert dest.resolve() == target, dest
    elif not dest.exists(): dest.symlink_to(target, target_is_directory=target.is_dir())


def setup(root):
    root = Path(root).resolve(); work = root / 'runtime'; work.mkdir(parents=True, exist_ok=True)
    for target, dest in [(root/'hourly', work/'data/pheme_multitask_hourly'),
                         (root/'history', work/'data/pheme_history'),
                         (ROOT/'models/pheme_bert', work/'models/pheme_bert'),
                         (ROOT/'external', work/'external'), (ROOT/'scripts', work/'scripts'),
                         (ROOT/'dsg2former', work/'dsg2former')]: link(target, dest)
    for name in ['threads.jsonl','action_labels.npz','raw_audit.json']: link(root/name, root/'hourly'/name)
    os.chdir(work)
    return ROOT, work


def configure():
    from scripts.np50_comparison import common as base
    base.SPEC.clear(); base.SPEC.update(SPEC); base.OUT = OUT/'base'
    base.setup = setup
    base.identify = lambda root: next(n for n,s in SPEC.items() if (ROOT/s['root']).resolve() == Path(root).resolve())
    from scripts.weibo_comparison import prepare_features
    prepare_features.setup = setup
    from scripts.np50_temporal_peak import common as peak
    peak.OUT = OUT/'optimized'
    from scripts.np50_tree_readout import common as tree
    tree.OUT = OUT/'tree'
    def tree_paths(name):
        s=SPEC[name]; dest=tree.OUT/s['slug']; dest.mkdir(parents=True,exist_ok=True)
        return ROOT/s['root']/'runtime', peak.OUT/s['slug'], dest
    tree.paths = tree_paths
