"""Matched-feature tree ablations, separate from pure-neural experiments."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.np50_comparison.common import ROOT, SPEC, SEEDS, sha, write

OUT = ROOT / 'results/np50_tree_readout'
ACTION_CONFIGS = {f'l{l}_r{r}': dict(leaves=l, regularization=r) for l in [15, 31] for r in [10, 100]}
COUNT_CONFIGS = {f'{loss}_l{l}': dict(loss=loss, leaves=l) for loss in ['log', 'poisson'] for l in [7, 15]}
MODELS = ['SDSformer', 'DSG2former']
WEIGHTS = [0., .25, .5, .75, 1.]
read = lambda p: json.loads(Path(p).read_text())


def paths(name):
    work = ROOT / SPEC[name]['root'] / 'runtime'
    parent = ROOT / 'results/np50_temporal_peak' / SPEC[name]['slug']
    out = OUT / SPEC[name]['slug']; out.mkdir(parents=True, exist_ok=True)
    return work, parent, out


def verify_plan():
    plan = read(OUT / 'plan.json')
    for p, h in plan['source_sha256'].items(): assert sha(ROOT / p) == h, p
    return plan
