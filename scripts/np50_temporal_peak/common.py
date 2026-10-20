"""Versioned optimization experiment; parent results and datasets are read only."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.np50_comparison.common import ROOT, SPEC, SEEDS, BACKBONES, sha, write

OUT = ROOT / 'results/np50_temporal_peak'
POLICIES = ['global', 'global_bootstrap']
MACRO_CONFIGS = {
    'peak_w025': dict(first_cap=1., continuation_cap=2., fine_tune=True, raw_weight=.25, penalty=.01),
    'peak_w1': dict(first_cap=1., continuation_cap=2., fine_tune=True, raw_weight=1., penalty=.01),
    'peak_w4': dict(first_cap=1., continuation_cap=2., fine_tune=True, raw_weight=4., penalty=.01),
}


def paths(name):
    work = ROOT / SPEC[name]['root'] / 'runtime'
    out = OUT / SPEC[name]['slug']
    out.mkdir(parents=True, exist_ok=True)
    return work, out


def read(path):
    return json.loads(Path(path).read_text())


def source_files():
    return [*sorted((ROOT / 'scripts/np50_temporal_peak').glob('*.py')),
            ROOT / 'dsg2former/causal_user_state.py', ROOT / 'dsg2former/peak_popularity.py',
            ROOT / 'dsg2former/sds_aligned.py', ROOT / 'dsg2former/enhanced_model.py',
            ROOT / 'scripts/macro_residual_study/train.py',
            ROOT / 'scripts/weibo_comparison/neural.py']


def verify_plan():
    plan = read(OUT / 'plan.json')
    for p, h in plan['source_sha256'].items():
        assert sha(ROOT / p) == h, p
    return plan
