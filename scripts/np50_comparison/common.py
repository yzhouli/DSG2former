"""Isolated workspaces for the fixed-configuration NP50 rerun."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/np50_comparison'
SEEDS = [42, 2025, 2026]
BACKBONES = ['d24_regular', 'd96_regular', 'd48_strong_reg']
METHODS = ['ICS-SVM', 'K-GraphSAGE', 'RAM-GAT', 'TRS-GCN', 'Rumor-DPM', 'TRGCN', 'SDSformer', 'DSG2former']
SPEC = {
    'PHEME': dict(slug='pheme', root='data/pheme_multitask_hourly_np50', parent='.', raw='data/pheme_multitask_hourly_np50', history='data/pheme_history'),
    'Weibo-V1': dict(slug='weibo_v1', root='data/weibo_v1_v2_np50_information', parent='data/weibo_v1_v2_information/runtime', raw='data/weibo_v1_v2_np50_information/hourly', history='data/weibo_v1_v2_information/history'),
    'Weibo-public': dict(slug='weibo_public', root='data/weibo_public_v2_np50_information', parent='data/weibo_public_v2_information/runtime', raw='data/weibo_public_v2_np50_information/hourly', history='data/weibo_public_v2_information/history'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def identify(root):
    root = Path(root).resolve()
    return next((name for name, spec in SPEC.items() if (ROOT / spec['root']).resolve() == root))


def link(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        assert destination.resolve() == source, destination
    elif not destination.exists():
        destination.symlink_to(source, target_is_directory=source.is_dir())


def setup(root):
    name = identify(root)
    spec = SPEC[name]
    root = ROOT / spec['root']
    work = root / 'runtime'
    work.mkdir(parents=True, exist_ok=True)
    for target, destination in [
        (ROOT / spec['raw'], work / 'data/pheme_multitask_hourly'),
        (ROOT / spec['history'], work / 'data/pheme_history'),
        (ROOT / ('models/pheme_bert' if name == 'PHEME' else 'models/weibo_bert_fp16'), work / 'models/pheme_bert'),
        (ROOT / 'scripts', work / 'scripts'), (ROOT / 'dsg2former', work / 'dsg2former'),
    ]:
        link(target, destination)
    os.chdir(work)
    return ROOT, work


def previous(name):
    spec = SPEC[name]
    if name == 'PHEME':
        base = ROOT / 'results/pheme_paper_baselines'
        trgcn = ROOT / 'results/pheme_trgcn_baseline'
    else:
        base = ROOT / spec['parent'] / 'results/baselines'
        trgcn = ROOT / spec['parent'] / 'results/trgcn'
    return base, trgcn
