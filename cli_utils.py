"""Portable configuration for the retained research training modules."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from functools import partial

ROOT = Path(__file__).resolve().parent
DATASETS = ("PHEME", "SocialNet", "Weibo", "Twitter15", "Twitter16", "TikTok")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value["dataset"] not in DATASETS:
        raise ValueError(f"Unknown dataset: {value['dataset']}")
    for key in ("data_dir", "output_dir"):
        value[key] = str((ROOT / value[key]).resolve())
    if not Path(value["output_dir"]).is_relative_to(ROOT):
        raise ValueError("output_dir must be inside this checkout for portable checkpoint manifests")
    if value.get("attention", "dense") not in ("dense", "sparse"):
        raise ValueError("attention must be dense or sparse")
    if not value.get("seeds") or len(set(value["seeds"])) != len(value["seeds"]):
        raise ValueError("Use a nonempty list of distinct seeds")
    return value


def link(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source:
            raise ValueError(f"Existing link points elsewhere: {destination}")
    elif destination.exists():
        if destination.resolve() != source:
            raise FileExistsError(destination)
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def configure(value):
    """Adapt paths and exposed settings; retain original models and objectives."""
    import torch
    from dsg2former.attention_backend import make_model
    from scripts.weibo_comparison import neural
    from scripts.np50_temporal_peak import common as peak
    from scripts.np50_comparison import common as base

    torch.set_num_threads(value.get("cpu_threads", 1))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    data = Path(value["data_dir"])
    out = Path(value["output_dir"])
    work = out / "runtime"
    work.mkdir(parents=True, exist_ok=True)
    input_path = data / "neural/graphs.pt"
    if not input_path.is_file():
        raise FileNotFoundError(f"Run preprocess.py first; missing {input_path}")
    if not (data / "hourly/threads.jsonl").is_file():
        raise FileNotFoundError("Source metadata is required for the causal author-memory count head")
    frozen = dict(configuration=value, input_sha256=digest(input_path),
                  threads_sha256=digest(data / "hourly/threads.jsonl"))
    record = out / "run_configuration.json"
    if record.exists() and json.loads(record.read_text()) != frozen:
        raise ValueError("Configuration/input changed: use a new output_dir to keep checkpoints separate")
    write_json(record, frozen)
    for source, target in [(data / "neural", work / "data/pheme_model_improvement"),
                           (data / "hourly", work / "data/pheme_multitask_hourly")]:
        link(source, target)
    neural.R = work / "results/neural"
    neural.R.mkdir(parents=True, exist_ok=True)
    neural.PUBLIC_ATTENTION = value.get("attention", "dense")
    neural.PaperAlignedDSG2former = partial(make_model, neural.PUBLIC_ATTENTION)
    neural.PUBLIC_TRAINING.update(value.get("training", {}))
    base.SEEDS[:] = value["seeds"]
    base.BACKBONES[:] = value["backbones"]
    internal = {"SocialNet": "Weibo-V1", "Weibo": "Weibo-public"}.get(value["dataset"], value["dataset"])
    base.SPEC.clear()
    base.SPEC[internal] = dict(slug=value["dataset"].lower(), root=str(data))
    peak.SEEDS[:] = value["seeds"]
    peak.BACKBONES[:] = value["backbones"]
    peak.OUT = out / "optimized"
    peak.OUT.mkdir(parents=True, exist_ok=True)
    peak.paths = lambda name: (work, peak.OUT)
    # Freeze training sources so resumed stages use the same implementation.
    from scripts.np50_temporal_peak import train as pt, infer as pi
    pt.paths = pi.paths = peak.paths
    pt.OUT = pi.OUT = peak.OUT
    pt.PaperAlignedDSG2former = partial(make_model, neural.PUBLIC_ATTENTION)
    files = [p for folder in (ROOT / "dsg2former", ROOT / "scripts")
             for p in folder.rglob("*.py")]
    files += [ROOT / name for name in ("cli_utils.py", "preprocess.py", "train.py", "test.py")]
    plan = dict(source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(files)},
                configuration_sha256=digest(record), selection_partition="validation")
    path = peak.OUT / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Training source changed: use a new output_dir")
    write_json(path, plan)
    os.chdir(work)
    return neural, pt, pi, internal
