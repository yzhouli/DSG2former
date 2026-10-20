"""Validate and import the benchmark's prepared, causal graph interface."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from cli_utils import config, digest, write_json


def validate(data, metadata, feature_dim=184):
    if not isinstance(data, list) or not data:
        raise ValueError("graphs.pt must contain a nonempty list of source items")
    meta = {str(t["thread_id"]): t for t in metadata}
    if len(meta) != len(metadata) or len({str(t["thread_id"]) for t in data}) != len(data):
        raise ValueError("Source item IDs must be unique")
    if set(meta) != {str(t["thread_id"]) for t in data}:
        raise ValueError("Source metadata and graph IDs differ")
    statistics = {s: Counter() for s in ("train", "validation", "test")}
    starts = {s: [] for s in statistics}
    for item in data:
        split = item["split"]
        if split not in statistics or split != meta[str(item["thread_id"])]["split"]:
            raise ValueError("Source-level split differs from its metadata")
        source = meta[str(item["thread_id"])]["source"]
        if not source.get("uid") or not np.isfinite(float(source["timestamp"])):
            raise ValueError("A source requires an author ID and finite publication time")
        starts[split].append(float(source["timestamp"]))
        statistics[split]["items"] += 1
        macro_windows = set()
        for branch in ("snapshots", "macro_snapshots"):
            last = -float("inf")
            for s in item[branch]:
                x, raw = np.asarray(s["x"]), np.asarray(s["raw_x"])
                n, count = len(x), int(s["user_count"])
                if x.shape != (n, feature_dim) or raw.shape != x.shape or not 0 < count < n:
                    raise ValueError("Expected 184 features, user nodes and a source token")
                arrays = (x, raw, np.asarray(s["drivers"]), np.asarray(s["adjacency"]), np.asarray(s["history_summary"]))
                if any(not np.isfinite(a).all() for a in arrays):
                    raise ValueError("Nonfinite graph input")
                if arrays[2].shape != (n, 2, 3) or arrays[3].shape != (n, n):
                    raise ValueError("Driver/adjacency shape mismatch")
                positions = np.asarray(s["positions"])
                role = np.asarray(s["role"])
                if positions.shape != (n, n) or not np.issubdtype(positions.dtype, np.integer) or np.any((positions < 0) | (positions >= 50)):
                    raise ValueError("Invalid relative-position table indices")
                if role.shape != (n,) or not np.isin(role, [0, 1]).all():
                    raise ValueError("Invalid node roles")
                if len(s["node_ids"]) != count or len(set(s["node_ids"])) != count:
                    raise ValueError("User node identities must be unique within each graph")
                y, ix = np.asarray(s["target"]), np.asarray(s["sample_nodes"])
                if y.shape != ix.shape or y.ndim != 1 or not np.isin(y, [0, 1, 2]).all():
                    raise ValueError("Action rows/labels do not align")
                if not np.issubdtype(ix.dtype, np.integer) or np.any((ix < 0) | (ix >= count)):
                    raise ValueError("Action indices must refer to user nodes")
                if float(s["cutoff"]) < last or float(s["end"]) != float(s["cutoff"]) + 3600:
                    raise ValueError("Snapshots must be chronological one-hour prediction windows")
                last = float(s["cutoff"])
                if last < float(source["timestamp"]):
                    raise ValueError("Prediction precedes source publication")
                if arrays[-1].ndim != 1 or len(arrays[-1]) < 13:
                    raise ValueError("Eight-hour count history is required")
                if not set(s["state_update_users"]).issubset(set(s["node_ids"])):
                    raise ValueError("State update identity is absent from graph")
                if not np.isfinite(s["popularity_target"]) or s["popularity_target"] < 0:
                    raise ValueError("Count targets must be finite and nonnegative")
                statistics[split][branch] += 1
                if branch == "snapshots":
                    statistics[split]["action_rows"] += len(y)
                else:
                    if len(y) or int(s["window"]) in macro_windows:
                        raise ValueError("Macro graphs must have no action candidates and one graph per window")
                    macro_windows.add(int(s["window"]))
        if not macro_windows:
            raise ValueError("Each source item requires at least one macro graph")
    if any(not starts[s] or not statistics[s]["action_rows"] for s in starts):
        raise ValueError("All three partitions require source items and action rows")
    if max(starts["train"]) > min(starts["validation"]) or max(starts["validation"]) > min(starts["test"]):
        raise ValueError("Source publication times violate chronological splitting")
    return {s: dict(v) for s, v in statistics.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--graphs", required=True, type=Path, help="Trusted local prepared PyTorch file")
    parser.add_argument("--threads", required=True, type=Path, help="Source metadata JSONL")
    args = parser.parse_args()
    value = config(args.config)
    graphs, threads = args.graphs.resolve(), args.threads.resolve()
    data = torch.load(graphs, weights_only=False, map_location="cpu")
    metadata = [json.loads(line) for line in threads.open() if line.strip()]
    statistics = validate(data, metadata)
    destination = Path(value["data_dir"])
    for source, target in [(graphs, destination / "neural/graphs.pt"), (threads, destination / "hourly/threads.jsonl")]:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(source):
            raise FileExistsError(f"Refusing to replace another data version: {target}")
        if target.resolve() != source:
            shutil.copyfile(source, target)
    write_json(destination / "import_manifest.json", dict(statistics=statistics,
               graphs_sha256=digest(graphs), threads_sha256=digest(threads),
               feature_causality="Must be established by the upstream feature-generation audit",
               importer="Validates the prepared interface; does not relabel or rebuild features"))
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
