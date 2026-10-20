"""Causal snapshot construction. Never uses final recv/fans or bundle node_order."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import io
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


LABELS = ["rumor", "non_rumor_participation", "inactive_next_window"]


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Unexpected executable pickle global: {module}.{name}")


def load_bundle(path):
    raw = Path(path).read_bytes()
    bundle = PlainUnpickler(io.BytesIO(raw)).load()
    for r in bundle["records"]:
        if r["mal"] not in (0, 1) or not np.isfinite(r["t"]):
            raise ValueError("Expected binary mal and finite timestamps")
    # Stable ordering makes tied-time last-action labels deterministic.
    records = sorted(bundle["records"], key=lambda r: (r["t"], str(r["uid"]), r["mal"]))
    return records, bundle.get("meta", {}), hashlib.sha256(raw).hexdigest()


@dataclass
class Protocol:
    windows: int = 30
    calibration_fraction: float = 0.1
    max_nodes: int = 512  # 0 = every user known during calibration
    bias_size: int = 50
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    gamma: float = 0.23
    directed: bool = True


def chronological_split(length, protocol):
    train = int(length * protocol.train_fraction)
    valid = train + max(1, int(length * protocol.validation_fraction))
    if train < 2 or valid >= length:
        raise ValueError("Need at least two training, one validation and one test snapshot")
    return train, valid


def plan(records, protocol):
    if protocol.bias_size < 3 or not 0 < protocol.calibration_fraction < 1:
        raise ValueError("Invalid protocol")
    times = np.array([r["t"] for r in records], dtype=float)
    # Retrospective event quantile schedule: keeps highly uneven timelines usable.
    # These boundaries are protocol metadata, never model features.
    bounds = np.unique(np.quantile(times, np.linspace(protocol.calibration_fraction, 1, protocol.windows + 1)))
    chronological_split(len(bounds) - 1, protocol)
    observed = [r for r in records if r["t"] <= bounds[0]]
    counts = Counter(r["uid"] for r in observed)
    degree = Counter()
    for r in observed:
        if r.get("parent") is not None:
            degree[r["uid"]] += 1
            degree[r["parent"]] += 1
            counts.setdefault(r["parent"], 0)
    nodes = sorted(counts, key=lambda u: (-degree[u], -counts[u], str(u)))
    if protocol.max_nodes:
        nodes = nodes[:protocol.max_nodes]
    if len(nodes) < 3:
        raise ValueError("Insufficient calibration cohort")
    return bounds, nodes


def snapshot(records, nodes, previous_cutoff, cutoff, next_cutoff, protocol):
    """Features/graph use t<=cutoff; labels use cutoff<t<=next_cutoff."""
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    historical = np.zeros((n, 2), dtype=float)
    originals = np.zeros(n)
    shares = np.zeros(n)
    recent = np.zeros((n, 2))
    prior = np.zeros((n, 2))
    last = np.full(n, 2, dtype=np.int64)
    target = np.full(n, 2, dtype=np.int64)
    recent_last = np.full(n, 2, dtype=np.int64)
    seen_pairs = set()
    ambiguous = set()
    labels_seen = {}
    for r in records:
        if r["t"] > next_cutoff:
            break
        i = index.get(r["uid"])
        k = 0 if r["mal"] == 1 else 1
        if r["t"] <= cutoff:
            if i is not None:
                historical[i, k] += 1
                last[i] = k
                if r.get("parent") is None:
                    originals[i] += 1
                else:
                    shares[i] += 1
                if r["t"] > previous_cutoff:
                    recent[i, k] += 1
                    recent_last[i] = k
                else:
                    prior[i, k] += 1
            j = index.get(r.get("parent"))
            if i is not None and j is not None and i != j:
                seen_pairs.add((i, j))
        elif i is not None:
            target[i] = k
            labels_seen.setdefault(i, set()).add(k)
    ambiguous = {i for i, values in labels_seen.items() if len(values) > 1}
    src = np.array([p[0] for p in sorted(seen_pairs)], dtype=np.int64)
    dst = np.array([p[1] for p in sorted(seen_pairs)], dtype=np.int64)
    adjacency = csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    if not protocol.directed:
        adjacency = adjacency.maximum(adjacency.T)
    distances = shortest_path(adjacency, directed=protocol.directed, unweighted=True)
    reachable = np.isfinite(distances)
    positions = np.zeros((n, n), dtype=np.int16)
    positions[reachable] = np.minimum(distances[reachable] + 1, protocol.bias_size - 1).astype(np.int16)

    # Proxy definitions: class-conditioned historical behavior, observed cascade
    # authority and neighbor stance, rather than absent text/follower features.
    total = historical.sum(1)
    interest = np.zeros((n, 3))
    interest[:, :2] = (historical + 0.5) / (total[:, None] + 1)
    interest[:, 2] = (recent_last == 2).astype(float)
    activity = np.log1p(shares + protocol.gamma * originals)
    propensity = (shares + 0.5) / (total + 1)
    intrinsic = interest * activity[:, None] * propensity[:, None]

    outgoing = np.asarray(adjacency.sum(1)).ravel()
    incoming = np.asarray(adjacency.sum(0)).ravel()
    authority = np.asarray(adjacency @ (incoming / max(incoming.mean(), 1.0))).ravel() / np.maximum(outgoing, 1)
    stance = np.eye(3)[last]
    peers = np.asarray(adjacency @ stance) / np.maximum(outgoing[:, None], 1)
    current_counts = np.bincount(recent_last, minlength=3)
    prior_last = np.where(prior.sum(1) > 0, prior.argmax(1), 2)
    prior_counts = np.bincount(prior_last, minlength=3)
    delta = (current_counts - prior_counts) / max(n, 1)
    extrinsic = authority[:, None] * peers * delta[None, :]
    drivers = np.stack([intrinsic, extrinsic], axis=1).astype(np.float32)
    return {
        "drivers": drivers, "positions": positions, "target": target,
        "last": last, "recent_last": recent_last,
        "cutoff": float(cutoff), "target_end": float(next_cutoff),
        "edges": len(seen_pairs), "ambiguous_targets": len(ambiguous),
        "clipped_pairs": int((reachable & (distances + 1 >= protocol.bias_size)).sum()),
    }


def prepare_event(path, protocol):
    records, meta, sha = load_bundle(path)
    bounds, nodes = plan(records, protocol)
    snapshots = [snapshot(records, nodes,
                          bounds[i - 1] if i else min(r["t"] for r in records) - 1,
                          bounds[i], bounds[i + 1], protocol)
                 for i in range(len(bounds) - 1)]
    train, valid = chronological_split(len(snapshots), protocol)
    # Fit per-driver scales exclusively to training inputs, with no target access.
    scale = np.maximum(np.std(np.concatenate([s["drivers"] for s in snapshots[:train]], axis=0), axis=0), 0.1)
    for s in snapshots:
        s["drivers"] = s["drivers"] / scale[None, :, :]
    information = {
        "event": Path(path).stem, "sha256": sha, "source_meta": meta,
        "protocol": asdict(protocol), "labels": LABELS,
        "cohort_nodes": len(nodes), "all_record_users": len({r["uid"] for r in records}),
        "known_at_calibration_only": True, "node_ids": nodes,
        "boundaries": bounds.tolist(), "train_end": train, "validation_end": valid,
        "driver_scale": scale.tolist(),
        "split_class_counts": {name: np.bincount(np.concatenate([s["target"] for s in snapshots[a:b]]), minlength=3).tolist()
                               for name, a, b in [("train", 0, train), ("validation", train, valid), ("test", valid, len(snapshots))]},
        "ambiguous_targets": sum(s["ambiguous_targets"] for s in snapshots),
        "clipped_pairs_per_snapshot": [s["clipped_pairs"] for s in snapshots],
    }
    return snapshots, information


def audit(data_dir):
    events = []
    for path in sorted(Path(data_dir).glob("*.pkl")):
        records, meta, sha = load_bundle(path)
        times = [r["t"] for r in records]
        events.append({
            "event": path.stem, "sha256": sha, "records": len(records),
            "users": len({r["uid"] for r in records}),
            "fields": sorted(set().union(*(r.keys() for r in records))),
            "mal_counts": dict(Counter(r["mal"] for r in records)),
            "start": min(times), "end": max(times), "hours": (max(times) - min(times)) / 3600,
            "source_meta": meta,
        })
    return {"events": events, "label_semantics": LABELS,
            "missing": ["explicit anti-rumor stance", "raw text", "full follower graph", "complete user histories"],
            "excluded_features": ["recv", "fans", "t0", "bundle.node_order"],
            "exclusion_reason": "Final counters and upstream preselection lack snapshot-time provenance."}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="external/CAGE/data/event_datasets")
    parser.add_argument("--output", default="artifacts/data_audit.json")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit(args.data_dir), indent=2, ensure_ascii=False))
    print(path)
