"""Chronological arriving-node classification: rumor vs. non-rumor."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .data import load_bundle


LABELS = ["rumor", "non_rumor"]


def stance_label(record):
    return 0 if record["mal"] == 1 else 1


def time_partitions(records):
    cutoffs = np.quantile([r["t"] for r in records], [.8, .9]).tolist()
    groups = [[r for r in records if r["t"] <= cutoffs[0]],
              [r for r in records if cutoffs[0] < r["t"] <= cutoffs[1]],
              [r for r in records if r["t"] > cutoffs[1]]]
    return cutoffs, groups


def statistics(data_dir):
    output = []
    for path in sorted(Path(data_dir).glob("*.pkl")):
        records, meta, digest = load_bundle(path)
        cutoffs, groups = time_partitions(records)
        counts = np.bincount([stance_label(r) for r in records], minlength=2)
        rumor_users = {r["uid"] for r in records if r["mal"] == 1}
        neutral_users = {r["uid"] for r in records if r["mal"] == 0}
        pairs = {(r["uid"], r["parent"]) for r in records if r.get("parent") is not None and r["uid"] != r["parent"]}
        output.append({
            "event": path.stem, "source_sha256": digest,
            "records": len(records), "author_users": len({r["uid"] for r in records}),
            "observed_cascade_edges": len(pairs),
            "rumor_records": int(counts[0]), "non_rumor_records": int(counts[1]),
            "rumor_users": len(rumor_users), "non_rumor_users": len(neutral_users),
            "users_with_both_original_labels": len(rumor_users & neutral_users),
            "cutoffs": cutoffs,
            "splits": {name: {"records": len(group), "author_users": len({r["uid"] for r in group}),
                              "class_counts": np.bincount([stance_label(r) for r in group], minlength=2).tolist()}
                       for name, group in zip(["train", "validation", "test"], groups)},
        })
    return {"task": "temporal_arriving_node_binary_classification", "labels": LABELS,
            "randomization": None,
            "split": "timestamp quantiles 80:10:10; equal timestamps stay together",
            "count_unit": "records; a user may have both original labels across time", "events": output}


@dataclass
class TemporalProtocol:
    batch_records: int = 512
    bias_size: int = 50
    gamma: float = 0.23
    directed: bool = True


def chronological_batches(group, batch_records):
    """Bound graph computation without fixing/dropping users or splitting ties."""
    if batch_records < 1:
        raise ValueError("batch_records must be positive")
    batch = []
    for record in group:
        if len(batch) >= batch_records and record["t"] != batch[-1]["t"]:
            yield batch
            batch = []
        batch.append(record)
    if batch:
        yield batch


class RollingGraphBuilder:
    def __init__(self, protocol):
        self.protocol = protocol
        self.counts = defaultdict(lambda: np.zeros(2))
        self.shares, self.originals = Counter(), Counter()
        self.last = {}
        self.parents = defaultdict(set)
        self.total_seen = set()
        self.class_totals = np.zeros(2)
        self.previous_totals = np.zeros(2)

    def build(self, records, split):
        authors = {r["uid"] for r in records}
        nodes = sorted(authors | {r["parent"] for r in records if r.get("parent") is not None}, key=lambda u: (type(u).__name__, str(u)))
        idx = {u: i for i, u in enumerate(nodes)}
        n = len(nodes)
        current_shares, current_originals = Counter(), Counter()
        pairs = {(idx[u], idx[v]) for u in nodes for v in self.parents.get(u, ()) if v in idx and u != v}
        for r in records:
            parent = r.get("parent")
            if parent is None:
                current_originals[r["uid"]] += 1
            else:
                current_shares[r["uid"]] += 1
                if parent != r["uid"]:
                    pairs.add((idx[r["uid"]], idx[parent]))
        pairs = sorted(pairs)
        src = np.array([p[0] for p in pairs], dtype=np.int64)
        dst = np.array([p[1] for p in pairs], dtype=np.int64)
        graph = csr_matrix((np.ones(len(pairs)), (src, dst)), shape=(n, n))
        if not self.protocol.directed:
            graph = graph.maximum(graph.T)
        distance = shortest_path(graph, directed=self.protocol.directed, unweighted=True)
        finite = np.isfinite(distance)
        positions = np.zeros((n, n), dtype=np.int16)
        positions[finite] = np.minimum(distance[finite] + 1, self.protocol.bias_size - 1).astype(np.int16)

        # No mal/stance from the current batch enters any input feature.
        counts = np.stack([self.counts.get(u, np.zeros(2)) for u in nodes])
        total = counts.sum(1)
        shares = np.array([self.shares[u] + current_shares[u] for u in nodes])
        originals = np.array([self.originals[u] + current_originals[u] for u in nodes])
        interest = (counts + .5) / (total[:, None] + 1)
        activity = np.log1p(shares + self.protocol.gamma * originals)
        propensity = (shares + .5) / (shares + originals + 1)
        inner = interest * activity[:, None] * propensity[:, None]
        outgoing = np.asarray(graph.sum(1)).ravel()
        incoming = np.asarray(graph.sum(0)).ravel()
        authority = np.asarray(graph @ (incoming / max(incoming.mean(), 1))).ravel() / np.maximum(outgoing, 1)
        historical_stance = np.zeros((n, 2))
        for i, u in enumerate(nodes):
            if u in self.last:
                historical_stance[i, self.last[u]] = 1
        peers = np.asarray(graph @ historical_stance) / np.maximum(outgoing[:, None], 1)
        trend = (self.class_totals - self.previous_totals) / max(1, (self.class_totals - self.previous_totals).sum())
        outer = authority[:, None] * peers * trend[None]
        target_nodes = np.array([idx[r["uid"]] for r in records], dtype=np.int64)
        last_labels = np.array([self.last.get(r["uid"], -1) for r in records], dtype=np.int64)
        seen = np.array([r["uid"] in self.last for r in records], dtype=bool)
        result = {"node_ids": nodes, "drivers": np.stack([inner, outer], axis=1).astype(np.float32),
                  "positions": positions, "target_nodes": target_nodes,
                  "target": np.array([stance_label(r) for r in records], dtype=np.int64),
                  "seen_before": seen, "last": last_labels, "split": split,
                  "cutoff": float(records[0]["t"]), "target_end": float(records[-1]["t"]),
                  "edges": len(pairs), "record_users": [r["uid"] for r in records]}
        # Reveal this batch's labels only to subsequent batches (rolling protocol).
        self.previous_totals = self.class_totals.copy()
        for r in records:
            user, k = r["uid"], stance_label(r)
            self.counts[user][k] += 1
            self.class_totals[k] += 1
            self.last[user] = k
            if r.get("parent") is None:
                self.originals[user] += 1
            else:
                self.shares[user] += 1
                if user != r["parent"]:
                    self.parents[user].add(r["parent"])
        self.total_seen.update(nodes)
        return result


def prepare_event(path, protocol):
    records, meta, sha = load_bundle(path)
    cutoffs, groups = time_partitions(records)
    builder = RollingGraphBuilder(protocol)
    snapshots = []
    split_ends = []
    for name, group in zip(["train", "validation", "test"], groups):
        snapshots.extend(builder.build(batch, name) for batch in chronological_batches(group, protocol.batch_records))
        split_ends.append(len(snapshots))
    train, valid = split_ends[:2]
    if train < 1 or train == valid or valid == len(snapshots):
        raise ValueError("Empty time partition")
    scales = np.maximum(np.std(np.concatenate([s["drivers"] for s in snapshots[:train]]), axis=0), .1)
    for s in snapshots:
        s["drivers"] /= scales[None]
    information = {"event": Path(path).stem, "source_sha256": sha, "source_meta": meta,
                   "task": "temporal_arriving_node_binary_classification", "labels": LABELS,
                   "protocol": asdict(protocol), "timestamp_cutoffs": cutoffs,
                   "train_end": train, "validation_end": valid,
                   "all_record_users": len({r["uid"] for r in records}),
                   "records": len(records), "fixed_cohort": False, "dropped_records": 0,
                   "driver_scale": scales.tolist(),
                   "snapshot_nodes": [len(s["node_ids"]) for s in snapshots],
                   "snapshot_records": [len(s["target"]) for s in snapshots],
                   "split_class_counts": {name: np.bincount([stance_label(r) for r in group], minlength=2).tolist()
                                          for name, group in zip(["train", "validation", "test"], groups)},
                   "split_unique_authors": {name: len({r["uid"] for r in group}) for name, group in zip(["train", "validation", "test"], groups)}}
    return snapshots, information


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="external/CAGE/data/event_datasets")
    parser.add_argument("--output", default="artifacts/temporal_binary_statistics.json")
    args = parser.parse_args()
    report = statistics(args.data_dir)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    rows = [{k: e[k] for k in ["event", "records", "author_users", "observed_cascade_edges", "rumor_records", "non_rumor_records"]}
            | {name + "_records": e["splits"][name]["records"] for name in ["train", "validation", "test"]}
            for e in report["events"]]
    with target.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
