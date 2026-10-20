"""Run DSG2former on the no-deduplication, time-split candidate dataset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import torch
from torch.nn import functional as F

from .data import load_bundle
from .experiment import code_fingerprint, json_write, set_seed
from .metrics import classification_metrics
from .model import DSG2former


LABELS = ["rumor", "non_rumor", "non_participation"]


def uid_key(user):
    return f"{type(user).__name__}:{user}"


class PastFeatures:
    """Contains actual raw observations only; never synthetic candidate history."""
    def __init__(self):
        self.counts = defaultdict(lambda: np.zeros(2))
        self.shares, self.originals = Counter(), Counter()
        self.parents = defaultdict(set)
        self.last = {}
        self.previous_users = set()
        self.previous_counts = np.zeros(3)
        self.before_previous_counts = np.zeros(3)

    def observe_window(self, records):
        active = set()
        last_in_window = {}
        for r in records:
            u = uid_key(r["uid"])
            k = 0 if r["mal"] == 1 else 1
            self.counts[u][k] += 1
            self.last[u] = k
            active.add(u)
            last_in_window[u] = k
            if r.get("parent") is None:
                self.originals[u] += 1
            else:
                self.shares[u] += 1
                v = uid_key(r["parent"])
                if u != v:
                    self.parents[u].add(v)
        self.before_previous_counts = self.previous_counts.copy()
        self.previous_counts = np.array([sum(k == 0 for k in last_in_window.values()),
                                         sum(k == 1 for k in last_in_window.values()),
                                         len(self.last) - len(active)], dtype=float)
        self.previous_users = active

    def graph(self, candidate_users, bias_size=50):
        # Sample rows stay intact. A user's graph state is shared by all its rows.
        candidate = set(candidate_users)
        context = {p for u in candidate for p in self.parents.get(u, ())}
        nodes = sorted(candidate | context)
        index = {u: i for i, u in enumerate(nodes)}
        n = len(nodes)
        pairs = [(index[u], index[v]) for u in nodes for v in self.parents.get(u, ()) if v in index]
        src = np.array([p[0] for p in pairs], dtype=np.int64)
        dst = np.array([p[1] for p in pairs], dtype=np.int64)
        adjacency = csr_matrix((np.ones(len(pairs)), (src, dst)), shape=(n, n))
        distances = shortest_path(adjacency, directed=True, unweighted=True)
        finite = np.isfinite(distances)
        positions = np.zeros((n, n), dtype=np.int16)
        positions[finite] = np.minimum(distances[finite] + 1, bias_size - 1).astype(np.int16)
        counts = np.stack([self.counts.get(u, np.zeros(2)) for u in nodes])
        total = counts.sum(1)
        interest = np.zeros((n, 3))
        interest[:, :2] = (counts + .5) / (total[:, None] + 1)
        interest[:, 2] = [float(u not in self.previous_users) for u in nodes]
        shares = np.array([self.shares[u] for u in nodes], dtype=float)
        originals = np.array([self.originals[u] for u in nodes], dtype=float)
        inner = interest * np.log1p(shares + .23 * originals)[:, None] * ((shares + .5) / (total + 1))[:, None]
        outgoing = np.asarray(adjacency.sum(1)).ravel()
        incoming = np.asarray(adjacency.sum(0)).ravel()
        authority = np.asarray(adjacency @ (incoming / max(incoming.mean(), 1))).ravel() / np.maximum(outgoing, 1)
        stance = np.zeros((n, 3))
        for i, u in enumerate(nodes):
            if u in self.last:
                stance[i, self.last[u] if u in self.previous_users else 2] = 1
        peers = np.asarray(adjacency @ stance) / np.maximum(outgoing[:, None], 1)
        trend = (self.previous_counts - self.before_previous_counts) / max(1, self.previous_counts.sum())
        outer = authority[:, None] * peers * trend[None]
        return {"node_ids": nodes, "drivers": np.stack([inner, outer], axis=1).astype(np.float32),
                "positions": positions, "sample_nodes": np.array([index[u] for u in candidate_users], dtype=np.int64),
                "seen_before": np.array([u in self.last for u in candidate_users], dtype=bool),
                "edges": len(pairs)}


def prepare_event(event, dataset_dir, raw_dir):
    dataset_dir = Path(dataset_dir)
    metadata = json.loads((dataset_dir / f"{event}.json").read_text())
    path = dataset_dir / f"{event}.npz"
    records, _, raw_sha = load_bundle(Path(raw_dir) / f"{event}.pkl")
    if raw_sha != metadata["source_sha256"] or metadata["labels"] != LABELS:
        raise ValueError("Source hash or class mapping mismatch")
    arrays = dict(np.load(path, allow_pickle=False))
    boundaries = arrays["boundaries"]
    raw_windows = np.searchsorted(boundaries[1:], np.array([r["t"] for r in records]), side="left")
    past = PastFeatures()
    snapshots = []
    for w in range(len(boundaries) - 1):
        selected = np.flatnonzero(arrays["window_id"] == w)
        actual_records = [records[i] for i in np.flatnonzero(raw_windows == w)]
        if len(selected):
            graph = past.graph(arrays["uid"][selected])
            graph.update({"target": arrays["label"][selected].astype(np.int64),
                          "sample_indices": selected, "window": w,
                          "stage": int(arrays["stage"][selected[0]]),
                          "cutoff": float(boundaries[w]), "end": float(boundaries[w + 1]),
                          "state_update_users": {uid_key(r["uid"]) for r in actual_records}})
            snapshots.append(graph)
        # Current labels, timestamps, and edges become history only AFTER the
        # graph has been built. Negative provenance never enters the encoder.
        past.observe_window(actual_records)
    train_end = sum(s["stage"] == 0 for s in snapshots)
    validation_end = sum(s["stage"] < 2 for s in snapshots)
    scale = np.maximum(np.concatenate([s["drivers"] for s in snapshots[:train_end]]).std(0), .1)
    for s in snapshots:
        s["drivers"] /= scale[None]
    information = {"event": event, "labels": LABELS, "task": "offline_candidate_next_window_participation",
                   "source_sha256": raw_sha, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "train_end": train_end, "validation_end": validation_end,
                   "driver_scale": scale.tolist(), "samples": len(arrays["label"]),
                   "split_counts": metadata["splits"], "boundaries": boundaries.tolist(),
                   "graph_nodes_per_window": [len(s["node_ids"]) for s in snapshots],
                   "record_weights_preserved": True, "fixed_candidate_set": False,
                   "feature_policy": "Strictly before target window; no provenance, current activity or target labels in features",
                   "history_policy": "Observed raw labels become available after each completed window; synthetic negatives never update feature history"}
    assert sum(len(s["target"]) for s in snapshots) == information["samples"]
    return snapshots, information


def to_tensors(snapshots):
    return [{**s, "drivers": torch.from_numpy(s["drivers"]),
             "positions": torch.from_numpy(s["positions"]),
             "sample_nodes": torch.from_numpy(s["sample_nodes"]),
             "target": torch.from_numpy(s["target"])} for s in snapshots]


def forward_window(model, s, cache, device):
    nodes = s["node_ids"]
    dim = model.embedding.out_features
    previous = torch.zeros(len(nodes), dim, device=device)
    mask = torch.zeros(len(nodes), dtype=torch.bool, device=device)
    # State is aligned by user identity, not the row position in a changing graph.
    present = [(i, cache[u]) for i, u in enumerate(nodes) if u in cache]
    if present:
        indices = torch.tensor([i for i, _ in present], device=device)
        previous[indices] = torch.stack([v for _, v in present]).to(device)
        mask[indices] = True
    logits, spatial, state, _ = model(s["drivers"].to(device), s["positions"].to(device=device, dtype=torch.long),
                                       previous=previous, previous_mask=mask)
    # Only real, subsequently observed authors advance identity state. An unseen
    # synthetic candidate cannot acquire history merely by being sampled.
    for i, u in enumerate(nodes):
        if u in s["state_update_users"]:
            cache[u] = state[i].detach()
    selected = s["sample_nodes"].to(device)
    return logits[selected], spatial[selected]


@torch.no_grad()
def evaluate(model, data, end, device):
    model.eval()
    cache, predictions = {}, []
    for s in data[:end]:
        logits, _ = forward_window(model, s, cache, device)
        predictions.append(logits.softmax(-1).cpu().numpy())
    return predictions


def evaluate_slice(data, predictions, start, end):
    targets = np.concatenate([s["target"].cpu().numpy() for s in data[start:end]])
    p = np.concatenate(predictions[start:end])
    metrics = classification_metrics(targets, p)
    seen = np.concatenate([s["seen_before"] for s in data[start:end]])
    metrics["seen_history_samples"] = classification_metrics(targets[seen], p[seen]) if seen.any() else None
    metrics["no_history_samples"] = classification_metrics(targets[~seen], p[~seen]) if (~seen).any() else None
    return metrics


def run_event(event, args):
    snapshots, info = prepare_event(event, args.dataset_dir, args.raw_dir)
    data = to_tensors(snapshots)
    root = Path(args.output) / event
    root.mkdir(parents=True, exist_ok=True)
    json_write(root / "protocol.json", info)
    train_end, valid_end = info["train_end"], info["validation_end"]
    train_y = np.concatenate([s["target"].numpy() for s in data[:train_end]])
    counts = np.bincount(train_y, minlength=3)
    prior = (counts + 1) / (counts.sum() + 3)
    test_y = np.concatenate([s["target"].numpy() for s in data[valid_end:]])
    majority = classification_metrics(test_y, np.broadcast_to(prior, (len(test_y), 3)))
    json_write(root / "majority_baseline.json", majority)
    print(json.dumps({"event": event, "nodes": info["graph_nodes_per_window"], "samples": info["samples"]}), flush=True)
    results = []
    for seed in args.seeds:
        out = root / f"seed_{seed}"
        out.mkdir(exist_ok=True)
        result_file = out / "metrics.json"
        if args.resume and result_file.exists():
            old = json.loads(result_file.read_text())
            if (old["dataset_sha256"] != info["dataset_sha256"] or old["code_sha256"] != code_fingerprint()
                    or old["configuration"] != {k: v for k, v in vars(args).items() if k not in ("resume", "events", "seeds")}):
                raise ValueError("Resume configuration/source mismatch")
            results.append(old)
            continue
        set_seed(seed)
        started = time.monotonic()
        options = {"dim": args.dim, "heads": 3, "blocks": 2, "dropout": .1}
        model = DSG2former(**options).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        best_score, best_loss, best_epoch, patience, best_state = -1., float("inf"), 0, 0, None
        trace = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            cache, losses = {}, []
            for s in data[:train_end]:
                optimizer.zero_grad(set_to_none=True)
                logits, spatial = forward_window(model, s, cache, args.device)
                y = s["target"].to(args.device)
                loss = F.cross_entropy(spatial, y) + F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(loss.item())
            prediction = evaluate(model, data, valid_end, args.device)
            metrics = evaluate_slice(data, prediction, train_end, valid_end)
            score, ce = metrics["macro_f1"], metrics["cross_entropy"]
            improved = score > best_score + 1e-8
            if improved or (abs(score - best_score) <= 1e-8 and ce < best_loss - 1e-6):
                best_score, best_loss, best_epoch = score, ce, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0 if improved else patience + 1
            trace.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": metrics})
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps({"event": event, "seed": seed, "epoch": epoch, "validation_macro_f1": score}), flush=True)
            if patience >= args.patience:
                break
        model.load_state_dict(best_state)
        predictions = evaluate(model, data, len(data), args.device)
        result = {"event": event, "seed": seed, "labels": LABELS,
                  "task": info["task"], "dataset_sha256": info["dataset_sha256"],
                  "source_sha256": info["source_sha256"], "code_sha256": code_fingerprint(),
                  "configuration": {k: v for k, v in vars(args).items() if k not in ("resume", "events", "seeds")},
                  "best_epoch": best_epoch, "epochs_executed": len(trace),
                  "seconds": time.monotonic() - started, "samples": info["samples"],
                  "torch_version": torch.__version__, "numpy_version": np.__version__,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                  "validation": evaluate_slice(data, predictions, train_end, valid_end),
                  "test": evaluate_slice(data, predictions, valid_end, len(data)),
                  "majority_test": majority}
        torch.save({"state_dict": best_state, "options": options, "information": info,
                    "seed": seed, "best_epoch": best_epoch}, out / "model.pt")
        json_write(out / "training_history.json", trace)
        offsets = np.concatenate([[0], np.cumsum([len(s["target"]) for s in data])])
        np.savez_compressed(out / "predictions.npz", probabilities=np.concatenate(predictions),
                            target=np.concatenate([s["target"].numpy() for s in data]), offsets=offsets,
                            sample_indices=np.concatenate([s["sample_indices"] for s in data]),
                            seen_before=np.concatenate([s["seen_before"] for s in data]),
                            train_end=train_end, validation_end=valid_end)
        json_write(result_file, result)
        results.append(result)
        print(json.dumps({"event": event, "seed": seed, "seconds": round(result["seconds"], 2),
                          "test_macro_f1": result["test"]["macro_f1"], "test_micro_f1": result["test"]["micro_f1"]}), flush=True)
    return results


def summarize(output):
    groups = defaultdict(list)
    for path in Path(output).glob("*/seed_*/metrics.json"):
        result = json.loads(path.read_text())
        groups[result["event"]].append(result)
    rows = []
    for event, runs in sorted(groups.items()):
        row = {"event": event, "runs": len(runs), "test_samples": sum(runs[0]["test"]["support"])}
        for name in ["macro_f1", "micro_f1", "auc_ovr_macro_all3", "auc_ovr_macro_available"]:
            values = [r["test"][name] for r in runs if r["test"][name] is not None]
            row[name + "_mean"] = float(np.mean(values)) if values else None
            row[name + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        row["majority_macro_f1"] = runs[0]["majority_test"]["macro_f1"]
        rows.append(row)
    json_write(Path(output) / "summary.json", rows)
    if rows:
        with (Path(output) / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def verify_results(output, expected=None):
    paths = sorted(Path(output).glob("*/seed_*/metrics.json"))
    if expected is not None and len(paths) != expected:
        raise ValueError(f"Expected {expected} runs, found {len(paths)}")
    for path in paths:
        result = json.loads(path.read_text())
        with np.load(path.parent / "predictions.npz") as arrays:
            offsets = arrays["offsets"]
            train, valid = int(arrays["train_end"]), int(arrays["validation_end"])
            if len(arrays["sample_indices"]) != result["samples"]:
                raise ValueError("Sample rows were dropped")
            np.testing.assert_array_equal(np.sort(arrays["sample_indices"]), np.arange(result["samples"]))
            for name, a, b in [("validation", offsets[train], offsets[valid]), ("test", offsets[valid], offsets[-1])]:
                recalculated = classification_metrics(arrays["target"][a:b], arrays["probabilities"][a:b])
                for key in ["macro_f1", "micro_f1", "cross_entropy", "auc_ovr_macro_available", "auc_ovr_macro_all3"]:
                    if recalculated[key] is None:
                        assert result[name][key] is None
                    else:
                        np.testing.assert_allclose(result[name][key], recalculated[key], rtol=1e-6, atol=1e-8)
                np.testing.assert_array_equal(result[name]["support"], recalculated["support"])
                np.testing.assert_array_equal(result[name]["confusion_matrix"], recalculated["confusion_matrix"])
    return {"verified_runs": len(paths), "all_sample_rows_preserved": True, "metrics_recomputed_from_saved_probabilities": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/temporal_participation")
    parser.add_argument("--raw-dir", default="external/CAGE/data/event_datasets")
    parser.add_argument("--output", default="results/temporal_participation")
    parser.add_argument("--events", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2025, 2026])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    available = sorted(p.stem for p in Path(args.dataset_dir).glob("*.npz"))
    events = available if args.events == ["all"] else args.events
    if set(events) - set(available) or not events:
        parser.error("Missing event datasets")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for event in events:
        run_event(event, args)
        summarize(args.output)


if __name__ == "__main__":
    main()
