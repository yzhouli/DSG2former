"""Construct a single source-disjoint PHEME dataset with auditable weak labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from sklearn.cluster import KMeans
import torch

from dsg2former.participation import PastFeatures, uid_key


class InformationHistory(PastFeatures):
    """Actual records update activity/edges, including unlabeled participation."""
    def __init__(self):
        super().__init__()
        self.observed = set()

    def observe_window(self, records):
        labeled = [r for r in records if r["action"] in (0, 1)]
        super().observe_window([{**r, "mal": 1 - r["action"]} for r in labeled])
        for r in records:
            u = uid_key(r["uid"])
            self.observed.add(u)
            if r["action"] not in (0, 1):
                if r.get("parent") is None:
                    self.originals[u] += 1
                else:
                    self.shares[u] += 1
                    v = uid_key(r["parent"])
                    if u != v:
                        self.parents[u].add(v)
        self.previous_users = {uid_key(r["uid"]) for r in records}
        self.previous_counts[2] = len(self.observed - self.previous_users)

    def graph(self, users):
        graph = super().graph(users)
        nodes = graph["node_ids"]
        # Activity includes related and unrelated observed replies. Stance
        # fractions describe only previously classifiable actions.
        counts = np.stack([self.counts.get(u, np.zeros(2)) for u in nodes])
        known = counts.sum(1)
        interest = np.zeros((len(nodes), 3))
        interest[:, :2] = (counts + .5) / (known[:, None] + 1)
        interest[:, 2] = [float(u not in self.previous_users) for u in nodes]
        shares = np.asarray([self.shares[u] for u in nodes], dtype=float)
        originals = np.asarray([self.originals[u] for u in nodes], dtype=float)
        inner = interest * np.log1p(shares + .23 * originals)[:, None] * ((shares + .5) / (shares + originals + 1))[:, None]
        graph["drivers"][:, 0] = inner
        graph["seen_before"] = np.asarray([u in self.observed for u in users])
        return graph


def make_records(thread, labels):
    by_id = {r["id"]: r["uid"] for r in [thread["source"]] + thread["reactions"]}
    def record(r, label):
        return {"uid": r["uid"], "t": r["timestamp"], "action": int(label),
                "parent": by_id.get(r["reply_to_id"], r["reply_to_uid"]), "id": r["id"]}
    root = record(thread["source"], 0)
    root["parent"] = None
    return root, [record(r, label) for r, label in zip(thread["reactions"], labels)]


def make_thread(thread, labels, other_pool, seed, windows):
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(thread["thread_id"]) % (2**32)]))
    initial, records = make_records(thread, labels)
    history = InformationHistory()
    history.observe_window([initial])
    snapshots, sample_rows = [], []
    if not records:
        return snapshots, sample_rows
    times = np.asarray([r["t"] for r in records])
    if np.any(times < initial["t"]):
        raise ValueError("Reaction predates its source")
    ends = np.unique(np.quantile(times, np.linspace(0, 1, windows + 1)[1:]))
    group = np.searchsorted(ends, times, side="left")
    offset = 0
    for w, end in enumerate(ends):
        indices = np.flatnonzero(group == w)
        current = [records[i] for i in indices]
        active = {r["uid"] for r in current}
        positive = [(int(i), records[i]) for i in indices if records[i]["action"] in (0, 1)]
        negatives = []
        future = [(int(i), records[i]) for i in np.flatnonzero(group == w + 2) if records[i]["uid"] not in active]
        count = len(positive)
        if future:
            for j in rng.integers(0, len(future), size=min(count, len(future))):
                i, r = future[j]
                negatives.append((r["uid"], "same_information_t_plus_2", thread["thread_id"], r["id"]))
        # Rejection sampling keeps raw-record frequency weights and replacement
        # duplicates without rebuilding a global unique-user pool.
        attempts = 0
        while len(negatives) < count:
            r = other_pool[int(rng.integers(len(other_pool)))]
            attempts += 1
            if r[1] != thread["thread_id"] and r[0] not in active:
                negatives.append((r[0], "other_information", r[1], r[2]))
            if attempts > 10000 + count * 100:
                raise RuntimeError("No eligible cross-information negative")
        rows = [{"uid": r["uid"], "label": r["action"], "reaction_index": i,
                 "origin": "actual", "source_thread_id": thread["thread_id"], "source_record_id": r["id"]}
                for i, r in positive]
        rows.extend({"uid": u, "label": 2, "reaction_index": -1, "origin": origin,
                     "source_thread_id": tid, "source_record_id": rid} for u, origin, tid, rid in negatives)
        assert all(row["uid"] not in active for row in rows if row["label"] == 2)
        if rows:
            snapshot = history.graph([uid_key(row["uid"]) for row in rows])
            snapshot.update({"target": np.asarray([row["label"] for row in rows], dtype=np.int64),
                             "window": w, "end": float(end),
                             "cutoff": float(initial["t"] if w == 0 else ends[w - 1]),
                             "state_update_users": {uid_key(r["uid"]) for r in current},
                             "row_start": offset})
            snapshots.append(snapshot)
            for row in rows:
                row["window"] = w
            sample_rows.extend(rows)
            offset += len(rows)
        history.observe_window(current)
    assert sorted(r["reaction_index"] for r in sample_rows if r["origin"] == "actual") == list(np.flatnonzero(np.asarray(labels) >= 0))
    return snapshots, sample_rows


def fit_stance_threshold(probability, training_related, target_anti_rate):
    """Match a training-only class target; identical scores keep one label."""
    if not 0 < target_anti_rate < 1:
        raise ValueError("target_anti_rate must lie strictly between 0 and 1")
    probability = np.asarray(probability, dtype=np.float64)
    scores = probability[:, 1] / np.maximum(probability[:, :2].sum(1), 1e-12)
    values, counts = np.unique(scores[training_related], return_counts=True)
    if len(values) < 2:
        raise ValueError("Training stance scores cannot support a threshold split")
    candidates = counts.sum() - np.cumsum(counts)[:-1]
    i = int(np.argmin(np.abs(candidates - target_anti_rate * counts.sum())))
    return float((values[i] + values[i + 1]) / 2)


def forecastable_mask(threads):
    return np.asarray([r["id"] != t["source"]["id"] and r["timestamp"] > t["source"]["timestamp"]
                       for t in threads for r in t["reactions"]], dtype=bool)


def build(root, seed=42, windows=30, output=None, target_anti_rate=None, exclude_initial_context=False):
    source_root = Path(root)
    root = Path(output) if output else source_root
    previous = json.loads((source_root / "dataset_manifest.json").read_text()) if (source_root / "dataset_manifest.json").exists() else None
    if (target_anti_rate is not None or exclude_initial_context) and root.resolve() == source_root.resolve():
        raise ValueError("Changed weak labels require a separate --output directory")
    if root.resolve() != source_root.resolve():
        root.mkdir(parents=True, exist_ok=True)
        if (root / "dataset_manifest.json").exists():
            raise FileExistsError("Refusing to overwrite a completed dataset version")
        for name in ["threads.jsonl", "raw_audit.json", "bert_similarity.npz", "embedding_metadata.json", "stance_probabilities.npz", "stance_metadata.json"]:
            shutil.copy2(source_root / name, root / name)
    threads = [json.loads(line) for line in (root / "threads.jsonl").read_text().splitlines()]
    similarity = np.load(root / "bert_similarity.npz")["similarity"]
    stance = np.load(root / "stance_probabilities.npz")
    offsets = stance["reaction_offsets"]
    assert len(similarity) == len(stance["label"]) == sum(len(t["reactions"]) for t in threads)
    training = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i, t in enumerate(threads) if t["split"] == "train"])
    # No human relatedness labels exist. A predeclared two-cluster calibration
    # on training similarities alone supplies a reproducible weak filter.
    if previous and root.resolve() != source_root.resolve():
        calibration = previous["similarity_calibration"].copy()
        centers, threshold = calibration["centers"], calibration["threshold"]
    else:
        clusters = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(similarity[training, None])
        centers = sorted(float(c) for c in clusters.cluster_centers_.ravel())
        threshold = float(np.mean(centers))
    related = similarity >= threshold
    forecastable = forecastable_mask(threads) if exclude_initial_context else np.ones(len(related), dtype=bool)
    probability = stance["probability"].astype(np.float64)
    deny_score = probability[:, 1] / np.maximum(probability[:, :2].sum(1), 1e-12)
    training_related = np.zeros(len(related), dtype=bool)
    training_related[training] = related[training] & forecastable[training]
    stance_threshold = .5 if target_anti_rate is None else fit_stance_threshold(probability, training_related, target_anti_rate)
    binary_label = (deny_score > stance_threshold).astype(np.int8)
    labels = np.where(related & forecastable, binary_label, -1).astype(np.int8)
    confidence = probability[np.arange(len(probability)), binary_label]
    binary_confidence = np.where(binary_label == 1, deny_score, 1 - deny_score)
    np.savez_compressed(root / "action_labels.npz", label=labels, related=related,
                        forecastable=forecastable,
                        similarity=similarity, reaction_offsets=offsets,
                        original_confidence=confidence,
                        conditional_binary_confidence=binary_confidence,
                        conditional_deny_score=deny_score,
                        teacher_binary_argmax=stance["label"],
                        original_argmax=stance["original_argmax"])
    pools = defaultdict(list)
    for t in threads:
        for r in [t["source"]] + t["reactions"]:
            pools[t["split"]].append((r["uid"], t["thread_id"], r["id"]))
    data, counts = [], defaultdict(Counter)
    with (root / "candidate_rows.jsonl").open("w") as stream:
        for i, t in enumerate(threads):
            a, b = offsets[i:i + 2]
            snapshots, rows = make_thread(t, labels[a:b], pools[t["split"]], seed, windows)
            data.append({"thread_id": t["thread_id"], "event": t["event"], "split": t["split"],
                         "snapshots": snapshots, "raw_reactions": len(t["reactions"]),
                         "related_reactions": int((labels[a:b] >= 0).sum()), "rows": len(rows)})
            counts[t["split"]].update(row["origin"] for row in rows)
            stream.write(json.dumps({"thread_id": t["thread_id"], "split": t["split"], "rows": rows}) + "\n")
    train_drivers = [s["drivers"] for t in data if t["split"] == "train" for s in t["snapshots"]]
    scale = np.maximum(np.concatenate(train_drivers).std(0), .1)
    for t in data:
        for s in t["snapshots"]:
            s["drivers"] /= scale[None]
    torch.save(data, root / "graphs.pt")
    report = {"dataset": "PHEME information-level action forecasting", "information_samples": len(data),
              "labels": ["pro_rumor", "anti_rumor", "non_participation"],
              "label_generation": "Frozen BERT similarity relevance filter, followed by frozen target-aware BERTweet; force support/deny choice as requested",
              "similarity_calibration": {"method": "Two-cluster KMeans on training pairs only; midpoint of centers", "centers": centers, "threshold": threshold},
              "stance_label_quality": "Weak labels, not human ground truth; RumourEval-trained teacher may overlap PHEME",
              "neutral_policy": "Related query/comment behaviors forced to support/deny by the documented conditional-deny threshold; all four original probabilities retained",
              "stance_threshold": {"score": "p(deny)/(p(support)+p(deny))", "decision": "anti-rumor iff score > threshold",
                                   "threshold": stance_threshold, "fit_split": "train" if target_anti_rate is not None else None,
                                   "target_training_anti_rate_among_participants": target_anti_rate,
                                   "actual_training_anti_rate_among_participants": float(binary_label[training_related].mean()),
                                   "changed_related_labels_from_teacher_argmax": int((related & forecastable & (binary_label != stance["label"])).sum())},
              "source_dataset": str(source_root),
              "related_reply_count": int(related.sum()),
              "supervised_related_reply_count": int((related & forecastable).sum()),
              "initial_context_exclusion": exclude_initial_context,
              "initial_context_rows_retained_without_supervision": int((~forecastable).sum()),
              "initial_context_policy": "All raw records retained; when enabled, source-ID copies and records at/before source timestamp are known context, excluded from next-step supervision and threshold fitting",
              "unrelated_reply_count": int((~related).sum()),
              "related_query_comment_forced_binary_count": int((related & (stance["original_argmax"] >= 2)).sum()),
              "related_original_confidence_below_half_count": int((related & (confidence < .5)).sum()),
              "unrelated_policy": "Raw records retained as observed activity/edges, excluded from stance supervision; never labeled inactive",
              "window_count_max": windows, "sampling_seed": seed, "deduplicate": False,
              "root_context": "Source post is known initial context, never a prediction target",
              "negative_policy": "One absent-user negative per labeled action; prefer same-thread t+2, then sample records with replacement from other information in the same split",
              "feature_policy": "Only source and previously completed windows; no target/future text, similarity, confidence, candidate provenance or future edges as model inputs",
              "driver_scale_training_only": scale.tolist(),
              "evaluation": "Compute metrics over each complete held-out information sequence, then equal-weight mean over evaluable information samples; empty samples reported as unavailable",
              "splits": {split: {"information_samples": sum(t["split"] == split for t in data),
                                  "evaluable_information_samples": sum(t["split"] == split and bool(t["rows"]) for t in data),
                                  "raw_reactions": sum(t["raw_reactions"] for t in data if t["split"] == split),
                                  "class_counts": np.bincount([int(y) for t in data if t["split"] == split for s in t["snapshots"] for y in s["target"]], minlength=3).tolist(),
                                  "candidate_origins": dict(counts[split])} for split in ["train", "validation", "test"]},
              "hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                         ["threads.jsonl", "bert_similarity.npz", "stance_probabilities.npz", "action_labels.npz", "candidate_rows.jsonl", "graphs.pt"]}}
    (root / "dataset_manifest.json").write_text(json.dumps(report, indent=2))
    import csv
    with (root / "statistics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "information_samples", "evaluable_information_samples", "pro_rumor_rows", "anti_rumor_rows", "non_participation_rows"])
        for name, subset in report["splits"].items():
            writer.writerow([name, subset["information_samples"], subset["evaluable_information_samples"], *subset["class_counts"]])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/pheme_information")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--windows", type=int, default=30)
    p.add_argument("--output")
    p.add_argument("--target-anti-rate", type=float)
    p.add_argument("--exclude-initial-context", action="store_true")
    a = p.parse_args()
    build(a.input, a.seed, a.windows, a.output, a.target_anti_rate, a.exclude_initial_context)
