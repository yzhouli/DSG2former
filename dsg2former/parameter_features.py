"""Explicit validation-study variants of otherwise frozen paper features."""
import json
from pathlib import Path

import numpy as np

from .paper_features import HistoricalCorpus, popularity_difference


def apply_configuration(data, root, history, config):
    """Transform only the supplied partitions; labels/graph relations are fixed.

    At the baseline settings, driver arrays remain bitwise unchanged. The
    class-conditioned variant is an extension, not a recovered paper formula.
    """
    k, half_life, external = config["top_k"], config["half_life"], config["external"]
    if k == 10 and half_life == 3600 and external == "broadcast":
        return
    root, history = Path(root), Path(history)
    threads = {t["thread_id"]: t for t in map(json.loads, (root / "threads.jsonl").read_text().splitlines())}
    records = list(map(json.loads, (history / "records.jsonl").read_text().splitlines()))
    embeddings = np.load(history / "embeddings.npy", mmap_mode="r")
    sentiment = np.load(history / "sentiment.npy", mmap_mode="r")
    stage = {"train": 0, "validation": 1, "test": 2}
    thread_stage = {tid: stage[t["split"]] for tid, t in threads.items()}
    record_stage = np.asarray([thread_stage.get(r["thread_id"], -1) for r in records])
    corpora = {split: HistoricalCorpus(records, embeddings, sentiment, k,
                                     allowed_records=record_stage <= stage[split]) for split in {t["split"] for t in data}}
    for item in data:
        thread = threads[item["thread_id"]]
        corpus = corpora[item["split"]]
        source = corpus.path_index[thread["source"]["archive_path"]]
        known = {thread["source"]["uid"]}
        previous_active, before_previous_active = known.copy(), set()
        previous_cutoff = thread["source"]["timestamp"]
        raw = thread["reactions"]
        for s in item["snapshots"]:
            n = len(s["node_ids"])
            cutoff = s["feature_cutoff_exclusive"]
            drivers = s["drivers"].copy()
            intrinsic = None
            if k != 10 or external != "broadcast":
                intrinsic = [corpus.intrinsic(u, cutoff, source) for u in s["node_ids"]]
            if k != 10:
                for j, (interest, activity, rate, _) in enumerate(intrinsic):
                    drivers[j, 0] = interest.astype(np.float64) * activity * rate
            if half_life != 3600:
                social = s["social_positions"] == 2
                indegree = social.sum(0)
                authority = np.zeros(n)
                if indegree.mean() > 0:
                    authority = (social @ (indegree / indegree.mean())) / np.maximum(social.sum(1), 1)
                friends = [corpus.neighbors(u, cutoff, source)[1] for u in s["node_ids"]]
                friend_score = np.array([len(v & known)/len(v) if v else 0. for v in friends])
                pop = popularity_difference(len(previous_active), len(before_previous_active),
                                            s["cutoff"]-previous_cutoff, half_life)
                drivers[:, 1] = np.repeat((authority*friend_score*pop)[:, None], 3, axis=1)
            if external == "class_conditioned_sentiment":
                # Decompose the scalar by prior social neighbors' historical
                # positive/negative/neutral interest. No stance labels enter.
                # Three times q preserves the average external scalar; uniform
                # q exactly recovers broadcasting. Semantic edges are not used
                # as follower/friend evidence.
                social = s["social_positions"] == 2
                usable = np.array([v[3] > 0 for v in intrinsic])
                interest = np.asarray([v[0] for v in intrinsic], dtype=np.float64)
                for j in range(n):
                    neighbors = (social[j] | social[:, j]) & usable
                    if neighbors.any():
                        q = interest[neighbors].mean(0)
                        q /= max(q.sum(), 1e-100)
                        drivers[j, 1] *= (3*q).astype(np.float32)
            s["drivers"] = drivers
            # The temporal bookkeeping is actual chain history, rather than
            # semantic context users from state_update_users.
            actual = {r["uid"] for r in raw if r["timestamp"] <= s["end"] and
                      (s["window"] == 0 or r["timestamp"] > s["cutoff"])}
            known |= actual
            before_previous_active, previous_active = previous_active, actual
            previous_cutoff = s["cutoff"]
