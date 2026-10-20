"""Fixed forecast horizons, record-preserving actions, and causal macro graphs."""
import math
import numpy as np

from .semantic_graph import augment_graph, eligible_evidence


def hourly_windows(thread, seconds=3600):
    if seconds <= 0:
        raise ValueError("Window duration must be positive")
    start = thread["source"]["timestamp"]
    times = np.array([r["timestamp"] for r in thread["reactions"]], dtype=np.float64)
    groups = np.ceil((times-start)/seconds).astype(np.int64)-1
    if np.any(times < start):
        raise ValueError("Reaction predates its source")
    count = max(1, int(groups.max())+1) if len(groups) else 1
    ends = start + np.arange(1, count+1)*seconds
    return ends, groups


def action_rows(thread, labels, groups, window, pool, rng):
    records = thread["reactions"]
    current = np.flatnonzero(groups == window)
    active = {records[int(i)]["uid"] for i in current}
    selected = [int(i) for i in current if labels[int(i)] >= 0]
    rows = [{"uid": records[i]["uid"], "label": int(labels[i]), "reaction_index": i,
             "origin": "actual", "source_thread_id": thread["thread_id"], "source_record_id": records[i]["id"],
             "window": window} for i in selected]
    count = len(rows)
    future = [int(i) for i in np.flatnonzero(groups == window+2) if records[int(i)]["uid"] not in active]
    negatives = []
    if future:
        for j in rng.integers(len(future), size=min(count, len(future))):
            r = records[future[int(j)]]
            negatives.append((r["uid"], "same_information_t_plus_2", thread["thread_id"], r["id"]))
    attempts = 0
    while len(negatives) < count:
        uid, tid, rid = pool[int(rng.integers(len(pool)))]
        if tid != thread["thread_id"] and uid not in active:
            negatives.append((uid, "other_information", tid, rid))
        attempts += 1
        if attempts > 10000+count*100:
            raise RuntimeError("No absent-user negative available")
    rows.extend({"uid": u, "label": 2, "reaction_index": -1, "origin": origin,
                 "source_thread_id": tid, "source_record_id": rid, "window": window}
                for u, origin, tid, rid in negatives)
    return rows


def new_participants(thread, indices, known_before):
    """Original records remain untouched; the target itself counts people."""
    source = thread["source"]
    actual = [thread["reactions"][int(i)] for i in indices
              if thread["reactions"][int(i)]["id"] != source["id"]
              and thread["reactions"][int(i)]["timestamp"] > source["timestamp"]]
    active = {r["uid"] for r in actual}
    return active-known_before, active, len(actual)


def semantic_view(corpus, candidates, known, cutoff, source_index, pop, metadata, info_matches, user_index):
    base = corpus.graph(candidates, known, cutoff, source_index, pop)
    base.update(metadata)
    visible = eligible_evidence(info_matches, corpus, cutoff, source_index)
    seeds = set(base["node_ids"]) | {corpus.records[int(i)]["uid"] for i in visible}
    pair_ids = user_index.candidates(seeds)
    return augment_graph(base, corpus, known, cutoff, source_index, info_matches, user_index.pairs, pair_ids)


def causal_macro_view(corpus, known, cutoff, source_index, pop, metadata, info_matches, user_index):
    # There is deliberately no candidate-user or current/future-count argument.
    # Macro context expands only from actual past chain participants and
    # time-visible information matches, never from sampled action candidates.
    macro_metadata = dict(metadata)
    macro_metadata["target"] = np.empty(0, dtype=np.int64)
    return semantic_view(corpus, [], known, cutoff, source_index, pop, macro_metadata, info_matches, user_index)
