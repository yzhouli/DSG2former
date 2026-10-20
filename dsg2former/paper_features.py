"""Paper Eq. (2)-(7), (9)-(10), using explicitly declared history proxies.

Only records preceding the cutoff are eligible, except the known source post at
the initial trigger. Text polarity is a feature, not a future action label.
"""
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class HistoricalCorpus:
    def __init__(self, records, embeddings, sentiment, top_k=10, gamma=.23, allowed_records=None):
        self.records, self.embeddings, self.sentiment = records, embeddings, sentiment
        self.top_k, self.gamma = top_k, gamma
        self.allowed = np.ones(len(records), dtype=bool) if allowed_records is None else np.asarray(allowed_records, dtype=bool)
        self.times = np.asarray([r["timestamp"] for r in records])
        if len(records) != len(embeddings) or len(records) != len(sentiment) or np.any(np.diff(self.times) < 0):
            raise ValueError("History arrays must align and be sorted chronologically")
        by_user, incoming = defaultdict(list), defaultdict(list)
        by_id = {}
        for i, r in enumerate(records):
            if self.allowed[i]:
                by_id.setdefault(r["id"], i)
        self.parent, ret = [], []
        self.path_index = {}
        for i, r in enumerate(records):
            uid = r["uid"]
            if self.allowed[i]:
                by_user[uid].append(i)
            parent = r.get("reply_to_uid")
            if parent is None and r.get("reply_to_id") in by_id:
                j = by_id[r["reply_to_id"]]
                if self.times[j] <= self.times[i]:
                    parent = records[j]["uid"]
            self.parent.append(parent)
            if self.allowed[i] and parent is not None and parent != uid:
                incoming[parent].append(i)
            # Raw PHEME reaction folders can contain the triggering source
            # itself. Retain the row, but do not invent a reply from its folder.
            ret.append((r["kind"] == "reaction" and r["id"] != r.get("thread_id"))
                       or r.get("reply_to_id") is not None or r["text"].startswith("RT @"))
            self.path_index[r["archive_path"]] = i
        self.ret = np.asarray(ret, dtype=bool)
        self.by_user = {u: np.asarray(v, dtype=np.int64) for u, v in by_user.items()}
        self.incoming = {u: np.asarray(v, dtype=np.int64) for u, v in incoming.items()}
        self.user_times = {u: self.times[v] for u, v in self.by_user.items()}
        self.incoming_times = {u: self.times[v] for u, v in self.incoming.items()}
        self.semantic_cache = {}

    def eligible(self, uid, cutoff, source_index):
        rows = self.by_user.get(uid, np.empty(0, dtype=np.int64))
        times = self.user_times.get(uid, np.empty(0))
        selected = rows[:np.searchsorted(times, cutoff, side="left")]
        # The triggering source is already known when forecasting its replies.
        source = self.records[source_index]
        if not self.allowed[source_index]:
            raise ValueError("Source is not available in this partition history")
        if source["uid"] == uid and source["timestamp"] == cutoff:
            selected = np.concatenate([selected, [source_index]])
        return selected

    def intrinsic(self, uid, cutoff, source_index):
        rows = self.eligible(uid, cutoff, source_index)
        key = (uid, len(rows), int(rows[-1]) if len(rows) else -1, source_index)
        if key in self.semantic_cache:
            return self.semantic_cache[key]
        interest = np.zeros(3)
        ret, ori, rate = 0., 0., 0.
        if len(rows):
            similarity = np.asarray(self.embeddings[rows] @ self.embeddings[source_index])
            nearest = rows[np.argsort(-similarity, kind="stable")[:self.top_k]]
            interest = np.asarray(self.sentiment[nearest]).mean(0)
            ret = float(self.ret[rows].sum())
            ori = float(len(rows) - ret)
            rate = ret / len(rows)
        activity = ret + self.gamma * ori
        result = (interest, activity, rate, len(rows))
        self.semantic_cache[key] = result
        return result

    def neighbors(self, uid, cutoff, source_index):
        rows = self.eligible(uid, cutoff, source_index)
        followed = {self.parent[int(i)] for i in rows if self.parent[int(i)] is not None and self.parent[int(i)] != uid}
        incoming = self.incoming.get(uid, np.empty(0, dtype=np.int64))
        times = self.incoming_times.get(uid, np.empty(0))
        incoming = incoming[:np.searchsorted(times, cutoff, side="left")]
        friends = followed | {self.records[int(i)]["uid"] for i in incoming}
        source = self.records[source_index]
        if source["timestamp"] == cutoff and self.parent[source_index] == uid:
            friends.add(source["uid"])
        friends.discard(uid)
        return followed, friends

    def graph(self, candidate_users, known_participants, cutoff, source_index, pop_difference, bias_size=50):
        nodes = sorted(set(candidate_users) | set(known_participants))
        index = {u: i for i, u in enumerate(nodes)}
        n = len(nodes)
        followed, friends = {}, {}
        for u in nodes:
            followed[u], friends[u] = self.neighbors(u, cutoff, source_index)
        pairs = [(index[u], index[v]) for u in nodes for v in followed[u] if v in index]
        src = np.asarray([u for u, v in pairs], dtype=np.int64)
        dst = np.asarray([v for u, v in pairs], dtype=np.int64)
        adjacency = csr_matrix((np.ones(len(pairs)), (src, dst)), shape=(n, n))
        distances = shortest_path(adjacency, directed=True, unweighted=True)
        finite = np.isfinite(distances)
        positions = np.zeros((n, n), dtype=np.int16)
        positions[finite] = np.minimum(distances[finite] + 1, bias_size - 1).astype(np.int16)
        incoming = np.asarray(adjacency.sum(0)).ravel()
        outgoing = np.asarray(adjacency.sum(1)).ravel()
        average_followers = float(incoming.mean())
        authority = np.zeros(n)
        if average_followers > 0:
            authority = np.asarray(adjacency @ (incoming / average_followers)).ravel() / np.maximum(outgoing, 1)
        friend_score = np.asarray([len(friends[u] & set(known_participants)) / len(friends[u]) if friends[u] else 0. for u in nodes])
        intrinsic = [self.intrinsic(u, cutoff, source_index) for u in nodes]
        interest = np.asarray([v[0] for v in intrinsic])
        activity = np.asarray([v[1] for v in intrinsic])
        retweet_rate = np.asarray([v[2] for v in intrinsic])
        inner = interest * activity[:, None] * retweet_rate[:, None]
        # Eq. (5)-(7) define scalar quantities. Broadcast their scalar product
        # across three payoffs, rather than inventing stance-specific variants.
        outer_scalar = authority * friend_score * pop_difference
        outer = np.repeat(outer_scalar[:, None], 3, axis=1)
        sampled = np.asarray([index[u] for u in candidate_users], dtype=np.int64)
        return {"node_ids": nodes, "drivers": np.stack([inner, outer], axis=1).astype(np.float32),
                "positions": positions, "sample_nodes": sampled,
                "seen_before": np.asarray([intrinsic[j][3] > 0 for j in sampled], dtype=bool),
                "seen_before_chain": np.asarray([u in known_participants for u in candidate_users], dtype=bool),
                "feature_components": {"user_interest": interest.astype(np.float32), "activity": activity.astype(np.float32),
                                       "retweet_rate": retweet_rate.astype(np.float32), "authority": authority.astype(np.float32),
                                       "friend_score": friend_score.astype(np.float32), "popularity_difference": float(pop_difference)},
                "edges": len(pairs)}


def popularity_difference(current_active, previous_active, elapsed_seconds, half_life_seconds):
    if half_life_seconds <= 0 or elapsed_seconds < 0:
        raise ValueError("Invalid popularity time scale")
    return float((current_active - previous_active) * .5 ** (elapsed_seconds / half_life_seconds))
