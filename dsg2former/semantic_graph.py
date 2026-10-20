"""Time-filtered user--information evidence and its user-graph projection.

Raw evidence rows are never deduplicated. Only graph vertices and adjacency are
sets, as required by the existing user-node Transformer representation.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def cosine64(query, rows):
    query = np.asarray(query, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    denominator = np.linalg.norm(rows, axis=1) * np.linalg.norm(query)
    return np.divide(rows @ query, denominator, out=np.zeros(len(rows)), where=denominator > 0)


def eligible_evidence(indices, corpus, cutoff, source_index):
    indices = np.asarray(indices, dtype=np.int64)
    # The source itself is known at the trigger; other same-second actions are
    # unavailable until their window has completed.
    visible = (corpus.times[indices] < cutoff) | ((indices == source_index) & (corpus.times[indices] == cutoff))
    return indices[visible & corpus.allowed[indices]]


def shortest_positions(adjacency, bias_size=50):
    distance = shortest_path(csr_matrix(adjacency), directed=True, unweighted=True)
    finite = np.isfinite(distance)
    result = np.zeros(distance.shape, dtype=np.int16)
    result[finite] = np.minimum(distance[finite] + 1, bias_size - 1).astype(np.int16)
    return result


class UserSemanticIndex:
    """Raw text-pair evidence indexed by its endpoint user, without deduping."""
    def __init__(self, records, record_pairs):
        self.pairs = np.asarray(record_pairs, dtype=np.int64).reshape(-1, 2)
        incident = {}
        for i, (left, right) in enumerate(self.pairs):
            for uid in (records[int(left)]["uid"], records[int(right)]["uid"]):
                incident.setdefault(uid, []).append(i)
        self.incident = {u: np.asarray(v, dtype=np.int64) for u, v in incident.items()}

    def candidates(self, users):
        arrays = [self.incident[u] for u in users if u in self.incident]
        return np.unique(np.concatenate(arrays)) if arrays else np.empty(0, dtype=np.int64)


def augment_graph(base, corpus, known_participants, cutoff, source_index, evidence_indices,
                  pair_records=None, pair_indices=None):
    """Add observable context users, then project a shared-information clique.

    Drivers use social edges only. Context users do not become supervised rows;
    repeated candidate users retain repeated sample indices and loss weights.
    """
    evidence = eligible_evidence(evidence_indices, corpus, cutoff, source_index)
    information_users = {corpus.records[int(i)]["uid"] for i in evidence}
    seeds = set(base["node_ids"]) | information_users
    pair_indices = np.asarray([] if pair_indices is None else pair_indices, dtype=np.int64)
    user_pairs = set()
    if len(pair_indices):
        pairs = np.asarray(pair_records)[pair_indices]
        visible = ((corpus.times[pairs] < cutoff) |
                   ((pairs == source_index) & (corpus.times[pairs] == cutoff))) & corpus.allowed[pairs]
        keep = visible.all(1)
        for j in np.flatnonzero(keep):
            left, right = pairs[j]
            u, v = corpus.records[int(left)]["uid"], corpus.records[int(right)]["uid"]
            if u != v and (u in seeds or v in seeds):
                user_pairs.add(tuple(sorted((u, v))))
            else:
                keep[j] = False
        pair_indices = pair_indices[keep]
    user_users = {u for pair in user_pairs for u in pair}
    users = information_users | user_users
    candidates = [base["node_ids"][int(i)] for i in base["sample_nodes"]]
    extra = sorted(users - set(base["node_ids"]))
    if extra:
        graph = corpus.graph(candidates + extra, known_participants, cutoff, source_index,
                             base["feature_components"]["popularity_difference"])
        for name in ("sample_nodes", "seen_before", "seen_before_chain"):
            graph[name] = graph[name][:len(candidates)]
    else:
        graph = dict(base)
    index = {u: i for i, u in enumerate(graph["node_ids"])}
    social = graph["positions"] == 2
    merged = social.copy()
    semantic = np.zeros(social.shape, dtype=bool)
    semantic_nodes = np.asarray([index[u] for u in sorted(information_users)], dtype=np.int64)
    if len(semantic_nodes):
        semantic[np.ix_(semantic_nodes, semantic_nodes)] = True
    for u, v in user_pairs:
        semantic[index[u], index[v]] = semantic[index[v], index[u]] = True
    np.fill_diagonal(semantic, False)
    merged |= semantic
    np.fill_diagonal(merged, False)
    # Retain all target/window metadata, never label the extra context nodes.
    for key in ("target", "window", "cutoff", "end", "row_start", "feature_cutoff_exclusive"):
        graph[key] = base[key]
    graph.update({"social_positions": graph["positions"],
                  "positions": shortest_positions(merged),
                  "social_edges": int(social.sum()),
                  "semantic_edges": int(semantic.sum()),
                  "information_projection_edges": len(information_users) * (len(information_users) - 1),
                  "user_user_edges": len(user_pairs) * 2,
                  "user_user_record_pair_indices": pair_indices,
                  "user_user_users": sorted(user_users),
                  "added_semantic_edges": int((merged & ~social).sum()),
                  "edges": int(merged.sum()),
                  "semantic_record_indices": evidence,
                  "semantic_users": sorted(information_users),
                  "added_context_nodes": len(extra),
                  "state_update_users": set(base["state_update_users"]) | users})
    return graph
