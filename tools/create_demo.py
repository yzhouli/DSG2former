"""Create deterministic synthetic interface fixtures, never paper results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from cli_utils import write_json


def fixture():
    rng = np.random.default_rng(42)
    data, metadata = [], []
    for item in range(10):
        split = "train" if item < 8 else "validation" if item == 8 else "test"
        start = 86400 * (item + 1)
        tid = str(item + 1)
        users = [f"source-{item}", f"history-{item}-1", f"history-{item}-2"]
        metadata.append(dict(thread_id=tid, split=split, source=dict(uid=users[0], timestamp=start)))
        t = dict(thread_id=tid, event="synthetic-smoke-only", split=split, snapshots=[], macro_snapshots=[])
        past = []
        for window in range(4):
            x = rng.normal(0, .2, (4, 184)).astype(np.float32)
            x[:, 0] = 1
            history = np.zeros(13, np.float32)
            history[:4] = [window + 1, past[-1] if past else 0, 2, window]
            if past:
                history[12 - len(past):12] = past
            count = float([1, 2, 0, 1][window])
            shared = dict(x=x, raw_x=x.copy(), drivers=rng.normal(size=(4, 2, 3)).astype(np.float32),
                          adjacency=np.ones((4, 4), np.float32), positions=np.full((4, 4), 2, np.int64),
                          role=np.array([0, 0, 0, 1]), user_count=3, node_ids=users,
                          state_update_users=users, window=window, cutoff=start + window * 3600,
                          end=start + (window + 1) * 3600, history_summary=history, popularity_target=count)
            np.fill_diagonal(shared["positions"], 1)
            t["snapshots"].append(dict(shared, sample_nodes=np.arange(3), target=np.arange(3)))
            t["macro_snapshots"].append(dict(shared, sample_nodes=np.empty(0, np.int64), target=np.empty(0, np.int64)))
            past.append(count)
        data.append(t)
    return data, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/demo_source"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, metadata = fixture()
    torch.save(data, args.output / "graphs.pt")
    (args.output / "threads.jsonl").write_text("".join(json.dumps(t) + "\n" for t in metadata))
    write_json(args.output / "SYNTHETIC_ONLY.json", dict(synthetic=True, purpose="Installation and interface verification; no experimental validity"))
    print(args.output)


if __name__ == "__main__":
    main()
