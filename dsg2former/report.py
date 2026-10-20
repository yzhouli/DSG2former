"""Export actual completed results and publication-style static figures."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .data import LABELS
from .experiment import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    root = Path(args.input)
    rows = summarize(root)
    if not rows:
        raise SystemExit("No completed runs")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "msdformer-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    protocols = [json.loads(p.read_text()) for p in root.glob("*/protocol.json")]
    assessment = {"task": "adapted_closed_cohort_next_window_behavior", "events": []}
    for protocol in protocols:
        counts = np.array(protocol["split_class_counts"]["test"])
        assessment["events"].append({"event": protocol["event"], "test_support": counts.tolist(),
                                      "inactive_fraction": float(counts[2] / counts.sum()),
                                      "three_class_auc_identifiable": bool((counts > 0).all()),
                                      "only_inactive_test": bool(counts[:2].sum() == 0)})
    (root / "assessment.json").write_text(json.dumps(assessment, indent=2))

    for event in sorted({row["event"] for row in rows}):
        event_rows = [row for row in rows if row["event"] == event]
        fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
        ax.bar([r["variant"] for r in event_rows], [r["macro_f1_mean"] for r in event_rows],
               yerr=[r["macro_f1_std"] or 0 for r in event_rows], capsize=3)
        ax.set(ylabel="Test Macro-F1 (three declared labels)", title=event, ylim=(0, 1))
        ax.tick_params(axis="x", rotation=65)
        fig.savefig(figure_dir / f"{event}_comparison.png", dpi=180)
        plt.close(fig)
        prediction_paths = sorted((root / event / "dsg2former").glob("seed_*/predictions.npz"))
        if not prediction_paths:
            continue
        predictions = [np.load(p) for p in prediction_paths]
        start = int(predictions[0]["validation_end"])
        truth = predictions[0]["target"][start:]
        probabilities = np.mean([p["probabilities"][start:] for p in predictions], axis=0)
        times = predictions[0]["target_ends"][start:]
        times = (times - predictions[0]["cutoffs"][0]) / 3600
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for k, ax in enumerate(axes):
            ax.plot(times, (truth == k).sum(1), "o-", label="Observed cohort")
            ax.plot(times, probabilities[:, :, k].sum(1), "s--", label="Predicted expectation")
            ax.set(xlabel="Hours since first prediction cutoff", ylabel="Users per window", title=LABELS[k])
            ax.legend(fontsize=8)
        fig.suptitle(event + " — held-out windows, closed cohort")
        fig.savefig(figure_dir / f"{event}_situation.png", dpi=180)
        plt.close(fig)
        for p in predictions:
            p.close()
    print(root / "summary.csv")
    print(root / "assessment.json")
    print(figure_dir)


if __name__ == "__main__":
    main()
