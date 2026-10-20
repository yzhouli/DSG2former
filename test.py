"""Frozen inference and pooled, seedwise metrics for the prepared benchmark."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from cli_utils import config, configure, digest, write_json
from dsg2former.pooled_metrics import pooled_popularity_metrics


def classification(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    if y.ndim != 1 or probability.shape != (len(y), 3) or not len(y):
        raise ValueError("Expected nonempty (N,) labels and (N,3) probabilities")
    if not np.isfinite(probability).all() or np.any(probability < 0) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Invalid labels/probabilities")
    np.testing.assert_allclose(probability.sum(1), 1., atol=3e-6, rtol=0)
    pred = probability.argmax(1)
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    aucs = [float(roc_auc_score(y == c, probability[:, c])) if 0 < (y == c).sum() < len(y) else None for c in range(3)]
    valid = [x for x in aucs if x is not None]
    return dict(accuracy=float(accuracy_score(y, pred)), precision=p.tolist(), recall=r.tolist(),
                f1=f.tolist(), class_support=support.tolist(), auc=aucs,
                macro_recall=float(r.mean()), macro_f1=float(f.mean()),
                macro_auc=float(np.mean(valid)) if valid else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pipeline", choices=("base", "optimized"), default="optimized")
    parser.add_argument("--replay", action="store_true", help="Recompute and compare predictions with the prior test output")
    args = parser.parse_args()
    value = config(args.config)
    neural, _, peak, name = configure(value)
    files = [p for p in neural.R.rglob("*.pt") if p.name not in ("common.pt", "development.pt", "development_final.pt")]
    plan = dict(frozen_before_test=True, checkpoint_sha256={str(p): digest(p) for p in files},
                source_sha256=digest(neural.__file__), input_sha256=digest("data/pheme_model_improvement/graphs.pt"))
    frozen = neural.R / "inference_plan.json"
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError("Checkpoints changed after freezing; use a new output_dir")
    write_json(frozen, plan)
    mode = "replay" if args.replay else "predict"
    for backbone in value["backbones"]:
        for seed in value["seeds"]:
            neural.predict(SimpleNamespace(backbone=backbone, seed=seed, device=args.device, mode=mode))
    if args.pipeline == "optimized":
        peak.freeze(name)
        for backbone in value["backbones"]:
            for seed in value["seeds"]:
                peak.infer(SimpleNamespace(dataset=name, backbone=backbone, seed=seed, device=args.device,
                                           mode="replay" if args.replay else "test"))
    stage = "exact_replay" if args.replay else "test_predictions"
    result = {}
    for variant in ("spatial", "temporal"):
        rows = []
        for seed in value["seeds"]:
            actions, counts = [], []
            for b in value["backbones"]:
                folder = (Path(value["output_dir"]) / "optimized" if args.pipeline == "optimized" else neural.R) / b / f"seed_{seed}" / variant / stage
                with np.load(folder / "action_predictions.npz", allow_pickle=False) as a:
                    actions.append({k: a[k] for k in a.files})
                with np.load(folder / "popularity_predictions.npz", allow_pickle=False) as c:
                    counts.append({k: c[k] for k in c.files})
                if args.replay:
                    for filename in ("action_predictions.npz", "popularity_predictions.npz"):
                        with np.load(folder.parent / "test_predictions" / filename, allow_pickle=False) as old, np.load(folder / filename, allow_pickle=False) as new:
                            for key in old.files:
                                np.testing.assert_array_equal(old[key], new[key], err_msg=f"Replay mismatch: {filename}:{key}")
            for a, c in zip(actions[1:], counts[1:]):
                for k in ("target", "thread_id", "window", "cutoff", "end"):
                    np.testing.assert_array_equal(actions[0][k], a[k])
                    np.testing.assert_array_equal(counts[0][k], c[k])
            probability = np.mean([a["probability"].astype(float) for a in actions], axis=0)
            prediction = np.mean([c["prediction"].astype(float) for c in counts], axis=0)
            micro = classification(actions[0]["target"], probability)
            macro = pooled_popularity_metrics(counts[0]["target"], prediction)
            actual = counts[0]["target"].astype(float)
            positive = actual > 0
            macro["mape"] = float(100 * np.mean(np.abs(prediction[positive] - actual[positive]) / actual[positive])) if positive.any() else None
            macro["mape_positive_windows"] = int(positive.sum())
            rows.append(dict(seed=seed, micro=micro, macro=macro))
        keys = ("accuracy", "macro_recall", "macro_f1", "macro_auc")
        def summary(entries):
            good = [v for v in entries if v is not None]
            return dict(mean=float(np.mean(good)) if good else None,
                        sample_sd=float(np.std(good, ddof=1)) if len(good) > 1 else None)
        result[variant] = dict(per_seed=rows,
                              micro={k: summary([r["micro"][k] for r in rows]) for k in keys},
                              macro={k: summary([r["macro"][k] for r in rows]) for k in ("mse", "rmse", "mape")})
    write_json(Path(value["output_dir"]) / ("replay_metrics.json" if args.replay else "metrics.json"),
               dict(dataset=value["dataset"], attention=value.get("attention", "dense"), pipeline=args.pipeline,
                    seeds=value["seeds"], variants=result, raw_rows_preserved=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
