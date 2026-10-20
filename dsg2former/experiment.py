"""Chronological training, ablations, probability metrics and saved predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.nn import functional as F

from .data import Protocol, prepare_event, LABELS
from .metrics import classification_metrics, situation_metrics
from .model import DSG2former


VARIANTS = {
    "dsg2former": {},
    "sdsformer": {"temporal": "none"},
    "self_only": {"feature": "self"},
    "external_only": {"feature": "external"},
    "payoff_only": {"feature": "payoff"},
    "no_pe": {"encoding": "none"},
    "ape": {"encoding": "ape"},
    "rpe": {"encoding": "rpe"},
    "rnn": {"temporal": "rnn"},
    "gru": {"temporal": "gru"},
    "temporal_mha": {"temporal": "mha"},
    "previous_spatial": {"history": "previous_spatial"},
    "mlp": {"temporal": "none", "blocks": 0},
}
BASELINES = ("majority", "persistence", "recent_persistence", "logistic")


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    x = np.exp(x)
    return x / x.sum(-1, keepdims=True)


def class_weights(snapshots, train_end, mode):
    counts = np.bincount(np.concatenate([s["target"] for s in snapshots[:train_end]]), minlength=3)
    weights = np.ones(3, dtype=np.float32)
    if mode == "sqrt":
        weights = np.sqrt(counts.sum() / (3 * np.maximum(counts, 1))).astype(np.float32)
        weights /= weights.mean()
    return weights


def baseline(variant, snapshots, train_end, weights, seed):
    labels = np.concatenate([s["target"] for s in snapshots[:train_end]])
    counts = np.bincount(labels, minlength=3)
    if variant == "majority":
        # Training empirical distribution; argmax is majority, scores stay calibrated.
        p = (counts + 1) / (counts.sum() + 3)
        return np.broadcast_to(p, (len(snapshots), len(snapshots[0]["target"]), 3)).copy(), {}
    if variant in ("persistence", "recent_persistence"):
        field = "last" if variant == "persistence" else "recent_last"
        predictions = np.stack([s[field] for s in snapshots])
        return np.eye(3)[predictions] * 0.998 + 0.002 / 3, {}
    features = np.stack([s["drivers"].reshape(len(s["target"]), -1) for s in snapshots])
    if len(np.unique(labels)) < 2:
        return baseline("majority", snapshots, train_end, weights, seed)
    estimator = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    estimator.fit(features[:train_end].reshape(-1, features.shape[-1]), labels, sample_weight=weights[labels])
    p = np.full((len(snapshots) * features.shape[1], 3), 1e-9)
    p[:, estimator.classes_] = estimator.predict_proba(features.reshape(-1, features.shape[-1]))
    p /= p.sum(-1, keepdims=True)
    return p.reshape(len(snapshots), features.shape[1], 3), {"coefficients": estimator.coef_.tolist(), "classes": estimator.classes_.tolist()}


def tensors(snapshots, device):
    # Hold only compact snapshots on CPU; each graph is transferred on demand.
    return [{"drivers": torch.from_numpy(s["drivers"]),
             "positions": torch.from_numpy(s["positions"].astype(np.int64)),
             "target": torch.from_numpy(s["target"])} for s in snapshots]


@torch.no_grad()
def evaluate_model(model, data, end, device):
    model.eval()
    predictions = []
    previous, history = None, []
    for s in data[:end]:
        logits, _, previous, h = model(s["drivers"].to(device), s["positions"].to(device), previous, history)
        if model.temporal == "mha":
            history.append(h)
        predictions.append(logits.softmax(-1).cpu().numpy())
    return np.stack(predictions)


def train_neural(snapshots, information, variant, seed, args, out):
    device = torch.device(args.device)
    train_end, valid_end = information["train_end"], information["validation_end"]
    options = dict(dim=args.dim, heads=args.heads, blocks=2, bias_size=information["protocol"]["bias_size"], dropout=args.dropout)
    options.update(VARIANTS[variant])
    model = DSG2former(**options).to(device)
    data = tensors(snapshots, device)
    weights = torch.tensor(class_weights(snapshots, train_end, args.class_weight), device=device)
    targets = np.stack([s["target"] for s in snapshots])
    trace = []
    stages = ["joint"] if args.training == "joint" or model.temporal == "none" else ["spatial", "temporal"]
    for stage in stages:
        if stage == "temporal":
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(("gate.", "temporal_head.", "cell.", "temporal_attention.")))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        best_score, best_loss, patience, best_state, best_epoch = -1, float("inf"), 0, None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            if stage == "temporal":
                # Frozen pretrained spatial encoder uses deterministic eval dropout.
                model.eval()
            previous, history = None, []
            losses = []
            for s in data[:train_end]:
                optimizer.zero_grad(set_to_none=True)
                logits, spatial, next_state, h = model(s["drivers"].to(device), s["positions"].to(device), previous, history)
                y = s["target"].to(device)
                spatial_loss = F.cross_entropy(spatial, y, weight=weights)
                if stage == "spatial" or model.temporal == "none":
                    loss = spatial_loss
                elif stage == "temporal":
                    loss = F.cross_entropy(logits, y, weight=weights)
                else:
                    loss = spatial_loss + args.temporal_weight * F.cross_entropy(logits, y, weight=weights)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss {variant}/{seed}/{stage}/{epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # Explicit one-snapshot truncated BPTT; identity order is frozen.
                previous = next_state.detach()
                if model.temporal == "mha":
                    history.append(h.detach())
                losses.append(loss.item())
            if stage == "spatial":
                temporal = model.temporal
                model.temporal = "none"
            prediction = evaluate_model(model, data, valid_end, device)
            if stage == "spatial":
                model.temporal = temporal
            metrics = classification_metrics(targets[train_end:valid_end], prediction[train_end:valid_end])
            score, valid_loss = metrics["macro_f1"], metrics["cross_entropy"]
            trace.append({"stage": stage, "epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics})
            improved = score > best_score + 1e-8
            tie_better_loss = abs(score - best_score) <= 1e-8 and valid_loss < best_loss - 1e-6
            if improved or tie_better_loss:
                best_score, best_loss, best_epoch = score, valid_loss, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            # Stopping monitors Macro-F1. CE breaks checkpoint ties but cannot
            # perpetually restart patience on majority-only validation windows.
            patience = 0 if improved else patience + 1
            if args.verbose and (epoch == 1 or epoch % 10 == 0):
                print(json.dumps({"event": information["event"], "variant": variant, "seed": seed,
                                  "stage": stage, "epoch": epoch, "val_macro_f1": score}), flush=True)
            if patience >= args.patience:
                break
        if best_state is None:
            raise RuntimeError("Training produced no checkpoint")
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "options": options, "stage": stage, "best_epoch": best_epoch,
                    "validation_macro_f1": best_score, "seed": seed, "information": information,
                    "args": vars(args)}, out / f"{stage}.pt")
    # Test predictions are produced only after all validation selection is finished.
    prediction = evaluate_model(model, data, len(data), device)
    json_write(out / "training_history.json", trace)
    return prediction, {"stages": stages, "parameters": sum(p.numel() for p in model.parameters()),
                        "best_epoch_final_stage": best_epoch, "options": options}


def run_one(event, snapshots, information, variant, seed, args):
    out = Path(args.output) / event / variant / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "metrics.json"
    if args.resume and result_path.exists():
        prior = json.loads(result_path.read_text())
        expected_args = {k: v for k, v in vars(args).items() if k not in ("resume", "verbose", "events", "variants", "seeds")}
        prior_args = {k: v for k, v in prior["configuration"].items() if k not in ("resume", "verbose", "events", "variants", "seeds")}
        if (prior["source_sha256"] != information["sha256"] or prior_args != expected_args
                or prior.get("code_sha256") != code_fingerprint()):
            raise ValueError(f"Resume configuration mismatch: {out}")
        print(f"SKIP {event}/{variant}/{seed}", flush=True)
        return prior
    set_seed(seed)
    start = time.monotonic()
    if variant in BASELINES:
        prediction, details = baseline(variant, snapshots, information["train_end"],
                                       class_weights(snapshots, information["train_end"], args.class_weight), seed)
    else:
        prediction, details = train_neural(snapshots, information, variant, seed, args, out)
    targets = np.stack([s["target"] for s in snapshots])
    train, valid = information["train_end"], information["validation_end"]
    result = {"event": event, "variant": variant, "seed": seed, "details": details,
              "elapsed_seconds": time.monotonic() - start, "labels": LABELS,
              "source_sha256": information["sha256"], "configuration": vars(args),
              "code_sha256": code_fingerprint(), "platform": platform.platform(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
              "device": args.device, "torch_version": torch.__version__, "numpy_version": np.__version__,
              "cohort_nodes": information["cohort_nodes"], "task": "adapted_next_window_behavior",
              "protocol": information["protocol"],
              "validation": classification_metrics(targets[train:valid], prediction[train:valid]),
              "test": classification_metrics(targets[valid:], prediction[valid:]),
              "test_situation": situation_metrics(targets[valid:], prediction[valid:])}
    np.savez_compressed(out / "predictions.npz", probabilities=prediction, target=targets,
                        cutoffs=np.array([s["cutoff"] for s in snapshots]),
                        target_ends=np.array([s["target_end"] for s in snapshots]),
                        train_end=train, validation_end=valid)
    json_write(result_path, result)
    print(json.dumps({"event": event, "variant": variant, "seed": seed,
                      "seconds": round(result["elapsed_seconds"], 2), "test": result["test"]}), flush=True)
    return result


def summarize(output):
    results = [json.loads(p.read_text()) for p in Path(output).glob("*/*/seed_*/metrics.json")]
    groups = {}
    for r in results:
        groups.setdefault((r["event"], r["variant"]), []).append(r)
    rows = []
    for (event, variant), runs in sorted(groups.items()):
        row = {"event": event, "variant": variant, "runs": len(runs), "nodes": runs[0]["cohort_nodes"]}
        for key in ("macro_f1", "micro_f1", "auc_ovr_macro_all3", "auc_ovr_macro_available"):
            values = [r["test"][key] for r in runs if r["test"][key] is not None]
            row[key + "_mean"] = float(np.mean(values)) if values else None
            row[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    json_write(Path(output) / "summary.json", rows)
    if rows:
        with (Path(output) / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="external/CAGE/data/event_datasets")
    parser.add_argument("--output", default="results/main")
    parser.add_argument("--events", nargs="+", default=["all"])
    parser.add_argument("--variants", nargs="+", default=["majority", "persistence", "logistic", "sdsformer", "dsg2former"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2025, 2026])
    parser.add_argument("--windows", type=int, default=30)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--bias-size", type=int, default=50)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--temporal-weight", type=float, default=1)
    parser.add_argument("--class-weight", choices=["none", "sqrt"], default="none")
    parser.add_argument("--training", choices=["joint", "staged"], default="joint")
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.max_nodes < 0:
        parser.error("epochs and patience must be positive; max_nodes must be nonnegative")
    unknown = set(args.variants) - set(VARIANTS) - set(BASELINES)
    if unknown:
        parser.error(f"Unknown variants {unknown}")
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = Protocol(windows=args.windows, max_nodes=args.max_nodes, bias_size=args.bias_size,
                        calibration_fraction=args.calibration_fraction, directed=not args.undirected)
    available = sorted(p.stem for p in Path(args.data_dir).glob("*.pkl"))
    events = available if args.events == ["all"] else args.events
    if not available or set(events) - set(available):
        parser.error(f"Missing events: {set(events) - set(available)}")
    json_write(output / "invocation.json", vars(args))
    for event in events:
        snapshots, info = prepare_event(Path(args.data_dir) / f"{event}.pkl", protocol)
        json_write(output / event / "protocol.json", info)
        print(json.dumps({"event": event, "nodes": info["cohort_nodes"], "class_counts": info["split_class_counts"]}), flush=True)
        for variant in args.variants:
            for seed in args.seeds:
                run_one(event, snapshots, info, variant, seed, args)
                summarize(output)


if __name__ == "__main__":
    main()
