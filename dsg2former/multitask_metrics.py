"""Action classification and nonnegative, archive-observed popularity metrics."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score, roc_auc_score


def action_metrics(target, probability):
    y = np.asarray(target, dtype=np.int64).reshape(-1)
    p = np.asarray(probability)
    if p.shape != (len(y), 3) or not len(y):
        raise ValueError("Action metrics need nonempty targets and an N x 3 probability array")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=1e-5):
        raise ValueError("Expected finite normalized class probabilities")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Unexpected action class")
    prediction = p.argmax(1)
    auc = {str(k): float(roc_auc_score(y == k, p[:, k])) for k in range(3)
           if (y == k).any() and (y != k).any()}
    return {"accuracy": float(accuracy_score(y, prediction)),
            "recall": float(recall_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
            "f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
            "auc": float(np.mean(list(auc.values()))) if len(auc) == 3 else None,
            "auc_available": float(np.mean(list(auc.values()))) if auc else None,
            "auc_by_class": auc,
            "recall_by_class": recall_score(y, prediction, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "f1_by_class": f1_score(y, prediction, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "support": np.bincount(y, minlength=3).tolist(),
            "cross_entropy": float(log_loss(y, p, labels=[0, 1, 2])),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist()}


def popularity_metrics(target, prediction):
    y, p = np.asarray(target, dtype=np.float64).reshape(-1), np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not len(y) or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Popularity metrics need aligned nonempty finite arrays")
    if np.any(y < 0) or np.any(p < 0):
        raise ValueError("MSE/RMSE/MSLE predictions and counts must be nonnegative")
    mse = float(np.mean((p-y)**2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "msle": float(np.mean((np.log1p(p)-np.log1p(y))**2))}


def mean_information_metrics(rows, keys):
    available = [r for r in rows if r["metrics"] is not None]
    result = {"information_samples": len(rows), "evaluable_information_samples": len(available)}
    for key in keys:
        values = [r["metrics"][key] for r in available if r["metrics"].get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
        result[key+"_information_count"] = len(values)
    return result
