import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, log_loss, confusion_matrix


def classification_metrics(target, probability):
    y = np.asarray(target).reshape(-1)
    p = np.asarray(probability).reshape(-1, 3)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Metrics require finite normalized probabilities")
    prediction = p.argmax(1)
    auc = {}
    for k in range(3):
        binary = y == k
        if binary.any() and not binary.all():
            auc[str(k)] = float(roc_auc_score(binary, p[:, k]))
    return {
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="micro", zero_division=0)),
        "auc_ovr_macro_available": float(np.mean(list(auc.values()))) if auc else None,
        "auc_by_class": auc, "auc_defined_classes": list(auc),
        "auc_ovr_macro_all3": float(np.mean(list(auc.values()))) if len(auc) == 3 else None,
        "cross_entropy": float(log_loss(y, p, labels=[0, 1, 2])),
        "f1_by_class": f1_score(y, prediction, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
        "support": np.bincount(y, minlength=3).tolist(),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
    }


def situation_metrics(target, probability):
    target = np.asarray(target)
    probability = np.asarray(probability)
    actual = np.stack([(target == k).sum(1) for k in range(3)], axis=1)
    expected = probability.sum(1)
    hard = np.stack([(probability.argmax(-1) == k).sum(1) for k in range(3)], axis=1)
    return {"actual": actual.tolist(), "expected": expected.tolist(), "argmax": hard.tolist(),
            "mae_by_class": np.abs(expected - actual).mean(0).tolist(),
            "rmse_by_class": np.sqrt(((expected - actual) ** 2).mean(0)).tolist()}
