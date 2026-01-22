import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

LOG_ORDER = ["trace", "debug", "info", "warn", "error"]
LOG2IDX = {k: i for i, k in enumerate(LOG_ORDER)}
IDX2LOG = {i: k for k, i in LOG2IDX.items()}

def normalize_level(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    s = s.replace("warning", "warn")
    return s

def levels_to_int(levels):
    y = []
    for v in levels:
        s = normalize_level(v)
        if s in LOG2IDX:
            y.append(LOG2IDX[s])
        else:
            y.append(LOG2IDX["info"])
    return np.asarray(y, dtype=np.int64)

def to_onehot(y_int, n_classes=5):
    y_int = np.asarray(y_int, dtype=np.int64)
    out = np.zeros((len(y_int), n_classes), dtype=np.float32)
    out[np.arange(len(y_int)), y_int] = 1.0
    return out

def ordinal_targets_from_int(y_int, n_classes=5):
    y_int = np.asarray(y_int, dtype=np.int64)
    out = np.zeros((len(y_int), n_classes), dtype=np.float32)
    for i, k in enumerate(y_int):
        out[i, : (k + 1)] = 1.0
    return out

def decode_ordinal_to_int(y_ord_probs, threshold=0.5):
    y_ord_probs = np.asarray(y_ord_probs, dtype=np.float32)
    bin_mat = (y_ord_probs >= threshold).astype(np.int64)
    k = np.sum(bin_mat, axis=1) - 1
    k = np.clip(k, 0, y_ord_probs.shape[1] - 1)
    return k

def ordinal_cumprob_to_classprob(cumprob):
    cumprob = np.asarray(cumprob, dtype=np.float32)
    cumprob = np.clip(cumprob, 0.0, 1.0)
    n = cumprob.shape[1]
    p = np.zeros_like(cumprob, dtype=np.float32)
    for k in range(n - 1):
        p[:, k] = np.maximum(cumprob[:, k] - cumprob[:, k + 1], 0.0)
    p[:, n - 1] = np.maximum(cumprob[:, n - 1], 0.0)
    s = np.sum(p, axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    return p / s

def safe_multiclass_auc(y_true_int, y_proba, n_classes=5):
    y_true_oh = to_onehot(y_true_int, n_classes=n_classes)
    y_proba = np.asarray(y_proba, dtype=np.float32)
    try:
        return float(roc_auc_score(y_true_oh, y_proba, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")

def aod_score(y_true_int, y_pred_int, left_boundary=0.0, right_boundary=4.0):
    y_true_int = np.asarray(y_true_int, dtype=np.float32)
    y_pred_int = np.asarray(y_pred_int, dtype=np.float32)
    lb = left_boundary
    rb = right_boundary
    max_dist = np.maximum(y_true_int - lb, rb - y_true_int)
    max_dist[max_dist == 0.0] = 1.0
    val = 1.0 - (np.abs(y_pred_int - y_true_int) / max_dist)
    return float(np.mean(val))

def compute_all_metrics(y_true_int, y_pred_int, y_proba):
    y_true_int = np.asarray(y_true_int, dtype=np.int64)
    y_pred_int = np.asarray(y_pred_int, dtype=np.int64)
    acc = float(accuracy_score(y_true_int, y_pred_int))
    auc = safe_multiclass_auc(y_true_int, y_proba, n_classes=5)
    aod = aod_score(y_true_int, y_pred_int)
    return {"accuracy": acc, "auc": auc, "aod": aod}
