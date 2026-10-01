"""Metric primitives. Every rate carries its denominator and a Wilson 95% interval; no accuracy anywhere
(the base rate is < 1%, so "99.4% accurate" is what approving everything scores)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

Z = 1.959963984540054


def wilson(k: float, n: float) -> tuple[float, float, float]:
    k, n = float(k), float(n)  # numpy int32 counts overflow in n * n
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def rate_row(name: str, k: float, n: float, min_n: int, **extra) -> dict:
    p, lo, hi = wilson(k, n)
    return {"metric": name, "value": p, "ci_low": lo, "ci_high": hi, "numerator": int(k), "denominator": int(n),
            "n_sufficient": int(n >= min_n), **extra}


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float) -> float:
    neg = np.sort(np.asarray(neg_scores, dtype=float))
    k = int(np.floor(fpr * len(neg)))
    return float(neg[len(neg) - k - 1]) if k < len(neg) else -np.inf


def recall_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> dict:
    thr = threshold_at_fpr(s[y == 0], fpr)
    tp = int((s[y == 1] > thr).sum())
    fp = int((s[y == 0] > thr).sum())
    return {"threshold": thr, "tp": tp, "n_pos": int((y == 1).sum()), "fp": fp, "n_neg": int((y == 0).sum()),
            "recall": tp / max((y == 1).sum(), 1), "realised_fpr": fp / max((y == 0).sum(), 1)}


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() and (y == 0).sum() else float("nan")


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if y.sum() and (y == 0).sum() else float("nan")


def partial_auc(y: np.ndarray, s: np.ndarray, max_fpr: float = 0.02) -> float:
    """McClish-standardised partial AUC over FPR in [0, max_fpr] (sklearn's max_fpr does the correction)."""
    return float(roc_auc_score(y, s, max_fpr=max_fpr)) if y.sum() and (y == 0).sum() else float("nan")


def bootstrap_ci(y: np.ndarray, s: np.ndarray, fn, B: int, seed: int = 0) -> tuple[float, float]:
    r = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(B):
        idx = r.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, fn, B: int, seed: int = 0) -> dict:
    """Delta fn(b) - fn(a) with a paired bootstrap 95% CI. A CI containing 0 = 'no measured effect'."""
    r = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(B):
        idx = r.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        d.append(fn(y[idx], s_b[idx]) - fn(y[idx], s_a[idx]))
    point = fn(y, s_b) - fn(y, s_a)
    lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d else (np.nan, np.nan)
    verdict = "no measured effect" if (np.isnan(lo) or lo <= 0 <= hi) else ("improves" if lo > 0 else "worsens")
    return {"delta": float(point), "delta_ci_low": lo, "delta_ci_high": hi, "verdict": verdict}


def recall_at_fpr_fn(fpr: float):
    def fn(y, s):
        return recall_at_fpr(y, s, fpr)["recall"]
    return fn
