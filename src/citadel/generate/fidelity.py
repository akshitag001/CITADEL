"""Is the synthetic data trivially separable? Build gates that fail the build.

The single-feature gate ("no raw column above 0.95 AUC") explodes categoricals per level, null PATTERNS are probed separately, and
the direction of every signal is carried so an anti-predictive column is not misread as a weak one.

When a gate fails, FIX THE GENERATOR, never the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from ..schema import COLUMNS, MULE_INITIATED_KINDS, feature_ok_columns
from ..timeutil import DAY_S

SINGLE_FEATURE_GATE = 0.95
JOINT_CEILING = 0.92        # recall at 0.5% FPR above this = "measuring the generator"
CONTRIBUTION_FLAG = 0.25    # one column delivering > 25% of the headline is flagged


def eval_mask(tx: pd.DataFrame) -> np.ndarray:
    return (~tx["row_kind"].isin(MULE_INITIATED_KINDS)).to_numpy()


def _auc(y: np.ndarray, v: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(v)) < 2:
        return 0.5
    return float(roc_auc_score(y, v))


def recall_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> tuple[float, float]:
    """Recall at the threshold where the realised FPR is <= fpr (ties broken against us)."""
    neg = np.sort(s[y == 0])
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan"), float("nan")
    k = int(np.floor(fpr * len(neg)))
    thr = neg[len(neg) - k - 1] if k < len(neg) else -np.inf
    tp = float((s[y == 1] > thr).mean())
    realised = float((neg > thr).mean())
    return tp, realised


def single_feature_probe(tx: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    m = eval_mask(tx)
    t = tx[m]
    y = t["is_attack"].to_numpy()
    rows = []
    columns = columns or feature_ok_columns()
    for c in columns:
        s = t[c]
        spec = COLUMNS.get(c)
        if spec is not None and spec.dtype == "category" or s.dtype == object:
            for lvl in pd.unique(s.astype(str)):
                ind = (s.astype(str) == lvl).to_numpy().astype(float)
                a = _auc(y, ind)
                rows.append((c, "level", str(lvl), a))
            continue
        v = s.to_numpy(dtype=float)
        nul = np.isnan(v)
        if nul.any():
            rows.append((c, "null_pattern", "isnull", _auc(y, nul.astype(float))))
        fill = np.where(nul, np.nanmedian(v) if (~nul).any() else 0.0, v)
        rows.append((c, "value", "", _auc(y, fill)))
    out = pd.DataFrame(rows, columns=["column", "probe", "level", "auc_raw"])
    out["direction"] = np.where(out["auc_raw"] >= 0.5, "higher=attack", "lower=attack")
    out["auc"] = np.maximum(out["auc_raw"], 1 - out["auc_raw"])
    out["gate"] = SINGLE_FEATURE_GATE
    out["passes"] = out["auc"] <= SINGLE_FEATURE_GATE
    out["n_rows"] = int(len(t))
    out["n_positives"] = int(y.sum())
    return out.sort_values("auc", ascending=False).reset_index(drop=True)


def _encode_raw(tx: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=tx.index)
    for c in feature_ok_columns():
        s = tx[c]
        if COLUMNS[c].dtype == "category":
            X[c] = s.cat.codes.astype(float) if hasattr(s, "cat") else pd.factorize(s)[0].astype(float)
        else:
            X[c] = s.to_numpy(dtype=float)
    return X


def time_forward_masks(ts: np.ndarray, train_frac: float = 0.6, gap_frac: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    day = (ts - ts.min()) // DAY_S
    D = day.max() + 1
    return day < D * train_frac, day >= D * (train_frac + gap_frac)


def joint_probe(X: pd.DataFrame, y: np.ndarray, ts: np.ndarray, mask: np.ndarray, refits: int = 5,
                fpr: float = 0.005, seed: int = 0, max_iter: int = 250) -> dict:
    """Time-forward booster; recall at an FPR budget; median over bootstrap refits + spread."""
    tr, te = time_forward_masks(ts)
    tr &= mask
    te &= mask
    r = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = X[tr].to_numpy(dtype=float), y[tr], X[te].to_numpy(dtype=float), y[te]
    recalls, praucs = [], []
    for i in range(refits):
        idx = r.integers(0, len(ytr), len(ytr)) if i else np.arange(len(ytr))
        m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=31,
                                           min_samples_leaf=40, l2_regularization=1.0, random_state=i)
        m.fit(Xtr[idx], ytr[idx], sample_weight=np.where(ytr[idx] == 1, 10.0, 1.0))
        s = m.predict_proba(Xte)[:, 1]
        recalls.append(recall_at_fpr(yte, s, fpr)[0])
        praucs.append(float(average_precision_score(yte, s)) if yte.sum() else float("nan"))
    rec = float(np.median(recalls))
    return {"recall_at_fpr": rec, "fpr_budget": fpr, "refit_spread": float(np.max(recalls) - np.min(recalls)),
            "pr_auc_median": float(np.median(praucs)), "refits": refits, "n_train": int(tr.sum()),
            "n_test": int(te.sum()), "n_test_positives": int(yte.sum()),
            "ceiling": JOINT_CEILING, "flag_measuring_generator": bool(rec > JOINT_CEILING)}


def raw_joint_probe(tx: pd.DataFrame, **kw) -> dict:
    return joint_probe(_encode_raw(tx), tx["is_attack"].to_numpy(), tx["ts"].to_numpy(), eval_mask(tx), **kw)


def label_shuffle_null(X: pd.DataFrame, y: np.ndarray, ts: np.ndarray, mask: np.ndarray, seed: int = 0) -> dict:
    """Refit on permuted labels: recall must be ~FPR budget and ROC-AUC ~0.5."""
    r = np.random.default_rng(seed)
    yp = y.copy()
    yp[mask] = r.permutation(y[mask])
    tr, te = time_forward_masks(ts)
    tr &= mask
    te &= mask
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                                       random_state=0)
    m.fit(X[tr].to_numpy(dtype=float), yp[tr])
    s = m.predict_proba(X[te].to_numpy(dtype=float))[:, 1]
    rec, _ = recall_at_fpr(yp[te], s, 0.005)
    auc = _auc(yp[te], s)
    return {"control": "label_shuffle_null", "roc_auc": auc, "recall_at_0.5pct_fpr": rec,
            "expected_recall": 0.005, "passes": bool(abs(auc - 0.5) < 0.08 and rec < 0.05),
            "n_test": int(te.sum()), "n_test_positives": int(yp[te].sum())}


def leakage_canary(X: pd.DataFrame, y: np.ndarray, ts: np.ndarray, mask: np.ndarray, seed: int = 0) -> dict:
    """Plant label+noise; BOTH probes must fire. A probe that silently stopped working must fail."""
    r = np.random.default_rng(seed)
    Xc = X.copy()
    Xc["canary"] = y + r.normal(0, 0.25, len(y))
    a = _auc(y[mask], Xc["canary"].to_numpy()[mask])
    jp = joint_probe(Xc, y, ts, mask, refits=1, max_iter=60)
    return {"control": "leakage_canary", "single_feature_auc": a, "joint_recall": jp["recall_at_fpr"],
            "single_probe_fires": bool(a > SINGLE_FEATURE_GATE),
            "joint_probe_fires": bool(jp["flag_measuring_generator"]),
            "passes": bool(a > SINGLE_FEATURE_GATE and jp["flag_measuring_generator"])}


def contribution_shares(tx: pd.DataFrame, joint_recall: float, fpr: float = 0.005) -> pd.DataFrame:
    m = eval_mask(tx)
    t = tx[m]
    y = t["is_attack"].to_numpy()
    rows = []
    for c in feature_ok_columns():
        s = t[c]
        v = s.cat.codes.to_numpy(dtype=float) if hasattr(s, "cat") else s.to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(v).any() else 0.0)
        best = 0.0
        for vv in (v, -v):
            best = max(best, recall_at_fpr(y, vv + 1e-9 * np.arange(len(vv)) / len(vv), fpr)[0])
        share = best / joint_recall if joint_recall and joint_recall > 0 else float("nan")
        rows.append({"column": c, "single_recall_at_fpr": best, "share_of_headline": share,
                     "flag": bool(share > CONTRIBUTION_FLAG)})
    return pd.DataFrame(rows).sort_values("share_of_headline", ascending=False).reset_index(drop=True)


# ---- realism ---------------------------------------------------------------------------------------
def _benford_mad(amounts: np.ndarray) -> float:
    a = amounts[amounts >= 1]
    first = np.array([int(str(int(x))[0]) for x in a])
    obs = np.bincount(first, minlength=10)[1:] / max(len(first), 1)
    exp = np.log10(1 + 1 / np.arange(1, 10))
    return float(np.mean(np.abs(obs - exp)))


def _overlap(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    ha, hb = ha / max(ha.sum(), 1), hb / max(hb.sum(), 1)
    return float(np.minimum(ha, hb).sum())


def realism_scores(tx: pd.DataFrame) -> pd.DataFrame:
    m = eval_mask(tx)
    t = tx[m]
    ben = t[t.is_attack == 0]
    atk = t[t.is_attack == 1]
    rows = []

    def add(metric, benign, attack, note):
        rows.append({"metric": metric, "benign": benign, "attack": attack, "note": note})

    add("benford_mad_first_digit", _benford_mad(ben.amount_inr.to_numpy()), _benford_mad(atk.amount_inr.to_numpy()),
        "MAD vs Benford; < 0.015 close conformity")
    add("round_number_mass_100", float((ben.amount_inr % 100 == 0).mean()), float((atk.amount_inr % 100 == 0).mean()),
        "share of amounts that are multiples of 100")
    hb = np.bincount(ben.hour, minlength=24) / len(ben)
    ha = np.bincount(atk.hour, minlength=24) / max(len(atk), 1)
    add("diurnal_shape_corr", 1.0, float(np.corrcoef(hb, ha)[0, 1]), "attack hour histogram vs benign (night bias on top)")
    wb = np.bincount(ben.dow, minlength=7) / len(ben)
    wa = np.bincount(atk.dow, minlength=7) / max(len(atk), 1)
    add("weekly_shape_corr", 1.0, float(np.corrcoef(wb, wa)[0, 1]), "attack weekday histogram vs benign")
    add("amount_tail_p99_over_p50", float(ben.amount_inr.quantile(0.99) / ben.amount_inr.median()),
        float(atk.amount_inr.quantile(0.99) / atk.amount_inr.median()), "tail heaviness")
    cv = ben.groupby("user_id").amount_inr.apply(lambda s: np.log1p(s).std()).dropna()
    add("per_user_log_amount_std_median", float(cv.median()), float("nan"), "per-customer consistency")
    indeg = t.groupby("payee_id").user_id.nunique()
    add("payee_indegree_p50", float(indeg.median()), float("nan"), "payee-graph shape")
    add("payee_indegree_p99", float(indeg.quantile(0.99)), float("nan"), "payee-graph shape (hubs)")
    add("scam_legit_log_amount_overlap", _overlap(np.log1p(ben.amount_inr.to_numpy()), np.log1p(atk.amount_inr.to_numpy())),
        float("nan"), "histogram overlap coefficient (1 = identical)")
    add("first_time_payee_share", float((ben.user_to_payee_prior_txn_count == 0).mean()),
        float((atk.user_to_payee_prior_txn_count == 0).mean()), "share of payments to a never-paid payee")
    return pd.DataFrame(rows)
