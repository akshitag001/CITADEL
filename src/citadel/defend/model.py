"""Scoring channels. Each is scored ALONE before any fusion (a dead channel in a
fusion silently dilutes the live one).

* L1  gradient-boosted trees over all causal features; NaN handled natively; number of boosting rounds
      chosen by early stopping on a TIME-FORWARD inner validation slice (sklearn's own early stopping
      uses a random split, which we do not allow); isotonic calibration on the calibration slice.
* L2  payee / mule-graph channel: a small interpretable logistic model over payee + graph features only.
* IF  isolation forest fitted on legitimate traffic only; kept only if it earns its place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ModelCfg
from ..features.registry import FEATURES, family_features, graph_or_payee_features

IF_FEATURES = family_features("raw", "user")


def _np(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return X[cols].to_numpy(dtype=np.float64)


@dataclass
class L1Model:
    model: HistGradientBoostingClassifier
    features: list[str]
    n_iter: int
    valid_curve: list[float]

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(_np(X, self.features))[:, 1]


def fit_l1(X: pd.DataFrame, y: np.ndarray, fit_mask: np.ndarray, valid_mask: np.ndarray, train_mask: np.ndarray,
           mc: ModelCfg, features: list[str] | None = None, seed: int = 0,
           fixed_iter: int | None = None) -> L1Model:
    features = features or FEATURES

    def make(n_iter: int) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(max_iter=n_iter, learning_rate=mc.learning_rate,
                                              max_leaf_nodes=mc.max_leaf_nodes, min_samples_leaf=mc.min_samples_leaf,
                                              l2_regularization=mc.l2_regularization, early_stopping=False,
                                              random_state=seed)

    curve: list[float] = []
    if fixed_iter is None:
        m = make(mc.max_iter)
        yf = y[fit_mask]
        m.fit(_np(X[fit_mask], features), yf, sample_weight=np.where(yf == 1, mc.positive_weight, 1.0))
        yv = y[valid_mask]
        Xv = _np(X[valid_mask], features)
        step = max(1, mc.max_iter // 40)
        best_i, best = mc.max_iter, -1.0
        for i, p in enumerate(m.staged_predict_proba(Xv), start=1):
            if i % step and i != mc.max_iter:
                continue
            ap = average_precision_score(yv, p[:, 1]) if yv.sum() else 0.0
            curve.append(float(ap))
            if ap > best + 1e-4:
                best, best_i = ap, i
        n_iter = max(best_i, 20)
    else:
        n_iter = fixed_iter
    final = make(n_iter)
    yt = y[train_mask]
    final.fit(_np(X[train_mask], features), yt, sample_weight=np.where(yt == 1, mc.positive_weight, 1.0))
    return L1Model(final, list(features), n_iter, curve)


@dataclass
class L2Model:
    pipe: object
    features: list[str]

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(_transform_l2(X, self.features))[:, 1]


def _transform_l2(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    A = X[cols].to_numpy(dtype=np.float64)
    heavy = np.sign(A) * np.log1p(np.abs(A))  # counts and amounts are heavy-tailed
    return heavy


def fit_l2(X: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> L2Model:
    cols = graph_or_payee_features()
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
    pipe.fit(_transform_l2(X[train_mask], cols), y[train_mask])
    return L2Model(pipe, cols)


@dataclass
class IFModel:
    pipe: object
    features: list[str]

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        imp, forest = self.pipe
        return -forest.score_samples(imp.transform(_np(X, self.features)))


def fit_if(X: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, seed: int = 0, max_rows: int = 60000) -> IFModel:
    legit = np.flatnonzero(train_mask & (y == 0))
    r = np.random.default_rng(seed)
    if len(legit) > max_rows:
        legit = r.choice(legit, max_rows, replace=False)
    imp = SimpleImputer(strategy="median").fit(_np(X.iloc[legit], IF_FEATURES))
    forest = IsolationForest(n_estimators=200, random_state=seed).fit(imp.transform(_np(X.iloc[legit], IF_FEATURES)))
    return IFModel((imp, forest), IF_FEATURES)


@dataclass
class Calibrator:
    """Isotonic calibration plus a 1e-7 strictly-monotone tie-break.

    Isotonic output is a step function; without the tie-break whole plateaus share one value and a
    volume-budget threshold cannot be placed (see ladder.budget_threshold). The tie-break preserves the
    raw ranking and changes the displayed probability by < 1e-7.
    """
    iso: IsotonicRegression

    def __call__(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return (1 - 1e-7) * self.iso.predict(s) + 1e-7 * (1 / (1 + np.exp(-np.clip(s, -50, 50))))


def fit_calibrator(s: np.ndarray, y: np.ndarray) -> Calibrator:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True)
    iso.fit(np.asarray(s, dtype=float), y)
    return Calibrator(iso)


def channel_alone(y: np.ndarray, channels: dict[str, np.ndarray], base_rate: float) -> pd.DataFrame:
    """Dead-channel diagnostic: PR-AUC, ROC-AUC and distinct values of every channel scored alone."""
    rows = []
    for name, s in channels.items():
        s = np.asarray(s, dtype=float)
        distinct = int(len(np.unique(s)))
        if distinct < 2 or y.sum() == 0:
            pr, roc = float("nan"), float("nan")
        else:
            pr, roc = float(average_precision_score(y, s)), float(roc_auc_score(y, s))
        dead = bool(distinct < 2 or not np.isfinite(pr) or pr < 2 * base_rate or roc < 0.55)
        anti = bool(np.isfinite(roc) and roc < 0.5)
        rows.append({"channel": name, "pr_auc": pr, "roc_auc": roc, "lift_over_base": pr / base_rate if base_rate else np.nan,
                     "distinct_values": distinct, "dead": dead, "anti_predictive": anti,
                     "n_rows": int(len(y)), "n_positives": int(y.sum())})
    return pd.DataFrame(rows)
