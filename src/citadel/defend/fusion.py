"""Score fusion (scores, never thresholded decisions). The rule is chosen by ablation on the STATS
slice, never on the test window, and every arm — including the losers — is published.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ARMS = ("L1_alone", "L1_L2_weighted", "equal_weights_all", "drop_dead_channels", "rank_average",
        "stacked_logistic")
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Fusion:
    arm: str
    channels: list[str]
    weights: dict[str, float] = field(default_factory=dict)
    ref: dict[str, np.ndarray] = field(default_factory=dict)     # sorted reference scores for ranks
    meta: LogisticRegression | None = None

    def rank(self, name: str, s: np.ndarray) -> np.ndarray:
        ref = self.ref[name]
        return np.searchsorted(ref, s, side="right") / max(len(ref), 1)

    def __call__(self, cal: dict[str, np.ndarray]) -> np.ndarray:
        if self.arm in ("L1_alone",):
            return cal["L1"]
        if self.arm in ("L1_L2_weighted", "equal_weights_all", "drop_dead_channels"):
            return sum(self.weights[c] * cal[c] for c in self.channels)
        if self.arm == "rank_average":
            return np.mean([self.rank(c, cal[c]) for c in self.channels], axis=0)
        if self.arm == "stacked_logistic":
            Z = np.column_stack([_logit(cal[c]) for c in self.channels])
            return self.meta.predict_proba(Z)[:, 1]
        raise ValueError(self.arm)


def build_arms(cal_calib: dict[str, np.ndarray], y_calib: np.ndarray, dead: set[str]) -> dict[str, Fusion]:
    names = list(cal_calib)
    refs = {c: np.sort(cal_calib[c]) for c in names}
    arms = {
        "L1_alone": Fusion("L1_alone", ["L1"]),
        "L1_L2_weighted": Fusion("L1_L2_weighted", ["L1", "L2"], {"L1": 0.8, "L2": 0.2}),
        "equal_weights_all": Fusion("equal_weights_all", names, {c: 1 / len(names) for c in names}),
    }
    live = [c for c in names if c not in dead] or ["L1"]
    arms["drop_dead_channels"] = Fusion("drop_dead_channels", live, {c: 1 / len(live) for c in live})
    arms["rank_average"] = Fusion("rank_average", names, ref=refs)
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(np.column_stack([_logit(cal_calib[c]) for c in names]), y_calib)
    arms["stacked_logistic"] = Fusion("stacked_logistic", names, meta=meta)
    return arms


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() else float("nan")
