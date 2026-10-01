"""Risk -> action ladder, priced by VOLUME budgets, not by raw score values.

A0 none | A1 nudge | A2 warning + cooling-off | A3 hold + analyst review. There is NO decline.

Thresholds are fitted on the STATS slice of the fused, calibrated risk (the same scale used at serve
time — thresholds fitted on one scale and applied to another never fire as intended), so a retrain re-prices
nothing. A threshold never lands on a mass point (alerting is risk >= t, so a quantile on a tie would
alert every tied row); a guard refuses to ship a collapsed or non-monotonic ladder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import LadderCfg

ACTIONS = ("A0", "A1", "A2", "A3")
BANDS = {"A0": "low", "A1": "elevated", "A2": "high", "A3": "critical"}
TXN_TREATMENT = {
    "pay": {"A1": "interstitial_nudge", "A2": "interstitial_warning_with_cooling_off",
            "A3": "hold_for_analyst_with_release_path"},
    "collect": {"A1": "collect_explainer", "A2": "approving_sends_money_screen_with_cooling_off",
                "A3": "hold_for_analyst_with_release_path"},
    "qr_pay": {"A1": "qr_explainer", "A2": "qr_is_for_paying_screen_with_cooling_off",
               "A3": "hold_for_analyst_with_release_path"},
}


class LadderCollapsedError(RuntimeError):
    pass


def budget_threshold(scores: np.ndarray, share: float) -> tuple[float, float]:
    """Smallest t with share(scores >= t) <= share, never on a mass point. Returns (t, realised share)."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(s)
    if n == 0:
        return float("inf"), 0.0
    k = int(np.floor(share * n))
    if k <= 0:
        return float(np.nextafter(s[0], np.inf)), 0.0
    t = s[k - 1]
    # rows scoring exactly t: if including them all exceeds the budget, move strictly above the tie
    if (s >= t).sum() > k:
        above = s[s > t]
        t = float(above.min()) if len(above) else float(np.nextafter(t, np.inf))
    realised = float((s >= t).mean())
    return float(t), realised


@dataclass
class Ladder:
    thresholds: dict[str, float]
    realised_shares: dict[str, float]
    budgets: dict[str, float]

    def assert_valid(self) -> None:
        t = [self.thresholds[a] for a in ("A1", "A2", "A3")]
        if not (t[0] < t[1] < t[2]):
            raise LadderCollapsedError(f"ladder thresholds not strictly monotonic: {self.thresholds}")
        if any(not np.isfinite(x) for x in t):
            raise LadderCollapsedError(f"ladder has a non-finite threshold: {self.thresholds}")

    def level(self, risk: np.ndarray) -> np.ndarray:
        r = np.asarray(risk, dtype=float)
        lvl = np.zeros(len(r), dtype=np.int8)
        for i, a in enumerate(("A1", "A2", "A3"), start=1):
            lvl[r >= self.thresholds[a]] = i
        return lvl


def fit_ladder(risk_stats: np.ndarray, lc: LadderCfg) -> Ladder:
    budgets = {"A1": lc.A1_max_share, "A2": lc.A2_max_share, "A3": lc.A3_max_share}
    th, rs = {}, {}
    for a, b in budgets.items():
        th[a], rs[a] = budget_threshold(risk_stats, b)
    lad = Ladder(th, rs, budgets)
    lad.assert_valid()
    return lad
