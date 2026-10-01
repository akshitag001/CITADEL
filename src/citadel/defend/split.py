"""Time-forward split with purge and embargo, plus the ENTITY-LEVEL sealed holdout.

TIME-FORWARD ONLY. No random splits anywhere: a random split leaks the future through entity
aggregates (the same customer, payee and mule appear on both sides), measuring interpolation within a
campaign instead of detection of one.

Windows (IST days):  TRAIN | purge | CALIBRATION | STATS | embargo | TEST
* PURGE drops rows whose trailing windows straddle the train boundary.
* CALIBRATION fits the isotonic calibrators and the fusion meta-learner.
* STATS fits ladder thresholds and the abstention band at operating prevalence; it also picks the
  fusion winner, so the test window is never used to choose anything.
* EMBARGO keeps the test rows' 24h/7d windows from reaching into fitted windows.

SEALED HOLDOUT: every row touching an entity (victim or mule) of an episode with the withheld variant
OR the withheld evasion is removed from TRAIN, CALIBRATION and STATS (benign rows included). They stay
in TEST, where generalisation is measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..schema import MULE_INITIATED_KINDS
from ..timeutil import DAY_S


@dataclass
class Split:
    train: np.ndarray
    inner_fit: np.ndarray
    inner_valid: np.ndarray
    calib: np.ndarray
    stats: np.ndarray
    test: np.ndarray
    purge: np.ndarray
    embargo: np.ndarray
    sealed_removed: np.ndarray
    boundaries: dict
    sealed_entities: int
    strict_sealing_would_remove: np.ndarray | None = None

    def table(self, y: np.ndarray) -> pd.DataFrame:
        rows = []
        for name in ("train", "inner_fit", "inner_valid", "calib", "stats", "test", "purge", "embargo",
                     "sealed_removed", "strict_sealing_would_remove"):
            m = getattr(self, name)
            if m is None:
                continue
            rows.append({"slice": name, "n_rows": int(m.sum()), "n_positives": int(y[m].sum()),
                         "day_from": self.boundaries.get(f"{name}_from", np.nan),
                         "day_to": self.boundaries.get(f"{name}_to", np.nan)})
        out = pd.DataFrame(rows)
        for c in ("day_from", "day_to"):
            out[c] = pd.array(out[c], dtype="Int64")
        return out


def make_split(tx: pd.DataFrame, episodes: pd.DataFrame, cfg: Config, t0: int) -> Split:
    sc = cfg.split
    D = cfg.traffic.days
    day = ((tx["ts"].to_numpy() - t0) // DAY_S).astype(int)
    a = int(round(D * sc.train_share))
    p = a + int(round(sc.purge_days))
    c = p + max(1, int(round(D * sc.calibration_share)))
    s = c + max(1, int(round(D * sc.stats_share)))
    e = s + int(round(sc.embargo_days))
    if e >= D:
        raise ValueError(f"split leaves no test window (days={D})")
    inner = a - max(1, int(round(a * sc.inner_valid_share)))
    eval_rows = ~tx["row_kind"].isin(MULE_INITIATED_KINDS).to_numpy()

    is_held = ((episodes.scam_variant == cfg.sealed_holdout.variant)
               | (episodes.evasion_technique == cfg.sealed_holdout.evasion))
    held, seen = episodes[is_held], episodes[~is_held]

    def mules_of(eps: pd.DataFrame) -> set[str]:
        out: set[str] = set()
        for ms in eps.get("mule_ids", pd.Series([], dtype=object)):
            out |= set(str(ms).split(",")) - {""}
        return out

    # Seal the withheld SCRIPT: its episodes, its victims (all their rows) and mule accounts used ONLY
    # by withheld episodes. Mules shared with seen scripts stay (D-006: strict sealing would discard
    # the majority of seen-variant training positives; the cost is reported in defend_split.csv).
    exclusive_mules = mules_of(held) - mules_of(seen)
    ent = set(held["victim_id"].astype(str)) | exclusive_mules
    touched = (tx["user_id"].astype(str).isin(ent) | tx["payee_id"].astype(str).isin(ent)
               | tx["episode_id"].astype(str).isin(set(held["episode_id"].astype(str)))).to_numpy()
    strict_ent = ent | mules_of(held)
    strict = (tx["user_id"].astype(str).isin(strict_ent) | tx["payee_id"].astype(str).isin(strict_ent)).to_numpy()

    train_w = day < a
    calib_w = (day >= p) & (day < c)
    stats_w = (day >= c) & (day < s)
    test_w = day >= e
    sp = Split(
        train=train_w & eval_rows & ~touched,
        inner_fit=(day < inner) & eval_rows & ~touched,
        inner_valid=(day >= inner) & train_w & eval_rows & ~touched,
        calib=calib_w & eval_rows & ~touched,
        stats=stats_w & eval_rows & ~touched,
        test=test_w & eval_rows,
        purge=(day >= a) & (day < p),
        embargo=(day >= s) & (day < e),
        sealed_removed=(train_w | calib_w | stats_w) & eval_rows & touched,
        boundaries={"train_from": 0, "train_to": a - 1, "inner_fit_from": 0, "inner_fit_to": inner - 1,
                    "inner_valid_from": inner, "inner_valid_to": a - 1, "purge_from": a, "purge_to": p - 1,
                    "calib_from": p, "calib_to": c - 1, "stats_from": c, "stats_to": s - 1,
                    "embargo_from": s, "embargo_to": e - 1, "test_from": e, "test_to": D - 1,
                    "sealed_removed_from": 0, "sealed_removed_to": s - 1},
        sealed_entities=len(ent),
    )
    sp.strict_sealing_would_remove = (train_w | calib_w | stats_w) & eval_rows & strict
    return sp
