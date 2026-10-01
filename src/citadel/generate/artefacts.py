"""Automated hunt for fraud-only namespaces.

Searches every non-truth column for a categorical value, an id PREFIX or a narrow numeric band that
scam rows carry and legitimate rows almost never do. Any hit is a generator defect: the defence would
learn "how the row was made", not "what the scam looks like".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import COLUMNS, ID, TRUTH_ONLY
from .fidelity import eval_mask

PURITY = 0.5          # P(attack | value) at or above this ...
MIN_SUPPORT = 0.05    # ... carried by at least 5% of attack rows (and >= 10 rows) is an artefact
ID_PREFIX_LEN = (1, 2, 3)
NUM_BINS = 200


MIN_DISTINCT_IDS = 5  # an id-prefix "namespace" must span several accounts, not one reused mule


def _check(values: pd.Series, y: np.ndarray, column: str, kind: str, n_pos: int,
           ids: pd.Series | None = None) -> list[dict]:
    df = pd.DataFrame({"v": values.astype(str).to_numpy(), "y": y})
    g = df.groupby("v")["y"].agg(["sum", "count"])
    g["purity"] = g["sum"] / g["count"]
    hit = g[(g["purity"] >= PURITY) & (g["sum"] >= max(10, MIN_SUPPORT * n_pos))]
    if ids is not None and len(hit):
        df["id"] = ids.astype(str).to_numpy()
        distinct = df[df.y == 1].groupby("v")["id"].nunique()
        hit = hit[distinct.reindex(hit.index).fillna(0) >= MIN_DISTINCT_IDS]
    return [{"column": column, "kind": kind, "value": v, "attack_rows": int(r["sum"]), "rows": int(r["count"]),
             "purity": float(r["purity"])} for v, r in hit.iterrows()]


def hunt(tx: pd.DataFrame) -> pd.DataFrame:
    m = eval_mask(tx)
    t = tx[m]
    y = t["is_attack"].to_numpy()
    n_pos = int(y.sum())
    hits: list[dict] = []
    for c, spec in COLUMNS.items():
        if spec.role == TRUTH_ONLY or c == "ts":
            continue
        s = t[c]
        if spec.role == ID:
            if spec.dtype == "str":
                for k in ID_PREFIX_LEN:
                    hits += _check(s.astype(str).str[:k], y, c, f"id_prefix_{k}", n_pos, ids=s)
                # id length / alphabet are namespaces too
                hits += _check(s.astype(str).str.len(), y, c, "id_length", n_pos, ids=s)
            continue
        if spec.dtype in ("category", "str"):
            hits += _check(s, y, c, "category_value", n_pos)
            continue
        v = s.to_numpy(dtype=float)
        nul = np.isnan(v)
        hits += _check(pd.Series(nul), y, c, "null_pattern", n_pos)
        vv = v[~nul]
        if len(vv) == 0:
            continue
        uniq = np.unique(vv)
        if len(uniq) <= 50:
            hits += _check(pd.Series(np.where(nul, np.nan, v)), y, c, "exact_value", n_pos)
        else:
            edges = np.unique(np.quantile(vv, np.linspace(0, 1, NUM_BINS + 1)))
            band = np.digitize(np.where(nul, np.nan, v), edges[1:-1])
            hits += _check(pd.Series(np.where(nul, -1, band)), y, c, "numeric_band", n_pos)
    out = pd.DataFrame(hits, columns=["column", "kind", "value", "attack_rows", "rows", "purity"])
    out["n_attack_rows_total"] = n_pos
    return out
