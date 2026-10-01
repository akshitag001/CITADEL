"""Causal trailing-window aggregates over an event stream.

Every aggregate answers "what had this entity done BEFORE this event". Rows sharing the event's
timestamp are excluded too: a burst that lands in the same second must not see itself.

Implementation: each (entity code, timestamp) pair is mapped to one int64 sort key
``code * STRIDE + (ts - T_MIN)``. Because every key of one entity lies in its own STRIDE-wide band,
a single ``searchsorted`` for ``(code, t - window)`` can never reach a neighbouring entity, so window
bounds for all rows come from two vectorised binary searches and prefix sums.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

T_MIN = -(10 ** 10)          # keys stay positive for any timestamp after ~1653 AD minus a window
STRIDE = 2 * 10 ** 10        # > any timestamp span we will ever see (~600 years)


def codes_of(*arrays: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Integer codes for several id arrays, drawn from ONE shared vocabulary."""
    sizes = [len(a) for a in arrays]
    joined = np.concatenate([np.asarray(a).astype(str) for a in arrays]) if arrays else np.array([], dtype=str)
    codes, uniques = pd.factorize(joined, sort=False)
    codes = codes.astype(np.int64)
    split = np.split(codes, np.cumsum(sizes)[:-1]) if sizes else []
    return list(split), len(uniques)


def _key(codes: np.ndarray, ts: np.ndarray) -> np.ndarray:
    t = np.asarray(ts, dtype=np.int64) - T_MIN
    return np.asarray(codes, dtype=np.int64) * STRIDE + np.clip(t, 0, STRIDE - 1)


class _Index:
    """Rows sorted by (entity, time) with their keys; ``bounds`` gives [lo, hi) per row."""

    def __init__(self, codes: np.ndarray, ts: np.ndarray):
        self.codes = np.asarray(codes, dtype=np.int64)
        self.ts = np.asarray(ts, dtype=np.int64)
        self.order = np.lexsort((self.ts, self.codes))
        self.keys = _key(self.codes[self.order], self.ts[self.order])

    def before(self, q_codes: np.ndarray, q_ts: np.ndarray) -> np.ndarray:
        """Position just past the last indexed row with the same code and time < q_ts."""
        return np.searchsorted(self.keys, _key(q_codes, q_ts), side="left")

    def since(self, q_codes: np.ndarray, q_ts: np.ndarray) -> np.ndarray:
        """Position of the first indexed row with the same code and time >= q_ts."""
        return np.searchsorted(self.keys, _key(q_codes, q_ts), side="left")

    def entity_start(self, q_codes: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.keys, np.asarray(q_codes, dtype=np.int64) * STRIDE, side="left")

    def prefix(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)[self.order]
        return np.concatenate([[0.0], np.cumsum(v)])


def _scatter(order: np.ndarray, sorted_values: np.ndarray) -> np.ndarray:
    out = np.empty_like(sorted_values)
    out[order] = sorted_values
    return out


def prior_rank(codes: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """How many events the same entity had strictly earlier in time."""
    if len(codes) == 0:
        return np.array([], dtype=np.int64)
    ix = _Index(codes, ts)
    c, t = ix.codes[ix.order], ix.ts[ix.order]
    return _scatter(ix.order, (ix.before(c, t) - ix.entity_start(c)).astype(np.int64))


def window_sum_count(codes: np.ndarray, ts: np.ndarray, values: np.ndarray,
                     window_s: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of the entity's values over [t - window, t)."""
    if len(codes) == 0:
        return np.array([]), np.array([])
    ix = _Index(codes, ts)
    c, t = ix.codes[ix.order], ix.ts[ix.order]
    hi, lo = ix.before(c, t), ix.since(c, t - window_s)
    ps = ix.prefix(values)
    return _scatter(ix.order, ps[hi] - ps[lo]), _scatter(ix.order, (hi - lo).astype(float))


def expanding_mean_std(codes: np.ndarray, ts: np.ndarray,
                       values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, population std and count of the entity's values over ALL strictly earlier events."""
    if len(codes) == 0:
        return np.array([]), np.array([]), np.array([])
    ix = _Index(codes, ts)
    c, t = ix.codes[ix.order], ix.ts[ix.order]
    hi, lo = ix.before(c, t), ix.entity_start(c)
    v = np.asarray(values, dtype=float)
    s1, s2 = ix.prefix(v), ix.prefix(v * v)
    n = (hi - lo).astype(float)
    tot, tot2 = s1[hi] - s1[lo], s2[hi] - s2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
        var = np.where(n > 1, tot2 / np.maximum(n, 1) - mean * mean, np.nan)
    return (_scatter(ix.order, mean), _scatter(ix.order, np.sqrt(np.maximum(var, 0))), _scatter(ix.order, n))


def cross_window_sum(src_codes: np.ndarray, src_ts: np.ndarray, src_values: np.ndarray,
                     q_codes: np.ndarray, q_ts: np.ndarray, window_s: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of SOURCE events with the query's code in [q_ts - window, q_ts)."""
    if len(src_codes) == 0:
        return np.zeros(len(q_codes)), np.zeros(len(q_codes))
    ix = _Index(src_codes, src_ts)
    q_ts = np.asarray(q_ts, dtype=np.int64)
    hi, lo = ix.before(q_codes, q_ts), ix.since(q_codes, q_ts - window_s)
    ps = ix.prefix(src_values)
    return ps[hi] - ps[lo], (hi - lo).astype(float)


def cross_count_before(src_codes: np.ndarray, src_ts: np.ndarray, q_codes: np.ndarray,
                       q_ts: np.ndarray) -> np.ndarray:
    """Number of SOURCE events with the query's code strictly before the query time."""
    if len(src_codes) == 0:
        return np.zeros(len(q_codes), dtype=np.int64)
    ix = _Index(src_codes, src_ts)
    return (ix.before(q_codes, q_ts) - ix.entity_start(q_codes)).astype(np.int64)


def cross_last_ts(src_codes: np.ndarray, src_ts: np.ndarray, q_codes: np.ndarray,
                  q_ts: np.ndarray) -> np.ndarray:
    """Time of the latest SOURCE event with the query's code strictly before q_ts (NaN if none)."""
    out = np.full(len(q_codes), np.nan)
    if len(src_codes) == 0 or len(q_codes) == 0:
        return out
    ix = _Index(src_codes, src_ts)
    pos = ix.before(q_codes, q_ts) - 1
    has = pos >= ix.entity_start(q_codes)
    out[has] = ix.ts[ix.order][pos[has]]
    return out


def last_event_ts(codes: np.ndarray, ts: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """Time of the entity's latest strictly-earlier event with ``flag`` set (NaN if none)."""
    sel = np.flatnonzero(np.asarray(flag, dtype=bool))
    return cross_last_ts(np.asarray(codes)[sel], np.asarray(ts)[sel], codes, ts)


def window_unique(codes: np.ndarray, ts: np.ndarray, secondary: np.ndarray, window_s: int) -> np.ndarray:
    """Distinct ``secondary`` values per entity in [t - window, t).

    Distinct counts have no prefix-sum form, so this walks each entity's time-ordered rows once with a
    sliding multiset; rows that share a timestamp are answered together before any of them is added.
    """
    n = len(codes)
    if n == 0:
        return np.array([], dtype=np.int32)
    ix = _Index(codes, ts)
    c = ix.codes[ix.order].tolist()
    t = ix.ts[ix.order].tolist()
    s = np.asarray(secondary)[ix.order].tolist()
    res = [0] * n
    start = 0
    while start < n:
        end = start
        while end < n and c[end] == c[start]:
            end += 1
        live: dict = {}
        tail = start
        i = start
        while i < end:
            j = i
            while j < end and t[j] == t[i]:
                j += 1
            floor = t[i] - window_s
            while tail < i and t[tail] < floor:
                k = s[tail]
                if live[k] == 1:
                    del live[k]
                else:
                    live[k] -= 1
                tail += 1
            for m in range(i, j):
                res[m] = len(live)
            for m in range(i, j):
                live[s[m]] = live.get(s[m], 0) + 1
            i = j
        start = end
    return _scatter(ix.order, np.asarray(res, dtype=np.int32))
