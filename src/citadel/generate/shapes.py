"""Legitimate traffic shaped like scam infrastructure (false-positive generators by design).

A mule account collects from many strangers and quickly pays out to several others. So do tutors on
fee day, charity organisers, small businesses paying contractors, friends splitting a bill and pop-up
festival stalls. If the defence learns "fan-in" or "fan-out" alone, it pays for it here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, rng
from ..timeutil import DAY_S, HOUR_S
from .benign import session_fields
from .entities import Population

SHAPES = ("tutor_fee_day", "split_bill_collect", "contractor_payout", "charity_spike", "festival_popup")


def _rows(r, pop, payer, payee, ts, txn_type, amount, archetype) -> pd.DataFrame:
    payer, payee, ts = np.asarray(payer), np.asarray(payee), np.asarray(ts, dtype=np.int64)
    txn_type = np.asarray(txn_type, dtype=object)
    fields = session_fields(r, pop, payer, payee, txn_type, ts, np.ones(len(payer), dtype=bool))
    return pd.DataFrame({"ts": ts, "_payer": payer, "_payee": payee, "txn_type": txn_type,
                         "amount_inr": np.round(np.asarray(amount, dtype=float), 2), **fields,
                         "row_kind": "shape", "hn_archetype": archetype})


def _alive_persons(pop: Population, ts: int) -> np.ndarray:
    acc = pop.accounts
    m = (acc.kind == "person") & (acc.created_ts < ts)
    return acc.loc[m, "idx"].to_numpy()


def generate_shapes(cfg: Config, pop: Population, n_benign: int) -> pd.DataFrame:
    r = rng(cfg, "shapes")
    target = int(cfg.traffic.shape_share * n_benign)
    acc = pop.accounts
    persons = acc[acc.kind == "person"]
    frames, made = [], 0
    days = cfg.traffic.days
    k = 0
    while made < target:
        kind = SHAPES[k % len(SHAPES)]
        k += 1
        day = int(r.integers(1, days))
        t_base = pop.t0 + day * DAY_S + int(r.integers(9, 19)) * HOUR_S
        alive = _alive_persons(pop, t_base)
        if kind == "tutor_fee_day":
            tutor = int(r.choice(alive))
            n = int(r.integers(15, 41))
            payers = r.choice(alive[alive != tutor], n, replace=False)
            ts = t_base + np.sort(r.uniform(0, 14 * HOUR_S, n)).astype(np.int64)
            fee = float(r.choice([500, 800, 1000, 1500, 2000]))
            f = _rows(r, pop, payers, np.full(n, tutor), ts, np.where(r.random(n) < 0.6, "pay", "qr_pay"),
                      np.full(n, fee), kind)
        elif kind == "split_bill_collect":
            initiator = int(r.choice(alive))
            n = int(r.integers(3, 7))
            payers = r.choice(alive[alive != initiator], n, replace=False)
            ts = t_base + np.sort(r.uniform(0, 1 * HOUR_S, n)).astype(np.int64)
            share = round(float(np.exp(r.normal(np.log(600), 0.6))), 0)
            f = _rows(r, pop, payers, np.full(n, initiator), ts, np.full(n, "collect"), np.full(n, share), kind)
            f["collect_request_age_s"] = np.exp(r.normal(np.log(300), 1.0, n)).round(1) + 6
            f["_phonebook"] = (r.random(n) < 0.5).astype(np.int8)
        elif kind == "contractor_payout":
            biz = int(r.choice(persons.loc[persons.user_age_band.isin(["26-40", "41-60"]), "idx"].to_numpy()))
            n = int(r.integers(15, 41))
            payees = r.choice(alive[alive != biz], n, replace=False)
            ts = t_base + 2 * HOUR_S + np.sort(r.uniform(0, 30 * 60, n)).astype(np.int64)
            amts = np.exp(r.normal(np.log(1500), 0.4, n)).round(-1)
            f = _rows(r, pop, np.full(n, biz), payees, ts, np.full(n, "pay"), amts, kind)
            client = int(r.choice(alive[alive != biz]))
            inbound = _rows(r, pop, [client], [biz], [t_base + int(r.uniform(0, HOUR_S))], ["pay"],
                            [round(float(amts.sum() * 1.1), -2)], kind)
            f = pd.concat([inbound, f], ignore_index=True)
        elif kind == "charity_spike":
            org = int(r.choice(alive))
            n = int(r.integers(40, 151))
            payers = r.choice(alive[alive != org], min(n, len(alive) - 1), replace=False)
            n = len(payers)
            ts = t_base + np.sort(r.uniform(0, 20 * HOUR_S, n)).astype(np.int64)
            amts = r.choice([101, 251, 501, 1001, 2100], n, p=[0.3, 0.3, 0.25, 0.12, 0.03])
            f = _rows(r, pop, payers, np.full(n, org), ts, np.where(r.random(n) < 0.5, "pay", "qr_pay"),
                      amts, kind)
        else:  # festival_popup: a young unverified stall gets many first-time payers on the festival day
            fday = min(cfg.traffic.festival_day, days - 1)
            t_f = pop.t0 + fday * DAY_S + 10 * HOUR_S
            merch = acc[(acc.kind == "merchant") & (acc.created_ts < t_f)]
            young = merch[merch.new_cohort == 1]
            stall = int(r.choice(young["idx"].to_numpy() if len(young) else merch["idx"].to_numpy()))
            n = int(r.integers(30, 81))
            alive_f = _alive_persons(pop, t_f)
            payers = r.choice(alive_f, n, replace=False)
            ts = t_f + np.sort(r.uniform(0, 11 * HOUR_S, n)).astype(np.int64)
            amts = np.exp(r.normal(np.log(250), 0.6, n)).round(0)
            f = _rows(r, pop, payers, np.full(n, stall), ts, np.full(n, "qr_pay"), amts, kind)
        frames.append(f)
        made += len(f)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["ts"] < pop.t_end]
    return out.sort_values("ts", kind="stable").reset_index(drop=True)
