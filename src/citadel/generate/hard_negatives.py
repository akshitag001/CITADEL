"""Legitimate payments engineered to look alarming (false-positive generators by design).

Each archetype mutates a real benign payment, exactly as attacks do, so the defence cannot separate
scams from hard negatives by "how the row was made". About a third of the selected rows stack 2-3
archetypes at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, rng
from ..timeutil import HOUR_S, hour_ist
from .entities import Population

ARCHETYPES = (
    "night_new_payee_new_device",   # first large transfer to a new payee at night on a new phone
    "reinstall_support_call",       # new phone + SIM swap + on a (genuine) support call + large payment
    "family_screen_share",          # a relative helps over a video-call screen share
    "new_merchant_quick_collect",   # fast approval of a collect from a new online merchant
    "split_to_new_payee",           # paying a new contractor in parts
    "large_balance_share",          # deposit / rent advance to a new landlord
    "qr_to_new_person",             # paying a street vendor's personal QR
)


def _new_person(r: np.random.Generator, pop: Population, payer: np.ndarray, ts: np.ndarray) -> np.ndarray:
    acc = pop.accounts
    persons = acc[(acc.kind == "person")]
    cand = persons["idx"].to_numpy()
    created = acc["created_ts"].to_numpy()
    out = r.choice(cand, len(payer))
    bad = (out == payer) | (created[out] > ts)
    while bad.any():
        out[bad] = r.choice(cand, int(bad.sum()))
        bad = (out == payer) | (created[out] > ts)
    return out


def apply_hard_negatives(cfg: Config, pop: Population, df: pd.DataFrame) -> pd.DataFrame:
    r = rng(cfg, "hard_negatives")
    acc = pop.accounts
    n_target = int(round(cfg.traffic.hard_negative_share * len(df)))
    if n_target == 0:
        return df
    # bias toward evening/night and older customers: where alarming-looking legitimate payments live
    hour = hour_ist(df["ts"].to_numpy())
    w = np.where((hour >= 20) | (hour <= 5), 2.0, 1.0)
    w = w * np.where(acc["user_age_band"].to_numpy()[df["_payer"].to_numpy()] == "60+", 1.6, 1.0)
    w = w * (df["row_kind"].to_numpy() == "benign")
    sel = r.choice(len(df), size=n_target, replace=False, p=w / w.sum())
    sel.sort()

    n_arch = np.where(r.random(n_target) < 0.33, r.integers(2, 4, n_target), 1)
    chosen = [r.choice(len(ARCHETYPES), size=k, replace=False) for k in n_arch]
    mask = np.zeros((n_target, len(ARCHETYPES)), dtype=bool)
    for i, c in enumerate(chosen):
        mask[i, c] = True
    # split_to_new_payee and qr_to_new_person imply their own payee/type; keep them exclusive of collect
    clash = mask[:, 3] & (mask[:, 4] | mask[:, 6])
    mask[clash, 3] = False
    mask[~mask.any(axis=1), 0] = True

    out = df.copy()
    rows = out.iloc[sel].copy()
    payer = rows["_payer"].to_numpy()
    ts = rows["ts"].to_numpy()
    balance = acc["account_balance_inr"].to_numpy()[payer]
    base = np.exp(acc["amount_mu"].to_numpy()[payer])
    screen_obs = acc["screen_obs"].to_numpy()[payer] == 1
    consent = acc["consent_call"].to_numpy()[payer] == 1
    n = len(rows)

    def setcol(col: str, m: np.ndarray, values) -> None:
        vals = rows[col].to_numpy().copy()
        vals[m] = np.asarray(values)[m] if np.ndim(values) else values
        rows[col] = vals

    new_payee = mask[:, 0] | mask[:, 4] | mask[:, 5] | mask[:, 6]
    setcol("_payee", new_payee, _new_person(r, pop, payer, ts))
    # new device (archetypes 0, 1)
    newdev = mask[:, 0] | mask[:, 1]
    keys = rows["_device_key"].to_numpy().copy()
    bound = rows["_bound_ts"].to_numpy().copy()
    for i in np.flatnonzero(newdev):
        created = int(ts[i] - r.uniform(0.5, 20) * HOUR_S)
        keys[i] = pop.new_device(created)
        bound[i] = created
    rows["_device_key"], rows["_bound_ts"] = keys, bound
    for i in np.flatnonzero(mask[:, 1]):
        pop.extra_sim_changes.append((int(payer[i]), int(ts[i] - r.uniform(1, 120) * HOUR_S)))
    setcol("call_in_progress", mask[:, 1] & consent, 1.0)
    setcol("screen_share_active", mask[:, 2] & screen_obs, 1.0)
    # family help happens over a video-call screen share, not a remote-CONTROL app (AnyDesk-class)
    setcol("session_duration_s", mask[:, 2], np.exp(r.normal(np.log(600), 0.5, n)).round(1))
    # amounts
    big = np.minimum(base * r.uniform(4, 15, n), balance * 0.9)
    setcol("amount_inr", mask[:, 0] | mask[:, 1], big.round(2))
    setcol("amount_inr", mask[:, 5], (balance * r.uniform(0.4, 0.85, n)).round(0))
    setcol("amount_inr", mask[:, 6], np.minimum(base * r.uniform(1, 4, n), balance * 0.9).round(0))
    # collect from a new merchant, approved quickly
    merch = acc[(acc.kind == "merchant")]
    young = merch[merch.new_cohort == 1]["idx"].to_numpy()
    m_pick = r.choice(young if len(young) else merch["idx"].to_numpy(), n)
    m_pick = np.where(acc["created_ts"].to_numpy()[m_pick] > ts, r.choice(merch["idx"].to_numpy(), n), m_pick)
    setcol("_payee", mask[:, 3], m_pick)
    setcol("txn_type", mask[:, 3], "collect")
    setcol("collect_request_age_s", mask[:, 3], r.uniform(6.5, 40, n).round(1))
    setcol("_phonebook", mask[:, 3], 0)
    setcol("amount_inr", mask[:, 3], np.minimum(base * r.uniform(1, 4, n), balance * 0.9).round(2))
    setcol("txn_type", mask[:, 6], "qr_pay")
    setcol("txn_type", mask[:, 4] | mask[:, 5] | mask[:, 0], "pay")
    setcol("collect_request_age_s", rows["txn_type"].to_numpy() != "collect", np.nan)
    rows["row_kind"] = "hard_negative"
    rows["hn_archetype"] = ["+".join(ARCHETYPES[j] for j in np.flatnonzero(m)) for m in mask]
    for c in out.columns:
        col = out[c].to_numpy().copy()
        col[sel] = rows[c].to_numpy().astype(col.dtype, copy=False)
        out[c] = col

    # split payments: 1-3 more parts to the same new payee within minutes
    split = rows[mask[:, 4]]
    extra = []
    for _, row in split.iterrows():
        for k in range(int(r.integers(1, 4))):
            e = row.copy()
            e["ts"] = int(row["ts"] + (k + 1) * r.uniform(2, 15) * 60)
            e["amount_inr"] = round(float(row["amount_inr"]) * r.uniform(0.7, 1.2), 0)
            e["session_duration_s"] = round(float(np.exp(r.normal(np.log(38), 0.65))), 1)
            extra.append(e)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    out = out.astype(df.dtypes.to_dict())
    return out.sort_values("ts", kind="stable").reset_index(drop=True)
