"""Turn internal event rows into the canonical schema.

This is the "switch-side state" a PSP/bank would hold: account attributes, payee history, device
bindings, SIM events and corroborated complaints. Every derived column is computed CAUSALLY from rows
strictly before the event. The same function runs offline (whole stream) and online in the API (on a
context slice), so serving can never compute a field differently from training.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..features import windows as W
from ..schema import COLUMNS
from ..timeutil import DAY_S, dow_ist, hour_ist

REPORTER_MIN_AGE_DAYS = 30.0  # complaints only count from aged reporter accounts (poisoning defence)


def _hx(prefix: str, values: np.ndarray, seed: int) -> np.ndarray:
    return np.array([hashlib.sha1(f"{seed}:{prefix}:{v}".encode()).hexdigest()[:12] for v in values], dtype=object)


def corroborated_complaints(complaints: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """One complaint per (payee, reporter), earliest arrival, only from reporters aged >= 30 days."""
    if complaints is None or len(complaints) == 0:
        return pd.DataFrame({"payee_idx": pd.Series([], dtype=np.int64), "arrival_ts": pd.Series([], dtype=np.int64)})
    c = complaints.copy()
    created = accounts.set_index("idx")["created_ts"]
    rep_age = (c["arrival_ts"].to_numpy() - created.reindex(c["reporter_idx"]).to_numpy()) / DAY_S
    c = c[rep_age >= REPORTER_MIN_AGE_DAYS]
    c = c.sort_values("arrival_ts").drop_duplicates(["payee_idx", "reporter_idx"], keep="first")
    return c[["payee_idx", "arrival_ts"]].reset_index(drop=True)


def enrich(df: pd.DataFrame, accounts: pd.DataFrame, devices: pd.DataFrame, sim_changes: pd.DataFrame,
           complaints: pd.DataFrame, seed: int, pair_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Internal rows (``_payer``, ``_payee``, ``_device_key``, ``_bound_ts``, ``_phonebook``) -> schema."""
    df = df.sort_values(["ts", "_rid"], kind="stable").reset_index(drop=True)
    ts = df["ts"].to_numpy(dtype=np.int64)
    payer = df["_payer"].to_numpy(dtype=np.int64)
    payee = df["_payee"].to_numpy(dtype=np.int64)
    acc = accounts.set_index("idx")
    a_payer = acc.reindex(payer)
    a_payee = acc.reindex(payee)
    out = pd.DataFrame(index=df.index)
    out["txn_id"] = df["txn_id"].to_numpy()
    out["ts"] = ts
    out["user_id"] = a_payer["acct_id"].to_numpy()
    out["payee_id"] = a_payee["acct_id"].to_numpy()
    out["device_id"] = _hx("dev", df["_device_key"].to_numpy(), seed)
    out["session_id"] = _hx("sess", [f"{p}:{d}:{t // 900}" for p, d, t in
                                     zip(payer, df["_device_key"].to_numpy(), ts)], seed)
    out["consent_call_signal"] = a_payer["consent_call"].to_numpy().astype(np.int8)
    out["txn_type"] = df["txn_type"].to_numpy()
    out["amount_inr"] = df["amount_inr"].to_numpy(dtype=float)
    out["hour"] = hour_ist(ts).astype(np.int8)
    out["dow"] = dow_ist(ts).astype(np.int8)
    out["user_age_band"] = a_payer["user_age_band"].to_numpy()
    out["user_tenure_days"] = np.round((ts - a_payer["created_ts"].to_numpy()) / DAY_S, 2)
    out["digital_literacy"] = a_payer["digital_literacy"].to_numpy()
    out["balance_band"] = a_payer["balance_band"].to_numpy()
    out["account_balance_inr"] = a_payer["account_balance_inr"].to_numpy(dtype=float)
    out["home_tier"] = a_payer["home_tier"].to_numpy().astype(np.int8)
    out["payee_age_days"] = np.round((ts - a_payee["created_ts"].to_numpy()) / DAY_S, 2)

    (pc, qc), _ = W.codes_of(payer.astype(str), payee.astype(str))
    pair = pd.factorize(pd.Series(payer).astype(str) + ">" + pd.Series(payee).astype(str))[0].astype(np.int64)
    prior = W.prior_rank(pair, ts)
    if pair_history is not None and len(pair_history):
        ph = pd.DataFrame({"payer_idx": payer, "payee_idx": payee}).merge(
            pair_history, on=["payer_idx", "payee_idx"], how="left")
        prior = prior + ph["n_prior"].fillna(0).to_numpy(dtype=np.int64)
    out["user_to_payee_prior_txn_count"] = prior.astype(np.int32)
    out["payee_inbound_unique_payers_24h"] = W.window_unique(qc, ts, pc, DAY_S).astype(np.int32)
    s24, _ = W.window_sum_count(qc, ts, df["amount_inr"].to_numpy(dtype=float), DAY_S)
    out["payee_inbound_amount_24h"] = np.round(s24, 2)

    cc = corroborated_complaints(complaints, accounts)
    if len(cc):
        (src, q), _ = W.codes_of(cc["payee_idx"].astype(str).to_numpy(), payee.astype(str))
        out["payee_complaint_count"] = W.cross_count_before(src, cc["arrival_ts"].to_numpy(), q, ts).astype(np.int32)
    else:
        out["payee_complaint_count"] = np.zeros(len(df), dtype=np.int32)
    out["payee_is_verified_merchant"] = a_payee["verified"].to_numpy().astype(np.int8)
    out["payee_name_similarity_to_known_contact"] = df["payee_name_similarity_to_known_contact"].to_numpy(dtype=float)

    dev_created = devices.drop_duplicates("device_key").set_index("device_key")["created_ts"]
    dc = dev_created.reindex(df["_device_key"].to_numpy()).to_numpy(dtype=float)
    out["device_age_days"] = np.round(np.maximum(ts - dc, 0) / DAY_S, 3)
    out["new_device_flag"] = ((ts - df["_bound_ts"].to_numpy(dtype=np.int64)) < DAY_S).astype(np.int8)
    if len(sim_changes):
        (sc, uq), _ = W.codes_of(sim_changes["idx"].astype(str).to_numpy(), payer.astype(str))
        last = W.cross_last_ts(sc, sim_changes["ts"].to_numpy(dtype=np.int64), uq, ts)
        out["sim_change_7d"] = ((ts - last) < 7 * DAY_S).astype(np.int8)  # NaN compares False
    else:
        out["sim_change_7d"] = np.zeros(len(df), dtype=np.int8)
    out["session_duration_s"] = df["session_duration_s"].to_numpy(dtype=float)
    out["attempts_in_session"] = df["attempts_in_session"].to_numpy().astype(np.int8)
    is_collect = df["txn_type"].to_numpy() == "collect"
    out["collect_request_age_s"] = np.where(is_collect, df["collect_request_age_s"].to_numpy(dtype=float), np.nan)
    known = ((prior > 0) | (df["_phonebook"].to_numpy() == 1)).astype(float)
    out["request_source_known_contact"] = np.where(is_collect, known, np.nan)
    for c in ("screen_share_active", "remote_access_app_detected", "call_in_progress"):
        out[c] = df[c].to_numpy(dtype=float)

    # truth
    for c, default in (("episode_id", ""), ("scam_variant", ""), ("evasion_technique", ""),
                       ("source_txn_id", ""), ("touched_fields", ""), ("hn_archetype", ""),
                       ("row_kind", "benign")):
        out[c] = df[c].fillna(default).to_numpy() if c in df else default
    for c in ("is_attack", "step_in_sequence", "label_visible"):
        out[c] = df[c].fillna(0).to_numpy().astype(np.int8) if c in df else np.int8(0)
    out["label_delay_days"] = df["label_delay_days"].to_numpy(dtype=float) if "label_delay_days" in df else np.nan
    for c in ("txn_type", "user_age_band", "digital_literacy", "balance_band"):
        out[c] = pd.Categorical(out[c], categories=list(COLUMNS[c].categories))
    return out[list(COLUMNS)]
