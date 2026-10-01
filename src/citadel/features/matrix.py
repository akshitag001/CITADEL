"""Build the causal feature matrix from schema rows.

Rules: every window is TRAILING and EXCLUDES the row being scored; features an institution cannot
construct are NaN (not zero); no TRUTH_ONLY column is ever read here (tests enforce it by building the
matrix from a frame with the truth columns dropped).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import COLUMNS
from ..timeutil import DAY_S, HOUR_S
from . import windows as W
from .graph import GRAPH_COLS, GraphState, snapshot_features
from .registry import FEATURES

MIN_HISTORY = 3
ORD = {
    "txn_type": {"pay": 0, "collect": 1, "qr_pay": 2},
    "user_age_band": {"18-25": 0, "26-40": 1, "41-60": 2, "60+": 3},
    "digital_literacy": {"low": 0, "med": 1, "high": 2},
    "balance_band": {"low": 0, "mid": 1, "high": 2},
}


def _ord(s: pd.Series, col: str) -> np.ndarray:
    return s.astype(str).map(ORD[col]).to_numpy(dtype=float)


def build_matrix(tx: pd.DataFrame, profiles: pd.DataFrame | None = None,
                 known_mules: pd.DataFrame | None = None,
                 graph_state: GraphState | None = None) -> pd.DataFrame:
    """``tx`` schema rows (truth columns optional and ignored) -> feature frame aligned to ``tx``.

    ``profiles``: user_id, prehist_n, prehist_mean_log_amt, prehist_std_log_amt (PSP profile store).
    ``known_mules``: payee_id, arrival_ts of confirmed scam labels (graph snapshots).
    ``graph_state``: when given (online scoring), graph features come from this state instead of
    nightly snapshots over ``tx``.
    """
    tx = tx[[c for c in tx.columns if c in COLUMNS and COLUMNS[c].role != "TRUTH_ONLY"]]
    n = len(tx)
    ts = tx["ts"].to_numpy(dtype=np.int64)
    amount = tx["amount_inr"].to_numpy(dtype=float)
    la = np.log1p(amount)
    (uc, pc), _ = W.codes_of(tx["user_id"].astype(str).to_numpy(), tx["payee_id"].astype(str).to_numpy())
    prior = tx["user_to_payee_prior_txn_count"].to_numpy(dtype=float)
    new_payee = (prior == 0).astype(float)
    f = pd.DataFrame(index=tx.index)

    # ---- raw ---------------------------------------------------------------------------------------
    f["f_amount_log"] = la
    f["f_txn_type"] = _ord(tx["txn_type"], "txn_type")
    f["f_hour"] = tx["hour"].to_numpy(dtype=float)
    f["f_dow"] = tx["dow"].to_numpy(dtype=float)
    f["f_age_band"] = _ord(tx["user_age_band"], "user_age_band")
    f["f_literacy"] = _ord(tx["digital_literacy"], "digital_literacy")
    f["f_tier"] = tx["home_tier"].to_numpy(dtype=float)
    f["f_tenure_days"] = tx["user_tenure_days"].to_numpy(dtype=float)
    f["f_balance_band"] = _ord(tx["balance_band"], "balance_band")
    f["f_payee_age_days"] = tx["payee_age_days"].to_numpy(dtype=float)
    f["f_payee_prior"] = prior
    f["f_payee_first_time"] = new_payee
    f["f_payee_unique_payers_24h"] = tx["payee_inbound_unique_payers_24h"].to_numpy(dtype=float)
    f["f_payee_inbound_amt_24h"] = tx["payee_inbound_amount_24h"].to_numpy(dtype=float)
    f["f_payee_complaints"] = tx["payee_complaint_count"].to_numpy(dtype=float)
    f["f_payee_verified"] = tx["payee_is_verified_merchant"].to_numpy(dtype=float)
    f["f_payee_name_sim"] = tx["payee_name_similarity_to_known_contact"].to_numpy(dtype=float)
    f["f_device_age_days"] = tx["device_age_days"].to_numpy(dtype=float)
    f["f_new_device"] = tx["new_device_flag"].to_numpy(dtype=float)
    f["f_sim_change"] = tx["sim_change_7d"].to_numpy(dtype=float)
    f["f_session_duration"] = tx["session_duration_s"].to_numpy(dtype=float)
    f["f_attempts"] = tx["attempts_in_session"].to_numpy(dtype=float)
    f["f_collect_age_s"] = tx["collect_request_age_s"].to_numpy(dtype=float)
    f["f_req_known_contact"] = tx["request_source_known_contact"].to_numpy(dtype=float)
    screen = tx["screen_share_active"].to_numpy(dtype=float)
    remote = tx["remote_access_app_detected"].to_numpy(dtype=float)
    f["f_screen_share"] = screen
    f["f_remote_access"] = remote
    f["f_call"] = tx["call_in_progress"].to_numpy(dtype=float)

    # ---- user baseline (in-window history combined with the PSP profile store) --------------------
    m_s, sd_s, n_s = W.expanding_mean_std(uc, ts, la)
    if profiles is not None and len(profiles):
        prof = profiles.set_index("user_id").reindex(tx["user_id"].astype(str))
        n0 = prof["prehist_n"].fillna(0).to_numpy(dtype=float)
        m0 = prof["prehist_mean_log_amt"].to_numpy(dtype=float)
        s0 = prof["prehist_std_log_amt"].to_numpy(dtype=float)
    else:
        n0, m0, s0 = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    m0z, s0z = np.nan_to_num(m0), np.nan_to_num(s0)
    m_sz, sd_sz = np.nan_to_num(m_s), np.nan_to_num(sd_s)
    n_tot = n0 + n_s
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (n0 * m0z + n_s * m_sz) / n_tot
        ex2 = (n0 * (s0z ** 2 + m0z ** 2) + n_s * (sd_sz ** 2 + m_sz ** 2)) / n_tot
        std = np.sqrt(np.maximum(ex2 - mean ** 2, 0))
    ok = n_tot >= MIN_HISTORY
    mean = np.where(ok, mean, np.nan)
    std = np.where(ok, np.maximum(std, 0.3), np.nan)
    f["f_user_txn_count_prior"] = n_tot
    f["f_is_cold_start"] = (n_tot < 5).astype(float)
    f["f_amt_z_user"] = (la - mean) / std
    f["f_amt_ratio_user"] = np.exp(la - mean)
    f["f_amt_over_user_p99"] = np.where(ok, (la > mean + 2.33 * std).astype(float), np.nan)
    bal = tx["account_balance_inr"].to_numpy(dtype=float)
    f["f_share_of_balance"] = np.where(bal > 0, amount / bal, np.nan)
    _, c1h = W.window_sum_count(uc, ts, amount, HOUR_S)
    s24, c24 = W.window_sum_count(uc, ts, amount, DAY_S)
    f["f_user_count_1h"] = c1h
    f["f_user_count_24h"] = c24
    f["f_user_amount_24h"] = s24
    np_mean, _, np_cnt = W.expanding_mean_std(uc, ts, new_payee)
    f["f_user_newpayee_rate"] = np.where(np_cnt >= MIN_HISTORY, np_mean, np.nan)
    ttype = tx["txn_type"].astype(str).to_numpy()
    (tc,), _ = W.codes_of(np.char.add(tx["user_id"].astype(str).to_numpy().astype(str), ttype.astype(str)))
    same_type_prior = W.prior_rank(tc, ts).astype(float)
    f["f_user_txn_type_share"] = np.where(n_s >= MIN_HISTORY, same_type_prior / np.maximum(n_s, 1), np.nan)
    ang = tx["hour"].to_numpy(dtype=float) / 24 * 2 * np.pi
    ms, _, cs = W.expanding_mean_std(uc, ts, np.sin(ang))
    mc, _, _ = W.expanding_mean_std(uc, ts, np.cos(ang))
    mean_ang = np.arctan2(ms, mc)
    diff = np.abs(np.angle(np.exp(1j * (ang - mean_ang)))) / (2 * np.pi) * 24
    f["f_hour_dev_user"] = np.where(cs >= MIN_HISTORY, diff, np.nan)
    ls = np.log1p(tx["session_duration_s"].to_numpy(dtype=float))
    msess, _, csess = W.expanding_mean_std(uc, ts, ls)
    f["f_session_ratio_user"] = np.where(csess >= MIN_HISTORY, np.exp(ls - msess), np.nan)

    # ---- sequence ----------------------------------------------------------------------------------
    anomaly = ((tx["new_device_flag"].to_numpy() == 1) | (tx["sim_change_7d"].to_numpy() == 1)
               | (np.nan_to_num(screen) == 1) | (np.nan_to_num(remote) == 1))
    last_anom = W.last_event_ts(uc, ts, anomaly)
    since = ts - last_anom
    f["f_secs_since_anomaly"] = since
    recent = anomaly | (np.nan_to_num(since, nan=1e12) <= 3600)
    f["f_anomaly_then_newpayee"] = (recent & (new_payee == 1)).astype(float)
    np1h = W.window_sum_count(uc, ts, new_payee, HOUR_S)[0]
    np24 = W.window_sum_count(uc, ts, new_payee, DAY_S)[0]
    f["f_newpayee_count_1h"] = np1h
    f["f_newpayee_count_24h"] = np24
    (pair,), _ = W.codes_of(np.char.add(np.char.add(tx["user_id"].astype(str).to_numpy().astype(str), ">"),
                                        tx["payee_id"].astype(str).to_numpy().astype(str)))
    sp_amt, sp_cnt = W.window_sum_count(pair, ts, amount, 1800)
    f["f_same_payee_count_30m"] = sp_cnt
    f["f_same_payee_amount_30m"] = sp_amt
    burst = (sp_cnt > 0) & (prior <= sp_cnt)
    f["f_burst_to_new_payee"] = np.where(burst, sp_cnt + 1, 0.0)
    f["f_burst_amount"] = np.where(burst, sp_amt + amount, 0.0)
    is_collect = ttype == "collect"
    cage = tx["collect_request_age_s"].to_numpy(dtype=float)
    lca = np.log1p(np.nan_to_num(cage))
    sel = np.flatnonzero(is_collect)
    lat = np.full(n, np.nan)
    if len(sel):
        mcol, _, ccol = W.expanding_mean_std(uc[sel], ts[sel], lca[sel])
        lat[sel] = np.where(ccol >= 2, np.exp(lca[sel] - mcol), np.nan)
    f["f_collect_latency_ratio"] = lat
    known = tx["request_source_known_contact"].to_numpy(dtype=float)
    f["f_collect_unknown_fast"] = np.where(is_collect, ((known == 0) & (cage < 120)).astype(float), np.nan)
    src = np.flatnonzero(new_payee == 1)
    inb_new = W.cross_window_sum(pc[src], ts[src], np.ones(len(src)), uc, ts, DAY_S)[0]
    inb_amt = W.cross_window_sum(pc, ts, amount, uc, ts, DAY_S)[0]
    f["f_inbound_from_new_24h"] = inb_new
    f["f_inbound_amt_24h"] = inb_amt

    # ---- payee windows -------------------------------------------------------------------------------
    f["f_payee_unique_payers_1h"] = W.window_unique(pc, ts, uc, HOUR_S).astype(float)
    f["f_payee_unique_payers_7d"] = W.window_unique(pc, ts, uc, 7 * DAY_S).astype(float)
    _, pin7 = W.window_sum_count(pc, ts, amount, 7 * DAY_S)
    pnew7, _ = W.window_sum_count(pc, ts, new_payee, 7 * DAY_S)
    f["f_payee_inbound_count_7d"] = pin7
    f["f_payee_new_payer_share_7d"] = np.where(pin7 > 0, pnew7 / np.maximum(pin7, 1), np.nan)
    out_amt, out_cnt = W.cross_window_sum(uc, ts, amount, pc, ts, DAY_S)
    f["f_payee_outbound_count_24h"] = out_cnt
    f["f_payee_passthrough_24h"] = out_amt / (tx["payee_inbound_amount_24h"].to_numpy(dtype=float) + amount)
    first_in = W.cross_last_ts(pc[_first_rows(pc, ts)], ts[_first_rows(pc, ts)], pc, ts)
    f["f_payee_activity_age_days"] = (ts - first_in) / DAY_S

    # ---- graph ---------------------------------------------------------------------------------------
    if graph_state is not None:
        g = graph_state.features(tx["user_id"].astype(str).to_numpy(), tx["payee_id"].astype(str).to_numpy(),
                                 tx["device_id"].astype(str).to_numpy())
        g = pd.DataFrame(g, columns=GRAPH_COLS, index=tx.index)
    else:
        g, _ = snapshot_features(tx, known_mules)
    for c in GRAPH_COLS:
        f[c] = g[c].to_numpy()
    return f[FEATURES].astype(np.float32)


def _first_rows(codes: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Index of each key's first event (its earliest timestamp)."""
    order = np.lexsort((ts, codes))
    first = np.ones(len(order), dtype=bool)
    first[1:] = codes[order][1:] != codes[order][:-1]
    return order[first]


def profiles_from_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    return accounts.rename(columns={"acct_id": "user_id"})[
        ["user_id", "prehist_n", "prehist_mean_log_amt", "prehist_std_log_amt"]]


def known_mules_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels is None or len(labels) == 0:
        return pd.DataFrame({"payee_id": pd.Series([], dtype=object), "arrival_ts": pd.Series([], dtype=np.int64)})
    k = labels.groupby("payee_id", as_index=False)["arrival_ts"].min()
    return k
