"""Legitimate UPI traffic.

Calendar shape comes from ONE timing module (timeutil): diurnal curve mixed with each customer's own
preferred hour, a weekly shape and a salary-window lift on days 1-5 of the month. Payees are habitual
(contacts + regular merchants, Zipf reuse) with occasional exploration of new payees, so "first-time
payee" is common in legitimate traffic too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, rng
from ..timeutil import DAY_S, WEEKLY, compose_ts, dow_ist, ist_day_number, sample_hours
from .entities import Population

BENIGN_RATE = {  # benign base rates of nullable session signals (when observable)
    "screen_share_active": 0.004,
    "remote_access_app_detected": 0.006,
    "call_in_progress": 0.08,
}
COLLECT_MIN_AGE_S = 6.0  # a human cannot read and approve a collect request faster than this


def _salary_lift(ts: np.ndarray) -> np.ndarray:
    # IST day-of-month approximation from the absolute day number (exact enough for a weighting)
    import datetime as _dt
    days = ist_day_number(ts)
    uniq, inv = np.unique(days, return_inverse=True)
    dom = np.array([(_dt.date(1970, 1, 1) + _dt.timedelta(days=int(d))).day for d in uniq])[inv]
    return np.where(dom <= 5, 1.25, 1.0)


def habitual_sets(cfg: Config, pop: Population) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per customer: padded arrays of habitual contacts and merchants (account idx), with counts."""
    r = rng(cfg, "benign/habits")
    acc = pop.accounts
    persons = acc[acc.kind == "person"]
    n_p = len(persons)
    tier = persons["home_tier"].to_numpy()
    merch_like = acc[(acc.kind == "merchant") | (acc.informal_merchant == 1)]
    m_idx = merch_like["idx"].to_numpy()
    m_tier = merch_like["home_tier"].to_numpy()
    m_pop = merch_like["popularity"].to_numpy()
    m_est = (merch_like["created_ts"].to_numpy() < pop.t0)

    KC, KM = 16, 20
    contacts = np.full((n_p, KC), -1, dtype=np.int64)
    merchants = np.full((n_p, KM), -1, dtype=np.int64)
    kc = np.clip(3 + r.poisson(5, n_p), 1, KC)
    km = np.clip(4 + r.poisson(6, n_p), 1, KM)
    by_tier = {t: np.flatnonzero(tier == t) for t in (1, 2, 3)}
    m_by_tier = {}
    for t in (1, 2, 3):
        sel = (m_tier == t) & m_est
        w = m_pop[sel]
        m_by_tier[t] = (m_idx[sel], w / w.sum())
    for i in range(n_p):
        t = tier[i]
        local = r.random(kc[i]) < 0.7
        c = np.where(local, r.choice(by_tier[t], kc[i]), r.integers(0, n_p, kc[i]))
        c = c[c != i]
        contacts[i, :len(c)] = c
        kc[i] = max(len(c), 1) if len(c) else 0
        ids, w = m_by_tier[t]
        merchants[i, :km[i]] = r.choice(ids, km[i], p=w)
    return contacts, kc, merchants, km


def generate_benign(cfg: Config, pop: Population) -> pd.DataFrame:
    r = rng(cfg, "benign/traffic")
    acc = pop.accounts
    persons = acc[acc.kind == "person"].reset_index(drop=True)
    n_p = len(persons)
    created = persons["created_ts"].to_numpy()
    start = np.maximum(pop.t0, created + 3600)
    active_days = np.maximum((pop.t_end - start) / DAY_S, 0)

    # oversample then thin by weekly/salary weights (keeps one calendar module)
    lam = persons["txn_rate"].to_numpy() * active_days * 1.3
    n_i = r.poisson(lam)
    payer = np.repeat(np.arange(n_p), n_i)
    u = r.random(len(payer))
    day_float = (start[payer] - pop.t0) / DAY_S + u * active_days[payer]
    day = np.floor(day_float).astype(np.int64)
    hours = sample_hours(r, len(payer), preferred=persons["pref_hour"].to_numpy()[payer], personal_share=0.45)
    ts = compose_ts(pop.t0, day, hours, r)
    w = WEEKLY[dow_ist(ts)] * _salary_lift(ts)
    keep = (r.random(len(ts)) < w / w.max()) & (ts >= start[payer]) & (ts < pop.t_end)
    payer, ts = payer[keep], ts[keep]
    n = len(payer)

    # ---- payee choice -----------------------------------------------------------------------------
    contacts, kc, merchants, km = habitual_sets(cfg, pop)
    pop.pair_history = pair_prehistory(cfg, pop, contacts, kc, merchants, km)
    explore = r.random(n) < persons["explore_p"].to_numpy()[payer]
    to_merchant_habit = r.random(n) < 0.6
    rank_u = r.random(n) ** 2.2
    c_rank = np.minimum((rank_u * kc[payer]).astype(int), np.maximum(kc[payer] - 1, 0))
    m_rank = np.minimum((rank_u * km[payer]).astype(int), km[payer] - 1)
    payee = np.where(to_merchant_habit, merchants[payer, m_rank], contacts[payer, c_rank])
    payee = np.where(payee < 0, merchants[payer, m_rank], payee)

    merch_like = acc[(acc.kind == "merchant") | (acc.informal_merchant == 1)]
    m_ids, m_w = merch_like["idx"].to_numpy(), merch_like["popularity"].to_numpy()
    sellers = persons.loc[persons.is_seller == 1, "idx"].to_numpy()
    n_exp = int(explore.sum())
    kind_u = r.random(n_exp)
    exp_payee = np.where(
        kind_u < 0.62, r.choice(m_ids, n_exp, p=m_w / m_w.sum()),
        np.where(kind_u < 0.75, r.choice(sellers, n_exp) if len(sellers) else r.integers(0, n_p, n_exp),
                 r.integers(0, n_p, n_exp)))
    payee[explore] = exp_payee
    # payee must exist at ts and differ from payer
    est = m_ids[merch_like["created_ts"].to_numpy() < pop.t0]
    bad = (acc["created_ts"].to_numpy()[payee] > ts) | (payee == payer)
    payee[bad] = r.choice(est, int(bad.sum()))

    merch_payee = is_merchant_like(pop, payee)

    # ---- transaction type -------------------------------------------------------------------------
    u2 = r.random(n)
    txn_type = np.where(merch_payee,
                        np.where(u2 < 0.55, "qr_pay", np.where(u2 < 0.88, "pay", "collect")),
                        np.where(u2 < 0.86, "pay", np.where(u2 < 0.97, "collect", "qr_pay")))

    # ---- amounts ----------------------------------------------------------------------------------
    mu = persons["amount_mu"].to_numpy()[payer]
    sig = persons["amount_sigma"].to_numpy()[payer]
    ticket = acc["ticket_median"].to_numpy()[payee]
    amt_merchant = np.exp(r.normal(np.log(np.maximum(ticket, 20)) + 0.3 * (mu - np.log(330)), 0.6))
    amt_p2p = np.exp(r.normal(mu + 0.45, sig))
    amount = np.where(merch_payee, amt_merchant, amt_p2p)
    round_it = (~merch_payee) & (r.random(n) < 0.55)
    amount = np.where(round_it, np.where(r.random(n) < 0.4, np.round(amount, -2), np.round(amount, -1)), amount)
    balance = persons["account_balance_inr"].to_numpy()[payer]
    amount = np.clip(np.minimum(amount, balance * r.uniform(0.6, 0.95, n)), 1, 100000).round(2)

    fields = session_fields(r, pop, payer, payee, txn_type, ts, explore)
    df = pd.DataFrame({"ts": ts, "_payer": payer, "_payee": payee, "txn_type": txn_type,
                       "amount_inr": amount, **fields, "row_kind": "benign", "hn_archetype": ""})
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def _rank_p(k: np.ndarray, j: np.ndarray) -> np.ndarray:
    """P(rank == j) under the rank draw floor(k * u**2.2) used for habitual payees."""
    k = np.maximum(k, 1)
    return np.clip(((j + 1) / k) ** (1 / 2.2) - (j / k) ** (1 / 2.2), 0, 1)


def pair_prehistory(cfg: Config, pop: Population, contacts, kc, merchants, km) -> pd.DataFrame:
    """Prior payment counts per (customer, habitual payee) from before the window opens.

    Customers have years of history before day 0; without this every habitual payee would look
    "first-time" on its first in-window payment and the defence would get "new payee" for free.
    """
    r = rng(cfg, "benign/prehistory")
    acc = pop.accounts
    n_pre = acc["prehist_n"].to_numpy()[: len(kc)]
    explore = acc["explore_p"].to_numpy()[: len(kc)]
    out = []
    for arr, k, share in ((merchants, km, 0.6), (contacts, kc, 0.4)):
        j = np.tile(np.arange(arr.shape[1]), (arr.shape[0], 1))
        p = _rank_p(k[:, None], j) * share * (1 - explore)[:, None] * (j < k[:, None])
        lam = p * n_pre[:, None]
        cnt = r.poisson(lam)
        ok = (arr >= 0) & (cnt > 0)
        rows, cols = np.nonzero(ok)
        out.append(pd.DataFrame({"payer_idx": rows.astype(np.int64), "payee_idx": arr[rows, cols],
                                 "n_prior": cnt[rows, cols].astype(np.int32)}))
    ph = pd.concat(out, ignore_index=True)
    return ph.groupby(["payer_idx", "payee_idx"], as_index=False)["n_prior"].sum()


def is_merchant_like(pop: Population, idx: np.ndarray) -> np.ndarray:
    acc = pop.accounts
    return (acc["kind"].to_numpy()[idx] == "merchant") | (acc["informal_merchant"].to_numpy()[idx] == 1)


def device_at(pop: Population, payer: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The payer's device key and binding time at ``ts`` (initial device or after a benign change)."""
    acc = pop.accounts
    dev_key = acc["device_key"].to_numpy()[payer].copy()
    bound = acc["device_bound_ts"].to_numpy()[payer].copy()
    dc = pop.device_changes
    if len(dc):
        n_acc = len(acc)
        chg_ts = np.full(n_acc, np.iinfo(np.int64).max)
        chg_key = np.full(n_acc, -1)
        chg_ts[dc["idx"].to_numpy()] = dc["ts"].to_numpy()
        chg_key[dc["idx"].to_numpy()] = dc["device_key"].to_numpy()
        after = ts >= chg_ts[payer]
        dev_key[after] = chg_key[payer][after]
        bound[after] = chg_ts[payer][after]
    return dev_key, bound


def session_fields(r: np.random.Generator, pop: Population, payer: np.ndarray, payee: np.ndarray,
                   txn_type: np.ndarray, ts: np.ndarray, new_payee: np.ndarray | None = None) -> dict:
    """Benign session, device and nullable-signal fields for the given rows."""
    acc = pop.accounts
    n = len(payer)
    if new_payee is None:
        new_payee = np.zeros(n, dtype=bool)
    merch = is_merchant_like(pop, payee)
    session = np.exp(r.normal(np.log(38), 0.65, n)).round(1)
    # retries after a wrong PIN / timeout: geometric, P(2)~13%, P(3)~2%, P(4+)~0.3% (D-010)
    attempts = np.minimum(r.geometric(0.85, n), 6).astype(np.int8)
    is_collect = np.asarray(txn_type) == "collect"
    c_age = np.where(merch, np.exp(r.normal(np.log(75), 0.9, n)), np.exp(r.normal(np.log(1500), 1.4, n)))
    c_age = np.where(is_collect, np.maximum(c_age, COLLECT_MIN_AGE_S + r.uniform(0, 2, n)), np.nan).round(1)
    phonebook = np.where(merch, 0, (r.random(n) < np.where(new_payee, 0.25, 0.6))).astype(np.int8)
    screen_obs = acc["screen_obs"].to_numpy()[payer] == 1
    consent = acc["consent_call"].to_numpy()[payer] == 1
    screen = np.where(screen_obs, (r.random(n) < BENIGN_RATE["screen_share_active"]).astype(float), np.nan)
    remote = np.where(screen_obs, (r.random(n) < BENIGN_RATE["remote_access_app_detected"]).astype(float),
                      np.nan)
    call = np.where(consent, (r.random(n) < BENIGN_RATE["call_in_progress"]).astype(float), np.nan)
    sim_name = np.where(r.random(n) < 0.04, r.uniform(0.8, 0.99, n), r.beta(1.6, 7, n)).round(3)
    dev_key, bound = device_at(pop, payer, ts)
    return {"session_duration_s": session, "attempts_in_session": attempts, "collect_request_age_s": c_age,
            "screen_share_active": screen, "remote_access_app_detected": remote, "call_in_progress": call,
            "payee_name_similarity_to_known_contact": sim_name, "_device_key": dev_key, "_bound_ts": bound,
            "_phonebook": phonebook}
