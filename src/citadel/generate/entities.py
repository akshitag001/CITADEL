"""Customers, merchants, devices and SIM events.

One account namespace for everybody: customers, merchants, informal sellers and (later) mule accounts
all get opaque 12-hex ids from the same function, so no id prefix can reveal a role.
A recently-onboarded cohort (~15% of customers, ~20% of merchants) exists so that "young payee" or
"new customer" is never a free fraud signal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, rng
from ..timeutil import DAY_S, start_epoch

AGE_BANDS = np.array(["18-25", "26-40", "41-60", "60+"])
AGE_P = np.array([0.24, 0.40, 0.26, 0.10])
LITERACY = np.array(["low", "med", "high"])
LITERACY_BY_AGE = np.array([[0.10, 0.40, 0.50], [0.15, 0.45, 0.40], [0.30, 0.45, 0.25], [0.55, 0.35, 0.10]])
MERCHANT_CATS = {  # category: (median ticket INR, popularity weight)
    "grocery": (180, 3.0), "food": (260, 2.5), "fuel": (700, 1.2), "pharmacy": (320, 1.0),
    "electronics": (2800, 0.4), "utilities": (1100, 0.8), "online": (650, 1.6), "education": (2400, 0.3),
    "transport": (120, 1.4), "recharge": (299, 1.5),
}


def hex_ids(seed: int, space: str, idx: np.ndarray) -> np.ndarray:
    return np.array([hashlib.sha1(f"{seed}:{space}:{int(i)}".encode()).hexdigest()[:12] for i in idx],
                    dtype=object)


def balance_band(balance: np.ndarray) -> np.ndarray:
    return np.where(balance < 10000, "low", np.where(balance < 75000, "mid", "high"))


@dataclass
class Population:
    accounts: pd.DataFrame        # persons + merchants, indexed by integer idx
    devices: pd.DataFrame         # device_id -> created_ts
    device_changes: pd.DataFrame  # idx, ts, device_id   (benign phone changes)
    sim_changes: pd.DataFrame     # idx, ts
    t0: int
    t_end: int
    n_persons: int
    n_merchants: int
    extra_devices: list = field(default_factory=list)   # (device_key, created_ts) added by attacks
    next_device_key: int = 0
    extra_sim_changes: list = field(default_factory=list)  # (idx, ts)
    pair_history: pd.DataFrame | None = None  # (payer_idx, payee_idx, n_prior) before the window opens

    def new_device(self, ts: int) -> int:
        key = self.next_device_key
        self.next_device_key += 1
        self.extra_devices.append((key, int(ts)))
        return key

    def all_devices(self) -> pd.DataFrame:
        extra = pd.DataFrame(self.extra_devices, columns=["device_key", "created_ts"])
        return pd.concat([self.devices, extra], ignore_index=True)

    def all_sim_changes(self) -> pd.DataFrame:
        extra = pd.DataFrame(self.extra_sim_changes, columns=["idx", "ts"])
        return pd.concat([self.sim_changes, extra], ignore_index=True)


def make_population(cfg: Config) -> Population:
    r = rng(cfg, "entities")
    pc = cfg.population
    t0 = start_epoch(cfg.time.start_date)
    t_end = t0 + cfg.traffic.days * DAY_S
    n_p, n_m = pc.users, pc.merchants

    # ---- persons --------------------------------------------------------------------------------
    age_i = r.choice(4, size=n_p, p=AGE_P)
    lit_i = np.array([r.choice(3, p=LITERACY_BY_AGE[a]) for a in age_i])
    tier = r.choice([1, 2, 3], size=n_p, p=[0.35, 0.35, 0.30])
    new_cohort = r.random(n_p) < pc.new_user_share
    created = np.where(
        new_cohort,
        t0 + r.uniform(-120, cfg.traffic.days * 0.6, n_p) * DAY_S,
        t0 - r.uniform(120, 3000, n_p) * DAY_S,
    ).astype(np.int64)
    tier_bal = np.array([0, 1.25, 1.0, 0.8])[tier]
    age_bal = np.array([0.45, 1.0, 1.5, 1.6])[age_i]
    balance = np.exp(r.normal(np.log(22000), 1.0, n_p)) * tier_bal * age_bal
    balance = np.clip(balance, 300, 2_500_000).round(0)
    amount_mu = r.normal(np.log(330), 0.55, n_p) + np.log(np.array([0.7, 1.05, 1.15, 0.95])[age_i])
    amount_sigma = r.uniform(0.65, 1.1, n_p)
    rate = np.exp(r.normal(0, 0.7, n_p))
    rate = rate / rate.mean() * cfg.traffic.txn_per_user_day
    rate = rate * np.array([1.15, 1.15, 0.85, 0.5])[age_i]
    pref_hour = r.choice(24, size=n_p, p=_pref_hour_p())
    explore = np.clip(r.beta(2, 14, n_p) + np.where(new_cohort, 0.08, 0.0), 0.01, 0.5)
    # profile state the PSP already holds: up to 90 days of pre-window history, estimated with noise
    pre_days = np.clip((t0 - created) / DAY_S, 0, 90)
    pre_n = r.poisson(rate_guess := cfg.traffic.txn_per_user_day * pre_days)
    del rate_guess
    pre_mean = amount_mu + 0.25 + r.normal(0, 1, n_p) * amount_sigma / np.sqrt(np.maximum(pre_n, 1))
    persons = pd.DataFrame({
        "idx": np.arange(n_p),
        "kind": "person",
        "created_ts": created,
        "user_age_band": AGE_BANDS[age_i],
        "digital_literacy": LITERACY[lit_i],
        "home_tier": tier.astype(np.int8),
        "account_balance_inr": balance,
        "balance_band": balance_band(balance),
        "amount_mu": amount_mu,
        "amount_sigma": amount_sigma,
        "txn_rate": rate,
        "pref_hour": pref_hour,
        "explore_p": explore,
        "consent_call": (r.random(n_p) < cfg.consent.call_in_progress_available_share).astype(np.int8),
        "screen_obs": (r.random(n_p) < pc.screen_signal_observable_share).astype(np.int8),
        "is_seller": (r.random(n_p) < np.where(age_i <= 1, 0.12, 0.06)).astype(np.int8),
        "informal_merchant": (r.random(n_p) < pc.informal_merchant_share).astype(np.int8),
        "verified": np.int8(0),
        "category": "",
        "ticket_median": 0.0,
        "popularity": 0.0,
        "new_cohort": new_cohort.astype(np.int8),
        "our_customer": np.int8(1),
        "prehist_n": pre_n.astype(np.int32),
        "prehist_mean_log_amt": np.where(pre_n > 0, pre_mean, np.nan),
        "prehist_std_log_amt": np.where(pre_n > 1, amount_sigma * r.uniform(0.85, 1.15, n_p), np.nan),
    })
    # informal sellers (tutors, shopkeepers on personal accounts) are payees like merchants, unverified
    inf = persons["informal_merchant"] == 1
    persons.loc[inf, "category"] = r.choice(["tutor", "kirana", "tiffin", "tailor"], size=int(inf.sum()))
    persons.loc[inf, "ticket_median"] = r.choice([150.0, 400.0, 1500.0, 2500.0], size=int(inf.sum()))
    persons.loc[inf, "popularity"] = r.uniform(0.6, 1.6, int(inf.sum()))

    # ---- merchants ------------------------------------------------------------------------------
    cats = np.array(list(MERCHANT_CATS))
    cat_w = np.array([MERCHANT_CATS[c][1] for c in cats])
    m_cat = r.choice(cats, size=n_m, p=cat_w / cat_w.sum())
    m_new = r.random(n_m) < pc.new_merchant_share
    m_created = np.where(m_new, t0 + r.uniform(-90, cfg.traffic.days * 0.7, n_m) * DAY_S,
                         t0 - r.uniform(365, 3000, n_m) * DAY_S).astype(np.int64)
    merchants = pd.DataFrame({
        "idx": np.arange(n_p, n_p + n_m),
        "kind": "merchant",
        "created_ts": m_created,
        "user_age_band": "26-40", "digital_literacy": "high",
        "home_tier": r.choice([1, 2, 3], size=n_m, p=[0.4, 0.35, 0.25]).astype(np.int8),
        "account_balance_inr": np.exp(r.normal(np.log(150000), 1.0, n_m)).round(0),
        "amount_mu": 0.0, "amount_sigma": 0.0, "txn_rate": 0.0, "pref_hour": 12, "explore_p": 0.0,
        "consent_call": np.int8(0), "screen_obs": np.int8(0), "is_seller": np.int8(0),
        "informal_merchant": np.int8(0),
        "verified": (r.random(n_m) < np.where(m_new, 0.45, 0.88)).astype(np.int8),
        "category": m_cat,
        "ticket_median": np.array([MERCHANT_CATS[c][0] for c in m_cat]) * np.exp(r.normal(0, 0.3, n_m)),
        "popularity": np.exp(r.normal(0, 1.0, n_m)),
        "new_cohort": m_new.astype(np.int8),
        "our_customer": np.int8(1),
        "prehist_n": np.int32(0), "prehist_mean_log_amt": np.nan, "prehist_std_log_amt": np.nan,
    })
    merchants["balance_band"] = balance_band(merchants["account_balance_inr"].to_numpy())
    accounts = pd.concat([persons, merchants], ignore_index=True)
    accounts["acct_id"] = hex_ids(cfg.seed, "acct", accounts["idx"].to_numpy())

    # ---- devices ---------------------------------------------------------------------------------
    dev_created = np.minimum(created + r.uniform(0, 1, n_p) * np.maximum(t0 - created, 0), t0 - 3600)
    dev_created = np.where(created > t0, created, dev_created).astype(np.int64)
    dev_idx = np.arange(n_p)
    # family phones: a share of customers use the device of another customer in the same tier
    fam = np.flatnonzero(r.random(n_p) < pc.family_device_share)
    for i in fam:
        same = np.flatnonzero(tier == tier[i])
        j = int(r.choice(same))
        if j != i:
            dev_idx[i] = dev_idx[j]
    accounts["device_key"] = np.concatenate([dev_idx, np.full(n_m, -1)])
    accounts["device_bound_ts"] = np.concatenate([np.maximum(created, dev_created[dev_idx]),
                                                  np.zeros(n_m, dtype=np.int64)])
    devices = pd.DataFrame({"device_key": np.arange(n_p), "created_ts": dev_created})

    # benign phone changes (~8% of customers) and SIM changes (~3%)
    chg = np.flatnonzero(r.random(n_p) < 0.08)
    chg_ts = (t0 + r.uniform(0, cfg.traffic.days, len(chg)) * DAY_S).astype(np.int64)
    chg_ts = np.maximum(chg_ts, created[chg] + DAY_S)
    keep = chg_ts < t_end
    chg, chg_ts = chg[keep], chg_ts[keep]
    new_keys = np.arange(n_p, n_p + len(chg))
    device_changes = pd.DataFrame({"idx": chg, "ts": chg_ts, "device_key": new_keys})
    devices = pd.concat([devices, pd.DataFrame({"device_key": new_keys, "created_ts": chg_ts})],
                        ignore_index=True)
    sim = np.flatnonzero(r.random(n_p) < 0.03)
    sim_ts = (t0 + r.uniform(-7, cfg.traffic.days, len(sim)) * DAY_S).astype(np.int64)
    sim_changes = pd.DataFrame({"idx": sim, "ts": sim_ts})
    return Population(accounts=accounts, devices=devices, device_changes=device_changes,
                      sim_changes=sim_changes, t0=t0, t_end=t_end, n_persons=n_p, n_merchants=n_m,
                      next_device_key=int(devices["device_key"].max()) + 1)


def _pref_hour_p() -> np.ndarray:
    from ..timeutil import DIURNAL
    p = DIURNAL ** 1.3
    return p / p.sum()
