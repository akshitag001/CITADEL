"""ONE shared mule registry across every scam variant.

Two kinds of mule account:
* recruited       an existing customer who rents out their account: real tenure, real benign history;
* purpose_opened  a new account SEASONED with small, unremarkable credits over the prior fortnight.

Mules belong to rings that share handsets and cash-out accounts, are reused across episodes until a
corroborated complaint gets them frozen, and fan money out to 2-6 downstream accounts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..timeutil import DAY_S, HOUR_S
from .entities import Population, hex_ids

#: Accounts opened for the purpose (mules, cash-outs) bank with THIS institution only this often. When
#: they do not, their outbound payments are invisible to us, exactly as for any other bank's customer.
OUR_CUSTOMER_SHARE = 0.5


@dataclass
class Mule:
    idx: int
    kind: str
    ring: int
    activate_ts: int
    retire_ts: int
    max_uses: int
    uses: int = 0
    frozen_ts: int | None = None


class MuleRegistry:
    def __init__(self, pop: Population, r: np.random.Generator, seed: int, n_mules: int):
        self.pop, self.r, self.seed = pop, r, seed
        self.mules: list[Mule] = []
        self.by_idx: dict[int, Mule] = {}
        self.rings: dict[int, dict] = {}
        self.pending: list[dict] = []
        self.seasoning: list[tuple[int, int, int, float]] = []  # (payer_idx, payee_idx, ts, amount)
        self._next_idx = int(pop.accounts["idx"].max()) + 1
        n_rings = max(2, n_mules // 3)
        for ring in range(n_rings):
            self._new_ring(ring)
        span = (pop.t0 - 5 * DAY_S, pop.t_end - DAY_S)
        while len(self.mules) < n_mules:
            ring = int(r.integers(n_rings))
            act = int(r.uniform(*span))
            if r.random() < 0.5:
                self._recruit(self._recruitable(act, min_tenure_days=60), ring, act)
            else:
                self._purpose_opened(ring, act)

    # --- account creation --------------------------------------------------------------------------
    def _new_ring(self, ring: int) -> None:
        devs = [self.pop.new_device(self.pop.t0 - int(self.r.uniform(10, 90)) * DAY_S)
                for _ in range(int(self.r.integers(1, 3)))]
        self.rings[ring] = {"members": [], "cashout": [], "devices": devs}

    def _new_account(self, created_ts: int, device_key: int) -> int:
        r = self.r
        idx = self._next_idx
        self._next_idx += 1
        age = "18-25" if r.random() < 0.6 else "26-40"
        bal = float(round(np.exp(r.normal(np.log(3000), 0.8)), 0))
        self.pending.append({
            "idx": idx, "kind": "person", "created_ts": int(created_ts), "user_age_band": age,
            "digital_literacy": "high" if r.random() < 0.5 else "med", "home_tier": int(r.choice([1, 2, 3])),
            "account_balance_inr": bal, "balance_band": "low" if bal < 10000 else "mid",
            "amount_mu": float(np.log(300)), "amount_sigma": 0.8, "txn_rate": 0.0, "pref_hour": 13,
            "explore_p": 0.0, "consent_call": int(r.random() < 0.35), "screen_obs": int(r.random() < 0.6),
            "is_seller": 0, "informal_merchant": 0, "verified": 0, "category": "", "ticket_median": 0.0,
            "popularity": 0.0, "new_cohort": 1, "device_key": int(device_key),
            "device_bound_ts": int(created_ts), "prehist_n": 0,
            "our_customer": int(r.random() < OUR_CUSTOMER_SHARE), "prehist_mean_log_amt": np.nan,
            "prehist_std_log_amt": np.nan, "acct_id": hex_ids(self.seed, "acct", np.array([idx]))[0],
        })
        return idx

    def _purpose_opened(self, ring: int, activate_ts: int) -> Mule:
        r = self.r
        created = activate_ts - int(r.uniform(10, 150) * DAY_S)
        # half of purpose-opened accounts are run from a shared ring handset, half from their own
        dev = int(r.choice(self.rings[ring]["devices"])) if r.random() < 0.5 else self.pop.new_device(created)
        idx = self._new_account(created, dev)
        # seasoning: small unremarkable credits over the month before first use
        alive = self._alive_persons(activate_ts - 30 * DAY_S)
        for _ in range(int(r.integers(4, 16))):
            t = int(activate_ts - r.uniform(0.2, 30) * DAY_S)
            if t > created:
                self.seasoning.append((int(r.choice(alive)), idx, t, float(round(r.uniform(100, 2000), -1))))
        m = Mule(idx, "purpose_opened", ring, activate_ts, self.pop.t_end + 30 * DAY_S, int(r.integers(2, 10)))
        self._register(m)
        return m

    def _recruit(self, idx: int, ring: int, activate_ts: int) -> Mule:
        m = Mule(int(idx), "recruited", ring, activate_ts, self.pop.t_end + 30 * DAY_S, int(self.r.integers(2, 10)))
        self._register(m)
        return m

    def _register(self, m: Mule) -> None:
        self.mules.append(m)
        self.by_idx[m.idx] = m
        self.rings[m.ring]["members"].append(m.idx)

    def _alive_persons(self, ts: int) -> np.ndarray:
        acc = self.pop.accounts
        m = (acc.kind == "person") & (acc.created_ts < ts) & (acc.txn_rate > 0)
        return acc.loc[m, "idx"].to_numpy()

    def _recruitable(self, ts: int, min_tenure_days: float) -> int:
        acc = self.pop.accounts
        m = ((acc.kind == "person") & (acc.created_ts < ts - min_tenure_days * DAY_S) & (acc.txn_rate > 0)
             & (acc.informal_merchant == 0) & ~acc.idx.isin(list(self.by_idx)))
        cand = acc[m]
        w = np.where(cand.user_age_band == "18-25", 3.0, 1.0) * np.where(cand.balance_band == "high", 0.3, 1.0)
        return int(self.r.choice(cand["idx"].to_numpy(), p=w / w.sum()))

    # --- use ---------------------------------------------------------------------------------------
    def tenure_days(self, idx: int, ts: int) -> float:
        acc = self.pop.accounts
        if idx < len(acc):
            created = int(acc["created_ts"].iloc[idx])
        else:
            created = next(p["created_ts"] for p in self.pending if p["idx"] == idx)
        return (ts - created) / DAY_S

    def pick(self, ts: int, aged: bool = False, exclude: set[int] | None = None,
             purpose_opened_share: float = 0.5) -> Mule:
        exclude = exclude or set()
        cand = [m for m in self.mules
                if m.activate_ts <= ts < m.retire_ts and m.uses < m.max_uses and m.idx not in exclude
                and (not aged or self.tenure_days(m.idx, ts) >= 365)]
        if not cand:
            ring = int(self.r.integers(len(self.rings)))
            if aged or self.r.random() > purpose_opened_share:
                m = self._recruit(self._recruitable(ts, 365 if aged else 60), ring, ts - HOUR_S)
            else:
                m = self._purpose_opened(ring, ts - HOUR_S)
        else:
            w = np.array([(1.0 + 0.3 * m.uses) * (purpose_opened_share if m.kind == "purpose_opened"
                                            else 1 - purpose_opened_share) for m in cand])
            m = cand[int(self.r.choice(len(cand), p=w / w.sum()))]
        m.uses += 1
        return m

    def recruit_contact(self, idx: int, ts: int) -> Mule:
        """A payee the victim already knows, whose account has been rented to the scammer."""
        if idx in self.by_idx:
            m = self.by_idx[idx]
            m.uses += 1
            return m
        m = self._recruit(idx, int(self.r.integers(len(self.rings))), ts - HOUR_S)
        m.uses += 1
        return m

    def retire(self, idx: int, ts: int) -> None:
        m = self.by_idx.get(idx)
        if m is not None:
            m.retire_ts = min(m.retire_ts, int(ts))

    def freeze(self, idx: int, ts: int) -> None:
        """The bank freezes a confirmed mule account: no further inbound or outbound payments."""
        m = self.by_idx.get(idx)
        if m is not None:
            m.frozen_ts = int(ts) if m.frozen_ts is None else min(m.frozen_ts, int(ts))
            m.retire_ts = min(m.retire_ts, int(ts))

    def frozen(self) -> dict[int, int]:
        return {m.idx: m.frozen_ts for m in self.mules if m.frozen_ts is not None}

    def downstream(self, mule_idx: int, n: int, ts: int) -> list[int]:
        """Layering accounts the mule pays out to. Rings share a cash-out pool that grows as it is used."""
        ring = self.rings[self.by_idx[mule_idx].ring]
        while len(ring["cashout"]) < n + 2:
            dev = int(self.r.choice(ring["devices"])) if self.r.random() < 0.5 else self.pop.new_device(ts)
            ring["cashout"].append(self._new_account(ts - int(self.r.uniform(10, 120)) * DAY_S, dev))
        if self.r.random() < 0.3:  # occasionally a fresh layering account
            ring["cashout"].append(self._new_account(ts - int(self.r.uniform(10, 120)) * DAY_S,
                                                     self.pop.new_device(ts)))
        pool = ring["cashout"]
        return [int(i) for i in self.r.choice(pool, size=min(n, len(pool)), replace=False)]

    def flush(self) -> None:
        if self.pending:
            add = pd.DataFrame(self.pending)
            self.pop.accounts = pd.concat([self.pop.accounts, add[self.pop.accounts.columns]], ignore_index=True)
            self.pending = []

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{"idx": m.idx, "kind": m.kind, "ring": m.ring, "activate_ts": m.activate_ts,
                              "retire_ts": m.retire_ts, "frozen_ts": m.frozen_ts, "uses": m.uses}
                             for m in self.mules])
