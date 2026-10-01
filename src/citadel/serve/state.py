"""Serving state: the run's bundle plus the switch-side history needed to score a NEW event online.

Online features are computed by the SAME code as training (generate.enrich.enrich + features.matrix
.build_matrix) on a context slice: every past row of the payer and the payee. Graph features come
from the end-of-history graph state; new events are therefore placed after the end of history, so
nothing the model sees is from the future.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, load_config, rng
from ..defend.bundle import Bundle, load_bundle
from ..features.graph import GraphState
from ..features.matrix import build_matrix, profiles_from_accounts
from ..generate.benign import session_fields
from ..generate.enrich import enrich
from ..generate.entities import Population, hex_ids
from ..timeutil import DAY_S, HOUR_S

EVENT_FIELDS = {"user_id", "payee_id", "txn_type", "amount_inr", "device_id", "ts", "session_duration_s",
                "attempts_in_session", "collect_request_age_s", "screen_share_active", "remote_access_app_detected",
                "call_in_progress", "payee_name_similarity_to_known_contact", "payee_in_contacts"}


class UnknownAccount(KeyError):
    pass


@dataclass
class ServingState:
    run: str
    bundle: Bundle
    accounts: pd.DataFrame
    devices: pd.DataFrame
    sim_changes: pd.DataFrame
    complaints: pd.DataFrame
    pair_history: pd.DataFrame
    internal: pd.DataFrame
    tx: pd.DataFrame
    features: pd.DataFrame | None
    decisions: pd.DataFrame | None
    episodes: pd.DataFrame
    labels: pd.DataFrame
    meta: dict
    graph: GraphState
    seed: int
    acct_idx: dict = field(default_factory=dict)
    dev_key: dict = field(default_factory=dict)
    by_party: dict = field(default_factory=dict)
    now_ts: int = 0
    _next_rid: int = 0

    # ------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, run: str) -> "ServingState":
        data, art = ROOT / "data" / run, ROOT / "artifacts" / run
        t = time.time()
        bundle = load_bundle(art / "bundle.pkl")
        rd = lambda n: pd.read_parquet(data / n)  # noqa: E731
        internal = rd("internal.parquet")
        tx = rd("transactions.parquet")
        labels = rd("labels.parquet")
        meta = json.loads((data / "generation_summary.json").read_text())
        g = GraphState()
        g.add_rows(tx["user_id"].astype(str).to_numpy(), tx["payee_id"].astype(str).to_numpy(),
                   tx["device_id"].astype(str).to_numpy())
        now = int(tx["ts"].max()) + 60
        if len(labels):
            km = labels.groupby("payee_id")["arrival_ts"].min()
            g.set_known(set(km[km < now].index.astype(str)))
        feats = pd.read_parquet(art / "features.parquet") if (art / "features.parquet").exists() else None
        dec = pd.read_parquet(art / "decisions_test.parquet") if (art / "decisions_test.parquet").exists() else None
        st = cls(run=run, bundle=bundle, accounts=rd("accounts.parquet"), devices=rd("devices.parquet"),
                 sim_changes=rd("sim_changes.parquet"), complaints=rd("complaints.parquet"),
                 pair_history=rd("pair_history.parquet"), internal=internal, tx=tx, features=feats, decisions=dec,
                 episodes=rd("episodes.parquet"), labels=labels, meta=meta, graph=g,
                 seed=load_config(ROOT / "configs" / "default.yaml").seed)
        cfg_path = art / "config_resolved.json"
        if cfg_path.exists():
            st.seed = int(json.loads(cfg_path.read_text()).get("seed", st.seed))
        st.acct_idx = dict(zip(st.accounts["acct_id"].astype(str), st.accounts["idx"].astype(int)))
        st.dev_key = dict(zip(tx["device_id"].astype(str), internal["_device_key"].astype(int)))
        st.by_party = {}
        for col in ("_payer", "_payee"):
            for k, idx in internal.groupby(col).indices.items():
                st.by_party.setdefault(int(k), []).append(idx)
        st.by_party = {k: np.unique(np.concatenate(v)) for k, v in st.by_party.items()}
        st.now_ts = now
        st._next_rid = int(internal["_rid"].max()) + 1
        st.load_seconds = time.time() - t
        return st

    # ------------------------------------------------------------------------------------------------
    def population(self) -> Population:
        return Population(accounts=self.accounts, devices=self.devices,
                          device_changes=pd.DataFrame(columns=["idx", "ts", "device_key"]),
                          sim_changes=self.sim_changes, t0=int(self.meta["t0"]), t_end=int(self.meta["t_end"]),
                          n_persons=int(self.meta["n_persons"]), n_merchants=int(self.meta["n_merchants"]),
                          next_device_key=int(self.devices["device_key"].max()) + 1, pair_history=self.pair_history)

    def idx_of(self, acct_id: str) -> int:
        try:
            return self.acct_idx[str(acct_id)]
        except KeyError as e:
            raise UnknownAccount(f"unknown account '{acct_id}'") from e

    def context(self, parties: set[int], before_ts: int) -> pd.DataFrame:
        """Every past row in which any of ``parties`` paid or was paid, strictly before ``before_ts``."""
        arrs = [self.by_party[p] for p in parties if p in self.by_party]
        if not arrs:
            return self.internal.iloc[0:0]
        idx = np.unique(np.concatenate(arrs))
        idx = idx[self.internal["ts"].to_numpy()[idx] < before_ts]
        return self.internal.iloc[idx]

    def new_rids(self, n: int) -> np.ndarray:
        out = np.arange(self._next_rid, self._next_rid + n)
        self._next_rid += n
        return out

    def event_to_internal(self, ev: dict) -> pd.DataFrame:
        """A raw API event -> one internal row (unknown accounts are rejected, never defaulted)."""
        payer, payee = self.idx_of(ev["user_id"]), self.idx_of(ev["payee_id"])
        ts = int(ev.get("ts") or self.now_ts)
        pop = self.population()
        r = rng(self.seed, f"serve/{payer}/{ts}")
        f = session_fields(r, pop, np.array([payer]), np.array([payee]), np.array([ev["txn_type"]], dtype=object),
                           np.array([ts]))
        row = {k: (v[0] if isinstance(v, np.ndarray) else v) for k, v in f.items()}
        dev = ev.get("device_id")
        if dev:
            if str(dev) in self.dev_key:
                row["_device_key"] = self.dev_key[str(dev)]
            else:  # a device we have never seen: a fresh binding, created now
                key = int(self.devices["device_key"].max()) + 1
                self.devices = pd.concat([self.devices, pd.DataFrame({"device_key": [key], "created_ts": [ts]})],
                                         ignore_index=True)
                self.dev_key[str(dev)] = key
                row["_device_key"], row["_bound_ts"] = key, ts
        for k in ("session_duration_s", "attempts_in_session", "collect_request_age_s", "screen_share_active",
                  "remote_access_app_detected", "call_in_progress", "payee_name_similarity_to_known_contact"):
            if ev.get(k) is not None:
                row[k] = float(ev[k])
        acc = self.accounts.set_index("idx")
        for k, consent_col in (("call_in_progress", "consent_call"), ("screen_share_active", "screen_obs"),
                               ("remote_access_app_detected", "screen_obs")):
            if ev.get(k) is None and int(acc.at[payer, consent_col]) == 0:
                row[k] = np.nan  # no consent / no SDK: missing, never zero
        if ev.get("payee_in_contacts") is not None:
            row["_phonebook"] = int(bool(ev["payee_in_contacts"]))
        row.update({"ts": ts, "_payer": payer, "_payee": payee, "txn_type": ev["txn_type"],
                    "amount_inr": float(ev["amount_inr"]), "row_kind": "api", "hn_archetype": ""})
        if row["txn_type"] != "collect":
            row["collect_request_age_s"] = np.nan
        return pd.DataFrame([row])

    def featurise(self, new_rows: pd.DataFrame, extra_context: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Enrich + featurise ``new_rows`` (internal) against history; returns (schema rows, features)."""
        new_rows = new_rows.copy()
        new_rows["_rid"] = self.new_rids(len(new_rows))
        new_rows["txn_id"] = hex_ids(self.seed, "apitxn", new_rows["_rid"].to_numpy())
        parties = set(new_rows["_payer"].astype(int)) | set(new_rows["_payee"].astype(int))
        ctx = self.context(parties, int(new_rows["ts"].max()) + 1)
        frames = [ctx] + ([extra_context] if extra_context is not None and len(extra_context) else []) + [new_rows]
        allrows = pd.concat(frames, ignore_index=True)
        devs = self.devices
        schema = enrich(allrows, self.accounts, devs, self.sim_changes, self.complaints, self.seed, self.pair_history)
        X = build_matrix(schema, profiles_from_accounts(self.accounts), graph_state=self.graph)
        sel = schema["txn_id"].isin(set(new_rows["txn_id"])).to_numpy()
        return schema[sel].reset_index(drop=True), X[sel].reset_index(drop=True)

    def stored_row(self, txn_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """An existing payment with its exact offline features (no online recomputation)."""
        if self.features is None:
            raise KeyError("features not available for this run")
        pos = np.flatnonzero(self.tx["txn_id"].to_numpy() == txn_id)
        if not len(pos):
            raise KeyError(txn_id)
        i = int(pos[0])
        X = self.features.iloc[[i]].drop(columns=["txn_id"]).reset_index(drop=True)
        return self.tx.iloc[[i]].reset_index(drop=True), X


def day_after(ts: int, hour: int) -> int:
    """Same IST hour on the next day after ``ts`` (keeps the victim's calendar habits)."""
    from ..timeutil import IST_OFFSET_S
    day = (ts + IST_OFFSET_S) // DAY_S + 1
    return int(day * DAY_S - IST_OFFSET_S + hour * HOUR_S + 600)


def load_state(run: str) -> ServingState:
    p = Path(ROOT / "artifacts" / run / "bundle.pkl")
    if not p.exists():
        raise FileNotFoundError(f"no bundle for run '{run}' — run `python -m citadel run --run {run}` first")
    return ServingState.load(run)
