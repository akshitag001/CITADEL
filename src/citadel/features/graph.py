"""Nightly account-graph snapshots.

Rows on IST day d only see the graph built from days < d, and only mules CONFIRMED by labels that
arrived before day d starts. Hubs (accounts above ``hub_degree`` counterparties: merchants, charities,
tutors) are excluded from shortest paths, otherwise everyone is two hops from everyone via a grocer.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from ..timeutil import DAY_S, IST_OFFSET_S, ist_day_number

FAR = 4  # "no confirmed mule within 3 hops"


class GraphState:
    def __init__(self, hub_degree: int = 40):
        self.hub_degree = hub_degree
        self.adj: dict[str, set] = defaultdict(set)
        self.device_accounts: dict[str, set] = defaultdict(set)
        self.account_devices: dict[str, set] = defaultdict(set)
        self.known: set[str] = set()

    def add_rows(self, users, payees, devices) -> None:
        for u, p, d in zip(users, payees, devices):
            self.adj[u].add(p)
            self.adj[p].add(u)
            self.device_accounts[d].add(u)
            self.account_devices[u].add(d)

    def set_known(self, known: set[str]) -> None:
        self.known = set(known)

    def distances(self, cutoff: int = 3) -> dict[str, int]:
        dist: dict[str, int] = {}
        q = deque()
        for k in self.known:
            dist[k] = 0
            q.append(k)
        while q:
            n = q.popleft()
            d = dist[n]
            if d >= cutoff:
                continue
            for m in self.adj.get(n, ()):
                if m in dist or (len(self.adj.get(m, ())) > self.hub_degree and m not in self.known):
                    continue  # hubs are neither reached nor traversed
                dist[m] = d + 1
                q.append(m)
        return dist

    def features(self, users, payees, devices, dist: dict[str, int] | None = None) -> np.ndarray:
        if dist is None:
            dist = self.distances()
        out = np.zeros((len(users), 5), dtype=float)
        known = self.known
        for i, (u, p, d) in enumerate(zip(users, payees, devices)):
            nb = self.adj.get(p, ())
            out[i, 0] = len(nb)
            out[i, 1] = dist.get(p, FAR)
            out[i, 2] = sum(1 for n in nb if n in known) if known and len(nb) <= 500 else 0
            devs = self.account_devices.get(p, ())
            out[i, 3] = max((len(self.device_accounts[x]) for x in devs), default=0)
            accts = self.device_accounts.get(d, ())
            out[i, 4] = len(accts) - (1 if u in accts else 0)
        return out


GRAPH_COLS = ["f_g_payee_degree", "f_g_payee_dist_known_mule", "f_g_payee_mule_neighbors",
              "f_g_payee_device_accounts", "f_g_user_device_accounts"]


def snapshot_features(tx: pd.DataFrame, known_mules: pd.DataFrame | None,
                      hub_degree: int = 40) -> tuple[pd.DataFrame, GraphState]:
    """Per-row graph features from nightly snapshots; also returns the end-of-stream state."""
    days = ist_day_number(tx["ts"].to_numpy())
    users = tx["user_id"].astype(str).to_numpy()
    payees = tx["payee_id"].astype(str).to_numpy()
    devices = tx["device_id"].astype(str).to_numpy()
    order = np.argsort(days, kind="stable")
    out = np.zeros((len(tx), 5))
    state = GraphState(hub_degree)
    km = known_mules if known_mules is not None else pd.DataFrame({"payee_id": [], "arrival_ts": []})
    km_ts = km["arrival_ts"].to_numpy() if len(km) else np.array([], dtype=np.int64)
    km_ids = km["payee_id"].astype(str).to_numpy() if len(km) else np.array([], dtype=object)
    uniq, starts = np.unique(days[order], return_index=True)
    ends = list(starts[1:]) + [len(order)]
    for day, s, e in zip(uniq, starts, ends):
        idx = order[s:e]
        day_start = int(day) * DAY_S - IST_OFFSET_S
        state.set_known(set(km_ids[km_ts < day_start]))
        out[idx] = state.features(users[idx], payees[idx], devices[idx])
        state.add_rows(users[idx], payees[idx], devices[idx])
    if len(km):
        state.set_known(set(km_ids))
    return pd.DataFrame(out, columns=GRAPH_COLS, index=tx.index), state
