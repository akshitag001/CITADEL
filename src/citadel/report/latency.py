"""Measure serving latency: server compute (reported by the API) vs HTTP round trip (measured here),
over a real uvicorn server on loopback. Writes artifacts/<run>/serve_latency.csv. No latency is ever
claimed that this did not measure."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import numpy as np
import pandas as pd
import uvicorn

from ..runctx import RunCtx
from ..serve.api import create_app
from ..serve.state import load_state


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def measure(ctx: RunCtx, n: int = 200) -> pd.DataFrame:
    import tempfile
    from pathlib import Path
    state = load_state(ctx.run)
    app = create_app(ctx.run, state=state, store_path=Path(tempfile.mkdtemp()) / "lat.sqlite",
                     rate_per_minute={"customer": 10 ** 6, "analyst": 10 ** 6})
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    H = {"Authorization": "Bearer demo-customer-token"}
    r = np.random.default_rng(0)
    test = state.decisions
    ids = test["txn_id"].to_numpy()[r.choice(len(test), n)]
    rows = state.tx.set_index("txn_id").loc[ids]
    out = []
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as c:
        for path, bodies in (
            ("stored_txn", [{"txn_id": t} for t in ids]),
            ("online_event", [{"event": {"user_id": u, "payee_id": p, "txn_type": str(tt), "amount_inr": float(a)}}
                              for u, p, tt, a in zip(rows.user_id, rows.payee_id, rows.txn_type, rows.amount_inr)]),
        ):
            c.post("/v1/score", json=bodies[0], headers=H)  # warm-up
            server_ms, rt_ms = [], []
            for b in bodies:
                t0 = time.perf_counter()
                resp = c.post("/v1/score", json=b, headers=H)
                rt_ms.append((time.perf_counter() - t0) * 1000)
                server_ms.append(resp.json().get("server_ms", np.nan))
            for kind, arr in (("server_compute", server_ms), ("round_trip", rt_ms)):
                a = np.asarray(arr, dtype=float)
                out.append({"path": f"{path}.{kind}", "n": len(a), "p50_ms": np.percentile(a, 50),
                            "p95_ms": np.percentile(a, 95), "p99_ms": np.percentile(a, 99),
                            "note": "loopback, single process, laptop CPU; online path recomputes features from history"})
    server.should_exit = True
    th.join(timeout=5)
    df = pd.DataFrame(out)
    ctx.csv(df, "serve_latency.csv")
    return df
