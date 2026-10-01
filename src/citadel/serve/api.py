"""Citadel serving API (FastAPI).

Rules: unknown request keys -> 422 (never silently defaulted); customer clients get a band + reasons and
NEVER a score; scoring is rate-limited per client; scoring never trains; every analyst view of an
evidence record is written to the audit log; every response carries the run id that produced it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ..config import ROOT
from ..defend.bundle import model_card
from .auth import RateLimiter, hmac_id, role_of
from .state import ServingState, UnknownAccount, load_state
from .store import Store
from .studio import IllegalComposition, compose, grammar

RESULT_FILES = ["defend_headline.csv", "defend_ladder.csv", "defend_alert_budget.csv", "eval_generalisation.csv",
                "eval_lovo.csv", "eval_layer_ablation.csv", "eval_baselines.csv", "eval_hard_negative_fpr.csv",
                "eval_time_to_alert_summary.csv", "eval_episode_trace.csv", "eval_per_variant.csv",
                "eval_per_evasion.csv", "eval_per_segment.csv", "eval_fairness.csv", "defend_reason_code_usage.csv",
                "defend_cost_summary.csv", "defend_fusion_arms.csv", "defend_channels_alone.csv", "eval_controls.csv",
                "eval_evidence_index.csv", "defend_split.csv", "defend_capacity.csv", "eval_sample_size_curve.csv",
                "eval_operating_curve.csv"]


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str
    payee_id: str
    txn_type: Literal["pay", "collect", "qr_pay"]
    amount_inr: float = Field(gt=0, le=100000)
    device_id: str | None = None
    ts: int | None = None
    session_duration_s: float | None = Field(default=None, ge=0)
    attempts_in_session: int | None = Field(default=None, ge=1, le=20)
    collect_request_age_s: float | None = None
    screen_share_active: Literal[0, 1] | None = None
    remote_access_app_detected: Literal[0, 1] | None = None
    call_in_progress: Literal[0, 1] | None = None
    payee_name_similarity_to_known_contact: float | None = Field(default=None, ge=0, le=1)
    payee_in_contacts: bool | None = None


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    txn_id: str | None = None
    event: Event | None = None
    lang: Literal["en", "hi"] = "en"


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ScoreRequest] = Field(max_length=500)


class AuthorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant: str
    evasion: str | None = "none"
    segment: str = "mainstream"
    seed: int = Field(default=0, ge=0, le=10_000)


class CaseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disposition: Literal["confirm_scam", "release", "needs_info"]
    note: str = Field(default="", max_length=500)


class AppealRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alert_id: str
    reason: str = Field(default="", max_length=500)


def create_app(run: str = "demo", state: ServingState | None = None, store_path: Path | None = None,
               rate_per_minute: dict[str, int] | None = None) -> FastAPI:
    app = FastAPI(title="Citadel UPI Scam-Sequence Shield", version="0.1.0",
                  description="PSP/bank-side scam-sequence scoring with a graded response. Never declines.")
    app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1", "http://localhost", "null"],
                       allow_methods=["*"], allow_headers=["*"])
    S: dict = {"state": state, "loaded_at": time.time(), "latency": []}
    limiter = RateLimiter(rate_per_minute or {"customer": 60, "analyst": 600})
    store = Store(store_path or (ROOT / "artifacts" / run / "alerts.sqlite"))

    def st() -> ServingState:
        if S["state"] is None:
            S["state"] = load_state(run)
        return S["state"]

    def auth(request: Request, authorization: str | None = Header(default=None)) -> dict:
        role, tok = role_of(authorization)
        if role is None:
            raise HTTPException(401, "missing or invalid bearer token")
        if not limiter.allow(tok, role):
            raise HTTPException(429, "rate limit exceeded")
        return {"role": role, "actor": f"{role}:{hmac_id(tok)[:6]}"}

    def analyst(who: dict = Depends(auth)) -> dict:
        if who["role"] != "analyst":
            raise HTTPException(403, "analyst role required")
        return who

    def seed_queue() -> None:
        """Load the test window's A2+ decisions into the analyst queue once per store."""
        s = st()
        if S.get("seeded") or s.decisions is None:
            return
        S["seeded"] = True
        if store.count_seeded():
            return
        d = s.decisions
        d = d[d["level"] >= 2].sort_values(["level", "risk"], ascending=[False, False]).head(300)
        tx = s.tx.set_index("txn_id")
        for _, r in d.iterrows():
            store.add_alert({**_alert_record(s, r, tx), "alert_id": f"t-{r['txn_id']}"})

    # ---- health / cards ---------------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        s = st()
        return {"status": "ok", "run_id": s.bundle.provenance.get("run_id"), "run": run,
                "model_version": s.bundle.provenance.get("run_id"), "rows_in_history": int(len(s.tx)),
                "loaded_seconds": round(getattr(s, "load_seconds", 0.0), 2)}

    @app.get("/v1/model-card")
    def card() -> dict:
        return model_card(st().bundle)

    @app.get("/v1/grammar")
    def get_grammar() -> dict:
        return grammar(sealed=_sealed(run))

    @app.get("/v1/results")
    def results() -> dict:
        art = ROOT / "artifacts" / run
        out = {}
        for f in RESULT_FILES:
            p = art / f
            if p.exists():
                out[f] = pd.read_csv(p).replace({np.nan: None}).to_dict(orient="records")
        rep = art / "REPORT.md"
        return {"run_id": st().bundle.provenance.get("run_id"), "files": out,
                "report_md": rep.read_text(encoding="utf-8") if rep.exists() else None}

    # ---- scoring ------------------------------------------------------------------------------------
    def score_one(req: ScoreRequest, who: dict) -> dict:
        s = st()
        t0 = time.perf_counter()
        try:
            if req.txn_id:
                raw, X = s.stored_row(req.txn_id)
            elif req.event is not None:
                raw, X = s.featurise(s.event_to_internal(req.event.model_dump()))
            else:
                raise HTTPException(422, "provide txn_id or event")
        except UnknownAccount as e:
            raise HTTPException(422, e.args[0]) from e
        except KeyError as e:
            raise HTTPException(404, f"unknown txn_id {e}") from e
        d = s.bundle.decide(X, raw).iloc[0]
        server_ms = (time.perf_counter() - t0) * 1000
        S["latency"].append(server_ms)
        completeness = float(np.isfinite(X.to_numpy(dtype=float)).mean())
        out = {"run_id": s.bundle.provenance.get("run_id"), "model_version": s.bundle.provenance.get("run_id"),
               "band": d["band"], "action": d["action"], "treatment": d["treatment"],
               "reason_codes": list(d["reasons"]),
               "message": s.bundle.message(d["action"], list(d["reasons"]), str(raw["txn_type"].iat[0]), req.lang,
                                           float(raw["amount_inr"].iat[0])),
               "feature_completeness": round(completeness, 3), "server_ms": round(server_ms, 2)}
        if d["level"] >= 2 and req.event is not None:
            out["alert_id"] = store.add_alert(_alert_record(s, pd.Series({**raw.iloc[0].to_dict(), **d.to_dict()}),
                                                            s.tx.set_index("txn_id")))
        if who["role"] == "analyst":
            out["analyst"] = {"risk": float(d["risk"]), "p_L1": float(d["p_L1"]), "p_L2": float(d["p_L2"]),
                              "l0_fired": list(d["l0_fired"]), "abstained": bool(d["abstained"]),
                              "score_level": int(d["score_level"])}
        return out

    @app.post("/v1/score")
    def score(req: ScoreRequest, who: dict = Depends(auth)) -> dict:
        return score_one(req, who)

    @app.post("/v1/score/batch")
    def score_batch(req: BatchRequest, who: dict = Depends(auth)) -> dict:
        return {"results": [score_one(i, who) for i in req.items]}

    @app.get("/v1/latency")
    def latency(who: dict = Depends(analyst)) -> dict:
        lat = np.array(S["latency"]) if S["latency"] else np.array([np.nan])
        return {"n": int(len(S["latency"])), "server_compute_ms_p50": float(np.nanpercentile(lat, 50)),
                "server_compute_ms_p95": float(np.nanpercentile(lat, 95)),
                "server_compute_ms_p99": float(np.nanpercentile(lat, 99)),
                "note": "server compute only; round-trip is measured client-side (scripts/measure_latency.py)"}

    # ---- studio -------------------------------------------------------------------------------------
    @app.post("/v1/scam/author")
    def author(req: AuthorRequest, who: dict = Depends(analyst)) -> dict:
        try:
            return compose(st(), req.variant, req.evasion, req.segment, req.seed)
        except IllegalComposition as e:
            raise HTTPException(422, str(e)) from e

    # ---- analyst queue ------------------------------------------------------------------------------
    @app.get("/v1/alerts")
    def alerts(status: str = "open", limit: int = 50, offset: int = 0, who: dict = Depends(analyst)) -> dict:
        seed_queue()
        return {"run_id": st().bundle.provenance.get("run_id"), "alerts": store.queue(status, min(limit, 200), offset)}

    @app.get("/v1/alerts/{alert_id}")
    def alert(alert_id: str, who: dict = Depends(analyst)) -> dict:
        rec = store.evidence(alert_id, who["actor"])
        if rec is None:
            raise HTTPException(404, "unknown alert")
        return rec

    @app.put("/v1/cases/{alert_id}")
    def case(alert_id: str, body: CaseUpdate, who: dict = Depends(analyst)) -> dict:
        if not store.disposition(alert_id, body.disposition, body.note, who["actor"]):
            raise HTTPException(404, "unknown alert")
        return {"alert_id": alert_id, "disposition": body.disposition}

    @app.post("/v1/appeal")
    def appeal(body: AppealRequest, who: dict = Depends(auth)) -> dict:
        rec = store.appeal(body.alert_id, body.reason, who["actor"])
        if rec is None:
            raise HTTPException(404, "unknown alert")
        return rec

    @app.get("/v1/appeals")
    def appeals(who: dict = Depends(analyst)) -> dict:
        return {"appeals": store.appeals()}

    @app.get("/v1/audit")
    def audit(who: dict = Depends(analyst)) -> dict:
        return {"audit": store.audit_log()}

    @app.exception_handler(UnknownAccount)
    def _unknown(_: Request, exc: UnknownAccount):
        return JSONResponse(status_code=422, content={"detail": exc.args[0]})

    front = ROOT / "frontend"
    if front.exists():
        app.mount("/", StaticFiles(directory=str(front), html=True), name="console")
    app.state.store = store
    return app


def _sealed(run: str) -> dict:
    import json
    p = ROOT / "artifacts" / run / "config_resolved.json"
    return json.loads(p.read_text())["sealed_holdout"] if p.exists() else {}


def _alert_record(s: ServingState, r: pd.Series, tx: pd.DataFrame) -> dict:
    """Analyst evidence: the payer's recent timeline + layer scores + reasons. Never ground truth."""
    user = r["user_id"]
    ts = int(r["ts"])
    hist = s.tx[(s.tx["user_id"] == user) & (s.tx["ts"] < ts) & (s.tx["ts"] >= ts - 7 * 86400)].tail(8)
    timeline = [{"ts": int(h.ts), "txn_type": str(h.txn_type), "amount_inr": float(h.amount_inr),
                 "payee_hash": hmac_id(h.payee_id), "first_time_payee": bool(h.user_to_payee_prior_txn_count == 0)}
                for h in hist.itertuples()]
    codes = s.bundle.codes
    reasons = r["reasons"] if isinstance(r["reasons"], list) else [c for c in str(r["reasons"]).split(",") if c]
    l0f = r["l0_fired"] if isinstance(r["l0_fired"], list) else [c for c in str(r["l0_fired"]).split(",") if c]
    payee = {}
    if r.get("txn_id") in tx.index:
        t = tx.loc[r["txn_id"]]
        payee = {"payee_age_days": float(t["payee_age_days"]), "inbound_unique_payers_24h": int(t["payee_inbound_unique_payers_24h"]),
                 "complaints": int(t["payee_complaint_count"]), "verified_merchant": bool(t["payee_is_verified_merchant"])}
    return {"txn_id": r["txn_id"], "event_ts": ts, "user_hash": hmac_id(user), "payee_hash": hmac_id(r["payee_id"]),
            "txn_type": str(r["txn_type"]), "amount_inr": float(r["amount_inr"]), "action": r["action"], "band": r["band"],
            "risk": float(r["risk"]), "reasons": reasons, "l0_fired": l0f,
            "evidence": {"timeline": timeline, "layers": {"p_L1": float(r["p_L1"]), "p_L2": float(r["p_L2"]),
                                                          "risk": float(r["risk"])},
                         "reasons_detail": [{"code": c, "analyst": codes[c]["analyst"]} for c in reasons if c in codes],
                         "payee": payee}}
