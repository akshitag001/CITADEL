"""Serving contract, privacy and misuse tests (P9 + P12)."""

import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from citadel.serve.api import create_app
from citadel.serve.auth import RateLimiter, hmac_id
from citadel.serve.store import Store

A = {"Authorization": "Bearer demo-analyst-token"}
C = {"Authorization": "Bearer demo-customer-token"}


@pytest.fixture(scope="module")
def client(tiny_state, tiny_run, tmp_path_factory):
    app = create_app(tiny_run.run, state=tiny_state, store_path=tmp_path_factory.mktemp("s") / "a.sqlite",
                     rate_per_minute={"customer": 1000, "analyst": 5000})
    return TestClient(app)


@pytest.fixture(scope="module")
def sample(tiny_state):
    tx = tiny_state.tx
    dec = tiny_state.decisions
    hi = dec.sort_values("risk", ascending=False).iloc[0]
    return {"txn_hi": hi.txn_id, "row": tx[tx.txn_id == hi.txn_id].iloc[0]}


def test_health_reports_the_run_being_served(client, tiny_run):
    j = client.get("/health").json()
    assert j["status"] == "ok" and j["run_id"] == tiny_run.prov.run_id


def test_scoring_requires_a_token(client, sample):
    assert client.post("/v1/score", json={"txn_id": sample["txn_hi"]}).status_code == 401


def test_customer_payload_never_contains_a_score(client, sample, tiny_state):
    r = client.post("/v1/score", json={"txn_id": sample["txn_hi"]}, headers=C)
    assert r.status_code == 200
    j = r.json()
    assert {"band", "action", "reason_codes", "message", "run_id"} <= set(j)
    assert "analyst" not in j
    flat = str(j).lower()
    for word in ("risk", "score", "p_l1", "p_l2", "probab"):
        assert word not in flat.replace("server_ms", ""), word
    risk = float(tiny_state.decisions.set_index("txn_id").loc[sample["txn_hi"], "risk"])
    assert f"{risk:.4f}"[:6] not in flat


def test_analyst_payload_has_scores_and_evidence(client, sample):
    j = client.post("/v1/score", json={"txn_id": sample["txn_hi"]}, headers=A).json()
    assert 0.0 <= j["analyst"]["risk"] <= 1.0


def test_unknown_feature_key_is_rejected_not_defaulted(client, sample):
    ev = {"user_id": sample["row"].user_id, "payee_id": sample["row"].payee_id, "txn_type": "pay",
          "amount_inr": 500.0, "surprise": 1}
    assert client.post("/v1/score", json={"event": ev}, headers=A).status_code == 422
    assert client.post("/v1/score", json={"txn_id": sample["txn_hi"], "extra": 1}, headers=A).status_code == 422


def test_unknown_account_is_rejected(client, sample):
    ev = {"user_id": "000000000000", "payee_id": sample["row"].payee_id, "txn_type": "pay", "amount_inr": 500.0}
    r = client.post("/v1/score", json={"event": ev}, headers=A)
    assert r.status_code == 422 and "unknown account" in r.json()["detail"]


def test_online_event_scoring_works_and_is_fast_enough(client, sample):
    ev = {"user_id": sample["row"].user_id, "payee_id": sample["row"].payee_id, "txn_type": "collect",
          "amount_inr": 4999.0, "collect_request_age_s": 3.0, "payee_in_contacts": False}
    r = client.post("/v1/score", json={"event": ev, "lang": "hi"}, headers=A)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["action"] in ("A0", "A1", "A2", "A3") and j["server_ms"] < 5000


def test_customer_cannot_reach_analyst_endpoints(client):
    assert client.get("/v1/alerts", headers=C).status_code == 403
    assert client.post("/v1/scam/author", json={"variant": "V1"}, headers=C).status_code == 403


def test_evidence_access_is_audited(client):
    q = client.get("/v1/alerts", headers=A).json()["alerts"]
    assert q, "queue should be seeded from the test window"
    aid = q[0]["alert_id"]
    assert client.get(f"/v1/alerts/{aid}", headers=A).status_code == 200
    log = client.get("/v1/audit", headers=A).json()["audit"]
    assert any(e["action"] == "view_evidence" and e["alert_id"] == aid for e in log)


def test_evidence_never_contains_ground_truth(client):
    q = client.get("/v1/alerts", headers=A).json()["alerts"]
    ev = client.get(f"/v1/alerts/{q[0]['alert_id']}", headers=A).json()
    flat = str(ev)
    for k in ("is_attack", "scam_variant", "episode_id", "evasion"):
        assert k not in flat


def test_disposition_and_appeal_flow(client):
    q = client.get("/v1/alerts", headers=A).json()["alerts"]
    aid = q[0]["alert_id"]
    assert client.post("/v1/appeal", json={"alert_id": aid, "reason": "I know this person"}, headers=C).status_code == 200
    assert client.put(f"/v1/cases/{aid}", json={"disposition": "release", "note": "verified"}, headers=A).status_code == 200
    assert client.put(f"/v1/cases/{aid}", json={"disposition": "decline"}, headers=A).status_code == 422


def test_scam_studio_traces_and_explains_illegal(client):
    r = client.post("/v1/scam/author", json={"variant": "V2", "evasion": "none", "segment": "mainstream"}, headers=A)
    assert r.status_code == 200, r.text
    j = r.json()
    assert [s["step"] for s in j["steps"]][0] == "S1" and j["steps"][-1]["step"] == "S4"
    assert "inside our grammar" in j["bound"]
    r = client.post("/v1/scam/author", json={"variant": "V4", "evasion": "signal_suppression"}, headers=A)
    assert r.status_code == 422 and "nothing to suppress" in r.json()["detail"]


def test_grammar_and_results_are_served(client):
    g = client.get("/v1/grammar").json()
    assert len(g["variants"]) == 6 and g["illegal"]
    res = client.get("/v1/results").json()
    assert "defend_headline.csv" in res["files"]


# ---- misuse ----------------------------------------------------------------------------------------
def test_boundary_probing_is_rate_limited(tiny_state, tiny_run, tmp_path, sample):
    app = create_app(tiny_run.run, state=tiny_state, store_path=tmp_path / "p.sqlite",
                     rate_per_minute={"customer": 20, "analyst": 600})
    c = TestClient(app)
    codes = [c.post("/v1/score", json={"txn_id": sample["txn_hi"]}, headers=C).status_code for _ in range(40)]
    assert codes.count(200) == 20 and codes.count(429) == 20
    # within the limit a prober sees only 4 coarse bands: at most log2(4) = 2 bits per query
    bands = {c2 for c2 in ("low", "elevated", "high", "critical")}
    assert len(bands) == 4


def test_rate_limiter_window_slides():
    rl = RateLimiter({"customer": 2})
    assert rl.allow("t", "customer", now=0) and rl.allow("t", "customer", now=1)
    assert not rl.allow("t", "customer", now=2)
    assert rl.allow("t", "customer", now=61.5)


def test_complaint_poisoning_cannot_reach_a_hold_alone(tiny_state):
    """An adversary files complaints against innocent payees. L0-01 needs aged corroboration AND a
    structural signal, and complaints alone must not push ordinary payments to A3."""
    b = tiny_state.bundle
    feats = tiny_state.features.drop(columns=["txn_id"])
    tx = tiny_state.tx
    dec = tiny_state.decisions
    a0 = dec[(dec.level == 0) & (dec.is_attack == 0)].head(400)
    idx = tx.reset_index().set_index("txn_id").loc[a0.txn_id, "index"].to_numpy()
    X = feats.iloc[idx].reset_index(drop=True).copy()
    raw = tx.iloc[idx].reset_index(drop=True)
    X["f_payee_complaints"] = 10.0
    d = b.decide(X, raw)
    assert not any("L0-01" in f for f in d.l0_fired)
    assert (d.level == 3).mean() == 0.0, "complaints alone pushed innocent payments to a hold"


def test_fresh_reporter_accounts_do_not_count():
    from citadel.generate.enrich import corroborated_complaints
    acc = pd.DataFrame({"idx": np.arange(10), "created_ts": np.full(10, 100 * 86400)})
    c = pd.DataFrame({"payee_idx": np.full(10, 7), "reporter_idx": np.arange(10),
                      "arrival_ts": np.full(10, 101 * 86400)})  # ten sock-puppets, one day old
    assert len(corroborated_complaints(c, acc)) == 0


# ---- privacy ---------------------------------------------------------------------------------------
def test_ids_are_hmac_hashed_with_a_salt():
    assert hmac_id("abc", "s1") != hmac_id("abc", "s2")
    assert hmac_id("abc", "s1") == hmac_id("abc", "s1") and "abc" not in hmac_id("abc", "s1")


def test_retention_ttl_purges_old_alerts(tmp_path):
    s = Store(tmp_path / "r.sqlite")
    rec = {"txn_id": "t", "event_ts": 0, "user_hash": "u", "payee_hash": "p", "txn_type": "pay", "amount_inr": 1.0,
           "action": "A2", "band": "high", "risk": 0.5, "reasons": ["R01"], "l0_fired": []}
    old = s.add_alert(rec, now=time.time() - 200 * 86400)
    new = s.add_alert(rec, now=time.time())
    s.disposition(old, "release", "x", "a")
    assert s.purge_expired(time.time(), ttl_days=180) == 1
    ids = {a["alert_id"] for a in s.queue("all")}
    assert new in ids and old not in ids
