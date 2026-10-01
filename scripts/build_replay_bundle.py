"""Record the replay bundle the console uses offline: frontend/replay/replay.js.

Every response is RECORDED from the real API (in-process TestClient against the run's bundle), never
hand-written.

    python scripts/build_replay_bundle.py --run demo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from citadel.generate.library import load_library  # noqa: E402
from citadel.serve.api import create_app  # noqa: E402
from citadel.serve.auth import hmac_id  # noqa: E402
from citadel.serve.state import load_state  # noqa: E402

A = {"Authorization": "Bearer demo-analyst-token"}
C = {"Authorization": "Bearer demo-customer-token"}
SAFETY_TESTS = [
    "tests/test_api.py::test_customer_payload_never_contains_a_score",
    "tests/test_api.py::test_unknown_feature_key_is_rejected_not_defaulted",
    "tests/test_api.py::test_boundary_probing_is_rate_limited",
    "tests/test_api.py::test_complaint_poisoning_cannot_reach_a_hold_alone",
    "tests/test_api.py::test_fresh_reporter_accounts_do_not_count",
    "tests/test_api.py::test_customer_cannot_reach_analyst_endpoints",
    "tests/test_api.py::test_evidence_access_is_audited",
    "tests/test_api.py::test_evidence_never_contains_ground_truth",
    "tests/test_api.py::test_retention_ttl_purges_old_alerts",
    "tests/test_api.py::test_ids_are_hmac_hashed_with_a_salt",
    "tests/test_features_l0.py::test_l0_zero_false_positives_on_legitimate_populations",
    "tests/test_features_l0.py::test_l0_runs_with_the_model_stack_unavailable",
    "tests/test_features_l0.py::test_no_truth_column_reaches_the_matrix",
    "tests/test_features_l0.py::test_unobservable_signals_stay_missing_not_zero",
    "tests/test_defend.py::test_held_payment_always_has_a_release_and_appeal_path",
    "tests/test_defend.py::test_messages_never_expose_a_score_or_accuse",
    "tests/test_defend.py::test_collapsed_ladder_is_refused",
    "tests/test_defend.py::test_sealed_holdout_is_entity_level",
    "tests/test_causality.py::test_feature_matrix_is_prefix_invariant",
    "tests/test_generate.py::test_no_fraud_only_id_namespace",
]
POLICY = {
    "collect": [
        {"signal": "payment (amount, type, time, payer/payee ids as salted hashes)", "why": "the decision itself", "consent": "core service"},
        {"signal": "payee-side aggregates (fan-in, pass-through, account age)", "why": "mule detection at S4 / S3", "consent": "core service"},
        {"signal": "device binding age, SIM-change flag", "why": "account-takeover step (S2)", "consent": "core service"},
        {"signal": "screen-share / remote-control app running (yes/no)", "why": "remote-access scams (V2)", "consent": "app SDK; missing when absent"},
        {"signal": "call in progress (yes/no)", "why": "phone-guided scams", "consent": "explicit opt-in; missing without consent"},
    ],
    "never": ["message, SMS or call content", "contact lists or names", "location", "the lure itself (S1)",
              "raw account numbers (stored only as salted HMACs)", "any score shown to the customer"],
    "false_positive": [
        "No automatic decline, ever: the strongest action is a hold with a human-release path.",
        "A warning (A2) costs a genuine customer a 60-second cooling-off; the Continue button unlocks after it.",
        "A hold (A3) says the money has not left, gives a review SLA, a release request and an appeal route.",
        "Analyst dispositions (release / confirm / needs-info) feed the feedback table for retraining.",
        "Complaints alone never trigger a hold; they need aged, independent reporters plus a structural signal.",
        "Alert volume is capped by budget (A2 <= 1% of payments, A3 <= 0.1%) and checked against analyst capacity.",
    ],
}


def signals_of(row: pd.Series) -> dict:
    out = {}
    for c in ("screen_share_active", "remote_access_app_detected", "call_in_progress", "new_device_flag",
              "sim_change_7d", "user_to_payee_prior_txn_count", "payee_age_days", "session_duration_s"):
        v = row[c]
        out[c] = None if pd.isna(v) else (round(float(v), 1) if isinstance(v, float) else int(v))
    return out


def miss_explanation(row: pd.Series, feats: pd.Series) -> str:
    """Generated from the payment's own features: which strong signals were absent."""
    absent = []
    if pd.isna(row["screen_share_active"]):
        absent.append("no screen-share/remote-access signal was observable (no SDK on this phone)")
    elif row["screen_share_active"] == 0 and row["remote_access_app_detected"] == 0:
        absent.append("no screen share or remote-control app was running")
    if row["user_to_payee_prior_txn_count"] > 0:
        absent.append(f"the customer had paid this account {int(row['user_to_payee_prior_txn_count'])} time(s) before")
    if np.isfinite(feats["f_amt_ratio_user"]) and feats["f_amt_ratio_user"] < 2.5:
        absent.append(f"the amount was within the customer's usual range ({feats['f_amt_ratio_user']:.1f}x typical)")
    if row["payee_age_days"] > 365:
        absent.append(f"the receiving account was {int(row['payee_age_days'])} days old")
    if pd.isna(row["call_in_progress"]):
        absent.append("call status was not consented")
    ev = row["evasion_technique"]
    tail = f" The scam used the '{ev}' evasion." if ev else ""
    return ("Citadel stayed quiet because " + "; ".join(absent[:4]) + "." if absent else
            "Citadel's risk stayed below the alert budget threshold.") + tail


def record(run: str) -> dict:
    t0 = time.time()
    state = load_state(run)
    app = create_app(run, state=state, store_path=Path(tempfile.mkdtemp()) / "replay.sqlite",
                     rate_per_minute={"customer": 100000, "analyst": 100000})
    c = TestClient(app)
    out: dict = {"meta": {"run_id": state.bundle.provenance["run_id"], "recorded_from": "in-process API",
                          "note": "recording of real API responses; do not edit by hand"}}
    out["health"] = c.get("/health").json()
    out["model_card"] = c.get("/v1/model-card").json()
    out["grammar"] = c.get("/v1/grammar").json()
    out["results"] = c.get("/v1/results").json()
    q = c.get("/v1/alerts?limit=60", headers=A).json()["alerts"]
    out["alerts"] = q
    out["evidence"] = {a["alert_id"]: c.get(f"/v1/alerts/{a['alert_id']}", headers=A).json() for a in q}
    out["appeals"] = []

    lib = load_library()
    studio = {}
    for v, spec in lib.variants.items():
        seg0 = "mainstream" if "mainstream" in spec["allowed_segments"] else spec["allowed_segments"][0]
        combos = [(v, e, seg0) for e in ["none", *lib.evasions] if not lib.explain_illegal(v, e, seg0)]
        combos += [(v, "none", s) for s in spec["allowed_segments"] if s != seg0]
        for v_, e, s in combos:
            r = c.post("/v1/scam/author", json={"variant": v_, "evasion": e, "segment": s, "seed": 0}, headers=A)
            if r.status_code == 200:
                studio[f"{v_}|{e}|{s}"] = r.json()
    out["studio"] = studio

    # ---- customer scenarios -------------------------------------------------------------------------
    scen = []
    for v, spec in lib.variants.items():
        for seed in (0, 1):
            seg = spec["allowed_segments"][seed % len(spec["allowed_segments"])]
            r = c.post("/v1/scam/author", json={"variant": v, "evasion": "none", "segment": seg, "seed": seed + 10},
                       headers=A)
            if r.status_code != 200:
                continue
            t = r.json()
            pays = [s for s in t["steps"] if s["step"] in ("S2", "S3") and s.get("observed")]
            if not pays:
                continue
            fired = next((s for s in pays if s["action"] in ("A2", "A3")), None)
            shown = fired or next((s for s in pays if s["step"] == "S3"), pays[0])
            s4 = t["steps"][-1]
            scen.append({
                "id": f"{v}-{seed}", "kind": "scam" if t["caught"] else "scam_missed", "variant": v,
                "withheld": t["composition"]["withheld_variant"],
                "title": f"{v} · {t['composition']['title']} ({seg})", "story": t["composition"]["story"],
                "payment": {"txn_type": shown["txn_type"], "amount_inr": shown["amount_inr"],
                            "payee_label": f"Account ••••{shown['payee_hash'][-4:]}", "time_ist": shown["time_ist"],
                            "first_time_payee": shown["first_time_payee"]},
                "decision": {"action": shown["action"], "band": shown["band"], "reason_codes": shown["reasons"],
                             "message": shown["message"]},
                "signals": shown["signals"],
                "after": {"en": f"Within minutes the money is split across {s4.get('mule_outflows', 0)} mule accounts. "
                                "UPI payments usually cannot be reversed.",
                          "hi": f"कुछ ही मिनटों में पैसा {s4.get('mule_outflows', 0)} म्यूल खातों में बँट जाता है। "
                                "UPI भुगतान आमतौर पर वापस नहीं होता।"},
                "explanation": None if t["caught"] else "This composed scam stayed under the alert budget; see the Studio trace.",
                "source": "POST /v1/scam/author (recorded)",
            })
    dec = state.decisions
    tx = state.tx.set_index("txn_id")
    feats = state.features.set_index("txn_id")
    hn = dec[(dec.row_kind == "hard_negative") & (dec.level <= 1) & dec.hn_archetype.str.contains(r"\+", regex=True)]
    for _, d in hn.sort_values("amount_inr", ascending=False).head(3).iterrows():
        row = tx.loc[d.txn_id]
        en = c.post("/v1/score", json={"txn_id": d.txn_id, "lang": "en"}, headers=C).json()
        hi = c.post("/v1/score", json={"txn_id": d.txn_id, "lang": "hi"}, headers=C).json()
        scen.append({
            "id": f"hn-{d.txn_id}", "kind": "legit", "variant": None, "withheld": False,
            "title": "Legitimate but alarming: " + d.hn_archetype.replace("_", " ").replace("+", " + "),
            "story": "A genuine customer's payment engineered to look like a scam. Citadel should stay calm here.",
            "payment": {"txn_type": str(row.txn_type), "amount_inr": float(row.amount_inr),
                        "payee_label": f"Account ••••{hmac_id(row.payee_id)[-4:]}", "time_ist": "",
                        "first_time_payee": bool(row.user_to_payee_prior_txn_count == 0)},
            "decision": {"action": en["action"], "band": en["band"], "reason_codes": en["reason_codes"],
                         "message": {"en": en["message"], "hi": hi["message"]}},
            "signals": signals_of(row), "after": None, "explanation": None,
            "source": "POST /v1/score (recorded, customer token)"})
    miss = dec[(dec.is_attack == 1) & (dec.level == 0)]
    for _, d in miss.drop_duplicates("scam_variant").head(2).iterrows():
        row = tx.loc[d.txn_id]
        en = c.post("/v1/score", json={"txn_id": d.txn_id, "lang": "en"}, headers=C).json()
        scen.append({
            "id": f"miss-{d.txn_id}", "kind": "scam_missed", "variant": d.scam_variant,
            "withheld": False, "title": f"Honest miss: {d.scam_variant} scam payment Citadel let through",
            "story": "A real scam payment from the test window that scored below every alert threshold.",
            "payment": {"txn_type": str(row.txn_type), "amount_inr": float(row.amount_inr),
                        "payee_label": f"Account ••••{hmac_id(row.payee_id)[-4:]}", "time_ist": "",
                        "first_time_payee": bool(row.user_to_payee_prior_txn_count == 0)},
            "decision": {"action": en["action"], "band": en["band"], "reason_codes": en["reason_codes"],
                         "message": {"en": en["message"], "hi": en["message"]}},
            "signals": signals_of(row),
            "after": {"en": "The money moved. This is the cost of keeping warnings under 1% of payments.",
                      "hi": "पैसा चला गया। चेतावनियों को 1% से कम रखने की यह क़ीमत है।"},
            "explanation": miss_explanation(row, feats.loc[d.txn_id]),
            "source": "POST /v1/score (recorded) + decisions_test.parquet"})
    order = {"scam": 0, "legit": 1, "scam_missed": 2}
    out["customer_scenarios"] = sorted(scen, key=lambda s: order.get(s["kind"], 3))
    out["policy"] = POLICY
    out["tests"] = run_safety_tests()
    out["meta"]["recording_seconds"] = round(time.time() - t0, 1)
    return out


def run_safety_tests() -> list[dict]:
    xml = Path(tempfile.mkdtemp()) / "junit.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", *SAFETY_TESTS],
                   cwd=ROOT, capture_output=True, text=True)
    import xml.etree.ElementTree as ET
    out = []
    if xml.exists():
        for tc in ET.parse(xml).getroot().iter("testcase"):
            failed = tc.find("failure") is not None or tc.find("error") is not None
            out.append({"id": f"{tc.get('classname', '').split('.')[-1]}::{tc.get('name')}",
                        "outcome": "failed" if failed else ("skipped" if tc.find("skipped") is not None else "passed")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="demo")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    global run_safety_tests
    if a.skip_tests:
        run_safety_tests = lambda: []  # noqa: E731
    data = record(a.run)
    dest = ROOT / "frontend" / "replay"
    dest.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=lambda o: None)
    (dest / "replay.js").write_text("window.CITADEL_REPLAY = " + text + ";\n", encoding="utf-8")
    print(f"replay bundle: {len(text) / 1e6:.2f} MB, {len(data['studio'])} studio traces, "
          f"{len(data['customer_scenarios'])} customer scenarios, {len(data['alerts'])} alerts, "
          f"{len(data['tests'])} recorded tests -> frontend/replay/replay.js")


if __name__ == "__main__":
    main()
