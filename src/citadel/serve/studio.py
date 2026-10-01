"""Scam Studio: compose a scam from the grammar (variant x evasion x segment), run it through the real
generator and the real decision path, and return the step-by-step trace.

The composed episode is placed AFTER the end of history (the victim's next day, at the hour of the real
benign payment it mutates), so every feature it sees is causal. It is new to the model, inside our
grammar — the page says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import rng
from ..generate.attacks import build_episode, choose_template, segment_of, victim_weights
from ..generate.campaign import _mule_rows
from ..generate.library import ScamLibrary, load_library
from ..generate.mules import MuleRegistry
from ..timeutil import DAY_S, fmt, hour_ist
from .auth import hmac_id
from .state import ServingState, day_after

SIGNAL_COLS = ["screen_share_active", "remote_access_app_detected", "call_in_progress", "new_device_flag",
               "sim_change_7d", "collect_request_age_s", "request_source_known_contact", "session_duration_s",
               "attempts_in_session", "payee_name_similarity_to_known_contact"]


class IllegalComposition(ValueError):
    pass


def grammar(lib: ScamLibrary | None = None, sealed: dict | None = None) -> dict:
    lib = lib or load_library()
    space = lib.composition_space()
    return {
        "variants": {k: {"title": v["title"], "story": v["story"], "allowed_segments": v["allowed_segments"],
                         "steps": {s: v["steps"][s].get("name") for s in ("S1", "S2", "S3", "S4")},
                         "step_columns": lib.step_columns(k), "lure_channel": v["steps"]["S1"].get("channel")}
                     for k, v in lib.variants.items()},
        "evasions": {k: {"title": e["title"], "levers": e["levers"]} for k, e in lib.evasions.items()},
        "segments": {k: s.get("title", k) for k, s in lib.segments.items()},
        "illegal": space["illegal"], "size_legal": space["size_legal"], "size_total": space["size_total"],
        "sealed_holdout": sealed or {},
    }


def _victims(state: ServingState, lib: ScamLibrary, variant: str, segment: str) -> pd.DataFrame:
    acc = state.accounts
    p = acc[(acc.kind == "person") & (acc.txn_rate > 0) & (acc.informal_merchant == 0)].copy()
    seg = lib.segments[segment]
    if "age_band" in seg:
        p = p[p.user_age_band.isin(seg["age_band"])]
    if "digital_literacy" in seg:
        p = p[p.digital_literacy.isin(seg["digital_literacy"])]
    if "max_tenure_days" in seg:
        p = p[(state.now_ts - p.created_ts) / DAY_S <= seg["max_tenure_days"]]
    if not hasattr(state, "_own_counts"):  # the victim's OWN legitimate payments (templates to mutate)
        own = state.internal[state.internal["row_kind"].isin(["benign", "hard_negative", "shape"])]
        state._own_counts = own.groupby("_payer").size()
    p = p[p.idx.map(state._own_counts).fillna(0) >= 5]
    return p


def compose(state: ServingState, variant: str, evasion: str | None, segment: str, seed: int = 0) -> dict:
    lib = load_library()
    ev = None if evasion in (None, "", "none") else evasion
    why = lib.explain_illegal(variant, ev, segment)
    if why:
        raise IllegalComposition(why)
    r = rng(seed, f"studio/{variant}/{ev}/{segment}")
    cand = _victims(state, lib, variant, segment)
    if cand.empty:
        raise IllegalComposition(f"no customer in segment '{segment}' with enough history in this run")
    w = victim_weights(lib, variant, cand)
    victim = int(cand["idx"].to_numpy()[int(r.choice(len(cand), p=w / w.sum()))])
    hist = state.internal.iloc[state.by_party[victim]]
    own = hist[(hist["_payer"] == victim) & (hist["row_kind"].isin(["benign", "hard_negative", "shape"]))]
    t_i = choose_template(r, own, 1.5, ev == "time_of_day_mimicry")
    template = own.iloc[t_i].to_dict()
    template["ts"] = day_after(state.now_ts, int(hour_ist(template["ts"])))
    template["_rid"] = -1

    pop = state.population()
    pop.accounts = state.accounts.copy()
    registry = MuleRegistry(pop, rng(seed, "studio/mules"), state.seed, 0)
    vrow = state.accounts.set_index("idx").loc[victim].to_dict()
    res = build_episode(lib=lib, registry=registry, r=r, episode_id=f"studio-{variant}-{seed}", variant=variant,
                        evasion=ev, template=template, victim=vrow, victim_history=hist, label_visible_share=0.0,
                        scam_devices=[])
    registry.flush()
    # the composed world: new mule accounts, devices and SIM events join the serving state for this trace
    accounts = pop.accounts
    devices = pop.all_devices()
    sims = pop.all_sim_changes()
    vic = pd.DataFrame(res.rows)
    mule = _mule_rows(rng(seed, "studio/mule_rows"), pop, list(res.mule_rows))
    seas = _mule_rows(rng(seed, "studio/seasoning"), pop, [(p, q, t, a, "seasoning") for p, q, t, a in registry.seasoning])
    new = pd.concat([f for f in (seas, mule, vic) if len(f)], ignore_index=True)
    for c in ("is_attack", "step_in_sequence"):
        if c in new:
            new[c] = new[c].fillna(0)
    saved = (state.accounts, state.devices, state.sim_changes)
    try:
        state.accounts, state.devices, state.sim_changes = accounts, devices, sims
        schema, X = state.featurise(new.drop(columns=[c for c in ("episode_id", "scam_variant", "evasion_technique",
                                                                     "source_rid", "touched_fields", "label_visible",
                                                                     "label_delay_days") if c in new.columns]))
    finally:
        state.accounts, state.devices, state.sim_changes = saved
    vic_mask = schema["ts"].isin(vic["ts"]).to_numpy() & schema["user_id"].eq(
        accounts.set_index("idx").at[victim, "acct_id"]).to_numpy()
    sv, Xv = schema[vic_mask].reset_index(drop=True), X[vic_mask].reset_index(drop=True)
    order = np.argsort(sv["ts"].to_numpy(), kind="stable")
    sv, Xv = sv.iloc[order].reset_index(drop=True), Xv.iloc[order].reset_index(drop=True)
    dec = state.bundle.decide(Xv, sv)
    steps_by_ts = dict(zip(vic["ts"], vic["step_in_sequence"]))
    return _trace(state, lib, variant, ev, segment, vrow, res, sv, Xv, dec, steps_by_ts, mule)


def _trace(state, lib, variant, ev, segment, vrow, res, sv, Xv, dec, steps_by_ts, mule) -> dict:
    v = lib.variants[variant]
    sealed = state.bundle.provenance
    steps = [{"step": "S1", "name": v["steps"]["S1"]["name"], "observed": False,
              "what_happens": v["steps"]["S1"].get("channel"),
              "note": "The lure happens off-platform (call, SMS, chat). Citadel never sees it and does not need to."}]
    fired = None
    stopped = False
    moved, protected = 0.0, 0.0
    for i in range(len(sv)):
        row, d = sv.iloc[i], dec.iloc[i]
        step = int(steps_by_ts.get(int(row["ts"]), 3))
        sig = {c: (None if pd.isna(row[c]) else float(row[c])) for c in SIGNAL_COLS}
        act = d["action"]
        if not stopped and d["level"] >= 2 and fired is None:
            fired = {"index": i, "step": f"S{step}", "action": act}
        status = "moved"
        if stopped:
            status = "not attempted (customer already protected)"
        elif d["level"] >= 2:
            status = "held for review" if act == "A3" else "warned + cooling-off"
            stopped = True
        if status == "moved":
            moved += float(row["amount_inr"])
        else:
            protected += float(row["amount_inr"])
        steps.append({
            "step": f"S{step}", "name": v["steps"][f"S{step}"]["name"], "observed": True,
            "ts": int(row["ts"]), "time_ist": fmt(int(row["ts"])), "txn_type": str(row["txn_type"]),
            "amount_inr": float(row["amount_inr"]), "payee_hash": hmac_id(row["payee_id"]),
            "payee_age_days": float(row["payee_age_days"]), "first_time_payee": bool(row["user_to_payee_prior_txn_count"] == 0),
            "signals": sig,
            "layers": {"L0_rules_fired": list(d["l0_fired"]), "L1_supervised": float(d["p_L1"]),
                       "L2_payee_graph": float(d["p_L2"]), "fused_risk": float(d["risk"]),
                       "abstained_to_human": bool(d["abstained"])},
            "action": act, "band": d["band"], "treatment": d["treatment"], "reasons": list(d["reasons"]),
            "message": {lang: state.bundle.message(act, list(d["reasons"]), str(row["txn_type"]), lang,
                                                   float(row["amount_inr"])) for lang in ("en", "hi")},
            "money": status,
        })
    s4 = mule[mule["row_kind"] == "mule_outflow"] if len(mule) else mule
    steps.append({"step": "S4", "name": v["steps"]["S4"]["name"], "observed": bool(len(s4)),
                  "mule_outflows": int(len(s4)), "mule_outflow_inr": float(s4["amount_inr"].sum()) if len(s4) else 0.0,
                  "note": ("If the money had moved, the mule fans it out to layering accounts within minutes; "
                           "that is why Citadel acts at S3, before the money moves.")})
    cfg_sealed = {}
    try:
        import json

        from ..config import ROOT
        cfg_sealed = json.loads((ROOT / "artifacts" / state.run / "config_resolved.json").read_text())["sealed_holdout"]
    except Exception:  # noqa: BLE001
        pass
    return {
        "composition": {"variant": variant, "title": v["title"], "evasion": ev or "none", "segment": segment,
                        "withheld_variant": cfg_sealed.get("variant") == variant,
                        "withheld_evasion": cfg_sealed.get("evasion") == ev, "story": v["story"]},
        "victim": {"user_hash": hmac_id(vrow["acct_id"]), "age_band": vrow["user_age_band"],
                   "digital_literacy": vrow["digital_literacy"], "home_tier": int(vrow["home_tier"]),
                   "segment": segment_of({"user_age_band": vrow["user_age_band"], "digital_literacy": vrow["digital_literacy"],
                                          "tenure_days": (state.now_ts - vrow["created_ts"]) / DAY_S}, lib)},
        "steps": steps, "fired_at": fired, "caught": fired is not None,
        "money_moved_inr": moved, "money_protected_inr": protected,
        "bound": "Composed from the grammar: new to the model, inside our grammar. Outside the grammar is untested.",
        "run_id": sealed.get("run_id"),
    }
