"""Scam episodes as MUTATIONS of real benign payments, driven by the YAML grammar.

For every episode we pick a plausibly-selected victim, pick one of their REAL benign payments as the
template and overwrite only the fields the scam touches. Additional payments in the episode (retries,
splits, escalations, the preliminary "fee") are clones of that template with a shifted time, so the
victim's own attributes, device history and calendar habits are never synthesised.

Each row records ``touched_fields``: the DECLARED set of columns this episode may write (variant signal
columns + attacker levers). tests/test_attack_is_mutation.py checks that nothing outside it changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..timeutil import DAY_S, HOUR_S, hour_ist, is_night
from .library import ScamLibrary
from .mules import MuleRegistry

LEVER_COLUMNS = {"_payee", "amount_inr", "txn_type", "_phonebook", "payee_name_similarity_to_known_contact",
                 "collect_request_age_s"}
DERIVED_SIGNALS = {"new_device_flag", "sim_change_7d"}   # realised as device/SIM events, not cell writes
LABEL_CHANNELS = {"complaint": (0.6, (0.5, 10.0)), "bank_report": (0.25, (14.0, 40.0)),
                  "analyst": (0.15, (0.1, 1.0))}


@dataclass
class EpisodeResult:
    rows: list[dict] = field(default_factory=list)           # victim payments (mutated / cloned)
    mule_rows: list[tuple] = field(default_factory=list)     # (payer, payee, ts, amount, kind, step)
    episode: dict = field(default_factory=dict)
    labels: list[dict] = field(default_factory=list)
    replaced_template: bool = True


def segment_of(row: dict | pd.Series, lib: ScamLibrary) -> str:
    for name, seg in lib.segments.items():
        if "age_band" in seg and row["user_age_band"] not in seg["age_band"]:
            continue
        if "digital_literacy" in seg and row["digital_literacy"] not in seg["digital_literacy"]:
            continue
        if "max_tenure_days" in seg and row["tenure_days"] > seg["max_tenure_days"]:
            continue
        return name
    return "mainstream"


def victim_weights(lib: ScamLibrary, vid: str, persons: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(persons))
    for col, table in lib.variants[vid].get("victim_weights", {}).items():
        vals = persons[col].astype(str).to_numpy()
        w *= np.array([float(table.get(v, 1.0)) for v in vals])
    return w


def _draw(spec: dict[str, Any], r: np.random.Generator) -> float:
    if "lognormal" in spec:
        med, s = spec["lognormal"]
        return float(np.exp(r.normal(np.log(med), s)))
    if "uniform" in spec:
        return float(r.uniform(*spec["uniform"]))
    if "int_range" in spec:
        lo, hi = spec["int_range"]
        return float(r.integers(lo, hi + 1))
    if "value" in spec:
        return float(spec["value"])
    return 1.0


def _amount(spec: dict, r: np.random.Generator, baseline: float, balance: float) -> float:
    if spec["base"] == "baseline":
        a = baseline * r.uniform(*spec["multiple"])
    elif spec["base"] == "balance":
        a = balance * r.uniform(*spec["share"])
    else:
        a = r.uniform(*spec["inr"])
    return float(a)


def build_episode(*, lib: ScamLibrary, registry: MuleRegistry, r: np.random.Generator, episode_id: str,
                  variant: str, evasion: str | None, template: dict, victim: dict,
                  victim_history: pd.DataFrame, label_visible_share: float,
                  scam_devices: list[int]) -> EpisodeResult:
    """Build one episode. ``template`` is a benign internal row of the victim; ``victim`` its account."""
    v = lib.variants[variant]
    steps = v["steps"]
    s2, s3, s4 = steps["S2"], steps["S3"], steps["S4"]
    ev = lib.evasions.get(evasion or "", {}) if evasion else {}
    pop = registry.pop
    res = EpisodeResult()
    t3 = int(template["ts"])
    baseline = float(np.exp(victim["amount_mu"]))
    balance = float(victim["account_balance_inr"])

    declared = set(LEVER_COLUMNS) | (lib.variant_signal_columns(variant) - DERIVED_SIGNALS)
    declared |= {"session_duration_s", "attempts_in_session"}

    # ---- S3 payment schedule -------------------------------------------------------------------
    k = int(r.integers(s3["payments"][0], s3["payments"][1] + 1))
    gaps = [0.0] + [float(r.uniform(*s3["gap_minutes"])) for _ in range(k - 1)]
    if evasion == "slow_drip":
        gaps = [0.0] + [float(r.uniform(*ev["gap_hours"])) * 60 for _ in range(k - 1)]
    totals = [_amount(s3["amount"], r, baseline, balance) for _ in range(k)]
    times = list(np.cumsum(gaps))
    pay_plan: list[tuple[float, float]] = []   # (minutes after t3, amount)
    for t_min, amt in zip(times, totals):
        if evasion == "amount_splitting":
            parts = int(r.integers(ev["parts"][0], ev["parts"][1] + 1))
            shares = r.dirichlet(np.full(parts, 4.0))
            off = 0.0
            for p_i, sh in enumerate(shares):
                if p_i:
                    off += float(r.uniform(*ev["part_gap_minutes"]))
                pay_plan.append((t_min + off, amt * sh))
        else:
            pay_plan.append((t_min, amt))
    # money cannot exceed the balance; UPI caps a single payment at 1 lakh
    total = sum(a for _, a in pay_plan)
    cap = 0.97 * balance
    if total > cap:
        pay_plan = [(t, a * cap / total) for t, a in pay_plan]
    pay_plan = [(t, float(np.clip(a, 1.0, 100000.0))) for t, a in pay_plan]

    # ---- payees (mules) -----------------------------------------------------------------------------
    aged = evasion == "aged_mule"
    first_time = r.random() < s3["first_time_payee_p"]
    prior_payees = victim_history.loc[(victim_history["ts"] < t3)
                                      & (victim_history["_payee"] < pop.n_persons), "_payee"].unique()
    if not first_time and len(prior_payees):
        mule = registry.recruit_contact(int(r.choice(prior_payees)), t3)
    else:
        mule = registry.pick(t3, aged=aged, purpose_opened_share=v["mule"]["purpose_opened_share"])
    mules_per_payment = [mule.idx] * len(pay_plan)
    if s3.get("new_mule_per_payment"):
        used = {mule.idx}
        for i in range(1, len(pay_plan)):
            m2 = registry.pick(t3 + int(pay_plan[i][0] * 60), aged=aged, exclude=used,
                               purpose_opened_share=v["mule"]["purpose_opened_share"])
            used.add(m2.idx)
            mules_per_payment[i] = m2.idx

    # ---- session signals (S2 + S3) ----------------------------------------------------------------
    sig_values: dict[str, float] = {}
    suppress = set(ev.get("suppress", [])) if evasion == "signal_suppression" else set()
    rebind = False
    sim_swap = False
    for col, spec in (s2.get("signals") or {}).items():
        fires = r.random() < spec["p"]
        if col == "new_device_flag":
            rebind = fires
            continue
        if col == "sim_change_7d":
            sim_swap = fires
            continue
        if fires and col not in suppress:
            sig_values[col] = _draw(spec, r)
    for col in suppress:
        sig_values[col] = 0.0
    declared |= suppress
    s3_sig = {col: spec for col, spec in (s3.get("signals") or {}).items()}

    if rebind:
        declared |= {"_device_key", "_bound_ts"}
    lead_first = 0.0
    prelim = s2.get("prelim") or {}
    prelim_plan: list[tuple[float, float]] = []
    if prelim.get("p", 0) > 0 and r.random() < prelim["p"]:
        n_pre = int(r.integers(prelim["count"][0], prelim["count"][1] + 1))
        leads = sorted((float(r.uniform(*prelim["lead_minutes"])) for _ in range(n_pre)), reverse=True)
        for ld in leads:
            prelim_plan.append((-ld, _amount(prelim["amount"], r, baseline, balance)))
        lead_first = leads[0]
    t_first = t3 - int(lead_first * 60)
    if rebind:
        created = t_first - int(r.uniform(5, 60) * 60)
        if r.random() < 0.3 and scam_devices:
            dev_key = int(r.choice(scam_devices))
        else:
            dev_key = pop.new_device(created)
            scam_devices.append(dev_key)
        bound_ts = created
    if sim_swap:
        pop.extra_sim_changes.append((int(template["_payer"]), int(t_first - r.uniform(1, 72) * HOUR_S)))

    name_sim = float(r.uniform(*ev["similarity"])) if evasion == "payee_name_mimicry" else None
    automation = r.random() < float(s3.get("automation_overlay_p", 0.0))

    def make_row(offset_min: float, amount: float, payee: int, step: int, clone: bool) -> dict:
        row = dict(template)
        row["ts"] = int(t3 + offset_min * 60)
        row["_payee"] = int(payee)
        row["amount_inr"] = round(float(amount), 2)
        row["txn_type"] = s3["txn_type"] if step == 3 else "pay"
        for col, val in sig_values.items():
            if not (isinstance(template.get(col), float) and np.isnan(template[col])):
                row[col] = val  # unobservable (NaN) stays NaN: the institution cannot see it
        if step == 3:
            for col, spec in s3_sig.items():
                if r.random() < spec["p"]:
                    row[col] = round(_draw(spec, r), 1)
        if row["txn_type"] == "collect":
            if automation:
                row["collect_request_age_s"] = round(float(r.uniform(2.0, 4.5)), 1)
            elif step != 3 or "collect_request_age_s" not in s3_sig:
                row["collect_request_age_s"] = round(float(np.exp(r.normal(np.log(75), 0.8))) + 6, 1)
        else:
            row["collect_request_age_s"] = np.nan
        row["_phonebook"] = int(r.random() < s3.get("in_phonebook_p", 0.0))
        row["payee_name_similarity_to_known_contact"] = (
            round(name_sim, 3) if name_sim is not None else round(float(r.beta(1.6, 7)), 3))
        if rebind:
            row["_device_key"], row["_bound_ts"] = dev_key, bound_ts
        row.update({"row_kind": "attack", "is_attack": 1, "episode_id": episode_id, "scam_variant": variant,
                    "evasion_technique": evasion or "", "step_in_sequence": step,
                    "source_rid": int(template["_rid"]),
                    "touched_fields": ",".join(sorted(declared | ({"ts"} if clone else set())))})
        return row

    for off, amt in prelim_plan:
        res.rows.append(make_row(off, amt, mules_per_payment[0], 2, True))
    for i, ((off, amt), payee) in enumerate(zip(pay_plan, mules_per_payment)):
        res.rows.append(make_row(off, amt, payee, 3, i > 0))

    # ---- lure credits (task scams pay small "rewards" first) -------------------------------------
    lc = v.get("lure_credits")
    if lc:
        t_lc = t_first - int(r.uniform(60, 720) * 60)
        for j in range(int(r.integers(lc["count"][0], lc["count"][1] + 1))):
            res.mule_rows.append((mules_per_payment[0], int(template["_payer"]), t_lc + j * int(r.uniform(30, 300) * 60),
                                  round(float(r.uniform(*lc["inr"])), 0), "lure_credit", 2))

    # ---- S4 mule fan-out, after each mule's last receipt ------------------------------------------
    received: dict[int, tuple[int, float]] = {}
    for row in res.rows:
        last_t, tot = received.get(row["_payee"], (0, 0.0))
        received[row["_payee"]] = (max(last_t, row["ts"]), tot + row["amount_inr"])
    delay = ev.get("fanout_delay_minutes") if aged else s4["delay_minutes"]
    for m_idx, (last_t, tot) in received.items():
        n_out = int(r.integers(s4["fanout"][0], s4["fanout"][1] + 1))
        t_out = last_t + int(r.uniform(*delay) * 60)
        outs = registry.downstream(m_idx, n_out, t_out)
        shares = r.dirichlet(np.full(len(outs), 3.0)) * tot * r.uniform(0.9, 0.98)
        for j, (dst, sh) in enumerate(zip(outs, shares)):
            res.mule_rows.append((m_idx, dst, t_out + j * int(r.uniform(20, 300)), round(float(sh), 0),
                                  "mule_outflow", 4))

    # ---- labels: most scams never get a visible label ----------------------------------------------
    visible = r.random() < label_visible_share
    first_loss = min(row["ts"] for row in res.rows)
    channel, delay_days = "", np.nan
    if visible:
        names = list(LABEL_CHANNELS)
        probs = np.array([LABEL_CHANNELS[c][0] for c in names])
        channel = names[int(r.choice(len(names), p=probs / probs.sum()))]
        delay_days = float(r.uniform(*LABEL_CHANNELS[channel][1]))
        arrival = int(first_loss + delay_days * DAY_S)
        for i, row in enumerate(res.rows):
            # a payment's label can never arrive before the payment itself (slow-drip episodes)
            row_arrival = max(arrival, int(row["ts"]) + int(r.uniform(1, 6) * HOUR_S))
            res.labels.append({"row_index": i, "episode_id": episode_id, "channel": channel,
                               "arrival_ts": row_arrival, "payee_idx": row["_payee"],
                               "reporter_idx": int(template["_payer"]), "row_ts": row["ts"]})
        if channel in ("complaint", "bank_report"):
            for m_idx in set(mules_per_payment):
                registry.freeze(m_idx, arrival + int(r.uniform(0, 1.0) * DAY_S))
    for row in res.rows:
        row["label_visible"] = int(visible)
        row["label_delay_days"] = delay_days

    hours = hour_ist(np.array([t3]))
    res.episode = {
        "episode_id": episode_id, "scam_variant": variant, "evasion_technique": evasion or "",
        "victim_idx": int(template["_payer"]), "s1_ts": int(t_first - r.uniform(*v["lure_lead_minutes"]) * 60),
        "first_payment_ts": first_loss, "s3_first_ts": t3, "n_victim_payments": len(res.rows),
        "n_prelim": len(prelim_plan), "total_loss_inr": round(sum(row["amount_inr"] for row in res.rows), 2),
        "mules": ",".join(str(m) for m in dict.fromkeys(mules_per_payment)),
        "first_time_payee": int(first_time), "night": int(is_night(hours)[0]),
        "signals_fired": ",".join(sorted(k for k, val in sig_values.items() if val == 1.0)),
        "device_rebind": int(rebind), "sim_swap": int(sim_swap), "automation_overlay": int(automation),
        "label_visible": int(visible), "label_channel": channel, "label_delay_days": delay_days,
    }
    return res


def choose_template(r: np.random.Generator, rows: pd.DataFrame, night_bias: float, mimic_time: bool) -> int:
    """Pick the benign payment to mutate. Night bias is applied ON TOP of the victim's own calendar."""
    if mimic_time:
        return int(r.integers(len(rows)))
    h = hour_ist(rows["ts"].to_numpy())
    w = 1.0 + night_bias * is_night(h) + 0.5 * ((h >= 19) & (h <= 22))
    return int(r.choice(len(rows), p=w / w.sum()))
