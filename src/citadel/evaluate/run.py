"""P7 + P8: the evaluation a fraud-model validator would trust.

``python -m citadel evaluate`` -> artifacts/<run>/*.csv + REPORT.md. Every metric row has n_rows and
n_positives; every rate a Wilson 95% CI; cells under the minimum positive count stay visible with
n_sufficient = 0; any delta whose CI includes zero is reported as "no measured effect"; numbers from a
reduced-scale profile are stamped reportable = False and cannot enter the PDF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import CONFIG_DIR
from ..defend import l0
from ..defend.bundle import Bundle, load_bundle
from ..defend.model import fit_l1
from ..defend.split import Split, make_split
from ..features.registry import FEATURES, family_features
from ..runctx import RunCtx
from ..timeutil import DAY_S
from . import metrics as M

HEED_DEFAULT = {"A0": 0.0}


@dataclass
class EvalState:
    ctx: RunCtx
    tx: pd.DataFrame
    X: pd.DataFrame
    episodes: pd.DataFrame
    bundle: Bundle
    sp: Split
    y: np.ndarray
    test: pd.DataFrame          # test rows (schema) aligned with dec
    dec: pd.DataFrame           # decisions on test rows
    Xt: pd.DataFrame
    yt: np.ndarray
    reportable: bool
    min_pos: int
    min_cell: int
    B: int


def _common(st: EvalState, n_pos: int) -> dict:
    return {"n_rows": int(len(st.yt)), "n_positives": int(n_pos), "reportable": bool(
        st.reportable and n_pos >= st.min_pos)}


# ----------------------------------------------------------------------------------------------------
def headline(st: EvalState) -> pd.DataFrame:
    y, risk = st.yt, st.dec["risk"].to_numpy()
    base = float(y.mean())
    rows = []
    pr = M.pr_auc(y, risk)
    lo, hi = M.bootstrap_ci(y, risk, M.pr_auc, st.B, seed=1)
    rows.append({"metric": "pr_auc", "value": pr, "ci_low": lo, "ci_high": hi, "headline": True,
                 "note": f"lift over base rate {pr / base:.1f}x (base {base:.4%})", **_common(st, y.sum())})
    for fpr in st.ctx.cfg.eval.fpr_budgets:
        r = M.recall_at_fpr(y, risk, fpr)
        p, lo, hi = M.wilson(r["tp"], r["n_pos"])
        rows.append({"metric": f"recall_at_{fpr:.1%}_fpr", "value": p, "ci_low": lo, "ci_high": hi, "headline": True,
                     "note": f"realised FPR {r['realised_fpr']:.4%} ({r['fp']}/{r['n_neg']})", **_common(st, y.sum())})
    pa = M.partial_auc(y, risk, 0.02)
    lo, hi = M.bootstrap_ci(y, risk, lambda a, b: M.partial_auc(a, b, 0.02), st.B, seed=2)
    rows.append({"metric": "partial_auc_fpr_0_2pct_mcclish", "value": pa, "ci_low": lo, "ci_high": hi,
                 "headline": False, "note": "McClish-standardised", **_common(st, y.sum())})
    k = capacity(st)["k_test"]
    top = np.argsort(-risk, kind="stable")[: max(int(k), 1)]
    tp = int(y[top].sum())
    p, lo, hi = M.wilson(tp, len(top))
    rows.append({"metric": "precision_at_k", "value": p, "ci_low": lo, "ci_high": hi, "headline": False,
                 "note": f"k={len(top)} from staffing; recall at k = {tp / max(y.sum(), 1):.3f}", **_common(st, y.sum())})
    a2 = st.dec["level"].to_numpy() >= 2
    amt = st.test["amount_inr"].to_numpy()
    vw = float(amt[(y == 1) & a2].sum() / max(amt[y == 1].sum(), 1))
    rows.append({"metric": "value_weighted_recall_A2plus", "value": vw, "ci_low": np.nan, "ci_high": np.nan,
                 "headline": False, "note": "share of scam INR at A2+ (warning or hold)", **_common(st, y.sum())})
    p, lo, hi = M.wilson(int((a2 & (y == 1)).sum()), int(y.sum()))
    rows.append({"metric": "recall_at_deployed_A2plus", "value": p, "ci_low": lo, "ci_high": hi, "headline": True,
                 "note": f"at the deployed ladder; A2+ alert share {a2.mean():.3%}", **_common(st, y.sum())})
    roc = M.roc_auc(y, risk)
    rows.append({"metric": "roc_auc", "value": roc, "ci_low": np.nan, "ci_high": np.nan, "headline": False,
                 "note": "computed, NOT a headline (dominated by true negatives at a <1% base rate)",
                 **_common(st, y.sum())})
    return pd.DataFrame(rows)


def operating_curve(st: EvalState) -> pd.DataFrame:
    """Recall vs alerts per 1,000 payments, with the deployed ladder points marked."""
    risk, y = st.dec["risk"].to_numpy(), st.yt
    order = np.sort(risk)[::-1]
    rows = []
    for share in np.unique(np.concatenate([np.geomspace(0.0002, 0.08, 40),
                                           list(st.bundle.ladder.budgets.values())])):
        k = max(int(round(share * len(risk))), 1)
        thr = order[k - 1]
        a = risk >= thr
        rows.append({"alerts_per_1000": 1000 * a.mean(), "recall": float(a[y == 1].mean()),
                     "precision": float(y[a].mean()), "fpr": float(a[y == 0].mean()),
                     "ladder_point": next((lv for lv, b in st.bundle.ladder.budgets.items() if abs(b - share) < 1e-12), ""),
                     **_common(st, y.sum())})
    return pd.DataFrame(rows)


def capacity(st: EvalState) -> dict:
    s = st.ctx.cfg.staffing
    per_day = s.analysts * s.cases_per_hour * s.shift_hours * s.shifts_per_day
    share = per_day / s.deployment_payments_per_day
    return {"analysts": s.analysts, "cases_per_hour": s.cases_per_hour, "shift_hours": s.shift_hours,
            "shifts_per_day": s.shifts_per_day, "cases_per_day": per_day,
            "deployment_payments_per_day": s.deployment_payments_per_day, "capacity_share": share,
            "k_test": share * len(st.yt), "k_per_1000": share * 1000}


def alert_budget(st: EvalState) -> tuple[pd.DataFrame, pd.DataFrame]:
    lvl = st.dec["level"].to_numpy()
    y = st.yt
    days = max((st.test["ts"].max() - st.test["ts"].min()) / DAY_S, 1e-9)
    cap = capacity(st)
    rows = []
    for i, a in enumerate(("A0", "A1", "A2", "A3")):
        m = lvl == i
        rows.append({"action": a, "n_alerts": int(m.sum()), "per_1000_payments": 1000 * m.mean(),
                     "scams_at_level": int((m & (y == 1)).sum()), "legit_at_level": int((m & (y == 0)).sum()),
                     "budget_share": st.bundle.ladder.budgets.get(a, np.nan), "realised_share": m.mean(),
                     **_common(st, y.sum())})
    ladder_df = pd.DataFrame(rows)
    a2p = lvl >= 2
    holds = lvl == 3
    ab = st.dec["abstained"].to_numpy()
    budget = pd.DataFrame([{
        "warnings_A2plus_per_1000": 1000 * a2p.mean(), "warning_budget_per_1000": 1000 * st.bundle.ladder.budgets["A2"],
        "holds_A3_per_1000": 1000 * holds.mean(), "hold_budget_per_1000": 1000 * st.bundle.ladder.budgets["A3"],
        "analyst_capacity_k_per_1000": cap["k_per_1000"], "holds_within_capacity": bool(holds.mean() <= cap["capacity_share"]),
        "scams_caught_A2plus": int((a2p & (y == 1)).sum()), "scams_total": int(y.sum()),
        "abstained_to_review": int(ab.sum()), "abstention_rate": float(ab.mean()),
        "abstention_cases_per_day_at_test_scale": float(ab.sum() / days),
        "abstention_precision": float(y[ab].mean()) if ab.any() else np.nan,
        "holds_per_day_at_deployment_scale": holds.mean() * cap["deployment_payments_per_day"],
        "analyst_cases_per_day_capacity": cap["cases_per_day"], **_common(st, y.sum())}])
    return ladder_df, budget


def time_to_alert(st: EvalState) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = st.test.assign(level=st.dec["level"].to_numpy(), action=st.dec["action"].to_numpy())
    vic = t[t["is_attack"] == 1].sort_values("ts")
    costs = yaml.safe_load((CONFIG_DIR / "costs.yaml").read_text())
    heed = {**HEED_DEFAULT, **costs["heed_rate"]}
    first_ts_all = st.tx[st.tx.is_attack == 1].groupby("episode_id")["ts"].min()
    rows = []
    for eid, g in vic.groupby("episode_id"):
        if g["ts"].min() > first_ts_all.get(eid, g["ts"].min()):
            continue  # episode started before the test window; its first loss is not observable here
        lv = g["level"].to_numpy()
        amt = g["amount_inr"].to_numpy()
        steps = g["step_in_sequence"].to_numpy()
        alerted = np.flatnonzero(lv >= 2)
        first_s3 = int(np.flatnonzero(steps == 3)[0]) if (steps == 3).any() else 0
        if len(alerted):
            j = int(alerted[0])
            act = g["action"].iat[j]
            saved = float(amt[j:].sum() * heed.get(act, 0.0))
            saved_perfect = float(amt[j:].sum())
            mins = float((g["ts"].iat[j] - g["ts"].iat[0]) / 60)
        else:
            j, act, saved, saved_perfect, mins = -1, "A0", 0.0, 0.0, np.nan
        rows.append({"episode_id": eid, "scam_variant": g["scam_variant"].iat[0],
                     "evasion_technique": g["evasion_technique"].iat[0] or "none",
                     "n_victim_payments": len(g), "total_loss_inr": float(amt.sum()),
                     "caught_any_A2plus": int(j >= 0), "first_alert_index": j,
                     "first_alert_step": int(steps[j]) if j >= 0 else 0, "first_alert_action": act,
                     "caught_before_first_loss": int(j == 0), "caught_by_first_harmful_payment": int(0 <= j <= first_s3),
                     "minutes_first_payment_to_alert": mins, "value_saved_inr_heed_adjusted": saved,
                     "value_saved_inr_if_every_alert_heeded": saved_perfect})
    ep = pd.DataFrame(rows)
    if ep.empty:
        return ep, pd.DataFrame()
    n = len(ep)
    summ = [M.rate_row("episodes_caught_any_A2plus", ep.caught_any_A2plus.sum(), n, st.min_cell),
            M.rate_row("episodes_caught_before_first_loss", ep.caught_before_first_loss.sum(), n, st.min_cell),
            M.rate_row("episodes_caught_by_first_harmful_payment", ep.caught_by_first_harmful_payment.sum(), n,
                       st.min_cell)]
    total = ep.total_loss_inr.sum()
    summ.append({"metric": "value_saved_share_heed_adjusted", "value": ep.value_saved_inr_heed_adjusted.sum() / total,
                 "ci_low": np.nan, "ci_high": np.nan, "numerator": int(ep.value_saved_inr_heed_adjusted.sum()),
                 "denominator": int(total), "n_sufficient": int(n >= st.min_cell),
                 "note": "heed rates from configs/costs.yaml (assumption)"})
    summ.append({"metric": "value_saved_share_upper_bound", "value": ep.value_saved_inr_if_every_alert_heeded.sum() / total,
                 "ci_low": np.nan, "ci_high": np.nan, "numerator": int(ep.value_saved_inr_if_every_alert_heeded.sum()),
                 "denominator": int(total), "n_sufficient": int(n >= st.min_cell), "note": "every alert heeded"})
    for step in (2, 3):
        summ.append(M.rate_row(f"first_alert_at_step_S{step}", (ep.first_alert_step == step).sum(), n, st.min_cell))
    s = pd.DataFrame(summ)
    s["n_episodes"] = n
    s["reportable"] = st.reportable and n >= st.min_cell
    return ep, s


def by_group(st: EvalState, col: pd.Series, name: str) -> pd.DataFrame:
    lvl = st.dec["level"].to_numpy()
    y = st.yt
    rows = []
    for g in sorted(col.astype(str).unique()):
        m = (col.astype(str) == g).to_numpy()
        pos, neg = m & (y == 1), m & (y == 0)
        r = M.rate_row("recall_A2plus", (pos & (lvl >= 2)).sum(), pos.sum(), st.min_cell, group=name, value_of=g)
        f = M.wilson((neg & (lvl >= 2)).sum(), neg.sum())
        r.update({"fpr_A2plus": f[0], "fpr_ci_low": f[1], "fpr_ci_high": f[2], "n_rows": int(m.sum()),
                  "n_positives": int(pos.sum())})
        rows.append(r)
    return pd.DataFrame(rows).sort_values("value")


def generalisation(st: EvalState) -> pd.DataFrame:
    cfg = st.ctx.cfg
    lvl = st.dec["level"].to_numpy()
    risk = st.dec["risk"].to_numpy()
    y = st.yt
    thr = M.threshold_at_fpr(risk[y == 0], 0.005)
    var = st.test["scam_variant"].astype(str).to_numpy()
    ev = st.test["evasion_technique"].astype(str).to_numpy()
    groups = {
        f"withheld_variant_{cfg.sealed_holdout.variant}": (y == 1) & (var == cfg.sealed_holdout.variant),
        "seen_variants": (y == 1) & (var != cfg.sealed_holdout.variant),
        f"withheld_evasion_{cfg.sealed_holdout.evasion}": (y == 1) & (ev == cfg.sealed_holdout.evasion),
        "no_evasion": (y == 1) & (ev == ""),
        "seen_evasions": (y == 1) & (ev != "") & (ev != cfg.sealed_holdout.evasion),
    }
    rows = []
    for g, m in groups.items():
        a = M.rate_row("recall_A2plus", (m & (lvl >= 2)).sum(), m.sum(), st.min_cell, group=g)
        b = M.wilson((m & (risk > thr)).sum(), m.sum())
        a.update({"recall_at_0.5pct_fpr": b[0], "r05_ci_low": b[1], "r05_ci_high": b[2], "n_positives": int(m.sum()),
                  "reportable": bool(st.reportable and m.sum() >= st.min_cell)})
        rows.append(a)
    df = pd.DataFrame(rows)
    v = df.set_index("group")["value"]
    seen = v.get("seen_variants", np.nan)
    for g in df["group"]:
        df.loc[df.group == g, "gap_vs_seen"] = seen - v[g] if g.startswith("withheld_variant") else np.nan
        df.loc[df.group == g, "retained_share_of_seen"] = v[g] / seen if g.startswith("withheld_variant") and seen else np.nan
    ne = v.get("no_evasion", np.nan)
    for g in df["group"]:
        if g.startswith("withheld_evasion"):
            df.loc[df.group == g, "gap_vs_seen"] = ne - v[g]
            df.loc[df.group == g, "retained_share_of_seen"] = v[g] / ne if ne else np.nan
    return df


def lovo(st: EvalState) -> pd.DataFrame:
    """Leave-one-variant-out: retrain L1 without each variant; recall on that variant (L1 alone)."""
    cfg = st.ctx.cfg
    y = st.y
    var_all = st.tx["scam_variant"].astype(str).to_numpy()
    X = st.X
    full = st.bundle.l1.raw(st.Xt)
    rows = []
    for v in cfg.attacks.variants:
        keep = st.sp.train & (var_all != v)
        m = fit_l1(X, y, keep, keep, keep, cfg.model, seed=cfg.seed, fixed_iter=st.bundle.l1.n_iter)
        s = m.raw(st.Xt)
        tvar = st.test["scam_variant"].astype(str).to_numpy()
        mask = (st.yt == 0) | (tvar == v)
        yy = st.yt[mask]
        r_out = M.recall_at_fpr(yy, s[mask], 0.005)
        r_in = M.recall_at_fpr(yy, full[mask], 0.005)
        lo, hi = M.wilson(r_out["tp"], r_out["n_pos"])[1:]
        rows.append({"variant": v, "recall_at_0.5pct_fpr_when_withheld": r_out["recall"], "ci_low": lo, "ci_high": hi,
                     "recall_at_0.5pct_fpr_when_trained": r_in["recall"],
                     "retained_share": r_out["recall"] / r_in["recall"] if r_in["recall"] else np.nan,
                     "n_positives": r_out["n_pos"], "n_sufficient": int(r_out["n_pos"] >= st.min_cell),
                     "reportable": bool(st.reportable and r_out["n_pos"] >= st.min_cell), "model": "L1 alone"})
    return pd.DataFrame(rows)


def hard_negative_fpr(st: EvalState) -> pd.DataFrame:
    lvl = st.dec["level"].to_numpy()
    kind = st.test["row_kind"].astype(str).to_numpy()
    arch = st.test["hn_archetype"].astype(str).to_numpy()
    y = st.yt
    rows = []
    pops = {"benign": kind == "benign", "hard_negative_all": kind == "hard_negative", "shape_all": kind == "shape",
            "all_legitimate": y == 0}
    for a in sorted({x for k, x in zip(kind, arch) if k in ("hard_negative", "shape") for x in x.split("+") if x}):
        pops[f"archetype:{a}"] = np.array([a in x.split("+") for x in arch]) & (y == 0)
    for name, m in pops.items():
        m = m & (y == 0)
        row = {"population": name, "n_rows": int(m.sum())}
        for lab, cond in (("A1plus", lvl >= 1), ("A2plus", lvl >= 2), ("A3", lvl == 3)):
            p, lo, hi = M.wilson((m & cond).sum(), m.sum())
            row.update({f"fpr_{lab}": p, f"fpr_{lab}_ci_low": lo, f"fpr_{lab}_ci_high": hi,
                        f"n_{lab}": int((m & cond).sum())})
        row["n_sufficient"] = int(m.sum() >= 100)
        rows.append(row)
    return pd.DataFrame(rows)


def _bands(st: EvalState) -> dict[str, pd.Series]:
    t = st.test
    amt = pd.cut(t["amount_inr"], [0, 500, 2000, 10000, 50000, 1e9],
                 labels=["<500", "500-2k", "2k-10k", "10k-50k", "50k+"]).astype(str)
    pa = pd.cut(t["payee_age_days"], [-1, 30, 180, 730, 1e9], labels=["<30d", "30-180d", "180d-2y", "2y+"]).astype(str)
    return {"txn_type": t["txn_type"].astype(str), "amount_band": amt, "payee_age_band": pa,
            "user_age_band": t["user_age_band"].astype(str), "digital_literacy": t["digital_literacy"].astype(str),
            "home_tier": t["home_tier"].astype(str)}


def worst_slices(st: EvalState) -> pd.DataFrame:
    lvl = st.dec["level"].to_numpy()
    y = st.yt
    dims = _bands(st)
    names = list(dims)
    combos = [(a,) for a in names] + [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    rows = []
    for c in combos:
        key = dims[c[0]] if len(c) == 1 else dims[c[0]] + " | " + dims[c[1]]
        for val in key.unique():
            m = (key == val).to_numpy()
            if m.sum() < 25:
                continue
            pos = m & (y == 1)
            if pos.sum() < 5:
                continue
            p, lo, hi = M.wilson((pos & (lvl >= 2)).sum(), pos.sum())
            neg = m & (y == 0)
            rows.append({"slice": " x ".join(c), "value": val, "recall_A2plus": p, "ci_low": lo, "ci_high": hi,
                         "fpr_A2plus": float((neg & (lvl >= 2)).sum() / max(neg.sum(), 1)),
                         "n_rows": int(m.sum()), "n_positives": int(pos.sum()),
                         "n_sufficient": int(pos.sum() >= st.min_cell)})
    return pd.DataFrame(rows).sort_values(["n_sufficient", "recall_A2plus"], ascending=[False, True]).reset_index(drop=True)


EXPERT_RULES = {
    "new_payee_and_3x_amount": lambda X: (X.f_payee_first_time == 1) & (X.f_amt_ratio_user >= 3),
    "remote_access": lambda X: X.f_remote_access == 1,
    "screen_share": lambda X: X.f_screen_share == 1,
    "call_and_new_payee": lambda X: (X.f_call == 1) & (X.f_payee_first_time == 1),
    "collect_from_unknown": lambda X: (X.f_txn_type == 1) & (X.f_req_known_contact == 0),
    "qr_to_new_unverified": lambda X: (X.f_txn_type == 2) & (X.f_payee_first_time == 1) & (X.f_payee_verified == 0),
    "new_device_or_sim": lambda X: (X.f_new_device == 1) | (X.f_sim_change == 1),
    "young_unverified_payee": lambda X: (X.f_payee_age_days < 90) & (X.f_payee_verified == 0),
    "half_the_balance": lambda X: X.f_share_of_balance >= 0.5,
    "burst_to_new_payee": lambda X: X.f_burst_to_new_payee >= 2,
}


def baselines(st: EvalState) -> pd.DataFrame:
    y = st.y
    sp = st.sp
    Xt, yt = st.Xt, st.yt
    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                       LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))
    Xtr = st.X[sp.train]
    lr.fit(np.sign(Xtr) * np.log1p(np.abs(Xtr)), y[sp.train])
    scores = {
        "citadel_deployed (fused, calibrated)": st.dec["risk"].to_numpy(),
        "citadel_L1_alone": st.bundle.l1.raw(Xt),
        "logistic_regression_all_features": lr.predict_proba(np.sign(Xt) * np.log1p(np.abs(Xt)))[:, 1],
        # float64 + an amount tie-break (D-013: in float32 a 1e-9 tie-break vanished and the rule scored 0)
        "payee_is_first_time_alone": Xt["f_payee_first_time"].to_numpy(dtype=float)
        + 1e-3 * np.nan_to_num(Xt["f_amount_log"].to_numpy(dtype=float)) / 20,
        "ten_expert_rules_count": sum(f(Xt).to_numpy().astype(float) for f in EXPERT_RULES.values())
        + 1e-3 * np.nan_to_num(Xt["f_amount_log"].to_numpy()) / 20,
    }
    rows = []
    ref = scores["citadel_deployed (fused, calibrated)"]
    for name, s in scores.items():
        s = np.nan_to_num(np.asarray(s, dtype=float))
        row = {"model": name, "pr_auc": M.pr_auc(yt, s), "roc_auc": M.roc_auc(yt, s)}
        for fpr in st.ctx.cfg.eval.fpr_budgets:
            r = M.recall_at_fpr(yt, s, fpr)
            p, lo, hi = M.wilson(r["tp"], r["n_pos"])
            row.update({f"recall_at_{fpr:.1%}_fpr": p, f"r{fpr}_ci_low": lo, f"r{fpr}_ci_high": hi,
                        f"realised_fpr_{fpr:.1%}": r["realised_fpr"]})
        if name != "citadel_deployed (fused, calibrated)":
            d = M.paired_delta(yt, s, ref, M.recall_at_fpr_fn(0.005), st.B, seed=3)
            row.update({"citadel_minus_this_recall_0.5pct": d["delta"], "delta_ci_low": d["delta_ci_low"],
                        "delta_ci_high": d["delta_ci_high"], "verdict": d["verdict"]})
        row.update(_common(st, yt.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def layer_ablation(st: EvalState) -> pd.DataFrame:
    cfg, sp, y = st.ctx.cfg, st.sp, st.y
    steps = [("raw schema", ["raw"]), ("+ user windows", ["raw", "user"]),
             ("+ sequence", ["raw", "user", "sequence"]), ("+ payee windows", ["raw", "user", "sequence", "payee"]),
             ("+ graph", ["raw", "user", "sequence", "payee", "graph"])]
    l0r = l0.evaluate(st.bundle.rules, l0.l0_frame(st.tx, st.X))
    Xl0 = pd.concat([st.X.reset_index(drop=True),
                     l0r.fired.astype(float).add_prefix("l0_").reset_index(drop=True)], axis=1)
    rows, prev = [], None
    full_cols = FEATURES
    for name, fams in steps + [("+ L0 rules as features", None)]:
        cols = family_features(*fams) if fams else full_cols + [c for c in Xl0.columns if c.startswith("l0_")]
        Xa = Xl0 if fams is None else st.X
        m = fit_l1(Xa, y, sp.train, sp.train, sp.train, cfg.model, features=cols, seed=cfg.seed,
                   fixed_iter=st.bundle.l1.n_iter)
        s = m.raw(Xa[sp.test].reset_index(drop=True))
        row = {"step": name, "n_features": len(cols), "pr_auc": M.pr_auc(st.yt, s),
               "recall_at_0.5pct_fpr": M.recall_at_fpr(st.yt, s, 0.005)["recall"],
               "recall_at_0.1pct_fpr": M.recall_at_fpr(st.yt, s, 0.001)["recall"]}
        if prev is not None:
            d = M.paired_delta(st.yt, prev, s, M.pr_auc, st.B, seed=4)
            row.update({"delta_pr_auc_vs_previous": d["delta"], "delta_ci_low": d["delta_ci_low"],
                        "delta_ci_high": d["delta_ci_high"], "verdict": d["verdict"]})
        row.update(_common(st, st.yt.sum()))
        rows.append(row)
        prev = s
    # label scarcity: train on the labels the institution would actually have by the train cutoff
    meta = st.ctx.gen_meta()
    cutoff = meta["t0"] + (sp.boundaries["train_to"] + 1) * DAY_S
    vis = (st.tx["label_visible"].to_numpy() == 1) & (
        st.tx["ts"].to_numpy() + np.nan_to_num(st.tx["label_delay_days"].to_numpy(), nan=1e6) * DAY_S < cutoff)
    y_vis = (y == 1) & vis
    m = fit_l1(st.X, y_vis.astype(int), sp.train, sp.train, sp.train, cfg.model, seed=cfg.seed,
               fixed_iter=st.bundle.l1.n_iter)
    s = m.raw(st.Xt)
    rows.append({"step": "ablation: visible matured labels only (no oracle)", "n_features": len(FEATURES),
                 "pr_auc": M.pr_auc(st.yt, s), "recall_at_0.5pct_fpr": M.recall_at_fpr(st.yt, s, 0.005)["recall"],
                 "recall_at_0.1pct_fpr": M.recall_at_fpr(st.yt, s, 0.001)["recall"],
                 "verdict": f"trained on {int(y_vis[sp.train].sum())} visible positives vs "
                            f"{int(y[sp.train].sum())} oracle; evaluated on oracle truth",
                 **_common(st, st.yt.sum())})
    return pd.DataFrame(rows)


def sample_size_curve(st: EvalState) -> pd.DataFrame:
    cfg, sp, y = st.ctx.cfg, st.sp, st.y
    r = np.random.default_rng(cfg.seed)
    idx = np.flatnonzero(sp.train)
    rows = []
    for frac in cfg.eval.sample_size_fracs:
        take = np.zeros(len(y), dtype=bool)
        take[r.choice(idx, max(int(frac * len(idx)), 100), replace=False)] = True
        m = fit_l1(st.X, y, take, take, take, cfg.model, seed=cfg.seed, fixed_iter=st.bundle.l1.n_iter)
        s = m.raw(st.Xt)
        rows.append({"train_fraction": frac, "n_train_rows": int(take.sum()), "n_train_positives": int(y[take].sum()),
                     "pr_auc": M.pr_auc(st.yt, s), "recall_at_0.5pct_fpr": M.recall_at_fpr(st.yt, s, 0.005)["recall"],
                     **_common(st, st.yt.sum())})
    return pd.DataFrame(rows)


def fairness(st: EvalState, tolerance: float = 2.0) -> pd.DataFrame:
    frames = [by_group(st, st.test[c], c) for c in ("user_age_band", "digital_literacy", "home_tier")]
    df = pd.concat(frames, ignore_index=True)
    lvl = st.dec["level"].to_numpy()
    overall = float(((lvl >= 2) & (st.yt == 0)).sum() / max((st.yt == 0).sum(), 1))
    df["overall_fpr_A2plus"] = overall
    df["fpr_ratio_to_overall"] = df["fpr_A2plus"] / overall if overall else np.nan
    df["flag_fpr_disparity"] = df["fpr_ratio_to_overall"] > tolerance
    df["tolerance"] = tolerance
    return df


def reason_usage(st: EvalState) -> pd.DataFrame:
    lvl = st.dec["level"].to_numpy()
    reasons = st.dec["reasons"]
    alert = lvl >= 1
    counts: dict[str, list[int]] = {}
    for rs, y, a in zip(reasons, st.yt, alert):
        if not a:
            continue
        for c in rs:
            counts.setdefault(c, [0, 0])
            counts[c][0 if y == 1 else 1] += 1
    n_alert = int(alert.sum())
    rows = [{"reason_code": c, "uses_on_scams": v[0], "uses_on_legit": v[1], "share_of_alerts": (v[0] + v[1]) / max(n_alert, 1)}
            for c, v in sorted(counts.items())]
    df = pd.DataFrame(rows)
    df["n_alerts"] = n_alert
    df["fallback_rate"] = float(st.dec["reason_fallback"].to_numpy()[alert].mean()) if n_alert else np.nan
    df["max_codes_per_decision"] = int(max((len(r) for r in reasons), default=0))
    return df


def cost_summary(st: EvalState) -> pd.DataFrame:
    c = yaml.safe_load((CONFIG_DIR / "costs.yaml").read_text())
    lvl = st.dec["level"].to_numpy()
    y = st.yt
    amt = st.test["amount_inr"].to_numpy()

    def total(review=1.0, hold=1.0, friction=1.0, heed=1.0, recovery=c["recovery_rate"]):
        h = {0: 0.0, 1: c["heed_rate"]["A1"] * heed, 2: c["heed_rate"]["A2"] * heed, 3: min(c["heed_rate"]["A3"] * heed, 1)}
        heeded = np.array([h[int(x)] for x in lvl])
        loss = float((amt * (1 - recovery) * (1 - heeded))[y == 1].sum())
        neg = y == 0
        fr = float(((lvl == 1) & neg).sum() * c["friction_inr"]["A1"] + ((lvl == 2) & neg).sum() * c["friction_inr"]["A2"]) * friction
        fh = float(((lvl == 3) & neg).sum() * c["false_hold_inr"]) * hold
        rv = float((lvl == 3).sum() * c["analyst_review_inr"]) * review
        return loss, fr, fh, rv

    base_loss = float((amt * (1 - c["recovery_rate"]))[y == 1].sum())
    rows = []
    for name, kw in [("assumed", {}), ("review_x2", {"review": 2}), ("review_x0.5", {"review": 0.5}),
                     ("false_hold_x2", {"hold": 2}), ("friction_x2", {"friction": 2}), ("heed_x0.5", {"heed": 0.5}),
                     ("heed_x1.5", {"heed": 1.5}), ("all_costs_x2_heed_x0.5", {"review": 2, "hold": 2, "friction": 2, "heed": 0.5})]:
        loss, fr, fh, rv = total(**kw)
        cit = loss + fr + fh + rv
        rows.append({"scenario": name, "approve_everything_inr": base_loss, "citadel_total_inr": cit,
                     "residual_scam_loss_inr": loss, "friction_inr": fr, "false_hold_inr": fh, "analyst_review_inr": rv,
                     "net_saving_inr": base_loss - cit, "net_saving_share": (base_loss - cit) / base_loss if base_loss else np.nan,
                     "assumptions": c["assumptions_note"], "provenance_tier": "derived (simulator-internal, assumption-driven)",
                     **_common(st, y.sum())})
    return pd.DataFrame(rows)


def evidence_index(st: EvalState, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    def add(cid, desc, value, artefact, selector, tier, n_rows, n_pos, rep):
        rows.append({"claim_id": cid, "description": desc, "value": value, "artefact": artefact, "selector": selector,
                     "provenance_tier": tier, "n_rows": n_rows, "n_positives": n_pos, "reportable": rep})

    h = tables["defend_headline.csv"].set_index("metric")
    for m in h.index:
        add(f"headline.{m}", f"test window {m}", h.loc[m, "value"], "defend_headline.csv", f"metric={m}", "measured",
            h.loc[m, "n_rows"], h.loc[m, "n_positives"], h.loc[m, "reportable"])
    g = tables["eval_generalisation.csv"]
    for _, r in g.iterrows():
        add(f"generalisation.{r['group']}", f"recall at A2+ for {r['group']}", r["value"], "eval_generalisation.csv",
            f"group={r['group']}", "measured", r["denominator"], r["n_positives"], r["reportable"])
    t = tables.get("eval_time_to_alert_summary.csv")
    if t is not None and len(t):
        for _, r in t.iterrows():
            add(f"time_to_alert.{r['metric']}", r["metric"], r["value"], "eval_time_to_alert_summary.csv",
                f"metric={r['metric']}", "derived" if "saved" in r["metric"] else "measured", r["denominator"],
                r["n_episodes"], r["reportable"])
    b = tables["defend_alert_budget.csv"].iloc[0]
    for k in ("warnings_A2plus_per_1000", "holds_A3_per_1000", "analyst_capacity_k_per_1000", "abstention_rate"):
        add(f"budget.{k}", k, b[k], "defend_alert_budget.csv", k, "measured" if "capacity" not in k else "derived",
            b["n_rows"], b["n_positives"], b["reportable"])
    hn = tables["eval_hard_negative_fpr.csv"].set_index("population")
    for p in ("hard_negative_all", "shape_all", "benign"):
        if p in hn.index:
            add(f"hard_negative.{p}.fpr_A2plus", f"FPR at A2+ on {p}", hn.loc[p, "fpr_A2plus"], "eval_hard_negative_fpr.csv",
                f"population={p}", "measured", hn.loc[p, "n_rows"], 0, st.reportable)
    for _, r in tables["eval_baselines.csv"].iterrows():
        add(f"baseline.{r['model']}", f"recall at 0.5% FPR, {r['model']}", r["recall_at_0.5%_fpr"], "eval_baselines.csv",
            f"model={r['model']}", "measured", r["n_rows"], r["n_positives"], r["reportable"])
    cs = tables["defend_cost_summary.csv"].set_index("scenario").loc["assumed"]
    add("cost.net_saving_share", "net saving vs approve-everything (assumed costs)", cs["net_saving_share"],
        "defend_cost_summary.csv", "scenario=assumed", "derived", cs["n_rows"], cs["n_positives"], cs["reportable"])
    add("design.A2_budget", "A2+ warning budget (design target)", st.bundle.ladder.budgets["A2"], "configs/default.yaml",
        "ladder_budgets.A2_max_share", "design-only", np.nan, np.nan, True)
    add("design.A3_budget", "A3 hold budget (design target)", st.bundle.ladder.budgets["A3"], "configs/default.yaml",
        "ladder_budgets.A3_max_share", "design-only", np.nan, np.nan, True)
    return pd.DataFrame(rows)


def evaluate(ctx: RunCtx) -> dict[str, pd.DataFrame]:
    from ..pipeline import load_features, log
    tx = ctx.read("transactions.parquet")
    X = load_features(ctx).drop(columns=["txn_id"])
    episodes = ctx.read("episodes.parquet")
    bundle = load_bundle(ctx.art / "bundle.pkl")
    sp = make_split(tx, episodes, ctx.cfg, ctx.gen_meta()["t0"])
    y = tx["is_attack"].to_numpy()
    test = tx[sp.test].reset_index(drop=True)
    Xt = X[sp.test].reset_index(drop=True)
    log("evaluate: scoring the test window")
    dec = bundle.decide(Xt, test)
    st = EvalState(ctx, tx, X, episodes, bundle, sp, y, test, dec, Xt, test["is_attack"].to_numpy(),
                   ctx.reportable(), ctx.cfg.eval.min_positives_for_headline, ctx.cfg.eval.min_positives_for_cell,
                   ctx.cfg.eval.bootstrap_resamples)
    out: dict[str, pd.DataFrame] = {}
    out["defend_headline.csv"] = headline(st)
    out["defend_ladder.csv"], out["defend_alert_budget.csv"] = alert_budget(st)
    out["eval_operating_curve.csv"] = operating_curve(st)
    ep, tta = time_to_alert(st)
    out["eval_episode_trace.csv"], out["eval_time_to_alert_summary.csv"] = ep, tta
    out["eval_generalisation.csv"] = generalisation(st)
    out["eval_hard_negative_fpr.csv"] = hard_negative_fpr(st)
    out["eval_per_variant.csv"] = by_group(st, test["scam_variant"].replace("", "benign"), "scam_variant")
    seg = episodes.set_index("episode_id")["segment"].reindex(test["episode_id"]).fillna("benign").reset_index(drop=True)
    out["eval_per_segment.csv"] = by_group(st, seg, "segment")
    out["eval_per_evasion.csv"] = by_group(st, test["evasion_technique"].replace("", "none"), "evasion")
    out["eval_worst_slices.csv"] = worst_slices(st)
    log("evaluate: baselines + layer ablation")
    out["eval_baselines.csv"] = baselines(st)
    out["eval_layer_ablation.csv"] = layer_ablation(st)
    if ctx.cfg.eval.lovo:
        log("evaluate: leave-one-variant-out")
        out["eval_lovo.csv"] = lovo(st)
    out["eval_sample_size_curve.csv"] = sample_size_curve(st)
    out["eval_fairness.csv"] = fairness(st)
    out["defend_reason_code_usage.csv"] = reason_usage(st)
    out["defend_cost_summary.csv"] = cost_summary(st)
    cap = capacity(st)
    out["defend_capacity.csv"] = pd.DataFrame([cap])
    fid_path = ctx.art / "generate_fidelity_summary.json"
    fid = json.loads(fid_path.read_text()) if fid_path.exists() else {}
    ctrl = []
    for k in ("label_shuffle_null", "leakage_canary"):
        if k in fid:
            ctrl.append({k2: v for k2, v in fid[k].items() if k2 != "_provenance"})
    if "raw_joint_probe" in fid:
        ctrl.append({"control": "raw_joint_probe", **{k: v for k, v in fid["raw_joint_probe"].items()}})
    if "derived_joint_probe" in fid:
        ctrl.append({"control": "derived_joint_probe", **{k: v for k, v in fid["derived_joint_probe"].items()}})
    out["eval_controls.csv"] = pd.DataFrame(ctrl)
    out["eval_evidence_index.csv"] = evidence_index(st, out)
    for name, df in out.items():
        df = df.copy()
        if "reportable" not in df.columns:
            df["reportable"] = st.reportable
        ctx.csv(df, name)
    keep = ["txn_id", "ts", "user_id", "payee_id", "txn_type", "amount_inr", "is_attack", "scam_variant",
            "evasion_technique", "step_in_sequence", "episode_id", "row_kind", "hn_archetype"]
    d = pd.concat([test[keep], dec.drop(columns=["reasons", "l0_fired"])], axis=1)
    d["reasons"] = [",".join(r) for r in dec["reasons"]]
    d["l0_fired"] = [",".join(r) for r in dec["l0_fired"]]
    d.to_parquet(ctx.art / "decisions_test.parquet", index=False)
    from .report import write_report
    write_report(ctx, out, fid)
    h = out["defend_headline.csv"].set_index("metric")
    log(f"evaluate: PR-AUC {h.loc['pr_auc', 'value']:.3f}, recall@0.5%FPR {h.loc['recall_at_0.5%_fpr', 'value']:.3f}, "
        f"n_pos {int(h.loc['pr_auc', 'n_positives'])}, reportable={st.reportable}")
    return out
