"""The numbers registry: every number the Round 1 PDF may quote, read from a pipeline CSV.

Nobody types a metric. The deck and the brief reference numbers by key; ``verify`` re-derives every
value from the CSVs and fails if anything drifted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Num:
    key: str
    value: float
    display: str
    artefact: str
    selector: str
    tier: str            # measured | derived | design-only
    n_rows: float
    n_positives: float
    reportable: bool
    description: str


def _pct(v: float, d: int = 1) -> str:
    return "—" if not np.isfinite(v) else f"{100 * v:.{d}f}%"


def _f(v: float, d: int = 3) -> str:
    return "—" if not np.isfinite(v) else f"{v:.{d}f}"


def build_numbers(art: Path, gen_summary: dict, fid: dict, reportable: bool) -> dict[str, Num]:
    rd = lambda n: pd.read_csv(art / n)  # noqa: E731
    out: dict[str, Num] = {}

    def add(key, value, display, artefact, selector, tier, n_rows=np.nan, n_pos=np.nan, rep=reportable, desc=""):
        out[key] = Num(key, float(value) if value is not None else float("nan"), display, artefact, selector, tier,
                       float(n_rows), float(n_pos), bool(rep), desc)

    h = rd("defend_headline.csv").set_index("metric")
    for m, key, d in (("pr_auc", "pr_auc", "PR-AUC on the test window"),
                      ("recall_at_0.5%_fpr", "recall_05", "recall at 0.5% FPR"),
                      ("recall_at_0.1%_fpr", "recall_01", "recall at 0.1% FPR"),
                      ("recall_at_deployed_A2plus", "recall_A2", "scam payments warned or held at the deployed ladder"),
                      ("precision_at_k", "precision_k", "precision of the top-k (k from analyst staffing)"),
                      ("value_weighted_recall_A2plus", "value_recall_A2", "share of scam rupees at A2+")):
        if m not in h.index:
            continue
        r = h.loc[m]
        rep = bool(r["reportable"])
        is_pct = m != "pr_auc"
        disp = (_pct(r["value"]) if is_pct else _f(r["value"]))
        if np.isfinite(r.get("ci_low", np.nan)):
            disp += f" (95% CI {(_pct if is_pct else _f)(r['ci_low'])}–{(_pct if is_pct else _f)(r['ci_high'])})"
        add(f"headline.{key}", r["value"], disp, "defend_headline.csv", f"metric={m}", "measured",
            r["n_rows"], r["n_positives"], rep, d)
    base = h.loc["pr_auc", "n_positives"] / h.loc["pr_auc", "n_rows"]
    add("headline.base_rate", base, _pct(base, 2), "defend_headline.csv", "n_positives/n_rows", "derived",
        h.loc["pr_auc", "n_rows"], h.loc["pr_auc", "n_positives"], reportable, "scam share of test payments")
    add("headline.lift", h.loc["pr_auc", "value"] / base, f"{h.loc['pr_auc', 'value'] / base:.0f}x", "defend_headline.csv",
        "pr_auc / base rate", "derived", h.loc["pr_auc", "n_rows"], h.loc["pr_auc", "n_positives"],
        bool(h.loc["pr_auc", "reportable"]), "PR-AUC lift over the base rate")
    add("data.n_test_rows", h.loc["pr_auc", "n_rows"], f"{int(h.loc['pr_auc', 'n_rows']):,}", "defend_headline.csv",
        "n_rows", "measured", desc="test-window payments")
    add("data.n_test_pos", h.loc["pr_auc", "n_positives"], f"{int(h.loc['pr_auc', 'n_positives']):,}",
        "defend_headline.csv", "n_positives", "measured", desc="test-window scam payments")

    b = rd("defend_alert_budget.csv").iloc[0]
    for k, d, fmt in (("warnings_A2plus_per_1000", "warnings (A2+) per 1,000 payments", "{:.1f}"),
                      ("holds_A3_per_1000", "holds (A3) per 1,000 payments", "{:.2f}"),
                      ("analyst_capacity_k_per_1000", "analyst capacity per 1,000 payments", "{:.2f}"),
                      ("abstention_rate", "share routed to a human because layers disagree", "{:.3%}")):
        tier = "derived" if "capacity" in k else "measured"
        add(f"budget.{k}", b[k], fmt.format(b[k]), "defend_alert_budget.csv", k, tier, b["n_rows"], b["n_positives"],
            bool(b["reportable"]), d)
    add("budget.holds_within_capacity", float(bool(b["holds_within_capacity"])),
        "yes" if bool(b["holds_within_capacity"]) else "NO", "defend_alert_budget.csv", "holds_within_capacity",
        "derived", desc="holds fit analyst capacity")

    t = rd("eval_time_to_alert_summary.csv").set_index("metric")
    for m, key in (("episodes_caught_any_A2plus", "caught_any"), ("episodes_caught_before_first_loss", "before_first_loss"),
                   ("episodes_caught_by_first_harmful_payment", "by_first_harmful"),
                   ("value_saved_share_heed_adjusted", "value_saved"), ("value_saved_share_upper_bound", "value_saved_ub")):
        if m in t.index:
            r = t.loc[m]
            disp = _pct(r["value"]) + (f" (95% CI {_pct(r['ci_low'])}–{_pct(r['ci_high'])})" if np.isfinite(r["ci_low"]) else "")
            add(f"tta.{key}", r["value"], disp, "eval_time_to_alert_summary.csv", f"metric={m}",
                "derived" if "saved" in m else "measured", r["denominator"], r["n_episodes"], bool(r["reportable"]),
                m.replace("_", " "))

    g = rd("eval_generalisation.csv").set_index("group")
    for grp in g.index:
        r = g.loc[grp]
        key = ("withheld_variant" if grp.startswith("withheld_variant") else "withheld_evasion"
               if grp.startswith("withheld_evasion") else grp)
        add(f"gen.{key}", r["value"], f"{_pct(r['value'])} (95% CI {_pct(r['ci_low'])}–{_pct(r['ci_high'])}, n={int(r['denominator'])})",
            "eval_generalisation.csv", f"group={grp}", "measured", r["denominator"], r["n_positives"],
            bool(r["reportable"]), f"recall at A2+ for {grp}")
        if np.isfinite(r.get("retained_share_of_seen", np.nan)):
            add(f"gen.{key}_retained", r["retained_share_of_seen"], _pct(r["retained_share_of_seen"], 0),
                "eval_generalisation.csv", f"group={grp}:retained_share_of_seen", "derived", r["denominator"],
                r["n_positives"], bool(r["reportable"]), "recall retained vs the seen comparison group")

    hn = rd("eval_hard_negative_fpr.csv").set_index("population")
    for p in ("benign", "hard_negative_all", "shape_all", "all_legitimate"):
        if p in hn.index:
            r = hn.loc[p]
            add(f"fpr.{p}", r["fpr_A2plus"], _pct(r["fpr_A2plus"], 2), "eval_hard_negative_fpr.csv",
                f"population={p}:fpr_A2plus", "measured", r["n_rows"], 0, reportable, f"A2+ false-positive rate, {p}")

    bl = rd("eval_baselines.csv").set_index("model")
    for m, key in (("payee_is_first_time_alone", "first_time"), ("ten_expert_rules_count", "expert_rules"),
                   ("logistic_regression_all_features", "logreg"), ("citadel_deployed (fused, calibrated)", "citadel"),
                   ("citadel_L1_alone", "l1_alone")):
        if m in bl.index:
            r = bl.loc[m]
            add(f"baseline.{key}", r["recall_at_0.5%_fpr"], _pct(r["recall_at_0.5%_fpr"]), "eval_baselines.csv",
                f"model={m}:recall_at_0.5%_fpr", "measured", r["n_rows"], r["n_positives"], bool(r["reportable"]),
                f"recall at 0.5% FPR, {m}")

    if "citadel_L1_alone" in bl.index:
        r = bl.loc["citadel_L1_alone"]
        add("fusion.vs_l1_alone_verdict", float(r["citadel_minus_this_recall_0.5pct"]),
            f"{r['verdict']} (deployed minus L1 alone: {r['citadel_minus_this_recall_0.5pct']:+.3f} recall, "
            f"95% CI {r['delta_ci_low']:+.3f} to {r['delta_ci_high']:+.3f})", "eval_baselines.csv",
            "model=citadel_L1_alone:citadel_minus_this_recall_0.5pct", "measured", r["n_rows"], r["n_positives"],
            bool(r["reportable"]), "does fusing L2 + isolation forest beat L1 alone?")
    la = rd("eval_layer_ablation.csv")
    raw = la.iloc[0]
    full = la[la.step == "+ graph"].iloc[0] if (la.step == "+ graph").any() else la.iloc[-2]
    add("ablation.raw_pr_auc", raw["pr_auc"], _f(raw["pr_auc"]), "eval_layer_ablation.csv", "step=raw schema", "measured",
        raw["n_rows"], raw["n_positives"], bool(raw["reportable"]), "PR-AUC, raw schema columns only")
    add("ablation.full_pr_auc", full["pr_auc"], _f(full["pr_auc"]), "eval_layer_ablation.csv", "step=+ graph", "measured",
        full["n_rows"], full["n_positives"], bool(full["reportable"]), "PR-AUC, all feature families")
    scar = la[la.step.str.startswith("ablation")]
    if len(scar):
        add("ablation.visible_labels_pr_auc", scar.iloc[0]["pr_auc"], _f(scar.iloc[0]["pr_auc"]), "eval_layer_ablation.csv",
            "step=ablation: visible matured labels only", "measured", scar.iloc[0]["n_rows"], scar.iloc[0]["n_positives"],
            bool(scar.iloc[0]["reportable"]), "PR-AUC when trained only on labels visible by the cutoff")
    n_neg = int(la.get("verdict", pd.Series(dtype=str)).isin(["no measured effect", "worsens"]).sum())
    add("ablation.n_no_effect_rows", n_neg, str(n_neg), "eval_layer_ablation.csv", "verdict in {no measured effect, worsens}",
        "measured", desc="ablation steps with no measured benefit (published)")

    cs = rd("defend_cost_summary.csv").set_index("scenario")
    add("cost.net_saving_share", cs.loc["assumed", "net_saving_share"], _pct(cs.loc["assumed", "net_saving_share"]),
        "defend_cost_summary.csv", "scenario=assumed", "derived", cs.loc["assumed", "n_rows"],
        cs.loc["assumed", "n_positives"], bool(cs.loc["assumed", "reportable"]), "net loss reduction vs approve-everything (assumed costs)")
    add("cost.net_saving_share_min", cs["net_saving_share"].min(), _pct(cs["net_saving_share"].min()),
        "defend_cost_summary.csv", "min over scenarios", "derived", desc="worst case across the cost sweep")

    fr = rd("eval_fairness.csv")
    add("fairness.max_fpr_ratio", fr["fpr_ratio_to_overall"].max(), f"{fr['fpr_ratio_to_overall'].max():.2f}x",
        "eval_fairness.csv", "max fpr_ratio_to_overall", "measured", desc="worst segment A2+ FPR relative to overall")
    add("fairness.n_flags", int(fr["flag_fpr_disparity"].sum()), str(int(fr["flag_fpr_disparity"].sum())),
        "eval_fairness.csv", "sum flag_fpr_disparity", "measured", desc="segments over the 2x FPR tolerance")

    ru = rd("defend_reason_code_usage.csv")
    if len(ru):
        add("reasons.fallback_rate", ru["fallback_rate"].iloc[0], _pct(ru["fallback_rate"].iloc[0]),
            "defend_reason_code_usage.csv", "fallback_rate", "measured", desc="alerts needing the generic reason card")

    add("data.n_rows", gen_summary["n_rows"], f"{gen_summary['n_rows']:,}", "generate_summary.json", "n_rows", "measured",
        desc="synthetic payments generated")
    add("data.days", gen_summary["days"], str(gen_summary["days"]), "generate_summary.json", "days", "measured")
    add("data.n_episodes", gen_summary["n_episodes"], f"{gen_summary['n_episodes']:,}", "generate_summary.json",
        "n_episodes", "measured", desc="scam episodes")
    add("data.n_users", gen_summary["n_users"], f"{gen_summary['n_users']:,}", "generate_summary.json", "n_users", "measured")
    add("data.attack_share", gen_summary["realised_attack_share"], _pct(gen_summary["realised_attack_share"], 2),
        "generate_summary.json", "realised_attack_share", "measured", desc="realised scam share of payments")
    add("fidelity.max_single_auc", fid.get("single_feature_max_auc", np.nan), _f(fid.get("single_feature_max_auc", np.nan)),
        "generate_fidelity_summary.json", "single_feature_max_auc", "measured", desc="best single raw column AUC (gate 0.95)")
    if "derived_joint_probe" in fid:
        add("fidelity.derived_probe", fid["derived_joint_probe"]["recall_at_fpr"],
            _pct(fid["derived_joint_probe"]["recall_at_fpr"]), "generate_fidelity_summary.json",
            "derived_joint_probe.recall_at_fpr", "measured", desc="separability ceiling probe (flag above 92%)")
        add("fidelity.raw_probe", fid["raw_joint_probe"]["recall_at_fpr"], _pct(fid["raw_joint_probe"]["recall_at_fpr"]),
            "generate_fidelity_summary.json", "raw_joint_probe.recall_at_fpr", "measured", desc="raw-schema probe")
    lat = art / "serve_latency.csv"
    if lat.exists():
        L = pd.read_csv(lat).set_index("path")
        for path in L.index:
            for q in ("p50_ms", "p95_ms", "p99_ms"):
                add(f"latency.{path}.{q}", L.loc[path, q], f"{L.loc[path, q]:.0f} ms", "serve_latency.csv",
                    f"path={path}:{q}", "measured", L.loc[path, "n"], 0, True, f"{path} {q}")
    add("design.A2_budget", 0.01, "1%", "configs/default.yaml", "ladder_budgets.A2_max_share", "design-only",
        rep=True, desc="design cap on warnings")
    add("design.A3_budget", 0.001, "0.1%", "configs/default.yaml", "ladder_budgets.A3_max_share", "design-only",
        rep=True, desc="design cap on holds")
    return out


def save(nums: dict[str, Num], path: Path) -> None:
    path.write_text(json.dumps({k: asdict(v) for k, v in nums.items()}, indent=1, default=float), encoding="utf-8")


def verify(saved_path: Path, fresh: dict[str, Num]) -> list[str]:
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    problems = []
    for k, s in saved.items():
        if k not in fresh:
            problems.append(f"{k}: no longer produced by the pipeline")
            continue
        a, b = s["value"], fresh[k].value
        if not ((np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9 * max(1, abs(a))):
            problems.append(f"{k}: PDF says {s['display']} but {fresh[k].artefact} now gives {fresh[k].display}")
    return problems
