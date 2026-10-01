"""REPORT.md — generated from the run's CSVs, never typed. Known limitations are derived from the
diagnostics, so a weakness cannot be quietly left out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..runctx import RunCtx


def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    cols = [c for c in cols if c in df.columns]
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = "".join("| " + " | ".join(fmt(r[c], digits) for c in cols) + " |\n" for _, r in df.iterrows())
    return head + body


def ci(r, v="value", lo="ci_low", hi="ci_high") -> str:
    if pd.isna(r.get(lo)):
        return fmt(r[v])
    return f"{fmt(r[v])} [{fmt(r[lo])}, {fmt(r[hi])}]"


def known_limitations(t: dict[str, pd.DataFrame], fid: dict) -> list[str]:
    out = []
    d = fid.get("derived_joint_probe", {})
    if d:
        out.append(f"**Synthetic-data ceiling.** A time-forward probe on the derived features reaches recall "
                   f"{fmt(d.get('recall_at_fpr'))} at 0.5% FPR (flag threshold {d.get('ceiling')}); deployed systems on "
                   f"real data typically sit at 0.5-0.85. Every number here is an UPPER BOUND on real-world performance.")
    pv = t.get("eval_per_variant.csv")
    if pv is not None:
        low = pv[(pv.value_of != "benign") & (pv.n_sufficient == 0)]
        if len(low):
            out.append("**Low-n variants** (below the per-cell minimum, shown but not claimable): "
                       + ", ".join(f"{r.value_of} (n={r.denominator})" for r in low.itertuples()) + ".")
        sc = pv[pv.value_of != "benign"]
        if len(sc):
            worst = sc.sort_values("value").iloc[0]
            out.append(f"**Weakest variant: {worst.value_of}** — recall at A2+ {fmt(worst.value)} "
                       f"[{fmt(worst.ci_low)}, {fmt(worst.ci_high)}] on {worst.denominator} payments.")
    g = t.get("eval_generalisation.csv")
    if g is not None:
        for _, r in g[g.group.str.startswith("withheld")].iterrows():
            out.append(f"**Generalisation gap ({r['group']}):** recall {fmt(r['value'])} vs seen "
                       f"(gap {fmt(r.get('gap_vs_seen'))}, retained {fmt(r.get('retained_share_of_seen'))}) on "
                       f"{r['denominator']} payments.")
    la = t.get("eval_layer_ablation.csv")
    if la is not None:
        flat = la[la.get("verdict", pd.Series(dtype=str)).isin(["no measured effect", "worsens"])]
        for _, r in flat.iterrows():
            out.append(f"**Layer with no measured benefit:** '{r['step']}' — delta PR-AUC {fmt(r['delta_pr_auc_vs_previous'])} "
                       f"[{fmt(r['delta_ci_low'])}, {fmt(r['delta_ci_high'])}] ({r['verdict']}).")
        scar = la[la.step.str.startswith("ablation")]
        if len(scar):
            out.append(f"**Label scarcity:** trained only on labels visible by the training cutoff, PR-AUC is "
                       f"{fmt(scar.iloc[0]['pr_auc'])} (oracle-label model in the headline). {scar.iloc[0]['verdict']}.")
    bl = t.get("eval_baselines.csv")
    if bl is not None and (bl.model == "citadel_L1_alone").any():
        r = bl[bl.model == "citadel_L1_alone"].iloc[0]
        if r.get("verdict") != "improves":
            out.append(f"**Fusion adds no measured lift over L1 alone:** deployed minus L1-alone recall at 0.5% FPR "
                       f"{fmt(r['citadel_minus_this_recall_0.5pct'])} [{fmt(r['delta_ci_low'])}, {fmt(r['delta_ci_high'])}] "
                       f"({r['verdict']}). The L2 payee channel is kept for readable payee evidence, not for lift.")
    ch = t.get("_channels")
    if ch is not None:
        dead = ch[(ch.slice == "stats") & ch.dead]
        for _, r in dead.iterrows():
            out.append(f"**Dead channel:** {r['channel']} alone has PR-AUC {fmt(r['pr_auc'])} on the stats slice.")
    fa = t.get("_fusion")
    if fa is not None:
        sel = fa[fa.selected]
        if len(sel):
            s = sel.iloc[0]
            best_test = fa.sort_values("test_pr_auc", ascending=False).iloc[0]
            if best_test["arm"] != s["arm"]:
                out.append(f"**Fusion choice is not the test-set best:** '{s['arm']}' was chosen on the stats slice; "
                           f"'{best_test['arm']}' scores higher on test ({fmt(best_test['test_pr_auc'])} vs "
                           f"{fmt(s['test_pr_auc'])}). We keep the pre-registered choice.")
    cs = t.get("defend_cost_summary.csv")
    if cs is not None and len(cs):
        neg = cs[cs.net_saving_inr < 0]
        if len(neg):
            out.append("**Cost sensitivity:** net saving turns NEGATIVE under: " + ", ".join(neg.scenario) + ".")
        else:
            rng_ = (cs.net_saving_share.min(), cs.net_saving_share.max())
            out.append(f"**Cost sensitivity:** net saving share ranges {fmt(rng_[0])}-{fmt(rng_[1])} across the "
                       f"assumption sweep; every rupee figure is simulator-internal.")
    fr = t.get("eval_fairness.csv")
    if fr is not None:
        f = fr[fr.flag_fpr_disparity & (fr.n_rows >= 500)]
        for _, r in f.iterrows():
            out.append(f"**FPR disparity:** {r['group']}={r['value_of']} has A2+ FPR {fmt(r['fpr_A2plus'], 4)} "
                       f"({fmt(r['fpr_ratio_to_overall'], 2)}x overall).")
    hn = t.get("eval_hard_negative_fpr.csv")
    if hn is not None:
        h = hn.set_index("population")
        if "hard_negative_all" in h.index and "benign" in h.index:
            out.append(f"**Customer harm concentrates on hard negatives:** A2+ FPR {fmt(h.loc['hard_negative_all', 'fpr_A2plus'], 4)} "
                       f"vs {fmt(h.loc['benign', 'fpr_A2plus'], 4)} on ordinary benign traffic.")
    out.append("**What the generator cannot know:** lures (S1) are unobserved; mule behaviour, label latency, heed "
               "rates and costs are assumptions; the grammar covers 6 variants and 6 evasions only — anything "
               "outside it is untested.")
    return out


def write_report(ctx: RunCtx, t: dict[str, pd.DataFrame], fid: dict) -> None:
    t = dict(t)
    for name, key in (("defend_channels_alone.csv", "_channels"), ("defend_fusion_arms.csv", "_fusion"),
                      ("defend_split.csv", "_split")):
        p = ctx.art / name
        if p.exists():
            t[key] = pd.read_csv(p)
    gs = ctx.gen_meta()
    rep = ctx.reportable()
    L = [f"# Citadel run report — `{ctx.run}`\n",
         f"{ctx.prov.line()}\n",
         "> Generated by `python -m citadel evaluate`. Every number below is read from a CSV in this directory. "
         "100% synthetic data: results are an upper bound on real-world performance.\n"]
    if not rep:
        L.append("> **NOT REPORTABLE.** This is a reduced-scale profile; numbers are smoke-test only and must not "
                 "enter the PDF.\n")
    L.append("## Data\n")
    L.append(f"- Payments: {gs['n_rows']:,} over {gs['days']} days; customers {gs['n_users']:,}; merchants "
             f"{gs['n_merchants']:,}; mule accounts {gs['n_mules']:,}\n"
             f"- Scam episodes: {gs['n_episodes']:,}; realised attack share {gs['realised_attack_share']:.3%} "
             f"(target {gs['target_attack_share']:.1%}); episodes with a visible label {gs['label_visible_share_episodes']:.1%}\n"
             f"- Row kinds: " + ", ".join(f"{k} {v:,}" for k, v in gs["row_kinds"].items()) + "\n")
    if "_split" in t:
        L.append("\n### Time-forward split (purge + embargo, sealed holdout)\n\n")
        L.append(md_table(t["_split"], ["slice", "n_rows", "n_positives", "day_from", "day_to"]))
    L.append("\n## Fidelity and leakage gates\n\n")
    if fid:
        L.append(f"- Max single-feature AUC: {fmt(fid.get('single_feature_max_auc'))} (gate {fid.get('single_feature_gate')}); "
                 f"artefacts found: {fid.get('artefacts_found')}\n")
        for k in ("raw_joint_probe", "derived_joint_probe"):
            if k in fid:
                v = fid[k]
                L.append(f"- {k.replace('_', ' ')}: recall {fmt(v['recall_at_fpr'])} at 0.5% FPR (refit spread "
                         f"{fmt(v['refit_spread'])}, ceiling {v['ceiling']}, flagged={v['flag_measuring_generator']})\n")
        for k in ("label_shuffle_null", "leakage_canary"):
            if k in fid:
                L.append(f"- {k}: passes={fid[k]['passes']} ({', '.join(f'{a}={fmt(b)}' for a, b in fid[k].items() if a not in ('control', 'passes', '_provenance'))})\n")
    L.append("\n## Headline (test window)\n\n")
    h = t["defend_headline.csv"].copy()
    h["value [95% CI]"] = [ci(r) for _, r in h.iterrows()]
    L.append(md_table(h, ["metric", "value [95% CI]", "n_rows", "n_positives", "headline", "reportable", "note"]))
    L.append("\n## Alert budget and capacity\n\n")
    L.append(md_table(t["defend_ladder.csv"], ["action", "n_alerts", "per_1000_payments", "scams_at_level",
                                              "legit_at_level", "budget_share", "realised_share"], 4))
    b = t["defend_alert_budget.csv"].T.reset_index()
    b.columns = ["quantity", "value"]
    L.append("\n" + md_table(b, ["quantity", "value"], 4))
    L.append("\n## Time to alert within the scam sequence\n\n")
    tt = t["eval_time_to_alert_summary.csv"]
    if len(tt):
        tt = tt.copy()
        tt["value [95% CI]"] = [ci(r) for _, r in tt.iterrows()]
        L.append(md_table(tt, ["metric", "value [95% CI]", "numerator", "denominator", "n_sufficient"]))
    L.append("\n## Generalisation (sealed holdout)\n\n")
    g = t["eval_generalisation.csv"].copy()
    g["recall A2+ [95% CI]"] = [ci(r) for _, r in g.iterrows()]
    L.append(md_table(g, ["group", "recall A2+ [95% CI]", "recall_at_0.5pct_fpr", "n_positives", "gap_vs_seen",
                          "retained_share_of_seen", "reportable"]))
    if "eval_lovo.csv" in t:
        L.append("\n### Leave-one-variant-out (L1 retrained without the variant)\n\n")
        L.append(md_table(t["eval_lovo.csv"], ["variant", "recall_at_0.5pct_fpr_when_withheld", "ci_low", "ci_high",
                                               "recall_at_0.5pct_fpr_when_trained", "retained_share", "n_positives"]))
    L.append("\n## Where customer harm lands: hard negatives and shapes\n\n")
    L.append(md_table(t["eval_hard_negative_fpr.csv"], ["population", "n_rows", "fpr_A1plus", "fpr_A2plus", "fpr_A3",
                                                         "n_A2plus"], 4))
    L.append("\n## Per variant / evasion / segment (worst first)\n\n")
    for name in ("eval_per_variant.csv", "eval_per_evasion.csv", "eval_per_segment.csv"):
        d = t[name].copy()
        d["recall A2+ [95% CI]"] = [ci(r) for _, r in d.iterrows()]
        L.append(md_table(d, ["value_of", "recall A2+ [95% CI]", "denominator", "fpr_A2plus", "n_rows", "n_sufficient"], 4) + "\n")
    L.append("\n### Worst slices (automated mining, >= 25 rows)\n\n")
    L.append(md_table(t["eval_worst_slices.csv"].head(12), ["slice", "value", "recall_A2plus", "ci_low", "ci_high",
                                                            "fpr_A2plus", "n_rows", "n_positives", "n_sufficient"], 4))
    L.append("\n## Baselines (same rows, same FPR budget)\n\n")
    L.append(md_table(t["eval_baselines.csv"], ["model", "pr_auc", "recall_at_0.1%_fpr", "recall_at_0.5%_fpr",
                                                "citadel_minus_this_recall_0.5pct", "delta_ci_low", "delta_ci_high",
                                                "verdict"]))
    L.append("\n## Layer ablation (negative rows published)\n\n")
    L.append(md_table(t["eval_layer_ablation.csv"], ["step", "n_features", "pr_auc", "recall_at_0.5pct_fpr",
                                                     "delta_pr_auc_vs_previous", "delta_ci_low", "delta_ci_high", "verdict"]))
    if "_channels" in t:
        L.append("\n## Channels alone and fusion arms\n\n")
        L.append(md_table(t["_channels"], ["slice", "channel", "pr_auc", "roc_auc", "lift_over_base", "dead"]))
        L.append("\n" + md_table(t["_fusion"], ["arm", "channels", "stats_pr_auc", "test_pr_auc",
                                                "test_recall_at_0.5pct_fpr", "ladder_non_degenerate", "selected"]))
    L.append("\n## Fairness (A2+ by segment)\n\n")
    L.append(md_table(t["eval_fairness.csv"], ["group", "value_of", "value", "denominator", "fpr_A2plus",
                                               "fpr_ratio_to_overall", "flag_fpr_disparity", "n_rows"], 4))
    L.append("\n## Reason codes\n\n")
    L.append(md_table(t["defend_reason_code_usage.csv"], ["reason_code", "uses_on_scams", "uses_on_legit",
                                                          "share_of_alerts", "fallback_rate"]))
    L.append("\n## Cost model (ASSUMPTIONS, swept)\n\n")
    L.append(md_table(t["defend_cost_summary.csv"], ["scenario", "approve_everything_inr", "citadel_total_inr",
                                                     "net_saving_inr", "net_saving_share"], 2))
    L.append("\n## Controls\n\n")
    L.append(md_table(t["eval_controls.csv"], [c for c in t["eval_controls.csv"].columns if c not in ("_provenance",)][:8]))
    L.append("\n## Sample-size curve\n\n")
    L.append(md_table(t["eval_sample_size_curve.csv"], ["train_fraction", "n_train_rows", "n_train_positives", "pr_auc",
                                                        "recall_at_0.5pct_fpr"]))
    L.append("\n## Known limitations (generated from the diagnostics)\n\n")
    for s in known_limitations(t, fid):
        L.append(f"- {s}\n")
    L.append("\n## Reporting rules\n\n- Every rate carries its denominator and a Wilson 95% CI; only precision/recall-type "
             "metrics are reported because the base rate is below 1%.\n"
             "- A delta whose CI includes zero is 'no measured effect'.\n- Rupee figures are simulator-internal.\n"
             "- Cells under the minimum positive count stay visible with n_sufficient = 0.\n")
    (ctx.art / "REPORT.md").write_text("".join(L), encoding="utf-8")
