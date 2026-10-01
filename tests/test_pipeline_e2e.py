"""End-to-end gates on the tiny run: artefacts exist, every metric has a denominator, controls fire,
reduced-scale numbers are stamped not reportable, and no accuracy figure appears anywhere."""

import json

import pandas as pd

REQUIRED = ["defend_headline.csv", "defend_split.csv", "defend_channels_alone.csv", "defend_fusion_arms.csv",
            "defend_ladder.csv", "defend_alert_budget.csv", "defend_cost_summary.csv", "defend_reason_code_usage.csv",
            "eval_generalisation.csv", "eval_hard_negative_fpr.csv", "eval_baselines.csv", "eval_layer_ablation.csv",
            "eval_time_to_alert_summary.csv", "eval_fairness.csv", "eval_evidence_index.csv", "eval_controls.csv",
            "generate_single_feature_auc.csv", "generate_artefact_hunt.csv", "generate_fidelity_scores.csv",
            "feature_registry.csv", "REPORT.md", "bundle.pkl"]


def test_all_artefacts_written(tiny_run):
    missing = [f for f in REQUIRED if not (tiny_run.art / f).exists()]
    assert not missing, missing


def test_every_csv_carries_the_run_id(tiny_run):
    for p in tiny_run.art.glob("*.csv"):
        df = pd.read_csv(p)
        assert "run_id" in df.columns and (df.run_id == tiny_run.prov.run_id).all(), p.name


def test_headline_has_denominators_and_no_accuracy(tiny_run):
    h = pd.read_csv(tiny_run.art / "defend_headline.csv")
    assert {"n_rows", "n_positives"} <= set(h.columns)
    assert not h.metric.str.contains("accuracy").any()
    rep = (tiny_run.art / "REPORT.md").read_text(encoding="utf-8").lower()
    assert "accuracy" not in rep


def test_reduced_scale_numbers_are_not_reportable(tiny_run):
    h = pd.read_csv(tiny_run.art / "defend_headline.csv")
    assert not h.reportable.any()
    assert "NOT REPORTABLE" in (tiny_run.art / "REPORT.md").read_text(encoding="utf-8")


def test_controls_fire(tiny_run):
    fid = json.loads((tiny_run.art / "generate_fidelity_summary.json").read_text())
    assert fid["leakage_canary"]["single_probe_fires"]
    assert abs(fid["label_shuffle_null"]["roc_auc"] - 0.5) < 0.15
    assert fid["single_feature_passes"] and fid["artefacts_found"] == 0


def test_layer_ablation_publishes_every_row(tiny_run):
    la = pd.read_csv(tiny_run.art / "eval_layer_ablation.csv")
    assert len(la) >= 7 and la.step.str.contains("visible matured labels").any()


def test_fusion_choice_never_uses_the_test_window(tiny_run):
    fa = pd.read_csv(tiny_run.art / "defend_fusion_arms.csv")
    assert fa.selected.sum() == 1
    eligible = fa[fa.ladder_non_degenerate]
    assert fa[fa.selected].stats_pr_auc.iloc[0] == eligible.stats_pr_auc.max()
