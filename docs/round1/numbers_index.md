# Numbers index

<!-- generated; verify with `python scripts/verify_numbers.py` -->

Every number the Round 1 PDF may quote. Tiers: **measured** (computed on the test window), **derived** (computed from measured numbers plus a stated assumption), **design-only** (a target, never a result). Only reportable rows may be quoted.

| key | display | tier | artefact | selector | n_rows | n_positives | reportable |
|---|---|---|---|---|---|---|---|
| headline.pr_auc | 0.767 (95% CI 0.725–0.801) | measured | defend_headline.csv | metric=pr_auc | 66615 | 432 | yes |
| headline.recall_05 | 80.1% (95% CI 76.1%–83.6%) | measured | defend_headline.csv | metric=recall_at_0.5%_fpr | 66615 | 432 | yes |
| headline.recall_01 | 66.2% (95% CI 61.6%–70.5%) | measured | defend_headline.csv | metric=recall_at_0.1%_fpr | 66615 | 432 | yes |
| headline.recall_A2 | 77.1% (95% CI 72.9%–80.8%) | measured | defend_headline.csv | metric=recall_at_deployed_A2plus | 66615 | 432 | yes |
| headline.precision_k | 100.0% (95% CI 96.4%–100.0%) | measured | defend_headline.csv | metric=precision_at_k | 66615 | 432 | yes |
| headline.value_recall_A2 | 91.3% | measured | defend_headline.csv | metric=value_weighted_recall_A2plus | 66615 | 432 | yes |
| headline.base_rate | 0.65% | derived | defend_headline.csv | n_positives/n_rows | 66615 | 432 | yes |
| headline.lift | 118x | derived | defend_headline.csv | pr_auc / base rate | 66615 | 432 | yes |
| data.n_test_rows | 66,615 | measured | defend_headline.csv | n_rows | — | — | yes |
| data.n_test_pos | 432 | measured | defend_headline.csv | n_positives | — | — | yes |
| budget.warnings_A2plus_per_1000 | 8.8 | measured | defend_alert_budget.csv | warnings_A2plus_per_1000 | 66615 | 432 | yes |
| budget.holds_A3_per_1000 | 1.74 | measured | defend_alert_budget.csv | holds_A3_per_1000 | 66615 | 432 | yes |
| budget.analyst_capacity_k_per_1000 | 1.54 | derived | defend_alert_budget.csv | analyst_capacity_k_per_1000 | 66615 | 432 | yes |
| budget.abstention_rate | 0.056% | measured | defend_alert_budget.csv | abstention_rate | 66615 | 432 | yes |
| budget.holds_within_capacity | NO | derived | defend_alert_budget.csv | holds_within_capacity | — | — | yes |
| tta.caught_any | 88.8% (95% CI 83.1%–92.7%) | measured | eval_time_to_alert_summary.csv | metric=episodes_caught_any_A2plus | 169 | 169 | yes |
| tta.before_first_loss | 69.2% (95% CI 61.9%–75.7%) | measured | eval_time_to_alert_summary.csv | metric=episodes_caught_before_first_loss | 169 | 169 | yes |
| tta.by_first_harmful | 76.3% (95% CI 69.4%–82.1%) | measured | eval_time_to_alert_summary.csv | metric=episodes_caught_by_first_harmful_payment | 169 | 169 | yes |
| tta.value_saved | 58.6% | derived | eval_time_to_alert_summary.csv | metric=value_saved_share_heed_adjusted | 1505482 | 169 | yes |
| tta.value_saved_ub | 93.4% | derived | eval_time_to_alert_summary.csv | metric=value_saved_share_upper_bound | 1505482 | 169 | yes |
| gen.withheld_variant | 27.3% (95% CI 16.3%–41.8%, n=44) | measured | eval_generalisation.csv | group=withheld_variant_V4 | 44 | 44 | yes |
| gen.withheld_variant_retained | 33% | derived | eval_generalisation.csv | group=withheld_variant_V4:retained_share_of_seen | 44 | 44 | yes |
| gen.seen_variants | 82.7% (95% CI 78.7%–86.2%, n=388) | measured | eval_generalisation.csv | group=seen_variants | 388 | 388 | yes |
| gen.withheld_evasion | 87.5% (95% CI 52.9%–97.8%, n=8) | measured | eval_generalisation.csv | group=withheld_evasion_signal_suppression | 8 | 8 | no |
| gen.withheld_evasion_retained | 112% | derived | eval_generalisation.csv | group=withheld_evasion_signal_suppression:retained_share_of_seen | 8 | 8 | no |
| gen.no_evasion | 77.8% (95% CI 72.5%–82.4%, n=266) | measured | eval_generalisation.csv | group=no_evasion | 266 | 266 | yes |
| gen.seen_evasions | 75.3% (95% CI 68.0%–81.4%, n=158) | measured | eval_generalisation.csv | group=seen_evasions | 158 | 158 | yes |
| fpr.benign | 0.21% | measured | eval_hard_negative_fpr.csv | population=benign:fpr_A2plus | 61442 | 0 | yes |
| fpr.hard_negative_all | 3.29% | measured | eval_hard_negative_fpr.csv | population=hard_negative_all:fpr_A2plus | 3683 | 0 | yes |
| fpr.shape_all | 0.19% | measured | eval_hard_negative_fpr.csv | population=shape_all:fpr_A2plus | 1036 | 0 | yes |
| fpr.all_legitimate | 0.39% | measured | eval_hard_negative_fpr.csv | population=all_legitimate:fpr_A2plus | 66183 | 0 | yes |
| baseline.first_time | 3.2% | measured | eval_baselines.csv | model=payee_is_first_time_alone:recall_at_0.5%_fpr | 66615 | 432 | yes |
| baseline.expert_rules | 4.6% | measured | eval_baselines.csv | model=ten_expert_rules_count:recall_at_0.5%_fpr | 66615 | 432 | yes |
| baseline.logreg | 47.5% | measured | eval_baselines.csv | model=logistic_regression_all_features:recall_at_0.5%_fpr | 66615 | 432 | yes |
| baseline.citadel | 80.1% | measured | eval_baselines.csv | model=citadel_deployed (fused, calibrated):recall_at_0.5%_fpr | 66615 | 432 | yes |
| baseline.l1_alone | 79.2% | measured | eval_baselines.csv | model=citadel_L1_alone:recall_at_0.5%_fpr | 66615 | 432 | yes |
| fusion.vs_l1_alone_verdict | no measured effect (deployed minus L1 alone: +0.009 recall, 95% CI -0.014 to +0.024) | measured | eval_baselines.csv | model=citadel_L1_alone:citadel_minus_this_recall_0.5pct | 66615 | 432 | yes |
| ablation.raw_pr_auc | 0.460 | measured | eval_layer_ablation.csv | step=raw schema | 66615 | 432 | yes |
| ablation.full_pr_auc | 0.774 | measured | eval_layer_ablation.csv | step=+ graph | 66615 | 432 | yes |
| ablation.visible_labels_pr_auc | 0.548 | measured | eval_layer_ablation.csv | step=ablation: visible matured labels only | 66615 | 432 | yes |
| ablation.n_no_effect_rows | 1 | measured | eval_layer_ablation.csv | verdict in {no measured effect, worsens} | — | — | yes |
| cost.net_saving_share | 57.6% | derived | defend_cost_summary.csv | scenario=assumed | 66615 | 432 | yes |
| cost.net_saving_share_min | 23.8% | derived | defend_cost_summary.csv | min over scenarios | — | — | yes |
| fairness.max_fpr_ratio | 3.43x | measured | eval_fairness.csv | max fpr_ratio_to_overall | — | — | yes |
| fairness.n_flags | 1 | measured | eval_fairness.csv | sum flag_fpr_disparity | — | — | yes |
| reasons.fallback_rate | 2.7% | measured | defend_reason_code_usage.csv | fallback_rate | — | — | yes |
| data.n_rows | 398,531 | measured | generate_summary.json | n_rows | — | — | yes |
| data.days | 50 | measured | generate_summary.json | days | — | — | yes |
| data.n_episodes | 898 | measured | generate_summary.json | n_episodes | — | — | yes |
| data.n_users | 25,001 | measured | generate_summary.json | n_users | — | — | yes |
| data.attack_share | 0.56% | measured | generate_summary.json | realised_attack_share | — | — | yes |
| fidelity.max_single_auc | 0.730 | measured | generate_fidelity_summary.json | single_feature_max_auc | — | — | yes |
| fidelity.derived_probe | 83.6% | measured | generate_fidelity_summary.json | derived_joint_probe.recall_at_fpr | — | — | yes |
| fidelity.raw_probe | 51.9% | measured | generate_fidelity_summary.json | raw_joint_probe.recall_at_fpr | — | — | yes |
| latency.stored_txn.server_compute.p50_ms | 24 ms | measured | serve_latency.csv | path=stored_txn.server_compute:p50_ms | 150 | 0 | yes |
| latency.stored_txn.server_compute.p95_ms | 61 ms | measured | serve_latency.csv | path=stored_txn.server_compute:p95_ms | 150 | 0 | yes |
| latency.stored_txn.server_compute.p99_ms | 65 ms | measured | serve_latency.csv | path=stored_txn.server_compute:p99_ms | 150 | 0 | yes |
| latency.stored_txn.round_trip.p50_ms | 26 ms | measured | serve_latency.csv | path=stored_txn.round_trip:p50_ms | 150 | 0 | yes |
| latency.stored_txn.round_trip.p95_ms | 64 ms | measured | serve_latency.csv | path=stored_txn.round_trip:p95_ms | 150 | 0 | yes |
| latency.stored_txn.round_trip.p99_ms | 68 ms | measured | serve_latency.csv | path=stored_txn.round_trip:p99_ms | 150 | 0 | yes |
| latency.online_event.server_compute.p50_ms | 185 ms | measured | serve_latency.csv | path=online_event.server_compute:p50_ms | 150 | 0 | yes |
| latency.online_event.server_compute.p95_ms | 216 ms | measured | serve_latency.csv | path=online_event.server_compute:p95_ms | 150 | 0 | yes |
| latency.online_event.server_compute.p99_ms | 241 ms | measured | serve_latency.csv | path=online_event.server_compute:p99_ms | 150 | 0 | yes |
| latency.online_event.round_trip.p50_ms | 187 ms | measured | serve_latency.csv | path=online_event.round_trip:p50_ms | 150 | 0 | yes |
| latency.online_event.round_trip.p95_ms | 225 ms | measured | serve_latency.csv | path=online_event.round_trip:p95_ms | 150 | 0 | yes |
| latency.online_event.round_trip.p99_ms | 270 ms | measured | serve_latency.csv | path=online_event.round_trip:p99_ms | 150 | 0 | yes |
| design.A2_budget | 1% | design-only | configs/default.yaml | ladder_budgets.A2_max_share | — | — | yes |
| design.A3_budget | 0.1% | design-only | configs/default.yaml | ladder_budgets.A3_max_share | — | — | yes |
