<!-- generated from artifacts/<run>/eval_baselines.csv; do not edit -->
| model | pr_auc | recall_at_0.1%_fpr | recall_at_0.5%_fpr | citadel_minus_this_recall_0.5pct | delta_ci_low | delta_ci_high | verdict |
|---|---|---|---|---|---|---|---|
| citadel_deployed (fused, calibrated) | 0.767 | 0.662 | 0.801 | — | — | — | — |
| citadel_L1_alone | 0.774 | 0.664 | 0.792 | 0.009 | -0.014 | 0.024 | no measured effect |
| logistic_regression_all_features | 0.397 | 0.317 | 0.475 | 0.326 | 0.280 | 0.370 | improves |
| payee_is_first_time_alone | 0.014 | 0.007 | 0.032 | 0.769 | 0.726 | 0.807 | improves |
| ten_expert_rules_count | 0.036 | 0.007 | 0.046 | 0.755 | 0.710 | 0.802 | improves |
