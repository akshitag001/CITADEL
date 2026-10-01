<!-- generated from artifacts/<run>/eval_layer_ablation.csv; do not edit -->
| step | n_features | pr_auc | recall_at_0.5pct_fpr | delta_pr_auc_vs_previous | delta_ci_low | delta_ci_high | verdict |
|---|---|---|---|---|---|---|---|
| raw schema | 27 | 0.460 | 0.502 | — | — | — | — |
| + user windows | 40 | 0.579 | 0.593 | 0.119 | 0.090 | 0.153 | improves |
| + sequence | 52 | 0.635 | 0.646 | 0.056 | 0.037 | 0.077 | improves |
| + payee windows | 59 | 0.742 | 0.757 | 0.107 | 0.080 | 0.133 | improves |
| + graph | 64 | 0.774 | 0.792 | 0.032 | 0.020 | 0.047 | improves |
| + L0 rules as features | 69 | 0.774 | 0.792 | 0.000 | 0.000 | 0.000 | no measured effect |
| ablation: visible matured labels only (no oracle) | 64 | 0.548 | 0.600 | — | — | — | trained on 257 visible positives vs 818 oracle; evaluated on oracle truth |
