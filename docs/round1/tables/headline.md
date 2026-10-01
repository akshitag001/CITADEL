<!-- generated from artifacts/<run>/defend_headline.csv; do not edit -->
| metric | value | ci_low | ci_high | n_rows | n_positives | reportable | note |
|---|---|---|---|---|---|---|---|
| pr_auc | 0.767 | 0.725 | 0.801 | 66,615 | 432 | yes | lift over base rate 118.3x (base 0.6485%) |
| recall_at_0.1%_fpr | 0.662 | 0.616 | 0.705 | 66,615 | 432 | yes | realised FPR 0.0997% (66/66183) |
| recall_at_0.5%_fpr | 0.801 | 0.761 | 0.836 | 66,615 | 432 | yes | realised FPR 0.4986% (330/66183) |
| partial_auc_fpr_0_2pct_mcclish | 0.908 | 0.892 | 0.923 | 66,615 | 432 | yes | McClish-standardised |
| precision_at_k | 1.000 | 0.964 | 1.000 | 66,615 | 432 | yes | k=102 from staffing; recall at k = 0.236 |
| value_weighted_recall_A2plus | 0.913 | — | — | 66,615 | 432 | yes | share of scam INR at A2+ (warning or hold) |
| recall_at_deployed_A2plus | 0.771 | 0.729 | 0.808 | 66,615 | 432 | yes | at the deployed ladder; A2+ alert share 0.883% |
| roc_auc | 0.988 | — | — | 66,615 | 432 | yes | computed, NOT a headline (dominated by true negatives at a <1% base rate) |
