# Preliminary validation — SYNTHETIC DATA

<!-- generated; numbers from numbers.json -->

> All results below come from Citadel's own synthetic generator. They show the design works as
> specified and where it breaks; they are an upper bound on real-world performance, not a claim about it.

**Setup.** 398,531 synthetic UPI payments over 50 days, 898 scam
episodes across six scripts and six evasions. Time-forward split with purge and embargo; one whole variant and
one whole evasion sealed out of training. Test window: 66,615 payments, 432
scam payments (base rate 0.65%).

**Is the data too easy?** Best single raw column AUC 0.730 (gate 0.95). A booster on
raw columns alone reaches 51.9% recall at 0.5% FPR; on all derived features
83.6% (we flag above 92% as "measuring the generator"). Label-shuffle null and a
planted leakage canary both behave as expected.

| Result | Value |
|---|---|
| PR-AUC | 0.767 (95% CI 0.725–0.801) — 118x the base rate |
| Recall at 0.5% / 0.1% false-positive rate | 80.1% (95% CI 76.1%–83.6%) / 66.2% (95% CI 61.6%–70.5%) |
| Scam payments warned or held at the deployed ladder | 77.1% (95% CI 72.9%–80.8%) |
| Episodes stopped at or before the first harmful payment | 76.3% (95% CI 69.4%–82.1%) |
| Warnings / holds per 1,000 payments | 8.8 / 1.74 |
| False warnings on alarming-but-legitimate payments | 3.29% (vs 0.21% on ordinary traffic) |
| Withheld variant (never trained on) | 27.3% (95% CI 16.3%–41.8%, n=44) vs seen 82.7% (95% CI 78.7%–86.2%, n=388) |
| Same rows: "first-time payee" rule / ten expert rules | 3.2% / 4.6% recall at 0.5% FPR |

**Caveats (generated from the diagnostics).**
- **Holds exceed analyst capacity on the test window** (1.74 vs 1.54 per 1,000): thresholds were fitted on an earlier slice and prevalence drifted. Mitigation for the finale: a capacity guard that downgrades overflow holds to A2 warnings.
- **Synthetic-data ceiling.** A time-forward probe on the derived features reaches recall 0.836 at 0.5% FPR (flag threshold 0.92); deployed systems on real data typically sit at 0.5-0.85. Every number here is an UPPER BOUND on real-world performance.
- **Weakest variant: V4** — recall at A2+ 0.273 [0.163, 0.418] on 44 payments.
- **Generalisation gap (withheld_variant_V4):** recall 0.273 vs seen (gap 0.555, retained 0.330) on 44 payments.
- **Generalisation gap (withheld_evasion_signal_suppression):** recall 0.875 vs seen (gap -0.097, retained 1.124) on 8 payments.
- **Layer with no measured benefit:** '+ L0 rules as features' — delta PR-AUC 0.000 [0.000, 0.000] (no measured effect).
- **Label scarcity:** trained only on labels visible by the training cutoff, PR-AUC is 0.548 (oracle-label model in the headline). trained on 257 visible positives vs 818 oracle; evaluated on oracle truth.
- **Fusion adds no measured lift over L1 alone:** deployed minus L1-alone recall at 0.5% FPR 0.009 [-0.014, 0.024] (no measured effect). The L2 payee channel is kept for readable payee evidence, not for lift.
- **Cost sensitivity:** net saving share ranges 0.238-0.773 across the assumption sweep; every rupee figure is simulator-internal.
- **FPR disparity:** user_age_band=60+ has A2+ FPR 0.0132 (3.43x overall).
- **Customer harm concentrates on hard negatives:** A2+ FPR 0.0329 vs 0.0021 on ordinary benign traffic.
- **What the generator cannot know:** lures (S1) are unobserved; mule behaviour, label latency, heed rates and costs are assumptions; the grammar covers 6 variants and 6 evasions only — anything outside it is untested.
