# PDF brief — Citadel: UPI Scam-Sequence Shield (RAKSHAM PS 2)

<!-- machine-assembled by `python -m citadel export-round1`. Quote numbers ONLY as written here; each maps to
numbers_index.md. Run id 3a888efe3099. -->

## 1. Scenario and user
AI made the scam lure perfect: fluent, personalised, endlessly varied. It did not change what the payment
looks like at the moment of harm. **Primary user:** the customer at the UPI approval screen, focusing on
first-time and senior users. **Secondary user:** the PSP/bank fraud analyst who reviews held payments.
**Harmful moment:** step 3 of 4, approving a collect request or paying a new payee, seconds before the money
moves and minutes before a mule splits it.

## 2. The scam pattern (grammar, declared as data)
- **V1 Fake-refund collect request**: A caller says a refund or cashback is due and sends a UPI collect request; the victim enters the PIN believing it receives money. Approving a collect SENDS money.
- **V2 Screen-share "bank support" scam**: A fake bank or wallet support agent asks the victim to install a screen-sharing or remote-control app "to fix a blocked account", then walks them through paying a new payee.
- **V3 Fake-KYC link, then payment**: An SMS says the account will be blocked unless KYC is updated. The link leads to a "verification fee" and then a larger payment; some victims re-register UPI on a new device on the scammer's instructions.
- **V4 QR "scan to receive money"**: A fake buyer on a marketplace sends a QR code "to receive payment". Scanning and entering the PIN PAYS the scammer. QR codes are only ever for paying.
- **V5 Utility / courier disconnection threat**: A call or SMS says electricity will be cut tonight or a parcel is held unless a small dues payment is made now; small amounts, urgency, often split across attempts.
- **V6 Task / job scam**: A "work from home" task pays small rewards first, then demands growing deposits to "unlock" earnings, each to a different account.

Evasions modelled: amount splitting, slow-drip pacing, aged/recruited mule, payee-name mimicry, time-of-day
mimicry, signal suppression (video call instead of screen share). Withheld from training: variant
**V4** and evasion **signal_suppression**.

## 3. Journey (before / after)
Figure `figures/F3_before_after.png`. Without Citadel the customer approves and the money is gone. With Citadel
the same moment shows a plain-language warning (English/Hindi) with one action to take, a 60-second cooling-off,
or a hold that says "your money has not left" with a human-release path and an appeal.

## 4. Architecture
Figure `figures/F1_architecture.png` (source `architecture.mmd`). Inputs: payment event, switch-side payee and
device state, consented session signals (missing, never zero, without consent). 64 causal features
(user baseline, sequence, payee windows, graph snapshots). Three layers that fail differently: **L0** rule cards
(work with the model down, can only add friction), **L1** gradient-boosted trees (calibrated), **L2** payee/mule
graph model, fused by a rule chosen on a held-out slice. Output: action ladder A0 none / A1 nudge / A2 warning +
cooling-off / A3 hold + analyst, priced by alert budgets; 1-3 fixed reason codes per decision. No auto-decline.

## 5. Detection logic and why it is defensible
- Time-forward split with purge and embargo; entity-level sealed holdout; no random splits.
- Leakage gates fail the build: best single column AUC 0.730; artefact hunter finds no
  fraud-only namespace; planted canary detected; label-shuffle null at chance.
- Reasons are rule cards (a code is shown only when its sentence is literally true for that payment);
  generic fallback rate 2.7% of alerts.

## 6. Preliminary validation (synthetic; upper bound)
- PR-AUC 0.767 (95% CI 0.725–0.801); 118x the base rate of 0.65%.
- Recall 80.1% (95% CI 76.1%–83.6%) at 0.5% FPR; 66.2% (95% CI 61.6%–70.5%) at 0.1% FPR.
- At the deployed ladder: 77.1% (95% CI 72.9%–80.8%) of scam payments warned or held with
  8.8 warnings and 1.74 holds per 1,000 payments.
- 76.3% (95% CI 69.4%–82.1%) of scam episodes are stopped at or before the first harmful payment.
- Baselines on the same rows: "first-time payee" alone 3.2%, ten expert rules
  4.6%, logistic regression 47.5% recall at 0.5% FPR.
- Raw columns only: PR-AUC 0.460; all layers: 0.774.
  1 ablation step(s) showed no measured effect and are published.
- Fusing the L2 payee-graph channel and an isolation forest on top of L1: no measured effect (deployed minus L1 alone: +0.009 recall, 95% CI -0.014 to +0.024).
  L2 stays for its readable payee evidence and as an independent check, not for measured lift.
- Generalisation: withheld variant 27.3% (95% CI 16.3%–41.8%, n=44) vs seen 82.7% (95% CI 78.7%–86.2%, n=388). **This is the
  weakness we lead with**, and why the finale adds new variants through the Scam Studio.

Figures: F4 ladder/budget, F5 operating curve, F6 generalisation, F7 ablation, F8 time-to-alert, F9 false positives.

## 7. Safeguards
- False positives: 3.29% false warnings on alarming-but-legitimate payments vs
  0.21% on ordinary traffic; hold = money stays, 30-min review SLA, release request, appeal.
- Bias: worst segment false-warning ratio 3.43x of overall; segments over tolerance
  1.
- Misuse: scoring is rate-limited and returns bands only; complaints from fresh accounts do not count;
  complaints alone cannot trigger a hold; customers never see a score; evidence access is audited (tests in repo).
- Privacy: salted-HMAC ids, no message content, retention TTL, consent flags; DPDP-style principles of purpose
  limitation, minimisation, consent and retention (no legal-compliance claim).

## 8. Adoption, integration, scale
Deployment position: PSP/bank side, as an SDK hook on the approval screen plus a scoring API. Owner: the
issuer/PSP fraud-risk team. Rollout: shadow mode -> A1 nudges only -> A2 with cooling-off -> A3 holds sized to
analyst capacity. Measured on a laptop over loopback: scoring a stored payment p95 61 ms server compute; an online event that recomputes its features from history p95 216 ms (round trip 225 ms). Capacity check: holds 1.74 vs analyst capacity
1.54 per 1,000 payments (within capacity: NO).

## 9. 48-hour finale plan
Already built: generator, pipeline, API, offline console, Scam Studio, replay bundle, 80+ tests. In 48 h:
(1) judges author a scam live in the Studio and watch where Citadel fires; (2) add one new variant to the
grammar and measure it as unseen; (3) harden the capacity overflow (holds above capacity fall back to A2);
(4) per-segment thresholds; (5) rehearse a 3/5/8-minute demo with the offline replay as fallback.

## 10. Known limitations (generated)
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

## Figure files
- F1: `figures/F1_architecture.png`
- F2: `figures/F2_scam_sequence.png`
- F3: `figures/F3_before_after.png`
- F4: `figures/F4_ladder_budget.png`
- F5: `figures/F5_operating_curve.png`
- F6: `figures/F6_generalisation.png`
- F7: `figures/F7_layer_ablation.png`
- F8: `figures/F8_time_to_alert.png`
- F9: `figures/F9_false_positives.png`
