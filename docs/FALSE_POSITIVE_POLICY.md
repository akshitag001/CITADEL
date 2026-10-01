# False-positive policy

A false positive is a genuine customer slowed down or stopped. Citadel is designed so that its worst false
positive costs the customer minutes, never money and never an accusation.

## What the customer sees
- **A1 nudge:** one line, "Check before you pay", with Continue. No delay.
- **A2 warning:** the reasons in plain language (English/Hindi), one action to take (call the bank on the number
  printed on the card, or ask a family member), Cancel, and "I know this person, continue", which unlocks after
  a 60-second cooling-off. The text never says "fraud" or "suspicious person" and never shows a score.
- **A3 hold:** "Your money has NOT left your account. A bank officer will review it within 30 minutes and may call
  you." Buttons: call the bank, request release, raise an appeal.

## How fast it is released
- A3 holds enter the analyst queue with a 30-minute SLA, ordered hold-first then by risk.
- Release is one analyst action (`PUT /v1/cases/{id}` with `release`), recorded in the audit log.
- Appeals (`POST /v1/appeal`) go to a human with a 24-hour SLA and are visible in the analyst inbox.

## How the volume is bounded
- Thresholds come from budgets: A1 <= 3%, A2 <= 1%, A3 <= 0.1% of payments, set on a held-out slice.
- Holds are checked against analyst capacity (`defend_alert_budget.csv`). On the current test window holds
  exceed capacity (DECISIONS.md D-018); the finale adds a guard that downgrades overflow holds to warnings.
- L0 rules are required to be zero-FP on every legitimate population in tests.

## Where harm concentrates (measured, `eval_hard_negative_fpr.csv`, `eval_fairness.csv`)
- Alarming-but-legitimate payments (new phone + support call, splitting a contractor payment, family helping
  over screen share) get false warnings far more often than ordinary payments.
- Customers aged 60+ get false warnings at ~3x the overall rate (D-019). Planned: per-segment thresholds with an
  FPR-ratio cap, and senior-first message testing.

## How feedback retrains
Analyst dispositions (confirm / release / needs-info) and appeal outcomes land in the `feedback` table with a
timestamp. Released holds become verified negatives with their reason codes, so a reason card whose precision
drops is visible before retraining. Retraining uses only labels matured by the cutoff (see the visible-label
ablation in `eval_layer_ablation.csv`).
