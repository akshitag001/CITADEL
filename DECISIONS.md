# DECISIONS.md

Template: **D-NNN — title** · date · phase · *what / why / evidence / regression test*.
Every defect a gate found is logged here, with the fix applied to the GENERATOR (never the gate).

---

**D-001 — PSP pre-history state (pair counts + customer profile)** · 2026-10-01 · P2
*What:* benign traffic showed 67% "first-time payee" payments because the simulation began with no history.
*Why it matters:* "new payee" would have been a nearly free scam signal. *Fix:* the population carries
pre-window state a PSP really holds: prior payment counts per (customer, habitual payee) and a noisy
amount profile. Benign first-time share is now ~26%. *Test:* `test_not_every_scam_is_to_a_first_time_payee`.

**D-002 — Mule fan-out goes through cash-out accounts, not sibling mules** · P3
*What:* graph features made the data "measure the generator": 64% of scam payees sat one hop from a confirmed
mule because mules paid each other directly; the derived probe hit 0.92 at 0.5% FPR (ceiling flag).
*Fix:* layering via a ring's cash-out pool; siblings are two hops apart. Probe 0.86, below the flag.

**D-003 — Institution observability of mule outflows** · P3
A PSP only sees payments its own customers make. Purpose-opened mules and cash-outs bank with this institution
50% of the time; otherwise their outbound rows are invisible (`mules.OUR_CUSTOMER_SHARE`).

**D-004 — Customer-ladder metrics exclude mule-initiated rows** · P2/P8
Mule fan-out (S4) and lure credits are payments by the criminal's own account. They are scored by the
payee-side path and excluded from customer-ladder metrics (`schema.MULE_INITIATED_KINDS`); counts are reported.

**D-005 — Train on oracle labels; publish the visible-label ablation** · P6
Headline model trains on simulated truth in the train window. The realistic alternative (only labels
visible by the training cutoff) is an ablation row in `eval_layer_ablation.csv`; evaluation is always on truth.

**D-006 — Seal the withheld SCRIPT, not shared mule infrastructure** · P6 · *deviation from the pack*
Strict entity sealing (remove every row of every mule a held-out episode touched) would discard 896 of
~1,250 training-window positives (most are seen-variant scams sharing those mules). We seal held-out
episodes, their victims (all rows) and mules used only by held-out episodes. The strict-sealing cost stays
visible in `defend_split.csv` (`strict_sealing_would_remove`). Real new scripts reuse existing mule networks.
*Test:* `test_sealed_holdout_is_entity_level`.

**D-007 — Confirmed mule accounts are frozen** · P5
L0-01 fired on genuine customers' first payments to recruited mule accounts after confirmation. Banks freeze
confirmed mules; the generator now blocks non-scam traffic to/from an account after its freeze time.

**D-008 — Reason attribution by rule cards, not SHAP** · P7
Each code is a fixed readable predicate; codes are ordered by permutation importance of their feature family
on the stats slice; L0 codes first. A code is shown only when its sentence is literally true. Fallback (R13)
rate is reported in `defend_reason_code_usage.csv`.

**D-009 — Calibrator tie-break** · P7
Isotonic output is a step function; whole plateaus tie and a volume budget cannot be placed. Risk =
(1-1e-7)·iso + 1e-7·sigmoid(raw): strictly monotone, inside [0,1]. *Test:* `test_calibration_is_monotonic`.

**D-010 — Realistic retry distribution** · P3 (artefact hunter)
`attempts_in_session == 3` was nearly scam-only (benign P = 0.09%). Benign retries are now geometric
(P2≈13%, P3≈2%). *Test:* `test_no_fraud_only_id_namespace` (hunter over every column).

**D-011 — L0-01 structural clause tightened** · P12 (misuse test)
The complaint-poisoning test planted complaints on innocent payees and L0-01 fired: "pass-through > 0.5" alone
is common in ordinary accounts. Now requires fan-in from >= 3 payers AND >= 2 onward payments, or a confirmed
mule, or a handset shared by >= 3 accounts. *Test:* `test_complaint_poisoning_cannot_reach_a_hold_alone`.

**D-012 — A label never arrives before its payment** · P2
Slow-drip episodes could get an episode label before a later payment. Per-row arrival = max(episode arrival,
row time + 1-6 h). *Test:* `test_labels_arrive_strictly_after_their_events`.

**D-013 — Baseline tie-break bug (unfair to the baseline)** · P8
"First-time payee alone" scored 0 recall because a 1e-9 tie-break vanished in float32; every first-time row
tied at the threshold. Fixed in float64 with an amount tie-break; numbers regenerated.

**D-014 — Artefact hunter: an id-prefix namespace must span >= 5 accounts** · P3
At small scale one heavily reused mule's 3-char id prefix looked like a namespace. A fraud-only namespace is a
property of many accounts, not one.

**D-015 — Family-help hard negative uses video-call screen share, not a remote-CONTROL app** · P5
To make L0-02 zero-FP we distinguish remote-control apps (AnyDesk-class) from screen share on a video call.
Disclosed because a generator change and a rule re-scope happened together: the claim it rests on is that
genuine family help rarely runs a remote-control app during a payment to a new payee.

**D-016 — NaN never satisfies an L0 comparison (including !=)** · P5
Unobservable signals cannot trigger a rule. *Test:* `test_nan_never_satisfies_a_rule`.

**D-017 — Console is a no-build static app, not Vite/React** · P10 · *deviation from the pack*
Opens from `file://` with the network unplugged (replay bundle loaded as a script, not fetched), needs no
npm in the venue, and is served by the API in live mode. Same rules: no hand-typed numbers, em dash for missing,
error-boundaried panels, no external requests.

**D-018 — OPEN RISK: holds exceed analyst capacity on the test window** · P7
1.74 holds / 1,000 vs capacity 1.54 (budget 1.0). Thresholds were fitted on an earlier slice; prevalence
drifted and abstentions add to A3. Planned mitigation (finale): a daily capacity guard that downgrades
overflow holds to A2 warnings, plus fitting the A3 budget net of abstentions.

**D-019 — OPEN RISK: false warnings concentrate on customers aged 60+** · P12
A2+ false-positive rate for 60+ is ~3x overall (flagged in `eval_fairness.csv`). These are the users the
product exists to protect, so friction matters. Planned: per-segment thresholds with a cap on segment FPR
ratio, and senior-first message testing.

**D-020 — Hubs excluded from mule-distance paths** · P4
Merchants, tutors and charities connect everyone; without exclusion every account is two hops from a mule.

**D-021 — Environment** · P0
Windows + Python 3.11.9. `make` is `mingw32-make` on this machine; every target is also a plain
`python -m citadel ...` command. Python 3.13 is not supported (pinned numpy/scipy predate cp313 wheels).
