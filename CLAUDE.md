# CITADEL — project instructions (read fully at the start of every session)

## Mission
Build "Citadel: UPI Scam-Sequence Shield" for the RAKSHAM AI Cybersecurity Hackathon (IIT Delhi, with Amazon),
Problem Statement 2: "Detect emerging AI-enabled scam workflows in UPI, banking, or wallet contexts and deliver
timely, useful alerts without creating alert fatigue."
Round 1 is scored on a PDF only (pitch deck + architecture). Round 2 is a 48-hour in-person build for the top 30
teams. Everything we build must (a) produce honest, reproducible evidence for the PDF and (b) become a
demonstrable, ready-to-use prototype for the finale.

## Scoring rubric (100 pts) — every design choice must serve one of these
- Problem fit & user understanding (20): specific India-relevant scam, one primary user, bounded problem.
- Technical architecture & detection logic (20): legible end-to-end design, defensible logic, defined I/O.
- Solution value & differentiation (15): clear before/after workflow; more than "generic AI detection".
- Feasibility & finale execution plan (15): narrow demonstrable MVP, named assumptions, provable progress in 48h.
- Safety, privacy & responsible AI (15): false positives, bias, misuse, adversarial adaptation, human escalation.
- Adoption, integration & scale (10): integration path, operating owner, rollout, latency/friction.
- Communication (5).
Organisers: "Substance matters more than slide polish. We reward products that are specific, buildable, and
safe." Avoid automatic blocking or accusations from a single weak signal. A detector score alone is not an
explanation.

## The product (core thesis)
AI makes the scam LURE perfect, but the PAYMENT BEHAVIOUR at the moment of harm still shows. We detect the scam
SEQUENCE, not a single transaction, and intervene at step 3 of 4, before money moves.
Sequence: S1 lure contact (mostly unobserved) -> S2 remote access / pressure (session signals) -> S3 collect
request or payment to a new payee (ALERT POINT) -> S4 mule fan-out (payee-side graph).
Primary user: the customer at the payment screen (focus: first-time and senior UPI users). Secondary user: the
PSP/bank fraud analyst who reviews held payments. Deployment position: PSP / bank side.

## Scam variants we model (the "grammar", configs/scam_library.yaml)
V1 fake-refund collect; V2 screen-share "bank support"; V3 fake-KYC link -> payment; V4 QR "scan to receive";
V5 utility/courier disconnection threat; V6 task/job scam. Evasions: amount splitting, slow-drip pacing,
aged/recruited mule, payee-name mimicry, time-of-day mimicry, signal suppression. One whole variant (V4) and one
whole evasion (signal_suppression) are always withheld from training.

## Architecture (layers fail differently, so they stay separate)
L0 deterministic guards (no model; work if the model is down; only add friction or hold, never decline);
L1 supervised gradient-boosted trees over causal features, calibrated; L2 payee/mule graph score (kept only if it
helps); fusion chosen by ablation on the stats slice -> calibrated risk -> ladder A0 none / A1 nudge / A2 warning
+ cooling-off / A3 hold + analyst -> fixed reason codes -> customer message (en/hi) + analyst evidence. NO
automatic decline. Alert budget: warnings <= 1.0% of payments, holds <= 0.1%, matched to analyst capacity.

## Hard rules (non-negotiable)
1. SYNTHETIC DATA ONLY. Disclose all AI-generated content, datasets and models (docs/DISCLOSURE.md, kept current).
2. NUMBERS POLICY. Every number in docs/PDF comes from a file the pipeline wrote under artifacts/<run>/, with
   denominators and a provenance tier (measured / derived / design-only). No accuracy figure anywhere.
3. NO LEAKAGE. Forbidden as features: episode_id, is_attack, scam_variant, evasion_technique, step_in_sequence,
   any label_* column, any id prefix that reveals role. Time-forward splits only (purge + embargo).
4. HONESTY. Report negative ablation rows, weak variants and known failures. Never fabricate a measurement.
5. SAFETY. No auto-decline. No action on one weak signal. Holds always have a human-release path and an appeal.
   Scores never reach end-user clients (band + reasons only). Scoring endpoints are rate limited.
6. PRIVACY. Hashed identifiers, no message content stored, retention TTLs, consent flags for optional signals.
7. SCOPE DISCIPLINE. Narrow UPI MVP. Non-goals: agentic commerce, card rails, crypto, LLM in the authorization
   path, full red-team genome loop (stretch only), real PSP integration.

## Engineering conventions
- Python 3.11 or 3.12 (NOT 3.13). Package src/citadel/. CLI: `python -m citadel <stage>`.
- One Config (YAML) + one seed => byte-identical dataset; key-derived RNG streams per component.
- run_id (seed + git sha + config hash + library hash) on every artefact and API response.
- All hour/day-of-week conventions go through `citadel/timeutil.py` (IST).
- Every stage writes CSV/JSON to artifacts/<run>/ and a generated REPORT.md. Nothing hand-typed.
- Tests are gates. A regression test must FAIL against the unfixed code. Never edit a gate to make it pass.
- Commit small: "<phase>: <what>". Keep PROGRESS.md and DECISIONS.md current.

## Workflow rules for the assistant
- At session start: read this file, PROGRESS.md, DECISIONS.md. State the current phase and next 3 actions.
- Before big changes: plan in <= 10 lines. Ask at most ONE question, only if blocked.
- After each phase: run the acceptance commands, update PROGRESS.md, list open risks.
- Never claim a result not computed in this repo.

## This repository (facts that save time)
- Commands: `python -m citadel run --run demo` (full, ~7 min) · `--quick` (~1 min, not reportable) ·
  `python -m pytest -q` (~20 s, builds a tiny run once) · `python -m citadel serve --run demo` (API + console at
  http://127.0.0.1:8000) · `python -m citadel export-round1 --run demo` · `python scripts/build_replay_bundle.py
  --run demo` · `python scripts/verify.py --run demo`. On Windows `make` is `mingw32-make`.
- Profiles: configs/default.yaml (reportable), quick.yaml, tiny.yaml (tests only).
- Data in data/<run>/ (gitignored); artefacts in artifacts/<run>/; Round 1 outputs in docs/round1/.
- Gates live in tests/; the reportability guard stamps reduced-scale numbers `reportable=False`.
