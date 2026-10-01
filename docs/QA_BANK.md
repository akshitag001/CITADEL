# Judge Q&A bank

Every answer points to evidence. Quote numbers only from `docs/round1/numbers_index.md`.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | What if a new voice-cloning tool makes the lure perfect? | We never score the lure. We score the payment behaviour at step 3: new payee, unusual amount, remote control, rushed approval, mule-shaped payee. | F2; `SCAM_SEQUENCE.md` |
| 2 | How do you avoid alert fatigue? | Thresholds come from volume budgets (warnings <= 1%, holds <= 0.1%), graded actions, and most alerts are a nudge. | `defend_alert_budget.csv`, F4, F5 |
| 3 | What happens on a false positive? | No decline. Warning = 60 s; hold = money stays, 30-min review, release, appeal. | `FALSE_POSITIVE_POLICY.md`, `messages.yaml` |
| 4 | Why synthetic data? What is the ceiling? | No real data is allowed. We gate separability (single column <= 0.95 AUC; probe flagged above 92%) and call every number an upper bound. | `generate_fidelity_summary.json`, REPORT controls |
| 5 | How do we know it is not leaking? | Truth columns cannot reach features (tested by scrambling); prefix invariance; artefact hunter; canary; label-shuffle null; time-forward split. | `tests/test_causality.py`, `test_features_l0.py`, `test_generate.py` |
| 6 | Does it work on scams it has never seen? | Partly, and we lead with it: the withheld variant scores far below seen ones; leave-one-variant-out per script is published. | `eval_generalisation.csv`, `eval_lovo.csv`, F6 |
| 7 | Who owns this in a bank? | The issuer/PSP fraud-risk team; analysts work the queue; product owns the customer screens. | PDF_BRIEF §8 |
| 8 | Latency? | Measured on a laptop over loopback: stored payment and online event p50/p95/p99 separately. | `serve_latency.csv` |
| 9 | How do you integrate with a UPI PSP? | Approval-screen SDK hook + scoring API; returns band, action, reasons, message, run id. Rollout shadow -> nudge -> warn -> hold. | `serve/api.py`, `/v1/model-card` |
| 10 | How do you retrain? | Dispositions and appeals feed a feedback table; labels arrive late; we show what a model trained only on matured labels scores. | `eval_layer_ablation.csv` (visible-label row) |
| 11 | Bias? | False-warning rate by age, literacy, tier with CIs; seniors are 3x overall, flagged as an open risk with a mitigation plan. | `eval_fairness.csv`, F9, D-019 |
| 12 | Misuse — can a fraudster probe it? | Customers get 4 bands, never a score; per-token rate limit; tests prove both. | `test_boundary_probing_is_rate_limited` |
| 13 | Can someone get an innocent payee blocked with fake complaints? | Complaints count only from distinct reporters aged >= 30 days, and holds also need a structural mule signal. A test plants the attack. | `test_complaint_poisoning_cannot_reach_a_hold_alone`, D-011 |
| 14 | Why not just block first-time payees? | Same rows, same FPR budget: that rule catches far less, because ~1 in 4 genuine payments go to a new payee. | `eval_baselines.csv` |
| 15 | Is the ML even needed vs expert rules? | Ten expert rules on the same rows are far behind; raw columns alone vs all layers shows what behaviour adds. | `eval_baselines.csv`, F7 |
| 16 | What did NOT work? | Adding L0 rules as model features had no measured effect, and fusing the L2 graph channel + isolation forest on top of L1 shows no measured effect either (CI includes zero). Both published. L2 stays for readable payee evidence, not lift. | F7, `eval_baselines.csv` (L1-alone row), `defend_fusion_arms.csv` |
| 17 | Why a hold and not a decline? | One signal is never proof; a hold keeps the money safe and gives a human 30 minutes. | `ladder.py`, `l0_rules.yaml` |
| 18 | What if the model is down? | L0 rules run with no model and can still warn or hold. | `test_l0_runs_with_the_model_stack_unavailable` |
| 19 | What do you collect? Privacy? | Payment, payee aggregates, device/SIM flags, consented session flags. No message content, contacts or location; salted-HMAC ids; retention TTL. | Safeguards screen, THREAT_MODEL |
| 20 | DPDP Act? | We follow its principles (purpose limitation, minimisation, consent, retention); we do not claim legal compliance. | THREAT_MODEL |
| 21 | Are the reasons real explanations? | Each code is a readable predicate shown only when literally true for that payment; generic fallback rate is measured. | `reason_codes.yaml`, `defend_reason_code_usage.csv` |
| 22 | What about mule accounts at other banks? | We only see our customers' outbound payments (modelled); the finale stretch adds a hashed mule-exchange. | D-003, PROGRESS P15 |
| 23 | How many analysts do you need? | Capacity is derived from staffing; on the test window holds exceed it, and we disclose that with the fix. | `defend_alert_budget.csv`, D-018 |
| 24 | Why gradient-boosted trees, not deep learning? | Tabular, NaN-native, calibratable, fast; the bottleneck is honest evaluation, not model class. | `defend/model.py` |
| 25 | How do you choose fusion? | Six arms scored on a stats slice the test never sees; losers published. | `defend_fusion_arms.csv` |
| 26 | What is the cost case? | Simulator-internal, from labelled assumptions, swept x0.5..x2. | `defend_cost_summary.csv` |
| 27 | What happens after step 3 if missed? | S4 payee-side features and graph distances catch repeat mules; confirmed mules are frozen. | `features/graph.py` |
| 28 | How reproducible is this? | One config + seed = byte-identical data; run id on every file and API response; CI checks determinism. | `.github/workflows/ci.yml` |
| 29 | What would you do with real data first? | Shadow mode, re-run every gate, re-fit budgets, measure the generalisation gap on real new scripts. | PDF_BRIEF §8 |
| 30 | What is out of scope? | Card rails, crypto, agentic commerce, any LLM in the authorisation path, real PSP integration. | CLAUDE.md |
