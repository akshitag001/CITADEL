# PROGRESS.md

Status as of 2026-10-01. Acceptance commands in brackets.

- [x] **P0** scaffold, config, provenance, IST time chokepoint, CLI [`python -m citadel --help`]
- [x] **P1** schema with roles, scam grammar as YAML, validator, composition space, `SCAM_SEQUENCE.md` [`tests/test_schema_library.py`]
- [x] **P2** generator: entities, benign, hard negatives, shapes, mules, attacks as mutations, labels, campaign [`tests/test_generate.py`]
- [x] **P3** fidelity gates: single-feature probe, joint probes, artefact hunter, null control, canary, realism [`python -m citadel fidelity`]
- [x] **P4** 64 causal features + registry, graph snapshots [`tests/test_causality.py`, `tests/test_features_l0.py`]
- [x] **P5** L0 rule cards, zero-FP gate, model-free engine
- [x] **P6** split with purge/embargo + sealed holdout, L1/L2/IF channels, channel-alone diagnostic, fusion ablation, atomic bundle
- [x] **P7** ladder from budgets, per-txn treatments, reason cards, en/hi messages, abstention, capacity, cost sweep
- [x] **P8** evaluation suite + generated REPORT.md + evidence index [`python -m citadel evaluate`]
- [x] **P9** FastAPI serving: score/batch/author/alerts/cases/appeal/model-card/health/results/latency [`tests/test_api.py`]
- [x] **P10** console: customer phone view (en/hi), analyst queue, Scam Studio, results, safeguards (static, offline)
- [x] **P11** replay bundle recorded from the real API [`python scripts/build_replay_bundle.py --run demo`]
- [x] **P12** privacy + misuse tests (rate limit, complaint poisoning, audit, retention, HMAC), fairness table, threat model
- [x] **P13** CI workflow, Dockerfile, `scripts/verify.py`
- [x] **P14** Round 1 export: tables, F1-F9, disclosure, numbers index + verifier, brief, deck PDF [`python -m citadel export-round1`]
- [ ] **P15** stretch: mule-hash exchange, on-device lure flag, lite red-team loop, per-segment thresholds
- [ ] **P16** finale freeze + rehearsal (checklist in `FINALE_CHECKLIST.md`)

## Before 6 Oct (team)
1. Fill team name/members in `src/citadel/report/deck.py` (`TEAM`) and re-run `python -m citadel export-round1 --run demo`.
2. Read `docs/round1/PDF_BRIEF.md` and the deck; run Prompt X ("did we claim too much?") on any rewritten slide text.
3. `python scripts/verify_numbers.py` must print OK after any re-run.

## Open risks
- D-018 holds above analyst capacity on the test window.
- D-019 false-warning disparity for customers aged 60+.
- Withheld variant V4 recall far below seen variants (the main generalisation weakness; we lead with it).
