# Citadel — UPI Scam-Sequence Shield

RAKSHAM AI Cybersecurity Hackathon · Problem Statement 2 (AI-driven scam pattern recognition)

AI made the scam **lure** perfect. It did not change what the **payment** looks like at the moment of harm.
Citadel models scams as four-step sequences (lure -> pressure / remote access -> collect or pay a new payee ->
mule fan-out). It scores the behaviour around each payment and steps in at **step 3, before the money moves**,
with a graded response instead of block/allow:

| Level | What the customer sees |
|---|---|
| A0 | nothing |
| A1 | a nudge: "Check before you pay" |
| A2 | a warning with plain-language reasons (English/Hindi), one action to take, and a 60 s cooling-off |
| A3 | a hold: "your money has not left", a 30-minute human review, a release request and an appeal |

There is no automatic decline. Alert volume is capped by budgets matched to analyst capacity. Every decision
carries 1–3 fixed reason codes. All results come from synthetic data with leakage gates, measured against scam
variants withheld from training.

## Quick start

```bash
pip install -r requirements.txt && pip install -e .     # Python 3.11 / 3.12
python -m citadel run --quick                            # ~1 min smoke run (not reportable)
python -m citadel run --run demo                         # full run, ~7 min -> artifacts/demo/REPORT.md
python -m pytest -q                                      # gates: leakage, causality, L0 zero-FP, API, misuse
python -m citadel serve --run demo                       # API + console at http://127.0.0.1:8000
python scripts/build_replay_bundle.py --run demo         # record the offline console bundle
python -m citadel export-round1 --run demo               # docs/round1/: figures, tables, deck PDF
python scripts/verify.py --run demo                      # one-screen PASS/FAIL
```

On Windows, `make` is `mingw32-make` (`mingw32-make demo`, `mingw32-make test`).
Offline demo: open `frontend/index.html` directly. It runs from the committed replay bundle with no network.

## Where things are

| Path | What |
|---|---|
| `configs/scam_library.yaml` | the scam grammar: 6 variants, 6 evasions, segments (data, validated) |
| `configs/l0_rules.yaml`, `reason_codes.yaml`, `messages.yaml`, `costs.yaml` | rule cards, closed reason list, en/hi text, labelled assumptions |
| `src/citadel/generate/` | synthetic UPI world: benign traffic, hard negatives, look-alike shapes, mule registry, attacks as mutations, labels; fidelity probes, artefact hunter |
| `src/citadel/features/` | 64 causal features, registry, nightly graph snapshots |
| `src/citadel/defend/` | L0 engine, L1/L2 channels, fusion, ladder, reasons, bundle, split |
| `src/citadel/evaluate/` | metrics with CIs, generalisation, baselines, ablation, fairness, cost, REPORT.md |
| `src/citadel/serve/` | FastAPI app, alert store, auth + rate limit, Scam Studio |
| `src/citadel/report/` | Round 1 numbers registry, figures, deck |
| `frontend/` | console: customer phone view, analyst queue, Scam Studio, results, safeguards |
| `docs/round1/` | everything the Round 1 PDF needs, generated |
| `DECISIONS.md` | every defect the gates caught and every deviation, with the fix |

## Honesty

- Numbers in documents come only from `artifacts/<run>/*.csv` via `docs/round1/numbers.json`.
  `scripts/verify_numbers.py` fails if any number drifts.
- Reduced-scale runs are stamped not reportable.
- Weak results are published: the withheld variant, layers with no measured effect, holds above capacity and
  the false-warning disparity for seniors are all in the report.
