# Demo script

Setup: `python -m citadel serve --run demo` -> http://127.0.0.1:8000 (live). Fallback: open
`frontend/index.html` from disk (replay mode, no network). The pill at top right says which mode is running.

## 60 seconds
| Time | Click | Say | Rubric |
|---|---|---|---|
| 0-10 s | Customer tab, first V1 scenario, **Without Citadel** -> Approve | "A senior gets a 'refund' collect request. Approving it sends money; the mule splits it in minutes." | Problem fit |
| 10-30 s | **With Citadel** -> Approve; switch to हिन्दी | "Same moment. Plain reasons, one action, 60 seconds to think. In Hindi too. No score, no accusation." | Value, safety |
| 30-45 s | Scam Studio: V2, signal_suppression, senior -> Run | "Judges can compose any scam from the grammar. Here's where Citadel fired, which layers, which reasons." | Architecture |
| 45-60 s | Results: KPI tiles + generalisation chart | "Synthetic, upper bound — and we show the variant it has never seen, where it's weakest." | Honesty |

## 3 minutes (add)
1. Customer: a **legitimate but alarming** scenario: Citadel stays calm or only nudges.
2. Customer: an **honest miss**, with its explanation generated from the payment's own features.
3. Analyst queue: open a held payment: timeline, layer bars, reason text; **release** it (live). Mention the
   audit log entry.
4. Studio: an **illegal** combination (V4 + signal suppression) is disabled with the reason.

## 5 / 8 minutes (add)
5. Results: operating curve with budget markers; ablation with the no-effect row; baselines; hard-negative FPR;
   fairness (seniors 3x — our open risk); evidence index (every number -> CSV).
6. Safeguards: limitations first; what we never collect; misuse tests recorded as passed.
7. Terminal: `python scripts/verify.py --run demo` — the one-screen PASS/FAIL, including the capacity FAIL we
   disclose.
8. Live: `curl` a score with the customer token -> band only; same call with the analyst token -> scores.

## If something breaks
- API down -> reload `frontend/index.html` from disk (replay mode). Every screen renders from the bundle.
- Laptop dies -> second laptop, `git clone` + open `frontend/index.html` (bundle is committed).
- Backup video: `docs/demo.mp4` (record during rehearsal).
