"""``python -m citadel export-round1``: everything the Round 1 PDF needs, generated from the pipeline.

docs/round1/
  tables/*.csv|md           headline, generalisation, baselines, ablation, budget, FPR, time-to-alert, fairness, reasons
  figures/F1..F9.png|svg     legible at slide size
  architecture.mmd           diagram source (Mermaid)
  numbers.json + numbers_index.md   every quoted number -> artefact -> tier -> denominators
  appendix_disclosure.md     datasets, models, libraries, reused components, AI assistance, assumptions
  preliminary_validation.md  one page, labelled synthetic, caveats generated from diagnostics
  PDF_BRIEF.md               machine-assembled brief for slide writing
  Citadel_Round1.pdf         the deck itself (10 slides + appendix), numbers by key from numbers.json
"""

from __future__ import annotations

import json
import platform
import textwrap
from pathlib import Path

import pandas as pd
import yaml

from ..config import CONFIG_DIR, ROOT
from ..defend.reasons import load_messages
from ..evaluate.report import known_limitations, md_table
from ..generate.library import load_library
from ..runctx import RunCtx
from . import numbers as NUM
from .figures import all_figures

OUT = ROOT / "docs" / "round1"
TABLES = {
    "headline": ("defend_headline.csv", ["metric", "value", "ci_low", "ci_high", "n_rows", "n_positives", "reportable", "note"]),
    "generalisation": ("eval_generalisation.csv", ["group", "value", "ci_low", "ci_high", "denominator", "gap_vs_seen",
                                                   "retained_share_of_seen", "reportable"]),
    "lovo": ("eval_lovo.csv", ["variant", "recall_at_0.5pct_fpr_when_withheld", "ci_low", "ci_high",
                               "recall_at_0.5pct_fpr_when_trained", "retained_share", "n_positives"]),
    "baselines": ("eval_baselines.csv", ["model", "pr_auc", "recall_at_0.1%_fpr", "recall_at_0.5%_fpr",
                                         "citadel_minus_this_recall_0.5pct", "delta_ci_low", "delta_ci_high", "verdict"]),
    "layer_ablation": ("eval_layer_ablation.csv", ["step", "n_features", "pr_auc", "recall_at_0.5pct_fpr",
                                                   "delta_pr_auc_vs_previous", "delta_ci_low", "delta_ci_high", "verdict"]),
    "alert_budget": ("defend_ladder.csv", ["action", "n_alerts", "per_1000_payments", "scams_at_level", "legit_at_level",
                                           "budget_share", "realised_share"]),
    "hard_negative_fpr": ("eval_hard_negative_fpr.csv", ["population", "n_rows", "fpr_A1plus", "fpr_A2plus",
                                                         "fpr_A2plus_ci_low", "fpr_A2plus_ci_high", "fpr_A3"]),
    "time_to_alert": ("eval_time_to_alert_summary.csv", ["metric", "value", "ci_low", "ci_high", "numerator",
                                                         "denominator"]),
    "fairness": ("eval_fairness.csv", ["group", "value_of", "value", "denominator", "fpr_A2plus", "fpr_ratio_to_overall",
                                       "flag_fpr_disparity"]),
    "reason_codes": ("defend_reason_code_usage.csv", ["reason_code", "uses_on_scams", "uses_on_legit", "share_of_alerts",
                                                      "fallback_rate"]),
    "per_variant": ("eval_per_variant.csv", ["value_of", "value", "ci_low", "ci_high", "denominator"]),
    "cost": ("defend_cost_summary.csv", ["scenario", "approve_everything_inr", "citadel_total_inr", "net_saving_share"]),
}

MERMAID = """flowchart LR
  subgraph PSP["PSP / bank boundary · salted-HMAC ids · no message content · retention TTL"]
    E[UPI event: pay / collect / QR] --> F[Causal features: user, sequence, payee, graph]
    S[Switch state: payee history, device, SIM] --> F
    C[Consented signals: screen share, call - missing not zero] -.-> F
    F --> L0[L0 deterministic rules - no model]
    F --> L1[L1 boosted trees - calibrated]
    F --> L2[L2 payee / mule graph]
    L1 --> FU[Fusion chosen by ablation -> calibrated risk]
    L2 --> FU
    FU --> LAD[Action ladder A0-A3 under alert budget]
    L0 --> LAD
    LAD --> RC[Reason codes - closed list, en/hi]
  end
  RC --> CU[Customer screen: band + reasons, never a score]
  RC --> AQ[Analyst queue: evidence, audit log]
  AQ --> FB[Feedback, appeals, late labels]
  FB -.retrain.-> L1
"""


def export_tables(art: Path, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (src, cols) in TABLES.items():
        p = art / src
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cols = [c for c in cols if c in df.columns]
        df[cols + ["run_id"]].to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
        (out / f"{name}.md").write_text(f"<!-- generated from artifacts/<run>/{src}; do not edit -->\n"
                                        + md_table(df, cols), encoding="utf-8")
        written.append(name)
    return written


def disclosure(ctx: RunCtx, gs: dict) -> str:
    import matplotlib
    import numpy
    import scipy
    import sklearn
    costs = yaml.safe_load((CONFIG_DIR / "costs.yaml").read_text())
    st = ctx.cfg.staffing
    return f"""# Appendix — Disclosure

<!-- generated by `python -m citadel export-round1`; edit the generator, not this file -->

## Data
- **100% synthetic.** No real bank, customer or platform data; nothing scraped. Generated by the Citadel
  generator (`src/citadel/generate/`) from `configs/{ctx.cfg.profile}.yaml` + `configs/scam_library.yaml`.
- Seed `{ctx.cfg.seed}` · config hash `{ctx.prov.config_hash}` · library hash `{ctx.prov.library_hash}` ·
  run id `{ctx.prov.run_id}` · code `{ctx.prov.git_sha}`.
- {gs['n_rows']:,} payments over {gs['days']} days; {gs['n_users']:,} customers; {gs['n_merchants']:,} merchants;
  {gs['n_episodes']:,} scam episodes; realised scam share {gs['realised_attack_share']:.3%}.
- Scams are mutations of real (synthetic) benign payments; six variants and six evasion techniques are
  declared as data. Results are an upper bound on real-world performance.

## Models and libraries
- scikit-learn {sklearn.__version__} HistGradientBoostingClassifier (L1), LogisticRegression (L2, meta-learner,
  baseline), IsolationForest (candidate channel), IsotonicRegression (calibration).
- numpy {numpy.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__},
  FastAPI, pydantic, uvicorn, networkx; Python {platform.python_version()}.
- No pretrained models, no LLM anywhere in the scoring path, no external APIs at runtime.

## Reused components
None. All code was written for Citadel; runtime libraries are pinned in `requirements.txt`.

## AI-assisted work
- Code, configuration and documentation were written with an AI coding assistant (Claude Code) under the
  team's direction and review. All numbers are produced by the pipeline, not by the assistant.
- The slide text may be drafted with AI assistance from `PDF_BRIEF.md`; every number in it is checked
  against `numbers_index.md` by `scripts/verify_numbers.py`.

## Assumptions (labelled, not measured)
- Cost model: analyst review INR {costs['analyst_review_inr']}, false hold INR {costs['false_hold_inr']}, friction
  A1/A2 INR {costs['friction_inr']['A1']}/{costs['friction_inr']['A2']}, heed rates {costs['heed_rate']},
  recovery {costs['recovery_rate']:.0%}. Swept x0.5..x2 in `defend_cost_summary.csv`.
- Label latency mix: complaint 0.5-10 days (60%), bank report 14-40 days (25%), analyst hours (15%);
  {ctx.cfg.attacks.label_visible_share:.0%} of episodes ever get a visible label.
- Analyst capacity: {st.analysts} analysts x {st.cases_per_hour:g} cases/h x {st.shift_hours:g} h x {st.shifts_per_day} shifts
  for an assumed {st.deployment_payments_per_day:,} payments/day.
- Signal observability: screen/remote-access signals for {ctx.cfg.population.screen_signal_observable_share:.0%} of
  customers (app SDK); call-in-progress only with consent ({ctx.cfg.consent.call_in_progress_available_share:.0%}).

## What we do not collect
Message, SMS or call content; contact names; location; the lure itself; raw account numbers (salted
HMACs only); any score shown to a customer.
"""


def numbers_index(nums: dict) -> str:
    rows = pd.DataFrame([{"key": k, "display": n.display, "tier": n.tier, "artefact": n.artefact,
                          "selector": n.selector, "n_rows": n.n_rows, "n_positives": n.n_positives,
                          "reportable": n.reportable} for k, n in nums.items()])
    return ("# Numbers index\n\n<!-- generated; verify with `python scripts/verify_numbers.py` -->\n\n"
            "Every number the Round 1 PDF may quote. Tiers: **measured** (computed on the test window), "
            "**derived** (computed from measured numbers plus a stated assumption), **design-only** (a target, "
            "never a result). Only reportable rows may be quoted.\n\n" + md_table(rows, list(rows.columns), 0))


def validation_page(nums: dict, lims: list[str]) -> str:
    n = lambda k: nums[k].display if k in nums else "—"  # noqa: E731
    return f"""# Preliminary validation — SYNTHETIC DATA

<!-- generated; numbers from numbers.json -->

> All results below come from Citadel's own synthetic generator. They show the design works as
> specified and where it breaks; they are an upper bound on real-world performance, not a claim about it.

**Setup.** {n('data.n_rows')} synthetic UPI payments over {n('data.days')} days, {n('data.n_episodes')} scam
episodes across six scripts and six evasions. Time-forward split with purge and embargo; one whole variant and
one whole evasion sealed out of training. Test window: {n('data.n_test_rows')} payments, {n('data.n_test_pos')}
scam payments (base rate {n('headline.base_rate')}).

**Is the data too easy?** Best single raw column AUC {n('fidelity.max_single_auc')} (gate 0.95). A booster on
raw columns alone reaches {n('fidelity.raw_probe')} recall at 0.5% FPR; on all derived features
{n('fidelity.derived_probe')} (we flag above 92% as "measuring the generator"). Label-shuffle null and a
planted leakage canary both behave as expected.

| Result | Value |
|---|---|
| PR-AUC | {n('headline.pr_auc')} — {n('headline.lift')} the base rate |
| Recall at 0.5% / 0.1% false-positive rate | {n('headline.recall_05')} / {n('headline.recall_01')} |
| Scam payments warned or held at the deployed ladder | {n('headline.recall_A2')} |
| Episodes stopped at or before the first harmful payment | {n('tta.by_first_harmful')} |
| Warnings / holds per 1,000 payments | {n('budget.warnings_A2plus_per_1000')} / {n('budget.holds_A3_per_1000')} |
| False warnings on alarming-but-legitimate payments | {n('fpr.hard_negative_all')} (vs {n('fpr.benign')} on ordinary traffic) |
| Withheld variant (never trained on) | {n('gen.withheld_variant')} vs seen {n('gen.seen_variants')} |
| Same rows: "first-time payee" rule / ten expert rules | {n('baseline.first_time')} / {n('baseline.expert_rules')} recall at 0.5% FPR |

**Caveats (generated from the diagnostics).**
""" + "".join(f"- {s}\n" for s in lims)


def brief(ctx: RunCtx, nums: dict, figs: dict, lims: list[str]) -> str:
    n = lambda k: nums[k].display if k in nums else "—"  # noqa: E731
    lib = load_library()
    var = "\n".join(f"- **{k} {v['title']}**: {' '.join(v['story'].split())}" for k, v in lib.variants.items())
    lat = ""
    if "latency.stored_txn.server_compute.p95_ms" in nums:
        lat = (f"Measured on a laptop over loopback: scoring a stored payment p95 "
               f"{n('latency.stored_txn.server_compute.p95_ms')} server compute; an online event that recomputes "
               f"its features from history p95 {n('latency.online_event.server_compute.p95_ms')} "
               f"(round trip {n('latency.online_event.round_trip.p95_ms')}).")
    return f"""# PDF brief — Citadel: UPI Scam-Sequence Shield (RAKSHAM PS 2)

<!-- machine-assembled by `python -m citadel export-round1`. Quote numbers ONLY as written here; each maps to
numbers_index.md. Run id {ctx.prov.run_id}. -->

## 1. Scenario and user
AI made the scam lure perfect: fluent, personalised, endlessly varied. It did not change what the payment
looks like at the moment of harm. **Primary user:** the customer at the UPI approval screen, focusing on
first-time and senior users. **Secondary user:** the PSP/bank fraud analyst who reviews held payments.
**Harmful moment:** step 3 of 4, approving a collect request or paying a new payee, seconds before the money
moves and minutes before a mule splits it.

## 2. The scam pattern (grammar, declared as data)
{var}

Evasions modelled: amount splitting, slow-drip pacing, aged/recruited mule, payee-name mimicry, time-of-day
mimicry, signal suppression (video call instead of screen share). Withheld from training: variant
**{ctx.cfg.sealed_holdout.variant}** and evasion **{ctx.cfg.sealed_holdout.evasion}**.

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
- Leakage gates fail the build: best single column AUC {n('fidelity.max_single_auc')}; artefact hunter finds no
  fraud-only namespace; planted canary detected; label-shuffle null at chance.
- Reasons are rule cards (a code is shown only when its sentence is literally true for that payment);
  generic fallback rate {n('reasons.fallback_rate')} of alerts.

## 6. Preliminary validation (synthetic; upper bound)
- PR-AUC {n('headline.pr_auc')}; {n('headline.lift')} the base rate of {n('headline.base_rate')}.
- Recall {n('headline.recall_05')} at 0.5% FPR; {n('headline.recall_01')} at 0.1% FPR.
- At the deployed ladder: {n('headline.recall_A2')} of scam payments warned or held with
  {n('budget.warnings_A2plus_per_1000')} warnings and {n('budget.holds_A3_per_1000')} holds per 1,000 payments.
- {n('tta.by_first_harmful')} of scam episodes are stopped at or before the first harmful payment.
- Baselines on the same rows: "first-time payee" alone {n('baseline.first_time')}, ten expert rules
  {n('baseline.expert_rules')}, logistic regression {n('baseline.logreg')} recall at 0.5% FPR.
- Raw columns only: PR-AUC {n('ablation.raw_pr_auc')}; all layers: {n('ablation.full_pr_auc')}.
  {n('ablation.n_no_effect_rows')} ablation step(s) showed no measured effect and are published.
- Fusing the L2 payee-graph channel and an isolation forest on top of L1: {n('fusion.vs_l1_alone_verdict')}.
  L2 stays for its readable payee evidence and as an independent check, not for measured lift.
- Generalisation: withheld variant {n('gen.withheld_variant')} vs seen {n('gen.seen_variants')}. **This is the
  weakness we lead with**, and why the finale adds new variants through the Scam Studio.

Figures: F4 ladder/budget, F5 operating curve, F6 generalisation, F7 ablation, F8 time-to-alert, F9 false positives.

## 7. Safeguards
- False positives: {n('fpr.hard_negative_all')} false warnings on alarming-but-legitimate payments vs
  {n('fpr.benign')} on ordinary traffic; hold = money stays, 30-min review SLA, release request, appeal.
- Bias: worst segment false-warning ratio {n('fairness.max_fpr_ratio')} of overall; segments over tolerance
  {n('fairness.n_flags')}.
- Misuse: scoring is rate-limited and returns bands only; complaints from fresh accounts do not count;
  complaints alone cannot trigger a hold; customers never see a score; evidence access is audited (tests in repo).
- Privacy: salted-HMAC ids, no message content, retention TTL, consent flags; DPDP-style principles of purpose
  limitation, minimisation, consent and retention (no legal-compliance claim).

## 8. Adoption, integration, scale
Deployment position: PSP/bank side, as an SDK hook on the approval screen plus a scoring API. Owner: the
issuer/PSP fraud-risk team. Rollout: shadow mode -> A1 nudges only -> A2 with cooling-off -> A3 holds sized to
analyst capacity. {lat} Capacity check: holds {n('budget.holds_A3_per_1000')} vs analyst capacity
{n('budget.analyst_capacity_k_per_1000')} per 1,000 payments (within capacity: {n('budget.holds_within_capacity')}).

## 9. 48-hour finale plan
Already built: generator, pipeline, API, offline console, Scam Studio, replay bundle, 80+ tests. In 48 h:
(1) judges author a scam live in the Studio and watch where Citadel fires; (2) add one new variant to the
grammar and measure it as unseen; (3) harden the capacity overflow (holds above capacity fall back to A2);
(4) per-segment thresholds; (5) rehearse a 3/5/8-minute demo with the offline replay as fallback.

## 10. Known limitations (generated)
""" + "".join(f"- {s}\n" for s in lims) + """
## Figure files
""" + "".join(f"- {k}: `figures/{Path(v[0]).name}`\n" for k, v in figs.items())


def export_round1(ctx: RunCtx) -> None:
    from ..pipeline import log
    art = ctx.art
    if not (art / "defend_headline.csv").exists():
        raise SystemExit(f"run '{ctx.run}' has not been evaluated; run `python -m citadel run --run {ctx.run}` first")
    OUT.mkdir(parents=True, exist_ok=True)
    if not (art / "serve_latency.csv").exists():
        from .latency import measure
        log("export-round1: measuring serving latency")
        measure(ctx)
    # stamp the documents with the run that PRODUCED the numbers, not the current code checkout
    ctx.prov.run_id = str(pd.read_csv(art / "defend_headline.csv")["run_id"].iloc[0])
    gs = json.loads((art / "generate_summary.json").read_text())
    fid = json.loads((art / "generate_fidelity_summary.json").read_text())
    nums = NUM.build_numbers(art, gs, fid, ctx.reportable())
    NUM.save(nums, OUT / "numbers.json")
    (OUT / "numbers_index.md").write_text(numbers_index(nums), encoding="utf-8")
    export_tables(art, OUT / "tables")
    figs = all_figures(art, OUT / "figures", load_messages())
    (OUT / "architecture.mmd").write_text(MERMAID, encoding="utf-8")
    tables = {n: pd.read_csv(art / n) for n in ("eval_per_variant.csv", "eval_generalisation.csv",
                                                 "eval_layer_ablation.csv", "defend_cost_summary.csv",
                                                 "eval_fairness.csv", "eval_hard_negative_fpr.csv",
                                                 "defend_channels_alone.csv", "defend_fusion_arms.csv",
                                                 "eval_baselines.csv")}
    tables["_channels"] = tables.pop("defend_channels_alone.csv")
    tables["_fusion"] = tables.pop("defend_fusion_arms.csv")
    lims = known_limitations(tables, fid)
    b = pd.read_csv(art / "defend_alert_budget.csv").iloc[0]
    if not bool(b["holds_within_capacity"]):
        lims.insert(0, f"**Holds exceed analyst capacity on the test window** ({b['holds_A3_per_1000']:.2f} vs "
                       f"{b['analyst_capacity_k_per_1000']:.2f} per 1,000): thresholds were fitted on an earlier slice and "
                       "prevalence drifted. Mitigation for the finale: a capacity guard that downgrades overflow holds to "
                       "A2 warnings.")
    (OUT / "appendix_disclosure.md").write_text(disclosure(ctx, gs), encoding="utf-8")
    (OUT / "preliminary_validation.md").write_text(validation_page(nums, lims), encoding="utf-8")
    (OUT / "PDF_BRIEF.md").write_text(brief(ctx, nums, figs, lims), encoding="utf-8")
    from .deck import build_deck
    build_deck(ctx, nums, lims, OUT)
    problems = NUM.verify(OUT / "numbers.json", NUM.build_numbers(art, gs, fid, ctx.reportable()))
    unrep = [k for k, v in nums.items() if not v.reportable]
    log(f"export-round1: {len(nums)} numbers ({len(unrep)} not reportable), {len(figs)} figures, "
        f"verify {'OK' if not problems else problems} -> docs/round1/")
    if not ctx.reportable():
        log("export-round1: WARNING this profile is not reportable; do not put these numbers in the PDF")


def _wrap(s: str, w: int) -> str:
    return "\n".join(textwrap.wrap(s, w))
