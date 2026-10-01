"""The Round 1 deck as a PDF, generated. Ten slides + appendix, 16:9. Every number is pulled by key from
numbers.json (the registry), so the deck can never disagree with the pipeline."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from ..runctx import RunCtx  # noqa: E402
from .figures import INK, INK2, S1, SURFACE  # noqa: E402

W, H = 13.333, 7.5
TEAM = "[Team name] · [members]"   # filled in by the team before upload


class Deck:
    def __init__(self, path: Path, nums: dict, run_id: str):
        self.pdf = PdfPages(path)
        self.nums, self.run_id, self.page = nums, run_id, 0
        self.preview = Path(path).parent / "deck_preview"

    def n(self, key: str) -> str:
        return self.nums[key].display if key in self.nums else "—"

    def new(self, title: str, kicker: str = ""):
        self.page += 1
        fig = plt.figure(figsize=(W, H))
        fig.patch.set_facecolor(SURFACE)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(Rectangle((0, H - 0.08), W, 0.08, color=S1))
        if kicker:
            ax.text(0.55, H - 0.45, kicker.upper(), fontsize=11, color=S1, fontweight="bold")
        ax.text(0.55, H - 0.95, title, fontsize=25, fontweight="bold", color=INK, va="center")
        ax.text(0.55, 0.25, "Citadel · RAKSHAM PS2 · synthetic data only · numbers from run " + self.run_id,
                fontsize=9, color=INK2)
        ax.text(W - 0.55, 0.25, str(self.page), fontsize=10, color=INK2, ha="right")
        return fig, ax

    def bullets(self, ax, x, y, items, width=58, fs=14, gap=0.12, color=INK):
        for it in items:
            sub = it.startswith("  ")
            txt = textwrap.fill(it.strip(), width - (4 if sub else 0))
            ax.text(x + (0.35 if sub else 0), y, ("– " if sub else "• ") + txt, fontsize=fs - (1 if sub else 0),
                    color=INK2 if sub else color, va="top", linespacing=1.3)
            y -= (txt.count("\n") + 1) * fs * 0.0195 + gap
        return y

    def image(self, fig, path: Path, rect):
        if not path.exists():
            return
        a = fig.add_axes(rect)
        a.imshow(mpimg.imread(path))
        a.axis("off")

    def tile(self, ax, x, y, w, h, value, label, sub=""):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#ffffff",
                                    ec="#dddcd6", lw=1.2))
        ax.text(x + 0.2, y + h - 0.35, label, fontsize=11.5, color=INK2, va="top", wrap=True)
        ax.text(x + 0.2, y + 0.55, value.split(" (")[0], fontsize=26, fontweight="bold", color=INK)
        if sub or " (" in value:
            ax.text(x + 0.2, y + 0.2, sub or "(" + value.split(" (", 1)[1], fontsize=9.5, color=INK2)

    def save(self, fig):
        self.pdf.savefig(fig)
        if self.preview is not None:
            self.preview.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.preview / f"slide_{self.page:02d}.png", dpi=80)
        plt.close(fig)

    def close(self):
        self.pdf.close()


def build_deck(ctx: RunCtx, nums: dict, lims: list[str], out: Path) -> Path:
    F = out / "figures"
    d = Deck(out / "Citadel_Round1.pdf", nums, ctx.prov.run_id)
    n = d.n

    # 1 — title
    fig, ax = d.new("", "")
    ax.text(0.8, 5.2, "Citadel", fontsize=54, fontweight="bold", color=INK)
    ax.text(0.8, 4.45, "UPI Scam-Sequence Shield", fontsize=28, color=S1)
    ax.text(0.8, 3.5, textwrap.fill("AI made the scam lure perfect. It did not change what the payment looks like at "
                                    "the moment of harm. Citadel watches the scam sequence and steps in at step 3 of 4, "
                                    "before the money moves, with a graded response instead of block/allow.", 80),
            fontsize=16, color=INK, va="top", linespacing=1.4)
    ax.text(0.8, 1.5, "RAKSHAM AI Cybersecurity Hackathon · Problem Statement 2: AI-driven scam pattern recognition",
            fontsize=13, color=INK2)
    ax.text(0.8, 1.05, TEAM, fontsize=13, color=INK2)
    d.save(fig)

    # 2 — problem & user
    fig, ax = d.new("One person, one moment, one bounded problem", "Problem fit & user")
    d.bullets(ax, 0.6, 5.9, [
        "Primary user: the customer at the UPI approval screen — focus on first-time and senior users, who are "
        "targeted by refund, 'bank support', KYC and QR scams.",
        "Secondary user: the PSP / bank fraud analyst who reviews held payments.",
        "The harmful moment: approving a collect request or paying a new payee. Seconds later a mule splits the "
        "money across several accounts and UPI payments are usually irreversible.",
        "Why lure detection is not enough: generative AI makes calls, SMS and chats fluent and endlessly varied. "
        "The lure (step 1) is also invisible to the bank.",
        "What does not change: a new payee, an unusual amount, a remote-control app, a rushed approval, and a "
        "payee that collects from strangers and pays straight out.",
    ], width=62)
    d.image(fig, F / "F2_scam_sequence.png", [0.5, 0.15, 0.49, 0.62])
    d.save(fig)

    # 3 — grammar
    from ..generate.library import load_library
    lib = load_library()
    fig, ax = d.new("Six scam scripts, six evasions — declared as data", "The scam pattern")
    y = 6.05
    for k, v in lib.variants.items():
        ax.text(0.6, y, f"{k}", fontsize=14, fontweight="bold", color=S1, va="top")
        ax.text(1.2, y, v["title"], fontsize=14, fontweight="bold", color=INK, va="top")
        ax.text(1.2, y - 0.33, textwrap.fill(" ".join(v["story"].split()), 135), fontsize=11, color=INK2, va="top")
        y -= 0.8
    ax.text(0.6, 1.1, textwrap.fill("Evasions: amount splitting · slow-drip pacing · aged/recruited mule · payee-name "
                                    "mimicry · time-of-day mimicry · signal suppression. Every signal fires "
                                    "probabilistically; the attacker can only pull attacker-controlled levers. One whole "
                                    f"variant ({ctx.cfg.sealed_holdout.variant}) and one evasion "
                                    f"({ctx.cfg.sealed_holdout.evasion}) are never shown to the model.", 140),
            fontsize=12, color=INK, va="top")
    d.save(fig)

    # 4 — journey
    fig, ax = d.new("Before and after: the same scam, the same moment", "Solution value")
    d.image(fig, F / "F3_before_after.png", [0.04, 0.12, 0.92, 0.72])
    d.save(fig)

    # 5 — architecture
    fig, ax = d.new("Three layers that fail differently, one graded response", "Architecture")
    d.image(fig, F / "F1_architecture.png", [0.03, 0.08, 0.94, 0.78])
    d.save(fig)

    # 6 — detection logic
    fig, ax = d.new("Sequence, payee and graph signals, with readable reasons", "Detection logic")
    d.bullets(ax, 0.6, 5.9, [
        "64 causal features: every window is trailing and excludes the payment being scored (tested against "
        "brute-force references and a prefix-invariance test).",
        "  user baseline: amount vs the customer's own history, usual hour, usual payment type",
        "  sequence: anomaly then new payee, split bursts, fast collect approvals, credits from strangers",
        "  payee: fan-in from strangers, pass-through, account age; graph: hops to confirmed mules, shared handsets",
        "L0 rule cards (no model needed; zero false positives on every legitimate population by test) can only add "
        "friction. L1 boosted trees and L2 payee graph are fused by the rule that won on a held-out slice.",
        "Ladder thresholds come from volume budgets (warnings <= 1%, holds <= 0.1% of payments), not raw scores.",
        f"Every alert carries 1-3 codes from a closed list of 15, shown only when literally true; generic "
        f"fallback on {n('reasons.fallback_rate')} of alerts.",
    ], width=60, fs=13)
    d.image(fig, F / "F4_ladder_budget.png", [0.53, 0.18, 0.45, 0.55])
    d.save(fig)

    # 7 — results
    fig, ax = d.new("Preliminary validation (synthetic data — an upper bound)", "Results")
    tiles = [("headline.recall_A2", "Scam payments warned or held\nat the deployed ladder"),
             ("tta.by_first_harmful", "Episodes stopped at or before\nthe first harmful payment"),
             ("headline.recall_05", "Recall at a 0.5%\nfalse-positive rate"),
             ("headline.pr_auc", f"PR-AUC ({n('headline.lift')} the\nbase rate of {n('headline.base_rate')})")]
    for i, (k, lab) in enumerate(tiles):
        d.tile(ax, 0.55 + i * 3.15, 4.55, 2.95, 1.75, n(k), lab)
    d.image(fig, F / "F5_operating_curve.png", [0.03, 0.08, 0.5, 0.5])
    d.bullets(ax, 7.3, 4.15, [
        f"Test window: {n('data.n_test_rows')} payments, {n('data.n_test_pos')} scam payments; time-forward split "
        "with purge and embargo, no random splits.",
        f"Alert load: {n('budget.warnings_A2plus_per_1000')} warnings and {n('budget.holds_A3_per_1000')} holds per "
        "1,000 payments.",
        f"Same rows, same budget: 'first-time payee' rule {n('baseline.first_time')}, ten expert rules "
        f"{n('baseline.expert_rules')}, logistic regression {n('baseline.logreg')} recall at 0.5% FPR.",
        f"Data sanity: best single column AUC {n('fidelity.max_single_auc')} (gate 0.95); separability probe "
        f"{n('fidelity.derived_probe')} (flag at 92%).",
    ], width=48, fs=12.5)
    d.save(fig)

    # 8 — generalisation & honesty
    fig, ax = d.new("What it has never seen — and what did not help", "Honest evaluation")
    d.image(fig, F / "F6_generalisation.png", [0.05, 0.4, 0.9, 0.45])
    d.image(fig, F / "F7_layer_ablation.png", [0.03, 0.07, 0.47, 0.31])
    d.bullets(ax, 6.9, 2.55, [
        f"Withheld variant: {n('gen.withheld_variant')} vs seen {n('gen.seen_variants')}. Generalisation to an "
        "unseen script is the main weakness; the finale adds scripts through the Scam Studio and re-measures.",
        f"Raw columns alone: PR-AUC {n('ablation.raw_pr_auc')}; all layers {n('ablation.full_pr_auc')}. "
        "Rows with no measured effect are kept.",
        f"Fusing L2 + isolation forest over L1 alone: {n('fusion.vs_l1_alone_verdict').split(' (')[0]}.",
    ], width=54, fs=12)
    d.save(fig)

    # 9 — safety
    fig, ax = d.new("Safety, privacy and responsible AI", "Safeguards")
    d.bullets(ax, 0.6, 5.9, [
        "No automatic decline. Strongest action: a hold that says the money has not left, a 30-minute human "
        "review, a release request and an appeal.",
        f"False warnings concentrate on alarming-but-legitimate payments: {n('fpr.hard_negative_all')} vs "
        f"{n('fpr.benign')} on ordinary traffic — measured, not hidden.",
        f"Bias: worst segment false-warning rate {n('fairness.max_fpr_ratio')} the overall rate; "
        f"{n('fairness.n_flags')} segment(s) over the 2x tolerance.",
        "Misuse tests in CI: rate-limited scoring with bands only (no score to the client); fresh-account "
        "complaints ignored; complaints alone cannot hold a payee; analyst evidence access audited.",
        "Privacy: salted-HMAC ids, no message content, retention TTL, consented signals stay missing (never zero) "
        "without consent; DPDP-style purpose limitation and minimisation.",
    ], width=60, fs=13)
    d.image(fig, F / "F9_false_positives.png", [0.47, 0.1, 0.52, 0.72])
    d.save(fig)

    # 10 — adoption & finale
    fig, ax = d.new("Adoption, integration and the 48-hour plan", "Adoption & feasibility")
    lat = (f"Measured latency (laptop, loopback): stored payment p95 {n('latency.stored_txn.server_compute.p95_ms')}; "
           f"online event with feature recomputation p95 {n('latency.online_event.server_compute.p95_ms')}.")
    d.bullets(ax, 0.6, 5.9, [
        "Where it runs: PSP / issuer bank, as an approval-screen SDK hook plus a scoring API (FastAPI, single "
        "model bundle, run id on every response). Owner: the fraud-risk team.",
        "Rollout: shadow mode -> A1 nudges -> A2 warnings with cooling-off -> A3 holds sized to analyst capacity.",
        lat,
        f"Capacity: holds {n('budget.holds_A3_per_1000')} vs analyst capacity "
        f"{n('budget.analyst_capacity_k_per_1000')} per 1,000 payments — over capacity on the test window; a "
        "capacity guard (overflow holds become warnings) is finale task 3.",
    ], width=60, fs=13)
    d.bullets(ax, 7.0, 5.9, [
        "Already built and tested: generator, pipeline, API, offline console, Scam Studio, replay bundle.",
        "Finale, 48 h:",
        "  1. Judges compose a scam live (variant x evasion x segment) and watch where Citadel fires",
        "  2. Add a new script to the grammar; measure it as unseen",
        "  3. Capacity guard + per-segment thresholds",
        "  4. Hindi/English customer screens with senior-friendly large text",
        "  5. 3 / 5 / 8-minute demo, offline replay as fallback",
    ], width=52, fs=13)
    d.save(fig)

    # appendix A — limitations
    fig, ax = d.new("Appendix A — Known limitations (generated)", "Appendix")
    clean = [s.replace("**", "") for s in lims]
    d.bullets(ax, 0.6, 6.0, clean[:9], width=135, fs=11.5, gap=0.08)
    d.save(fig)

    # appendix B — disclosure
    fig, ax = d.new("Appendix B — Disclosure", "Appendix")
    d.bullets(ax, 0.6, 6.0, [
        f"Data: 100% synthetic from the Citadel generator; {n('data.n_rows')} payments, {n('data.n_users')} customers, "
        f"{n('data.n_episodes')} scam episodes; seed {ctx.cfg.seed}, config {ctx.prov.config_hash}, library "
        f"{ctx.prov.library_hash}, run {ctx.prov.run_id}. No real bank, customer or platform data; nothing scraped.",
        "Models: scikit-learn HistGradientBoosting, LogisticRegression, IsolationForest (candidate), isotonic "
        "calibration. No pretrained models; no LLM in the scoring path.",
        "Reused components: none; all code written for Citadel. Libraries pinned in requirements.txt.",
        "AI assistance: code, configuration and documentation written with an AI coding assistant under the team's "
        "direction; all numbers come from the pipeline and are checked against numbers_index.md.",
        "Assumptions (labelled, swept): cost model, heed rates, label latency, analyst capacity, signal observability.",
        "Not collected: message/call content, contact names, location, the lure, raw account numbers.",
    ], width=135, fs=12, gap=0.14)
    d.save(fig)
    d.close()
    return out / "Citadel_Round1.pdf"
