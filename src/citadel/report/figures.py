"""Round 1 figures F1-F9. One message per figure, large fonts, labelled axes, one y-axis per chart.

Colours follow the validated reference palette (light mode, print): categorical slots in fixed order
for identity, a one-hue blue ordinal ramp for the ordered action ladder, text in ink tokens only.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"            # categorical slots 1-3
LADDER = {"A0": "#c9c8c3", "A1": "#86b6ef", "A2": "#3987e5", "A3": "#0d366b"}  # ordinal blue ramp
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "font.family": "DejaVu Sans",
})


def _save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _box(ax, x, y, w, h, text, fc="#ffffff", ec=INK2, fs=11, bold=False, color=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, a, b, color=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4, ls=ls))


# ---- F1 ----------------------------------------------------------------------------------------------
def f1_architecture(out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(15, 7.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 0.35), 11.25, 6.85, boxstyle="round,pad=0.02", fc="none", ec=S1, lw=1.6, ls="--"))
    ax.text(0.35, 6.95, "PSP / bank boundary · salted-HMAC ids · no message content · retention TTL",
            fontsize=11, color=S1, fontweight="bold")
    _box(ax, 0.4, 4.9, 2.2, 1.6, "UPI event\npay · collect · QR\namount · time", fs=11)
    _box(ax, 0.4, 3.0, 2.2, 1.6, "Switch-side state\npayee history\ndevice + SIM", fs=11)
    _box(ax, 0.4, 1.0, 2.2, 1.7, "Consented signals\nscreen share · call\n(missing ≠ 0)", fs=11, ls="--")
    _box(ax, 3.1, 2.6, 2.0, 2.7, "Causal features\n64, trailing,\nexclude self\n\nuser · sequence\npayee · graph", fs=11)
    _box(ax, 5.6, 5.1, 2.1, 1.2, "L0 rules\nno model needed", fs=11, fc="#eef4fc")
    _box(ax, 5.6, 3.35, 2.1, 1.2, "L1 boosted trees\ncalibrated", fs=11, fc="#eef4fc")
    _box(ax, 5.6, 1.6, 2.1, 1.2, "L2 payee /\nmule graph", fs=11, fc="#eef4fc")
    _box(ax, 8.2, 3.0, 1.6, 1.9, "Fusion\n(chosen by\nablation)\n→ risk", fs=11)
    _box(ax, 10.1, 4.4, 1.15, 2.0, "Ladder\nA0–A3\nalert\nbudget", fs=10, fc="#e8f0fb")
    _box(ax, 10.1, 1.5, 1.15, 2.5, "Reason\ncodes\n(closed\nlist)\nen / hi", fs=10, fc="#e8f0fb")
    _box(ax, 11.8, 5.1, 3.0, 1.5, "Customer screen\nband + reasons,\nnever a score\nnudge · warn · hold", fs=11, fc="#ffffff", ec=S1)
    _box(ax, 11.8, 2.9, 3.0, 1.6, "Analyst queue\nevidence · audit log\nrelease · confirm", fs=11, fc="#ffffff", ec=S1)
    _box(ax, 11.8, 0.8, 3.0, 1.5, "Feedback + appeals\nlabels arrive late\n→ retrain", fs=11, fc="#ffffff", ec=S1)
    for y in (5.7, 3.8, 1.85):
        _arrow(ax, (2.6, y), (3.1, 3.95))
    for y in (5.7, 3.95, 2.2):
        _arrow(ax, (5.1, 3.95), (5.6, y))
    _arrow(ax, (7.7, 3.95), (8.2, 3.95))
    _arrow(ax, (7.7, 2.2), (8.2, 3.4))
    _arrow(ax, (7.7, 5.7), (10.1, 5.6))
    _arrow(ax, (9.8, 4.3), (10.1, 5.0))
    _arrow(ax, (9.8, 3.6), (10.1, 3.0))
    _arrow(ax, (11.25, 5.4), (11.8, 5.8))
    _arrow(ax, (11.25, 4.6), (11.8, 3.8))
    _arrow(ax, (13.3, 2.9), (13.3, 2.3))
    ax.plot([11.8, 4.1], [1.2, 1.2], color=S2, lw=1.4, ls="--")
    _arrow(ax, (4.1, 1.2), (4.1, 2.58), color=S2, ls="--")
    ax.text(7.9, 0.8, "human-review & feedback path (late labels, dispositions, appeals)", color=S2, fontsize=11,
            ha="center")
    ax.text(0.35, 0.05, "No automatic decline anywhere: the strongest action is a hold with a human-release path.",
            fontsize=12, color=INK, fontweight="bold")
    return _save(fig, out, "F1_architecture")


# ---- F2 ----------------------------------------------------------------------------------------------
def f2_sequence(out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    steps = [("S1  Lure", "call · SMS · chat\n(AI makes it perfect)", "NOT observed by the bank", "#ffffff"),
             ("S2  Pressure", "remote access · screen share\nnew device · SIM swap · 'fee'", "session signals", "#ffffff"),
             ("S3  Collect / pay a new payee", "the money is about to move", "CITADEL ACTS HERE", "#e8f0fb"),
             ("S4  Mule fan-out", "money split to 2-6 accounts\nwithin minutes", "payee-side graph", "#ffffff")]
    for i, (t, d, obs, fc) in enumerate(steps):
        x = 0.3 + i * 3.7
        _box(ax, x, 2.3, 3.3, 2.0, f"{t}\n\n{d}", fs=12, fc=fc, ec=S1 if i == 2 else INK2, bold=False)
        ax.text(x + 1.65, 1.95, obs, ha="center", fontsize=12, color=S1 if i == 2 else INK2,
                fontweight="bold" if i == 2 else "normal")
        if i < 3:
            _arrow(ax, (x + 3.3, 3.3), (x + 3.7, 3.3))
    ax.text(0.3, 4.75, "A scam is a sequence, not a transaction", fontsize=16, fontweight="bold")
    _box(ax, 7.7, 0.15, 3.3, 1.45, "Customer sees:\n“Stop. This looks like a scam pattern.”\n+ reasons, cooling-off, call your bank",
         fs=10.5, fc="#ffffff", ec=S1)
    _arrow(ax, (9.35, 1.85), (9.35, 1.62), color=S1)
    ax.text(0.3, 0.6, "Lures get better with AI. The payment behaviour at\nthe moment of harm still shows.", fontsize=12, color=INK2)
    return _save(fig, out, "F2_scam_sequence")


# ---- F3 ----------------------------------------------------------------------------------------------
def f3_journey(out: Path, msgs: dict) -> list[Path]:
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    rows = [("Without Citadel", 3.6, [
        ("Collect request\n“Refund ₹4,999”\nfrom unknown number", "#ffffff"),
        ("Approve &\nenter UPI PIN", "#ffffff"),
        ("₹4,999 SENT\n(the victim thought\nthey would receive)", "#fde8e7"),
        ("Mule splits it to\n4 accounts in minutes.\nUsually irreversible.", "#fde8e7")]),
        ("With Citadel", 0.4, [
            ("Same request,\nsame moment", "#ffffff"),
            ("\n".join(textwrap.wrap(msgs["actions"]["A2"]["title"]["en"], 22))
             + "\n• request from someone\n  not in your contacts\n• you are on a call", "#fff3e0"),
            ("“Approving this request\nSENDS money. You never\nneed a PIN to receive.”\n60 s cooling-off", "#fff3e0"),
            ("Call bank / ask family\n→ cancel. Money safe.\nGenuine? Continue after\nthe wait.", "#e7f5ec")])]
    for label, y, cards in rows:
        ax.text(0.2, y + 2.55, label, fontsize=15, fontweight="bold", color=S2 if label.startswith("Without") else S1)
        for i, (t, fc) in enumerate(cards):
            x = 0.2 + i * 3.7
            _box(ax, x, y, 3.2, 2.3, t, fs=12, fc=fc)
            if i < 3:
                _arrow(ax, (x + 3.2, y + 1.15), (x + 3.7, y + 1.15))
    return _save(fig, out, "F3_before_after")


# ---- F4 ----------------------------------------------------------------------------------------------
def f4_ladder(out: Path, ladder: pd.DataFrame, budget: pd.Series) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    levels = ["A1", "A2", "A3"]
    names = {"A1": "A1 nudge", "A2": "A2 warning +\ncooling-off", "A3": "A3 hold +\nanalyst"}
    lad = ladder.set_index("action")
    meas = [lad.loc[a, "per_1000_payments"] for a in levels]
    bud = [lad.loc[a, "budget_share"] * 1000 for a in levels]
    x = np.arange(3)
    ax.bar(x, meas, width=0.55, color=[LADDER[a] for a in levels], zorder=3, edgecolor=SURFACE, linewidth=2)
    for i, (m, b) in enumerate(zip(meas, bud)):
        ax.hlines(b, i - 0.36, i + 0.36, color=INK, lw=2, linestyles="--", zorder=4)
        ax.text(i + 0.38, b, f"budget {b:g}", va="center", fontsize=11, color=INK2)
        ax.text(i, m, f"{m:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
    cap = budget["analyst_capacity_k_per_1000"]
    ax.hlines(cap, 1.62, 2.38, color=S2, lw=2.2, zorder=4)
    ax.text(1.6, cap, f"analyst capacity {cap:.2f}", ha="right", va="center", fontsize=11, color=S2)
    ax.set_xticks(x, [names[a] for a in levels])
    ax.set_yscale("log")
    ax.set_ylabel("alerts per 1,000 payments (log)")
    ax.set_title("Alert volume on the test window vs the budgets that set the thresholds")
    ax.text(0, -0.24, "No decline level exists. Thresholds come from volume budgets on a held-out slice, not raw scores.",
            transform=ax.transAxes, fontsize=11, color=INK2)
    return _save(fig, out, "F4_ladder_budget")


# ---- F5 ----------------------------------------------------------------------------------------------
def f5_operating(out: Path, curve: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    c = curve.sort_values("alerts_per_1000")
    ax.plot(c.alerts_per_1000, c.recall, color=S1, lw=2.2, zorder=3)
    for _, r in c[c.ladder_point.fillna("") != ""].iterrows():
        a = r.ladder_point
        ax.axvline(r.alerts_per_1000, color=LADDER[a], ls="--", lw=1.6, zorder=2)
        ax.scatter([r.alerts_per_1000], [r.recall], s=70, color=LADDER[a], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"{a} budget: {r.recall:.0%} of scams", (r.alerts_per_1000, r.recall), xytext=(8, -18),
                    textcoords="offset points", fontsize=11, color=INK)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("alerts per 1,000 payments (log scale)")
    ax.set_ylabel("share of scam payments alerted")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Recall rises fast at low alert volume, then flattens")
    return _save(fig, out, "F5_operating_curve")


# ---- F6 ----------------------------------------------------------------------------------------------
def _whisker_bars(ax, labels, vals, lo, hi, colors):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.6, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none",
                ecolor=INK, capsize=4, lw=1.3, zorder=4)
    for yy, v in zip(y, vals):
        ax.text(min(v + 0.02, 0.9), yy + 0.3, f"{v:.0%}", fontsize=12, color=INK, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def f6_generalisation(out: Path, gen: pd.DataFrame, lovo: pd.DataFrame | None) -> list[Path]:
    fig, axes = plt.subplots(1, 2 if lovo is not None and len(lovo) else 1, figsize=(15, 5.4))
    axes = np.atleast_1d(axes)
    g = gen.set_index("group")
    order = [i for i in g.index if i == "seen_variants"] + [i for i in g.index if i.startswith("withheld_variant")] + \
            [i for i in g.index if i == "no_evasion"] + [i for i in g.index if i.startswith("withheld_evasion")]
    labels = [f"{i.replace('_', ' ')}\n(n={int(g.loc[i, 'denominator'])}"
              + (", too few to claim)" if g.loc[i, "denominator"] < 20 else ")") for i in order]
    cols = [S2 if i.startswith("withheld") else S1 for i in order]
    _whisker_bars(axes[0], labels, g.loc[order, "value"], g.loc[order, "ci_low"], g.loc[order, "ci_high"], cols)
    axes[0].set_title("Sealed holdout: recall at the deployed ladder (A2+)")
    axes[0].set_xlabel("scam payments warned or held · bars: 95% Wilson CI")
    if len(axes) > 1:
        lv = lovo.sort_values("recall_at_0.5pct_fpr_when_withheld")
        y = np.arange(len(lv))
        axes[1].barh(y + 0.2, lv["recall_at_0.5pct_fpr_when_trained"], height=0.38, color=S1, label="trained on it",
                     zorder=3, edgecolor=SURFACE, linewidth=2)
        axes[1].barh(y - 0.2, lv["recall_at_0.5pct_fpr_when_withheld"], height=0.38, color=S2, label="never seen",
                     zorder=3, edgecolor=SURFACE, linewidth=2)
        axes[1].set_yticks(y, [f"{v} (n={n})" for v, n in zip(lv.variant, lv.n_positives)])
        axes[1].set_xlim(0, 1)
        axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        axes[1].set_title("Leave-one-variant-out (L1 alone, 0.5% FPR)")
        axes[1].set_xlabel("")
    fig.tight_layout()
    return _save(fig, out, "F6_generalisation")


# ---- F7 ----------------------------------------------------------------------------------------------
def f7_ablation(out: Path, abl: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    a = abl.copy()
    a["label"] = a.step.str.replace("ablation: ", "", regex=False)
    flat = a.verdict.fillna("").isin(["no measured effect", "worsens"])
    scarce = a.step.str.startswith("ablation")
    colors = [S2 if f else (S3 if s else S1) for f, s in zip(flat, scarce)]
    y = np.arange(len(a))[::-1]
    ax.barh(y, a.pr_auc, color=colors, height=0.6, zorder=3, edgecolor=SURFACE, linewidth=2)
    for yy, v, f, vd in zip(y, a.pr_auc, flat, a.verdict.fillna("")):
        ax.text(v + 0.01, yy, f"{v:.3f}" + ("  ← " + vd if f else ""), va="center", fontsize=12, color=INK)
    ax.set_yticks(y, a.label)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC on the test window (base rate < 1%)")
    ax.set_title("Each feature layer, added in order — rows with no measured effect are kept")
    return _save(fig, out, "F7_layer_ablation")


# ---- F8 ----------------------------------------------------------------------------------------------
def f8_time_to_alert(out: Path, ep: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(ep)
    cats = [("before any money moved\n(first payment)", (ep.first_alert_index == 0).sum()),
            ("by the first harmful\npayment (S3)", ((ep.first_alert_index > 0) & (ep.caught_by_first_harmful_payment == 1)).sum()),
            ("on a later payment", ((ep.caught_any_A2plus == 1) & (ep.caught_by_first_harmful_payment == 0)).sum()),
            ("never (missed)", (ep.caught_any_A2plus == 0).sum())]
    vals = np.array([c[1] for c in cats]) / max(n, 1)
    cols = [S1, S1, "#86b6ef", S2]
    ax.bar(range(4), vals, color=cols, zorder=3, edgecolor=SURFACE, linewidth=2, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(range(4), [c[0] for c in cats])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel(f"share of {n} test-window scam episodes")
    ax.set_title("Where in the sequence the first warning or hold fires")
    return _save(fig, out, "F8_time_to_alert")


# ---- F9 ----------------------------------------------------------------------------------------------
def f9_fpr(out: Path, hn: pd.DataFrame, fair: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    h = hn[hn.population.str.startswith("archetype:") | hn.population.isin(["benign"])].copy()
    h["label"] = h.population.str.replace("archetype:", "", regex=False).str.replace("_", " ")
    h = h.sort_values("fpr_A2plus")
    y = np.arange(len(h))
    cols = [S1 if p == "benign" else S2 for p in h.population]
    axes[0].barh(y, h.fpr_A2plus, color=cols, height=0.62, zorder=3, edgecolor=SURFACE, linewidth=2)
    axes[0].errorbar(h.fpr_A2plus, y, xerr=[h.fpr_A2plus - h.fpr_A2plus_ci_low, h.fpr_A2plus_ci_high - h.fpr_A2plus],
                     fmt="none", ecolor=INK, capsize=3, lw=1.1, zorder=4)
    axes[0].set_yticks(y, [f"{lab} (n={n:,})" for lab, n in zip(h.label, h.n_rows)])
    axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[0].set_title("Legitimate look-alikes: false warnings (A2+)")
    axes[0].set_xlabel("false-positive rate at A2+ · 95% CI")
    f = fair.copy()
    f["label"] = f.group.str.replace("_", " ") + ": " + f.value_of.astype(str)
    f = f.sort_values("fpr_A2plus")
    y = np.arange(len(f))
    axes[1].barh(y, f.fpr_A2plus, color=S1, height=0.62, zorder=3, edgecolor=SURFACE, linewidth=2)
    axes[1].errorbar(f.fpr_A2plus, y, xerr=[f.fpr_A2plus - f.fpr_ci_low, f.fpr_ci_high - f.fpr_A2plus], fmt="none",
                     ecolor=INK, capsize=3, lw=1.1, zorder=4)
    axes[1].axvline(f.overall_fpr_A2plus.iloc[0], color=INK2, ls="--")
    axes[1].set_yticks(y, f.label)
    axes[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=2))
    axes[1].set_title("False warnings by customer segment (dashed: overall)")
    axes[1].set_xlabel("false-positive rate at A2+ · 95% CI")
    fig.tight_layout()
    return _save(fig, out, "F9_false_positives")


def all_figures(art: Path, out: Path, msgs: dict) -> dict[str, list[Path]]:
    rd = lambda n: pd.read_csv(art / n)  # noqa: E731
    lovo = rd("eval_lovo.csv") if (art / "eval_lovo.csv").exists() else None
    return {
        "F1": f1_architecture(out), "F2": f2_sequence(out), "F3": f3_journey(out, msgs),
        "F4": f4_ladder(out, rd("defend_ladder.csv"), rd("defend_alert_budget.csv").iloc[0]),
        "F5": f5_operating(out, rd("eval_operating_curve.csv")),
        "F6": f6_generalisation(out, rd("eval_generalisation.csv"), lovo),
        "F7": f7_ablation(out, rd("eval_layer_ablation.csv")),
        "F8": f8_time_to_alert(out, rd("eval_episode_trace.csv")),
        "F9": f9_fpr(out, rd("eval_hard_negative_fpr.csv"), rd("eval_fairness.csv")),
    }
