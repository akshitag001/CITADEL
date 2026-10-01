"""P6/P7 training stage: channels -> calibration -> channel-alone diagnostic -> fusion ablation ->
ladder -> abstention -> reason-family weights -> ONE serving bundle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..config import CONFIG_DIR
from ..features.registry import FAMILIES, FEATURES, family_features
from ..generate.fidelity import recall_at_fpr
from ..runctx import RunCtx
from .bundle import Bundle
from .fusion import ARMS, build_arms, pr_auc
from .ladder import LadderCollapsedError, fit_ladder
from .model import channel_alone, fit_calibrator, fit_if, fit_l1, fit_l2
from .reasons import load_messages
from .split import Split, make_split


@dataclass
class TrainResult:
    bundle: Bundle
    split: Split
    X: pd.DataFrame
    tx: pd.DataFrame


def family_importance(l1, X: pd.DataFrame, y: np.ndarray, mask: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Permutation importance of each feature FAMILY on the stats slice (PR-AUC drop)."""
    r = np.random.default_rng(seed)
    Xs = X[mask].reset_index(drop=True)
    ys = y[mask]
    base = average_precision_score(ys, l1.raw(Xs)) if ys.sum() else 0.0
    out = {}
    for fam in FAMILIES:
        cols = family_features(fam)
        Xp = Xs.copy()
        perm = r.permutation(len(Xp))
        Xp[cols] = Xp[cols].to_numpy()[perm]
        out[fam] = float(max(base - average_precision_score(ys, l1.raw(Xp)), 0.0)) if ys.sum() else 0.0
    # rule and model codes always outrank feature cards
    out["rule"], out["model"] = 10.0, 9.0
    return out


def train(ctx: RunCtx, tx: pd.DataFrame, X: pd.DataFrame, episodes: pd.DataFrame) -> TrainResult:
    cfg = ctx.cfg
    meta = ctx.gen_meta()
    y = tx["is_attack"].to_numpy()
    sp = make_split(tx, episodes, cfg, meta["t0"])
    ctx.csv(sp.table(y), "defend_split.csv")

    # ---- channels ---------------------------------------------------------------------------------
    l1 = fit_l1(X, y, sp.inner_fit, sp.inner_valid, sp.train, cfg.model, seed=cfg.seed)
    l2 = fit_l2(X, y, sp.train)
    iforest = fit_if(X, y, sp.train, seed=cfg.seed)
    raw = {"L1": l1.raw(X), "L2": l2.raw(X), "IF": iforest.raw(X)}
    cals = {k: fit_calibrator(v[sp.calib], y[sp.calib]) for k, v in raw.items()}
    cal = {k: cals[k](v) for k, v in raw.items()}
    pd.DataFrame({"iteration_checkpoint": np.arange(1, len(l1.valid_curve) + 1),
                  "inner_valid_pr_auc": l1.valid_curve, "chosen_n_iter": l1.n_iter}).pipe(
        lambda d: ctx.csv(d, "defend_l1_early_stopping.csv"))

    # ---- channel-alone diagnostic (stats slice decides; test reported) ----------------------------
    base_stats = float(y[sp.stats].mean())
    ca_stats = channel_alone(y[sp.stats], {k: v[sp.stats] for k, v in cal.items()}, base_stats)
    ca_stats["slice"] = "stats"
    ca_test = channel_alone(y[sp.test], {k: v[sp.test] for k, v in cal.items()}, float(y[sp.test].mean()))
    ca_test["slice"] = "test"
    ctx.csv(pd.concat([ca_stats, ca_test], ignore_index=True), "defend_channels_alone.csv")
    dead = set(ca_stats.loc[ca_stats.dead, "channel"])

    # ---- fusion ablation ---------------------------------------------------------------------------
    arms = build_arms({k: v[sp.calib] for k, v in cal.items()}, y[sp.calib], dead)
    rows, eligible = [], {}
    for name in ARMS:
        fz = arms[name]
        f_all = fz(cal)
        fcal = fit_calibrator(f_all[sp.calib], y[sp.calib])
        risk = fcal(f_all)
        try:
            lad = fit_ladder(risk[sp.stats], cfg.ladder_budgets)
            ok = True
        except LadderCollapsedError:
            lad, ok = None, False
        row = {"arm": name, "channels": "+".join(fz.channels), "ladder_non_degenerate": ok,
               "stats_pr_auc": pr_auc(y[sp.stats], risk[sp.stats]),
               "test_pr_auc": pr_auc(y[sp.test], risk[sp.test]),
               "test_recall_at_0.5pct_fpr": recall_at_fpr(y[sp.test], risk[sp.test], 0.005)[0],
               "test_recall_at_0.1pct_fpr": recall_at_fpr(y[sp.test], risk[sp.test], 0.001)[0],
               "n_test_rows": int(sp.test.sum()), "n_test_positives": int(y[sp.test].sum())}
        rows.append(row)
        if ok:
            eligible[name] = (row["stats_pr_auc"], fz, fcal, lad, risk)
    arms_df = pd.DataFrame(rows)
    winner = max(eligible, key=lambda k: eligible[k][0])
    arms_df["selected"] = arms_df["arm"] == winner
    arms_df["selection_rule"] = "max stats-slice PR-AUC subject to a non-degenerate ladder (test never used)"
    ctx.csv(arms_df, "defend_fusion_arms.csv")
    _, fusion, final_cal, ladder, risk = eligible[winner]

    # ---- abstention (priced, capped) ---------------------------------------------------------------
    refs = {k: np.sort(cal[k][sp.stats]) for k in ("L1", "L2")}
    lvl = ladder.level(risk)
    r1 = np.searchsorted(refs["L1"], cal["L1"], side="right") / len(refs["L1"])
    r2 = np.searchsorted(refs["L2"], cal["L2"], side="right") / len(refs["L2"])
    dis = np.abs(r1 - r2)
    cand = sp.stats & (lvl == 2) & (dis >= cfg.ladder_budgets.abstain_disagreement)
    n_stats = int(sp.stats.sum())
    cap = int(np.floor(cfg.ladder_budgets.abstain_max_share * n_stats))
    if cand.sum() > cap:
        min_risk = float(np.sort(risk[cand])[::-1][max(cap - 1, 0)]) if cap > 0 else float("inf")
    else:
        min_risk = 0.0
    abstain = {"disagreement": cfg.ladder_budgets.abstain_disagreement, "min_risk": min_risk,
               "max_share": cfg.ladder_budgets.abstain_max_share}

    fam_w = family_importance(l1, X, y, sp.stats, seed=cfg.seed)
    ctx.csv(pd.DataFrame([{"family": k, "pr_auc_drop_when_permuted": v} for k, v in fam_w.items()]),
            "defend_family_importance.csv")

    bundle = Bundle(l1=l1, l2=l2, iforest=iforest if "IF" in fusion.channels else None,
                    calibrators={k: v for k, v in cals.items() if k in ("L1", "L2") or k in fusion.channels},
                    fusion=fusion, final_calibrator=final_cal, ladder=ladder, abstain=abstain, rank_refs=refs,
                    family_weight=fam_w, features=list(FEATURES),
                    l0_yaml=(CONFIG_DIR / "l0_rules.yaml").read_text(encoding="utf-8"),
                    reasons_yaml=(CONFIG_DIR / "reason_codes.yaml").read_text(encoding="utf-8"),
                    messages=load_messages(), provenance=ctx.prov.to_dict())
    bundle.save(ctx.art / "bundle.pkl")
    lad_rows = [{"action": a, "budget_share": ladder.budgets[a], "threshold_on_risk": ladder.thresholds[a],
                 "realised_share_stats": ladder.realised_shares[a],
                 "realised_share_test": float((risk[sp.test] >= ladder.thresholds[a]).mean()),
                 "n_stats_rows": n_stats, "n_test_rows": int(sp.test.sum())} for a in ("A1", "A2", "A3")]
    ctx.csv(pd.DataFrame(lad_rows), "defend_ladder_thresholds.csv")
    return TrainResult(bundle, sp, X, tx)
