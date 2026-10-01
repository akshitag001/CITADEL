"""Stage functions behind ``python -m citadel <stage>``. Each stage reads the previous stage's files and
writes machine-readable outputs to data/<run>/ or artifacts/<run>/."""

from __future__ import annotations

import time

import pandas as pd

from .features.matrix import build_matrix, known_mules_from_labels, profiles_from_accounts
from .features.registry import registry_frame
from .generate import artefacts, fidelity
from .generate.campaign import generate, write
from .generate.library import load_library
from .runctx import RunCtx


def log(msg: str) -> None:
    print(f"[citadel {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_generate(ctx: RunCtx) -> None:
    log(f"generate: profile={ctx.cfg.profile} seed={ctx.cfg.seed}")
    res = generate(ctx.cfg, load_library())
    write(res, ctx.data)
    ctx.json(res.summary, "generate_summary.json")
    ctx.json(ctx.cfg.to_dict(), "config_resolved.json")
    log(f"generate: {res.summary['n_rows']} rows, {res.summary['n_episodes']} episodes, "
        f"realised attack share {res.summary['realised_attack_share']:.4%}")


def load_features(ctx: RunCtx) -> pd.DataFrame:
    return pd.read_parquet(ctx.art / "features.parquet")


def stage_features(ctx: RunCtx) -> pd.DataFrame:
    tx = ctx.read("transactions.parquet")
    X = build_matrix(tx, profiles_from_accounts(ctx.read("accounts.parquet")),
                     known_mules_from_labels(ctx.read("labels.parquet")))
    out = X.copy()
    out.insert(0, "txn_id", tx["txn_id"].to_numpy())
    out.to_parquet(ctx.art / "features.parquet", index=False)
    ctx.csv(registry_frame(), "feature_registry.csv")
    log(f"features: {X.shape[1]} features x {len(X)} rows")
    return X


def stage_fidelity(ctx: RunCtx, X: pd.DataFrame | None = None) -> dict:
    tx = ctx.read("transactions.parquet")
    sf = fidelity.single_feature_probe(tx)
    ctx.csv(sf, "generate_single_feature_auc.csv")
    hunt = artefacts.hunt(tx)
    ctx.csv(hunt, "generate_artefact_hunt.csv")
    ctx.csv(fidelity.realism_scores(tx), "generate_fidelity_scores.csv")
    refits = 5 if ctx.reportable() else 2
    raw = fidelity.raw_joint_probe(tx, refits=refits)
    out = {"single_feature_max_auc": float(sf["auc"].max()), "single_feature_gate": fidelity.SINGLE_FEATURE_GATE,
           "single_feature_passes": bool(sf["passes"].all()), "artefacts_found": int(len(hunt)),
           "raw_joint_probe": raw}
    if X is None and (ctx.art / "features.parquet").exists():
        X = load_features(ctx).drop(columns=["txn_id"])
    if X is not None:
        y, ts, m = tx["is_attack"].to_numpy(), tx["ts"].to_numpy(), fidelity.eval_mask(tx)
        derived = fidelity.joint_probe(X, y, ts, m, refits=refits)
        out["derived_joint_probe"] = derived
        out["raw_vs_derived_gap"] = derived["recall_at_fpr"] - raw["recall_at_fpr"]
        out["label_shuffle_null"] = fidelity.label_shuffle_null(X, y, ts, m)
        out["leakage_canary"] = fidelity.leakage_canary(X, y, ts, m)
        ctx.csv(fidelity.contribution_shares(tx, derived["recall_at_fpr"]), "generate_contribution_shares.csv")
    out["reportable"] = ctx.reportable()
    ctx.json(out, "generate_fidelity_summary.json")
    log(f"fidelity: max single AUC {out['single_feature_max_auc']:.3f}, artefacts {len(hunt)}, raw probe "
        f"{raw['recall_at_fpr']:.3f}" + (f", derived {out['derived_joint_probe']['recall_at_fpr']:.3f}" if X is not None else ""))
    return out


def stage_train(ctx: RunCtx, X: pd.DataFrame | None = None):
    from .defend.train import train
    tx = ctx.read("transactions.parquet")
    if X is None:
        X = load_features(ctx).drop(columns=["txn_id"])
    res = train(ctx, tx, X, ctx.read("episodes.parquet"))
    log(f"train: L1 {res.bundle.l1.n_iter} rounds, fusion arm '{res.bundle.fusion.arm}', bundle saved")
    return res


def stage_evaluate(ctx: RunCtx):
    from .evaluate.run import evaluate
    return evaluate(ctx)


def stage_all(ctx: RunCtx) -> None:
    t = time.time()
    stage_generate(ctx)
    X = stage_features(ctx)
    stage_fidelity(ctx, X)
    stage_train(ctx, X)
    stage_evaluate(ctx)
    log(f"run complete in {time.time() - t:.0f}s -> artifacts/{ctx.run}/REPORT.md")
