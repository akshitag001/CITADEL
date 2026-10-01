"""Orchestrate one synthetic dataset: population -> benign -> shapes -> hard negatives -> mules ->
scam episodes -> labels -> enrichment -> data/<run>/*.parquet.

The attack share is a TARGET; the REALISED share is measured and written to generation_summary.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, Config, rng
from ..schema import MULE_INITIATED_KINDS, validate_frame
from ..timeutil import DAY_S
from .attacks import build_episode, choose_template, segment_of, victim_weights
from .benign import generate_benign, session_fields
from .enrich import enrich
from .entities import Population, hex_ids, make_population
from .hard_negatives import apply_hard_negatives
from .library import ScamLibrary, load_library
from .mules import MuleRegistry
from .shapes import generate_shapes

INTERNAL_COLS = ["ts", "_payer", "_payee", "txn_type", "amount_inr", "session_duration_s",
                 "attempts_in_session", "collect_request_age_s", "screen_share_active",
                 "remote_access_app_detected", "call_in_progress", "payee_name_similarity_to_known_contact",
                 "_device_key", "_bound_ts", "_phonebook", "row_kind", "hn_archetype"]


@dataclass
class GenResult:
    transactions: pd.DataFrame
    internal: pd.DataFrame          # internal representation (for serving + the mutation test)
    pre_attack: pd.DataFrame        # internal rows before attack mutation (mutation test)
    episodes: pd.DataFrame
    labels: pd.DataFrame
    complaints: pd.DataFrame
    population: Population
    mules: pd.DataFrame
    summary: dict


def data_dir(run: str) -> Path:
    return ROOT / "data" / run


def _mule_rows(r, pop: Population, specs: list[tuple]) -> pd.DataFrame:
    if not specs:
        return pd.DataFrame(columns=INTERNAL_COLS)
    payer = np.array([s[0] for s in specs], dtype=np.int64)
    payee = np.array([s[1] for s in specs], dtype=np.int64)
    ts = np.array([s[2] for s in specs], dtype=np.int64)
    txn = np.full(len(specs), "pay", dtype=object)
    f = session_fields(r, pop, payer, payee, txn, ts)
    df = pd.DataFrame({"ts": ts, "_payer": payer, "_payee": payee, "txn_type": txn,
                       "amount_inr": [float(s[3]) for s in specs], **f,
                       "row_kind": [s[4] for s in specs], "hn_archetype": ""})
    if len(specs[0]) > 5:
        df["step_in_sequence"] = [s[5] for s in specs]
    return df


def run_attacks(cfg: Config, pop: Population, lib: ScamLibrary, base: pd.DataFrame,
                next_rid: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MuleRegistry, int]:
    r = rng(cfg, "attacks")
    acc = pop.accounts
    n_target_rows = cfg.attacks.target_share * len(base)
    n_episodes = max(6, int(round(n_target_rows / 2.2)))
    registry = MuleRegistry(pop, rng(cfg, "mules"), cfg.seed, max(6, int(n_episodes * cfg.attacks.mules_per_episode)))

    persons = acc[(acc.kind == "person") & (acc.txn_rate > 0) & (acc.informal_merchant == 0)].copy()
    persons = persons[~persons.idx.isin(list(registry.by_idx))]
    benign_rows = base[(base.row_kind == "benign") & (base.ts >= pop.t0 + DAY_S)]
    by_user = benign_rows.groupby("_payer").indices
    persons = persons[persons.idx.isin(list(by_user))]
    hist_by_user = base.groupby("_payer").indices

    variants = list(cfg.attacks.variants)
    vw = np.array([cfg.attacks.variant_weights.get(v, 1.0) for v in variants])
    plan = []
    used: set[int] = set()
    for _ in range(n_episodes):
        vid = variants[int(r.choice(len(variants), p=vw / vw.sum()))]
        evasion = None
        if r.random() < cfg.attacks.evasion_share:
            opts = lib.compatible_evasions(vid)
            evasion = str(r.choice(opts)) if opts else None
        w = victim_weights(lib, vid, persons) * (~persons.idx.isin(list(used))).to_numpy()
        victim = int(persons["idx"].to_numpy()[int(r.choice(len(persons), p=w / w.sum()))])
        used.add(victim)
        rows = benign_rows.iloc[by_user[victim]]
        t_i = choose_template(r, rows, cfg.attacks.night_bias, evasion == "time_of_day_mimicry")
        plan.append((int(rows.iloc[t_i]["_rid"]), vid, evasion, victim))
    rid_pos = pd.Series(np.arange(len(base)), index=base["_rid"].to_numpy())
    base_ts = base["ts"].to_numpy()
    plan.sort(key=lambda p: int(base_ts[int(rid_pos[p[0]])]))
    attack_rows, mule_specs, episodes, labels = [], [], [], []
    replaced: set[int] = set()
    scam_devices: list[int] = []
    acc_idx = acc.set_index("idx")
    for e, (rid, vid, evasion, victim) in enumerate(plan):
        if rid in replaced:
            continue
        template = base.iloc[int(rid_pos[rid])].to_dict()
        vrow = acc_idx.loc[victim].to_dict()
        hist = base.iloc[hist_by_user[victim]]
        eid = hex_ids(cfg.seed, "episode", np.array([e]))[0]
        res = build_episode(lib=lib, registry=registry, r=r, episode_id=eid, variant=vid, evasion=evasion,
                            template=template, victim=vrow, victim_history=hist,
                            label_visible_share=cfg.attacks.label_visible_share, scam_devices=scam_devices)
        replaced.add(rid)
        start_rid = next_rid
        for row in res.rows:
            row["_rid"] = next_rid
            next_rid += 1
        for lab in res.labels:
            lab["_rid"] = start_rid + lab.pop("row_index")
        seg_row = {"user_age_band": vrow["user_age_band"], "digital_literacy": vrow["digital_literacy"],
                   "tenure_days": (res.episode["s3_first_ts"] - vrow["created_ts"]) / DAY_S}
        res.episode["segment"] = segment_of(seg_row, lib)
        res.episode["victim_age_band"] = vrow["user_age_band"]
        res.episode["victim_literacy"] = vrow["digital_literacy"]
        attack_rows.extend(res.rows)
        for spec in res.mule_rows:
            mule_specs.append((*spec, eid))
        episodes.append(res.episode)
        labels.extend(res.labels)
    registry.flush()

    atk = pd.DataFrame(attack_rows)
    mr = _mule_rows(rng(cfg, "attacks/mule_rows"), pop, [s[:6] for s in mule_specs])
    if len(mr):
        mr["episode_id"] = [s[6] for s in mule_specs]
        mr["_rid"] = np.arange(next_rid, next_rid + len(mr))
        next_rid += len(mr)
    seas = _mule_rows(rng(cfg, "attacks/seasoning"), pop,
                      [(p, q, t, a, "seasoning") for p, q, t, a in registry.seasoning])
    if len(seas):
        seas["_rid"] = np.arange(next_rid, next_rid + len(seas))
        next_rid += len(seas)
    if len(mr):  # outbound payments of accounts at another institution are not in our stream
        ours = pop.accounts.set_index("idx")["our_customer"].reindex(mr["_payer"]).to_numpy() == 1
        mr = mr[ours]
    keep = base[~base._rid.isin(replaced)]
    out = pd.concat([keep, atk, mr, seas], ignore_index=True)
    out = out[(out.ts >= pop.t0) & (out.ts < pop.t_end)]
    # frozen (confirmed) mule accounts can no longer send or receive non-scam payments
    frozen = registry.frozen()
    if frozen:
        fz = pd.Series(frozen, dtype="int64")
        to_f = out["_payee"].map(fz).to_numpy(dtype=float)
        from_f = out["_payer"].map(fz).to_numpy(dtype=float)
        blocked = ((out["ts"].to_numpy() >= to_f) | (out["ts"].to_numpy() >= from_f)) & (out["row_kind"] != "attack").to_numpy()
        out = out[~blocked]
    return out, pd.DataFrame(episodes), pd.DataFrame(labels), registry, next_rid


def benign_disputes(cfg: Config, pop: Population, df: pd.DataFrame) -> pd.DataFrame:
    """Genuine customer disputes against legitimate merchants and sellers (false-positive generator for R08)."""
    r = rng(cfg, "labels/disputes")
    acc = pop.accounts
    merch_like = (acc["kind"].to_numpy() == "merchant") | (acc["informal_merchant"].to_numpy() == 1)
    pool = df[(df.row_kind.isin(["benign", "shape", "hard_negative"])) & merch_like[df["_payee"].to_numpy()]]
    sel = pool[r.random(len(pool)) < 0.0025]
    return pd.DataFrame({"payee_idx": sel["_payee"].to_numpy(), "reporter_idx": sel["_payer"].to_numpy(),
                         "arrival_ts": (sel["ts"].to_numpy() + r.uniform(1, 10, len(sel)) * DAY_S).astype(np.int64),
                         "source": "dispute"})


def generate(cfg: Config, lib: ScamLibrary | None = None) -> GenResult:
    lib = lib or load_library()
    pop = make_population(cfg)
    benign = generate_benign(cfg, pop)
    shapes = generate_shapes(cfg, pop, len(benign))
    base = pd.concat([benign, shapes], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    base = apply_hard_negatives(cfg, pop, base)
    base["_rid"] = np.arange(len(base), dtype=np.int64)
    pre_attack = base.copy()
    df, episodes, labels, registry, _ = run_attacks(cfg, pop, lib, base, len(base))

    # truth defaults + ids
    for c, d in (("is_attack", 0), ("step_in_sequence", 0), ("label_visible", 0)):
        df[c] = df[c].fillna(d).astype(np.int8) if c in df else np.int8(d)
    rid_to_txn = dict(zip(df["_rid"].to_numpy(), hex_ids(cfg.seed, "txn", df["_rid"].to_numpy())))
    df["txn_id"] = df["_rid"].map(rid_to_txn)
    pre_attack["txn_id"] = hex_ids(cfg.seed, "txn", pre_attack["_rid"].to_numpy())
    src_map = dict(zip(pre_attack["_rid"], pre_attack["txn_id"]))
    df["source_txn_id"] = df["source_rid"].map(src_map).fillna("") if "source_rid" in df else ""
    df.loc[df.row_kind.isin(MULE_INITIATED_KINDS), "step_in_sequence"] = df.loc[
        df.row_kind.isin(MULE_INITIATED_KINDS), "step_in_sequence"].replace(0, 4)

    # complaints = corroborating label channels on mules + genuine merchant disputes
    lab = labels.copy()
    if len(lab):
        lab["txn_id"] = lab["_rid"].map(rid_to_txn)
    mule_complaints = lab[lab.channel.isin(["complaint", "bank_report"])].drop_duplicates(["episode_id", "payee_idx"])
    complaints = pd.concat([
        pd.DataFrame({"payee_idx": mule_complaints["payee_idx"].to_numpy(), "reporter_idx": mule_complaints["reporter_idx"].to_numpy(),
                      "arrival_ts": mule_complaints["arrival_ts"].to_numpy(), "source": "scam_report"}),
        benign_disputes(cfg, pop, df)], ignore_index=True)

    tx = enrich(df, pop.accounts, pop.all_devices(), pop.all_sim_changes(), complaints, cfg.seed,
                pop.pair_history)
    validate_frame(tx)
    internal = df.set_index("txn_id").loc[tx["txn_id"], [c for c in INTERNAL_COLS + ["_rid"] if c in df]].reset_index()

    acct = pop.accounts.set_index("idx")["acct_id"]
    episodes["victim_id"] = acct.reindex(episodes["victim_idx"]).to_numpy()
    episodes["mule_ids"] = [",".join(acct.reindex([int(x) for x in str(m).split(",") if x]).astype(str))
                            for m in episodes["mules"]]
    if len(lab):
        lab["payee_id"] = acct.reindex(lab["payee_idx"]).to_numpy()
    eval_pop = ~tx.row_kind.isin(MULE_INITIATED_KINDS)
    summary = {
        "n_rows": int(len(tx)), "n_eval_rows": int(eval_pop.sum()),
        "n_attack_rows": int(tx.is_attack.sum()),
        "realised_attack_share": float(tx.loc[eval_pop, "is_attack"].mean()),
        "target_attack_share": cfg.attacks.target_share,
        "n_episodes": int(len(episodes)), "n_users": int((pop.accounts.kind == "person").sum()),
        "n_merchants": int((pop.accounts.kind == "merchant").sum()), "n_mules": int(len(registry.mules)),
        "days": cfg.traffic.days,
        "row_kinds": {k: int(v) for k, v in tx.row_kind.value_counts().items()},
        "variants": {k: int(v) for k, v in episodes.scam_variant.value_counts().items()},
        "evasions": {k: int(v) for k, v in episodes.evasion_technique.replace("", "none").value_counts().items()},
        "label_visible_share_episodes": float(episodes.label_visible.mean()) if len(episodes) else 0.0,
        "n_complaints": int(len(complaints)),
    }
    return GenResult(transactions=tx, internal=internal, pre_attack=pre_attack, episodes=episodes, labels=lab,
                     complaints=complaints, population=pop, mules=registry.table(), summary=summary)


def write(res: GenResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    res.transactions.to_parquet(out_dir / "transactions.parquet", index=False)
    res.episodes.to_parquet(out_dir / "episodes.parquet", index=False)
    res.labels.to_parquet(out_dir / "labels.parquet", index=False)
    res.complaints.to_parquet(out_dir / "complaints.parquet", index=False)
    res.internal.to_parquet(out_dir / "internal.parquet", index=False)
    res.population.accounts.to_parquet(out_dir / "accounts.parquet", index=False)
    res.population.all_devices().to_parquet(out_dir / "devices.parquet", index=False)
    res.population.all_sim_changes().to_parquet(out_dir / "sim_changes.parquet", index=False)
    res.mules.to_parquet(out_dir / "mules.parquet", index=False)
    res.population.pair_history.to_parquet(out_dir / "pair_history.parquet", index=False)
    meta = {"t0": res.population.t0, "t_end": res.population.t_end, "n_persons": res.population.n_persons,
            "n_merchants": res.population.n_merchants}
    (out_dir / "generation_summary.json").write_text(json.dumps({**res.summary, **meta}, indent=2))


def load_transactions(run: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir(run) / "transactions.parquet")
