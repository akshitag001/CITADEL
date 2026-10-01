"""Generator gates: determinism, attacks are mutations, no fraud-only namespaces, labels after events."""

import dataclasses
import hashlib
import re

import numpy as np
import pandas as pd

from citadel.generate import artefacts
from citadel.generate.campaign import generate
from citadel.generate.enrich import REPORTER_MIN_AGE_DAYS, corroborated_complaints
from citadel.schema import MULE_INITIATED_KINDS


def _digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def test_same_config_and_seed_is_byte_identical(tiny_cfg, tiny_gen):
    again = generate(tiny_cfg)
    assert _digest(again.transactions) == _digest(tiny_gen.transactions)
    cols = [c for c in again.episodes.columns if c not in ("mules", "mule_ids")]
    assert _digest(again.episodes[cols]) == _digest(tiny_gen.episodes[cols])


def test_different_seed_differs(tiny_cfg, tiny_gen):
    other = generate(dataclasses.replace(tiny_cfg, seed=tiny_cfg.seed + 1))
    assert _digest(other.transactions) != _digest(tiny_gen.transactions)


def test_attack_rows_are_mutations_of_real_benign_rows(tiny_gen):
    internal = tiny_gen.internal.set_index("txn_id")
    tx = tiny_gen.transactions
    atk = tx[tx.row_kind == "attack"]
    pre = tiny_gen.pre_attack.set_index("txn_id")
    assert len(atk) > 0
    compare = [c for c in internal.columns if c not in ("_rid", "row_kind", "hn_archetype")]
    for _, a in atk.iterrows():
        assert a.source_txn_id in pre.index, "attack row without a benign source"
        src, cur = pre.loc[a.source_txn_id], internal.loc[a.txn_id]
        touched = set(a.touched_fields.split(","))
        for c in compare:
            same = (pd.isna(src[c]) and pd.isna(cur[c])) or src[c] == cur[c]
            if not same:
                assert c in touched, f"{c} changed but was not declared touched ({a.scam_variant})"
        assert src["_payer"] == cur["_payer"], "victim identity must never change"
        assert src["row_kind"] == "benign"


def test_victim_attributes_come_from_the_real_account(tiny_gen):
    tx = tiny_gen.transactions
    acc = tiny_gen.population.accounts.set_index("acct_id")
    atk = tx[tx.row_kind == "attack"]
    a = acc.loc[atk.user_id]
    assert (a.user_age_band.to_numpy() == atk.user_age_band.astype(str).to_numpy()).all()
    assert (a.digital_literacy.to_numpy() == atk.digital_literacy.astype(str).to_numpy()).all()


def test_no_fraud_only_id_namespace(tiny_gen):
    tx = tiny_gen.transactions
    for col in ("txn_id", "user_id", "payee_id", "device_id", "session_id"):
        assert tx[col].astype(str).map(lambda s: bool(re.fullmatch(r"[0-9a-f]{12}", s))).all(), col
    hits = artefacts.hunt(tx)
    assert hits.empty, hits.to_string()


def test_artefact_hunter_finds_a_planted_namespace(tiny_gen):
    tx = tiny_gen.transactions.copy()
    m = tx.is_attack == 1
    tx.loc[m, "payee_id"] = "zz" + tx.loc[m, "payee_id"].str[2:]
    hits = artefacts.hunt(tx)
    assert ((hits.column == "payee_id") & hits.kind.str.startswith("id_prefix")).any()


def test_labels_arrive_strictly_after_their_events(tiny_gen):
    lab = tiny_gen.labels
    assert len(lab) > 0
    assert (lab.arrival_ts > lab.row_ts).all()


def test_most_scams_never_get_a_visible_label(tiny_gen):
    assert tiny_gen.episodes.label_visible.mean() < 0.7


def test_not_every_scam_is_to_a_first_time_payee(tiny_gen):
    tx = tiny_gen.transactions
    atk = tx[tx.is_attack == 1]
    assert (atk.user_to_payee_prior_txn_count > 0).any()


def test_realised_attack_share_is_recorded_and_near_target(tiny_gen, tiny_cfg):
    s = tiny_gen.summary["realised_attack_share"]
    assert 0.25 * tiny_cfg.attacks.target_share < s < 4 * tiny_cfg.attacks.target_share


def test_attack_hours_use_the_shared_calendar(tiny_gen):
    tx = tiny_gen.transactions
    atk = tx[tx.is_attack == 1]
    assert set(atk.hour.unique()) <= set(tx[tx.row_kind == "benign"].hour.unique())


def test_mule_initiated_rows_are_marked(tiny_gen):
    tx = tiny_gen.transactions
    mk = tx.row_kind.isin(MULE_INITIATED_KINDS)
    assert (tx.loc[mk, "is_attack"] == 0).all()
    assert (tx.loc[tx.row_kind == "mule_outflow", "step_in_sequence"] == 4).all()


def test_complaints_require_distinct_aged_reporters():
    acc = pd.DataFrame({"idx": [1, 2, 3], "created_ts": [0, 0, 10 * 86400]})
    c = pd.DataFrame({"payee_idx": [9, 9, 9, 9], "reporter_idx": [1, 1, 2, 3],
                      "arrival_ts": [40 * 86400, 41 * 86400, 42 * 86400, 20 * 86400]})
    out = corroborated_complaints(c, acc)
    assert len(out) == 2  # reporter 1 deduplicated; reporter 3 younger than 30 days at report time
    assert REPORTER_MIN_AGE_DAYS == 30.0
    assert np.all(out.payee_idx == 9)
