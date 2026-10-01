"""Split, sealed holdout, bundle, calibration, ladder and message gates."""

import numpy as np
import pandas as pd
import pytest

from citadel.defend.bundle import load_bundle
from citadel.defend.ladder import LadderCollapsedError, budget_threshold, fit_ladder
from citadel.defend.model import fit_calibrator
from citadel.defend.reasons import customer_message, load_messages, load_reason_codes
from citadel.defend.split import make_split


@pytest.fixture(scope="module")
def run_data(tiny_run):
    tx = tiny_run.read("transactions.parquet")
    ep = tiny_run.read("episodes.parquet")
    sp = make_split(tx, ep, tiny_run.cfg, tiny_run.gen_meta()["t0"])
    return tiny_run, tx, ep, sp


def test_split_windows_are_time_ordered_with_purge_and_embargo(run_data):
    ctx, tx, _, sp = run_data
    ts = tx.ts.to_numpy()
    assert ts[sp.train].max() < ts[sp.calib].min()
    assert ts[sp.calib].max() < ts[sp.stats].min()
    assert ts[sp.stats].max() < ts[sp.test].min()
    gap_purge = (ts[sp.calib].min() - ts[sp.train].max()) / 86400
    gap_embargo = (ts[sp.test].min() - ts[sp.stats].max()) / 86400
    assert gap_purge >= ctx.cfg.split.purge_days - 1
    assert gap_embargo >= ctx.cfg.split.embargo_days - 1
    fitted = sp.train | sp.calib | sp.stats
    assert not (fitted & sp.test).any() and not (sp.purge & fitted).any() and not (sp.embargo & fitted).any()


def test_sealed_holdout_is_entity_level(run_data):
    ctx, tx, ep, sp = run_data
    held = ep[(ep.scam_variant == ctx.cfg.sealed_holdout.variant) | (ep.evasion_technique == ctx.cfg.sealed_holdout.evasion)]
    fitted = sp.train | sp.calib | sp.stats
    f = tx[fitted]
    assert not f.episode_id.isin(set(held.episode_id)).any(), "held-out episode rows in a fitted window"
    assert not f.user_id.isin(set(held.victim_id)).any(), "held-out victims' benign rows in a fitted window"
    assert not f.payee_id.isin(set(held.victim_id)).any()
    # the withheld variant is genuinely present in the test window (a zero gap would mean it was not held out)
    t = tx[sp.test]
    assert (t.scam_variant == ctx.cfg.sealed_holdout.variant).any() or len(held) == 0


def test_bundle_roundtrip_reproduces_decisions(run_data, tmp_path):
    ctx, tx, _, sp = run_data
    b = load_bundle(ctx.art / "bundle.pkl")
    X = pd.read_parquet(ctx.art / "features.parquet").drop(columns=["txn_id"])
    Xt, raw = X[sp.test].reset_index(drop=True).head(300), tx[sp.test].reset_index(drop=True).head(300)
    d1 = b.decide(Xt, raw)
    b.save(tmp_path / "b.pkl")
    d2 = load_bundle(tmp_path / "b.pkl").decide(Xt, raw)
    assert np.allclose(d1.risk, d2.risk) and (d1.action == d2.action).all()
    assert b.provenance["run_id"] == ctx.prov.run_id


def test_calibration_is_monotonic():
    r = np.random.default_rng(0)
    s = r.random(5000)
    y = (r.random(5000) < s ** 3).astype(int)
    cal = fit_calibrator(s, y)
    grid = np.linspace(0, 1, 1001)
    assert np.all(np.diff(cal(grid)) > 0), "calibrated risk must be strictly increasing (tie-break included)"


def test_ladder_is_monotonic_and_respects_budgets(tiny_cfg):
    r = np.random.default_rng(0)
    s = r.random(20000)
    lad = fit_ladder(s, tiny_cfg.ladder_budgets)
    t = lad.thresholds
    assert t["A1"] < t["A2"] < t["A3"]
    for a, b in lad.budgets.items():
        assert (s >= t[a]).mean() <= b + 1e-12


def test_collapsed_ladder_is_refused(tiny_cfg):
    """Regression: a score with one value used to give A1 == A2 == A3; the guard must refuse it."""
    with pytest.raises(LadderCollapsedError):
        fit_ladder(np.full(10000, 0.3), tiny_cfg.ladder_budgets)


def test_threshold_never_lands_on_a_mass_point():
    s = np.concatenate([np.full(500, 0.9), np.random.default_rng(0).random(9500) * 0.5])
    t, share = budget_threshold(s, 0.01)  # 1% budget, but 5% of rows tie at 0.9
    assert share <= 0.01 and t > 0.9


def test_every_reason_and_action_has_a_message_in_both_languages():
    m = load_messages()
    codes = load_reason_codes()
    for c in codes:
        for lang in ("en", "hi"):
            assert m["reasons"][c][lang].strip(), (c, lang)
    for a in ("A1", "A2", "A3"):
        for lang in ("en", "hi"):
            assert m["actions"][a]["title"][lang] and m["actions"][a]["body"][lang]
        for b in m["actions"][a]["buttons"]:
            assert m["buttons"][b]["en"] and m["buttons"][b]["hi"]
    for t in ("pay", "collect", "qr_pay"):
        assert m["txn_type_screens"][t]["en"] and m["txn_type_screens"][t]["hi"]


def test_messages_never_expose_a_score_or_accuse():
    m = load_messages()
    for c in m["reasons"]:
        for lang in ("en", "hi"):
            txt = m["reasons"][c][lang].lower()
            assert "score" not in txt and "fraudster" not in txt and "%" not in txt
    msg = customer_message(m, "A3", ["R03", "R01"], "collect", "en", amount=4999)
    flat = str(msg).lower()
    assert "risk" not in flat and "score" not in flat and "probab" not in flat
    assert "4,999" in msg["screen"]  # the amount the customer is about to SEND is shown, not a score


def test_held_payment_always_has_a_release_and_appeal_path():
    m = load_messages()
    assert {"request_release", "appeal"} <= set(m["actions"]["A3"]["buttons"])
    assert "cancel" in m["actions"]["A2"]["buttons"] and "continue_after" in m["actions"]["A2"]["buttons"]
