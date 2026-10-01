import copy

import numpy as np
import pytest
import yaml

from citadel import schema
from citadel import timeutil as T
from citadel.config import CONFIG_DIR, load_config, rng
from citadel.generate.library import LibraryError, load_library
from citadel.schema import COLUMNS, FEATURE_OK, ID, TRUTH_ONLY

EXPECTED_TRUTH = {"episode_id", "is_attack", "scam_variant", "evasion_technique", "step_in_sequence",
                  "label_visible", "label_delay_days"}
RAW = yaml.safe_load((CONFIG_DIR / "scam_library.yaml").read_text(encoding="utf-8"))


# ---- schema ---------------------------------------------------------------------------------------
def test_every_column_has_exactly_one_known_role():
    assert all(spec.role in (ID, FEATURE_OK, TRUTH_ONLY) for spec in COLUMNS.values())


def test_truth_columns_are_truth_only_and_never_feature_ok():
    truth = set(schema.truth_columns())
    assert EXPECTED_TRUTH <= truth
    assert not (truth & set(schema.feature_ok_columns()))
    for c in schema.feature_ok_columns():
        assert not c.startswith("label_")


def test_nullable_signals_are_declared_nullable():
    for c in ("screen_share_active", "remote_access_app_detected", "call_in_progress", "collect_request_age_s"):
        assert COLUMNS[c].nullable


def test_generated_frame_matches_schema(tiny_gen):
    schema.validate_frame(tiny_gen.transactions)


# ---- scam grammar ---------------------------------------------------------------------------------
def test_library_has_six_variants_and_six_evasions():
    lib = load_library()
    assert set(lib.variants) == {"V1", "V2", "V3", "V4", "V5", "V6"}
    assert len(lib.evasions) == 6


def test_unknown_column_fails_load():
    raw = copy.deepcopy(RAW)
    raw["variants"]["V1"]["steps"]["S2"]["signals"]["victim_mood"] = {"p": 0.5}
    with pytest.raises(LibraryError, match="not a schema column"):
        load_library(raw=raw)


def test_truth_only_signal_fails_load():
    raw = copy.deepcopy(RAW)
    raw["variants"]["V1"]["steps"]["S2"]["signals"]["is_attack"] = {"p": 0.5}
    with pytest.raises(LibraryError, match="TRUTH_ONLY"):
        load_library(raw=raw)


def test_always_firing_flag_fails_load():
    raw = copy.deepcopy(RAW)
    raw["variants"]["V2"]["steps"]["S2"]["signals"]["screen_share_active"] = {"p": 1.0}
    with pytest.raises(LibraryError, match="probabilistic"):
        load_library(raw=raw)


def test_evasion_cannot_pull_a_victim_lever():
    raw = copy.deepcopy(RAW)
    raw["evasions"]["slow_drip"]["levers"].append("user_tenure_days")
    with pytest.raises(LibraryError, match="not attacker-controllable"):
        load_library(raw=raw)


def test_illegal_composition_returns_the_failed_constraint():
    lib = load_library()
    assert "nothing to suppress" in lib.explain_illegal("V4", "signal_suppression", "mainstream")
    assert "QR" in lib.explain_illegal("V4", "payee_name_mimicry", "mainstream")
    assert "not targeted" in lib.explain_illegal("V6", None, "senior")
    assert lib.explain_illegal("V2", "signal_suppression", "senior") is None


def test_composition_space_is_counted_and_explained():
    space = load_library().composition_space()
    assert space["size_total"] == 6 * 7 * 5
    assert 0 < space["size_legal"] < space["size_total"]
    assert all(r["reason"] for r in space["illegal"])


# ---- time + config --------------------------------------------------------------------------------
def test_ist_hour_and_dow():
    t0 = T.start_epoch("2026-06-01")  # Monday 2026-06-01 00:00 IST
    assert int(T.hour_ist(t0)) == 0 and int(T.dow_ist(t0)) == 0
    assert int(T.hour_ist(t0 + 13 * 3600 + 59)) == 13
    assert int(T.dow_ist(t0 + 6 * 86400)) == 6


def test_night_definition_is_single_source():
    assert T.is_night(np.array([23, 0, 3, 5])).all()
    assert not T.is_night(np.array([6, 12, 22])).any()


def test_unknown_config_key_fails():
    with pytest.raises(KeyError):
        load_config("configs/quick.yaml", traffic={"dayz": 3})


def test_rng_streams_are_deterministic_and_independent():
    cfg = load_config("configs/tiny.yaml")
    a1, a2 = rng(cfg, "x").random(5), rng(cfg, "x").random(5)
    b = rng(cfg, "y").random(5)
    assert np.array_equal(a1, a2) and not np.array_equal(a1, b)
