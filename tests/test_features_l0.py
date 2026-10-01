"""Feature registry, forbidden-feature and L0 gates."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from citadel.defend import l0
from citadel.features.matrix import build_matrix, known_mules_from_labels, profiles_from_accounts
from citadel.features.registry import FEATURES, REGISTRY, registry_frame
from citadel.schema import COLUMNS, TRUTH_ONLY, truth_columns

FORBIDDEN = {"episode_id", "is_attack", "scam_variant", "evasion_technique", "step_in_sequence"}


def test_registry_matches_matrix(tiny_features):
    assert list(tiny_features.columns) == FEATURES
    assert len(set(FEATURES)) == len(FEATURES)
    df = registry_frame()
    assert df["causal"].all()
    assert set(df["family"]) <= {"raw", "user", "sequence", "payee", "graph"}


def test_registry_requires_only_real_non_truth_columns():
    for f in REGISTRY:
        for c in f.requires:
            assert c in COLUMNS, (f.name, c)
            assert COLUMNS[c].role != TRUTH_ONLY, (f.name, c)


def test_no_truth_column_reaches_the_matrix(tiny_gen, tiny_features):
    for c in FEATURES:
        assert c not in truth_columns() and not c.startswith("label_")
    assert not (FORBIDDEN & set(tiny_features.columns))
    # scrambling every truth column cannot change a single feature value
    tx = tiny_gen.transactions.copy()
    r = np.random.default_rng(1)
    for c in truth_columns():
        tx[c] = r.permutation(tx[c].to_numpy())
    X2 = build_matrix(tx, profiles_from_accounts(tiny_gen.population.accounts),
                      known_mules_from_labels(tiny_gen.labels))
    a, b = tiny_features.to_numpy(dtype=float), X2.to_numpy(dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_unobservable_signals_stay_missing_not_zero(tiny_gen, tiny_features):
    tx = tiny_gen.transactions
    no_consent = tx["consent_call_signal"].to_numpy() == 0
    assert np.isnan(tiny_features["f_call"].to_numpy()[no_consent]).all()


# ---- L0 -------------------------------------------------------------------------------------------
def test_l0_loader_rejects_unknown_and_truth_columns_and_calls():
    base = {"id": "X", "title": "t", "min_action": "A2", "reason_codes": ["R14"], "rationale": "r"}
    with pytest.raises(l0.RuleError, match="unknown"):
        l0.load_rules(raw={"rules": [{**base, "condition": "f_nonexistent > 1"}]})
    with pytest.raises(l0.RuleError, match="TRUTH_ONLY"):
        l0.load_rules(raw={"rules": [{**base, "condition": "is_attack == 1"}]})
    with pytest.raises(l0.RuleError, match="not allowed"):
        l0.load_rules(raw={"rules": [{**base, "condition": "__import__('os')"}]})
    with pytest.raises(l0.RuleError):
        l0.load_rules(raw={"rules": [{**base, "condition": "f_call == 1", "min_action": "decline"}]})


def test_l0_rules_never_decline_and_all_carry_reason_codes():
    for r in l0.load_rules():
        assert r.min_action in l0.ACTIONS
        assert r.reason_codes and "R14" in r.reason_codes


def test_l0_zero_false_positives_on_legitimate_populations(tiny_gen, tiny_features):
    tx = tiny_gen.transactions
    res = l0.evaluate(l0.load_rules(), l0.l0_frame(tx, tiny_features))
    legit = tx["row_kind"].isin(["benign", "hard_negative", "shape", "seasoning"]).to_numpy()
    for rule in l0.load_rules():
        if rule.log_only:
            continue
        hits = int((res.fired[rule.id].to_numpy() & legit).sum())
        assert hits == 0, f"{rule.id} fired on {hits} legitimate payments; re-scope the rule"


def test_nan_never_satisfies_a_rule():
    import pandas as pd
    rules = l0.load_rules(raw={"rules": [{"id": "T", "title": "t", "condition": "f_call == 1 or f_call != 1",
                                          "min_action": "A1", "reason_codes": ["R14"], "rationale": "r"}]})
    frame = pd.DataFrame({"f_call": [np.nan, 1.0]})
    out = l0.evaluate(rules, frame)
    assert out.fired["T"].tolist() == [False, True]  # NaN satisfies neither == nor !=
    rules = l0.load_rules(raw={"rules": [{"id": "T", "title": "t", "condition": "f_call == 1",
                                          "min_action": "A1", "reason_codes": ["R14"], "rationale": "r"}]})
    assert l0.evaluate(rules, frame).fired["T"].tolist() == [False, True]


def test_l0_runs_with_the_model_stack_unavailable():
    code = textwrap.dedent("""
        import sys
        for m in ("sklearn", "sklearn.ensemble", "sklearn.linear_model"):
            sys.modules[m] = None  # model down
        import pandas as pd, numpy as np
        from citadel.defend import l0
        rules = l0.load_rules()
        frame = pd.DataFrame({c: [np.nan] for c in __import__("citadel.features.registry", fromlist=["x"]).FEATURES})
        for c in l0.RAW_FOR_L0:
            frame[c] = ["collect"] if c == "txn_type" else [0.0]
        frame["collect_request_age_s"] = [2.0]
        res = l0.evaluate(rules, frame)
        assert res.min_level[0] == 2, res.min_level
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr
