"""Every window routine vs a brute-force reference, and feature prefix-invariance (the future cannot
change a past row's features)."""

import numpy as np
import pytest

from citadel.features import windows as W
from citadel.features.matrix import build_matrix, known_mules_from_labels, profiles_from_accounts

R = np.random.default_rng(0)
N = 600
CODES = R.integers(0, 12, N)
TS = np.sort(R.integers(0, 5000, N))
TS[::7] = TS[::7] // 10 * 10  # same-timestamp ties on purpose
VALS = R.random(N)
SEC = R.integers(0, 6, N)


def brute(fn):
    return np.array([fn(i) for i in range(N)])


def test_window_sum_count():
    s, c = W.window_sum_count(CODES, TS, VALS, 300)
    ref_s = brute(lambda i: VALS[(CODES == CODES[i]) & (TS < TS[i]) & (TS >= TS[i] - 300)].sum())
    ref_c = brute(lambda i: ((CODES == CODES[i]) & (TS < TS[i]) & (TS >= TS[i] - 300)).sum())
    assert np.allclose(s, ref_s) and np.allclose(c, ref_c)


def test_window_unique():
    u = W.window_unique(CODES, TS, SEC, 300)
    ref = brute(lambda i: len(set(SEC[(CODES == CODES[i]) & (TS < TS[i]) & (TS >= TS[i] - 300)])))
    assert np.array_equal(u, ref)


def test_prior_rank():
    ref = brute(lambda i: ((CODES == CODES[i]) & (TS < TS[i])).sum())
    assert np.array_equal(W.prior_rank(CODES, TS), ref)


def test_expanding_mean_std():
    m, s, c = W.expanding_mean_std(CODES, TS, VALS)
    for i in range(N):
        prior = VALS[(CODES == CODES[i]) & (TS < TS[i])]
        assert c[i] == len(prior)
        if len(prior):
            assert np.isclose(m[i], prior.mean())
        if len(prior) > 1:
            assert np.isclose(s[i], prior.std(), atol=1e-6)


def test_cross_routines():
    q_codes = R.integers(0, 12, 200)
    q_ts = R.integers(0, 5000, 200)
    s, c = W.cross_window_sum(CODES, TS, VALS, q_codes, q_ts, 400)
    last = W.cross_last_ts(CODES, TS, q_codes, q_ts)
    cnt = W.cross_count_before(CODES, TS, q_codes, q_ts)
    for j in range(200):
        m = (CODES == q_codes[j]) & (TS < q_ts[j])
        w = m & (TS >= q_ts[j] - 400)
        assert np.isclose(s[j], VALS[w].sum()) and c[j] == w.sum()
        assert cnt[j] == m.sum()
        assert (np.isnan(last[j]) and not m.any()) or last[j] == TS[m].max()


def test_a_window_that_includes_the_current_row_would_be_caught():
    """Regression guard: the reference itself must distinguish 'before' from 'including'."""
    s, _ = W.window_sum_count(CODES, TS, VALS, 300)
    including = brute(lambda i: VALS[(CODES == CODES[i]) & (TS <= TS[i]) & (TS >= TS[i] - 300)].sum())
    assert not np.allclose(s, including)


@pytest.mark.slow
def test_feature_matrix_is_prefix_invariant(tiny_gen):
    """Features of rows before T are identical whether or not rows after T exist."""
    tx = tiny_gen.transactions
    prof = profiles_from_accounts(tiny_gen.population.accounts)
    km = known_mules_from_labels(tiny_gen.labels)
    full = build_matrix(tx, prof, km)
    T = int(np.quantile(tx.ts, 0.6))
    T = ((T + 19800) // 86400) * 86400 - 19800  # an IST midnight, so nightly snapshots align
    early = (tx.ts < T).to_numpy()
    part = build_matrix(tx[early].reset_index(drop=True), prof, km)
    a = full[early].reset_index(drop=True).to_numpy(dtype=float)
    b = part.to_numpy(dtype=float)
    same = (a == b) | (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-5, atol=1e-6)
    bad = [full.columns[j] for j in np.flatnonzero((~same).any(axis=0))]
    assert not bad, f"features changed when the future was removed: {bad}"
