"""Tests for the synthetic |S11| simulator."""

import numpy as np

from pipeline_monitoring.shm_classification.data import (
    CLASS_NAMES,
    N_POINTS,
    add_gaussian_noise,
    frequency_grid,
    generate_dataset,
    s11_trace_db,
    sample_trace_params,
    stratified_split,
    zscore_per_trace,
)


def test_frequency_grid_matches_paper():
    f = frequency_grid()
    assert f.shape == (91,)
    assert f[0] == 1.0 and f[-1] == 10.0
    assert np.allclose(np.diff(f), 0.1)


def test_dataset_shape_and_balance():
    X, y, params = generate_dataset(n_per_class=100, seed=0)
    assert X.shape == (400, N_POINTS)
    assert y.shape == (400,)
    assert len(params) == 400
    assert np.array_equal(np.bincount(y), [100, 100, 100, 100])
    assert np.isfinite(X).all()


def test_dataset_is_deterministic_given_seed():
    X1, y1, _ = generate_dataset(seed=7)
    X2, y2, _ = generate_dataset(seed=7)
    X3, _, _ = generate_dataset(seed=8)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert not np.array_equal(X1, X3)


def test_class_signatures_move_resonance_as_documented():
    """Crack shifts the dip up; corrosion and leak shift it down; corrosion is shallowest."""
    f = frequency_grid()
    rng = np.random.default_rng(0)
    dips = {}
    depths = {}
    for label, name in enumerate(CLASS_NAMES):
        p = sample_trace_params(rng, label)
        tr = s11_trace_db(f, p)
        dips[name] = f[np.argmin(tr)]
        depths[name] = tr.max() - tr.min()
    assert dips["crack"] > dips["normal"] > dips["corrosion"] > dips["leak"]
    assert depths["corrosion"] < depths["normal"]


def test_zscore_per_trace():
    X, _, _ = generate_dataset(seed=0)
    Z = zscore_per_trace(X)
    assert np.allclose(Z.mean(axis=1), 0.0, atol=1e-10)
    assert np.allclose(Z.std(axis=1), 1.0, atol=1e-10)


def test_noise_has_requested_std_and_zero_sigma_is_identity():
    X = np.zeros((2000, N_POINTS))
    for sigma in (0.1, 0.3, 1.0):
        N = add_gaussian_noise(X, sigma, seed=1)
        assert abs(N.std() - sigma) < 0.02 * max(sigma, 0.1) + 1e-3
        assert abs(N.mean()) < 0.01
    assert np.array_equal(add_gaussian_noise(X, 0.0), X)
    assert np.array_equal(add_gaussian_noise(X, 0.3, seed=3), add_gaussian_noise(X, 0.3, seed=3))


def test_stratified_split_is_balanced_disjoint_and_seeded():
    _, y, _ = generate_dataset(seed=0)
    tr, te = stratified_split(y, test_fraction=0.2, seed=0)
    assert len(tr) == 320 and len(te) == 80
    assert np.array_equal(np.bincount(y[te]), [20, 20, 20, 20])
    assert np.intersect1d(tr, te).size == 0
    tr2, te2 = stratified_split(y, test_fraction=0.2, seed=0)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)
