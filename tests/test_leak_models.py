"""Tests for the physics residuals, the PINN training loop and the RTTM baseline."""

from dataclasses import replace

import numpy as np
import pytest

from pipeline_monitoring.leak_detection.data import (
    REGIMES,
    DatasetConfig,
    PipelineConfig,
    ScenarioSpec,
    SimulationConfig,
    generate_dataset,
    simulate,
)
from pipeline_monitoring.leak_detection.features import (
    Normaliser,
    binned_channels,
    build_features,
    build_targets,
    sensor_window,
)
from pipeline_monitoring.leak_detection.physics import residuals_on_snapshots
from pipeline_monitoring.leak_detection.pinn import PINNConfig, predict, train_pinn
from pipeline_monitoring.leak_detection.rttm import choose_threshold, run_rttm

PIPE = PipelineConfig()
SIM = SimulationConfig()
INTERIOR = slice(5, None)  # skip the inlet boundary layer of the finite-volume solution


@pytest.fixture(scope="module")
def small_dataset():
    return generate_dataset(DatasetConfig(n_scenarios=9, seed=3))


def _run(regime, leak, x=1200.0, frac=0.03):
    spec = ScenarioSpec(regime, 3.0, leak, leak_x_m=x, leak_fraction=frac, leak_t_s=120.0)
    out = simulate([spec], PIPE, SIM)
    k = np.searchsorted(out.t_snap, 200.0)
    return out, k


@pytest.mark.parametrize("regime", ["stratified", "annular"])
def test_no_leak_steady_state_residuals_are_small(regime):
    out, k = _run(regime, False)
    res = residuals_on_snapshots(
        out.snap_p[0, k:],
        out.snap_alpha_g[0, k:],
        out.snap_u_g[0, k:],
        out.snap_u_l[0, k:],
        out.t_snap[k:],
        out.x_centers,
        0.0,
        0.0,
        REGIMES[regime].drag_coeff,
        PIPE,
    )
    for name, r in res.items():
        rms = np.sqrt((r[:, INTERIOR] ** 2).mean())
        assert rms < 0.1, f"{name} scaled residual rms {rms:.3f} on a steady no-leak state"


def test_correct_leak_sink_lowers_the_gas_mass_residual():
    regime = "annular"
    out, k = _run(regime, True, x=1000.0, frac=0.04)
    ag = REGIMES[regime].inlet_gas_fraction
    mdot = PIPE.area_m2 * 3.0 * (ag * PIPE.rho_gas_ref + (1 - ag) * PIPE.rho_liquid)
    args = (
        out.snap_p[0, k:],
        out.snap_alpha_g[0, k:],
        out.snap_u_g[0, k:],
        out.snap_u_l[0, k:],
        out.t_snap[k:],
        out.x_centers,
    )
    with_sink = residuals_on_snapshots(*args, 0.04 * mdot, 1000.0, REGIMES[regime].drag_coeff, PIPE)
    no_sink = residuals_on_snapshots(*args, 0.0, 0.0, REGIMES[regime].drag_coeff, PIPE)
    rms = lambda r: np.sqrt((r[:, INTERIOR] ** 2).mean())  # noqa: E731
    assert rms(with_sink["mass_g"]) < rms(no_sink["mass_g"])


def test_features_and_targets(small_dataset):
    d = small_dataset
    idx = np.arange(len(d["is_leak"]))
    norm = Normaliser.fit(binned_channels(d, idx))
    X = build_features(d, idx, norm)
    assert X.shape == (9, 60 * 5 + 4) and np.isfinite(X).all()
    y = build_targets(d, idx)
    assert set(y) == {"is_leak", "loc", "rate"}
    assert ((y["loc"] >= 0) & (y["loc"] <= 1)).all() and ((y["rate"] >= 0) & (y["rate"] <= 1)).all()
    t_w, s_w = sensor_window(d, idx)
    assert t_w[0] >= 200.0 and t_w[-1] <= 300.0 and s_w.shape == (9, len(t_w), 4)


def test_pinn_trains_and_loss_decreases(small_dataset):
    d = small_dataset
    idx = np.arange(len(d["is_leak"]))
    norm = Normaliser.fit(binned_channels(d, idx))
    X = build_features(d, idx, norm)
    y = build_targets(d, idx)
    t_w, s_w = sensor_window(d, idx)
    cfg = PINNConfig(steps=60, batch_scenarios=9, n_collocation=64, n_sensor_points=16, seed=1)
    model, log = train_pinn(d, idx, X, y, t_w, s_w, cfg)
    assert log[-1]["total"] < log[0]["total"]
    assert log[-1]["physics"] < log[0]["physics"]
    out = predict(model, X)
    assert out["prob"].shape == (9,) and ((out["loc"] >= 0) & (out["loc"] <= 1)).all()


def test_pinn_is_deterministic(small_dataset):
    d = small_dataset
    idx = np.arange(len(d["is_leak"]))
    norm = Normaliser.fit(binned_channels(d, idx))
    X = build_features(d, idx, norm)
    y = build_targets(d, idx)
    t_w, s_w = sensor_window(d, idx)
    cfg = PINNConfig(steps=20, batch_scenarios=9, n_collocation=32, n_sensor_points=8, seed=4)
    m1, _ = train_pinn(d, idx, X, y, t_w, s_w, cfg)
    m2, _ = train_pinn(d, idx, X, y, t_w, s_w, replace(cfg))
    assert np.allclose(predict(m1, X)["prob"], predict(m2, X)["prob"], atol=1e-5)


def test_rttm_detects_and_bounds_location(small_dataset):
    d = small_dataset
    idx = np.arange(len(d["is_leak"]))
    r = run_rttm(d, idx)
    assert r.loc_m.shape == (9,) and ((r.loc_m >= 0) & (r.loc_m <= PIPE.length_m)).all()
    y = d["is_leak"].astype(int)
    if y.min() != y.max():
        thr = choose_threshold(r.dq_rel, y)
        assert ((r.dq_rel > thr).astype(int) == y).mean() >= 0.8
