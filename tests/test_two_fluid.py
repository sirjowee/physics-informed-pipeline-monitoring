"""Tests for the two-fluid solver and the leak scenario generator (short runs)."""

from dataclasses import replace

import numpy as np
import pytest

from pipeline_monitoring.leak_detection.data import (
    REGIME_NAMES,
    DatasetConfig,
    PipelineConfig,
    ScenarioSpec,
    SimulationConfig,
    event_split,
    generate_dataset,
    mass_residual_steady,
    sample_scenarios,
    simulate,
)

PIPE = PipelineConfig()
SIM = SimulationConfig(duration_s=120.0)


def _specs(leak: bool, regime: str = "stratified", u: float = 3.0) -> list[ScenarioSpec]:
    return [ScenarioSpec(regime, u, leak, leak_x_m=900.0, leak_fraction=0.03, leak_t_s=60.0)]


def test_output_shapes():
    out = simulate(_specs(False) + _specs(True, "annular"), PIPE, SIM)
    n_t = int(SIM.duration_s / SIM.sample_dt_s) + 1
    n_k = int(SIM.duration_s / SIM.snapshot_dt_s) + 1
    assert out.p_in.shape == (2, n_t) and out.q_out.shape == (2, n_t)
    assert out.snap_p.shape == (2, n_k, SIM.n_cells)
    assert np.isfinite(out.snap_p).all() and (out.snap_p > 0).all()
    assert ((out.snap_alpha_g > 0) & (out.snap_alpha_g < 1)).all()


def test_mass_is_conserved_without_leak():
    out = simulate(_specs(False), PIPE, SIM)
    stored = out.mass_final - out.mass_initial
    through = out.mass_in_total - out.mass_out_total
    assert abs(stored[0] - through[0]) < 1e-6 * out.mass_in_total[0]
    assert out.mass_leaked_total[0] == 0.0


def test_leak_mass_is_accounted_for():
    out = simulate(_specs(True), PIPE, SIM)
    stored = out.mass_final - out.mass_initial
    through = out.mass_in_total - out.mass_out_total - out.mass_leaked_total
    assert abs(stored[0] - through[0]) < 1e-6 * out.mass_in_total[0]
    assert out.mass_leaked_total[0] > 0.0


@pytest.mark.parametrize("regime", ["stratified", "annular"])
def test_no_leak_steady_state_residual_is_small(regime):
    """At the end of a no-leak run the continuity residual is tiny relative to the flux."""
    sim = SimulationConfig(duration_s=200.0)
    out = simulate(_specs(False, regime), PIPE, sim)
    dx = PIPE.length_m / sim.n_cells
    ag, ug, ul, p = (
        a[0, -1].astype(float) for a in (out.snap_alpha_g, out.snap_u_g, out.snap_u_l, out.snap_p)
    )
    r_g, r_l = mass_residual_steady(ag, ug, ul, p, PIPE, dx)
    flux_g = (ag * p / PIPE.gas_sound_speed**2 * ug).mean()
    flux_l = ((1 - ag) * PIPE.rho_liquid * ul).mean()
    assert np.abs(r_g).max() * dx < 0.05 * flux_g
    assert np.abs(r_l).max() * dx < 0.05 * flux_l
    # and the sensors have settled: inlet pressure barely changes over the last 20 s
    assert np.ptp(out.p_in[0, -40:]) < 1e-3 * out.p_in[0, -1]


def test_leak_reduces_outlet_flow_after_onset():
    out = simulate(_specs(False) + _specs(True), PIPE, SIM)
    k = np.searchsorted(out.t_sensor, 100.0)
    assert out.q_out[1, k:].mean() < out.q_out[0, k:].mean()
    # before onset the two runs are identical
    j = np.searchsorted(out.t_sensor, 55.0)
    assert np.allclose(out.q_out[0, :j], out.q_out[1, :j])


def test_solver_is_deterministic():
    a = simulate(_specs(True, "slug"), PIPE, SIM)
    b = simulate(_specs(True, "slug"), PIPE, SIM)
    assert np.array_equal(a.p_in, b.p_in) and np.array_equal(a.snap_u_l, b.snap_u_l)


def test_scenario_sampler_and_split():
    cfg = DatasetConfig(n_scenarios=30, seed=5)
    specs = sample_scenarios(cfg)
    assert len(specs) == 30
    assert {s.regime for s in specs} == set(REGIME_NAMES)
    for s in specs:
        if s.is_leak:
            assert 0.0 <= s.leak_x_m <= cfg.pipe.length_m
            assert 0.001 <= s.leak_fraction <= 0.05
    assert [s.regime for s in sample_scenarios(cfg)] == [s.regime for s in specs]
    split = event_split(specs, cfg.split_fractions, cfg.seed)
    all_idx = np.concatenate(list(split.values()))
    assert sorted(all_idx.tolist()) == list(range(30))
    assert len(split["test"]) >= 4 and len(split["val"]) >= 4


def test_generate_small_dataset():
    cfg = DatasetConfig(n_scenarios=6, seed=1, sim=replace(SIM, duration_s=60.0))
    d = generate_dataset(cfg)
    assert d["sensors"].shape == (6, int(60 / SIM.sample_dt_s) + 1, 4)
    assert d["sensors"].dtype == np.float32
    assert np.isfinite(d["sensors"]).all()
    assert set(d) >= {"train_idx", "val_idx", "test_idx", "is_leak", "leak_x_m", "regime_id"}
