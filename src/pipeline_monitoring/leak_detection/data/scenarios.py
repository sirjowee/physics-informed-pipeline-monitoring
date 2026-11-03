"""Scenario sampling, sensor-noise injection, event-based splitting and dataset I/O.

Paper (section III-A, III-D): leak locations uniform on [0, L]; leak rates 0.1-5 % of
nominal flow; stratified, slug and annular regimes; Gaussian sensor noise and transient
events; event-based disjoint train / validation / test sets; time series of pressure and
flow "at different positions" (here: inlet and outlet, as the setup paragraph says).

Chosen here (RESULTS.md 1.4): numbers of scenarios, the leak-onset window, the kinds and
sizes of disturbances, sensor noise levels, and the split fractions. See ``DatasetConfig``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import REGIME_NAMES, DatasetConfig, ScenarioSpec
from .two_fluid_solver import SimulationOutput, simulate

SENSOR_NAMES: tuple[str, ...] = ("p_in", "p_out", "q_in", "q_out")


def sample_scenarios(cfg: DatasetConfig) -> list[ScenarioSpec]:
    """Draw ``cfg.n_scenarios`` specs, regimes and nominal velocities cycled evenly."""
    rng = np.random.default_rng(cfg.seed)
    specs: list[ScenarioSpec] = []
    for i in range(cfg.n_scenarios):
        regime = REGIME_NAMES[i % len(REGIME_NAMES)]
        u_nom = cfg.pipe.nominal_velocities[
            (i // len(REGIME_NAMES)) % len(cfg.pipe.nominal_velocities)
        ]
        is_leak = bool(rng.random() < cfg.leak_probability)
        leak_x = float(rng.uniform(0.0, cfg.pipe.length_m)) if is_leak else 0.0
        leak_frac = float(rng.uniform(*cfg.leak_fraction_range)) if is_leak else 0.0
        leak_t = float(rng.uniform(*cfg.leak_onset_range_s)) if is_leak else 0.0
        has_step = rng.random() < cfg.disturbance_probability
        step_t = float(rng.uniform(*cfg.disturbance_time_range_s)) if has_step else -1.0
        step_f = float(rng.uniform(*cfg.inlet_step_range)) if has_step else 1.0
        has_ramp = rng.random() < cfg.disturbance_probability
        ramp_t = float(rng.uniform(*cfg.disturbance_time_range_s)) if has_ramp else -1.0
        dp = float(rng.uniform(*cfg.outlet_dp_range_pa)) if has_ramp else 0.0
        specs.append(
            ScenarioSpec(
                regime=regime,
                nominal_velocity=u_nom,
                is_leak=is_leak,
                leak_x_m=leak_x,
                leak_fraction=leak_frac,
                leak_t_s=leak_t,
                inlet_step_t_s=step_t,
                inlet_step_factor=step_f,
                outlet_dp_pa=dp,
                outlet_ramp_t_s=ramp_t,
            )
        )
    return specs


def event_split(
    specs: list[ScenarioSpec], fractions: tuple[float, float, float], seed: int
) -> dict[str, np.ndarray]:
    """Whole scenarios go to exactly one of train / val / test, stratified by (regime, leak)."""
    rng = np.random.default_rng(seed + 1)
    keys = np.array([f"{s.regime}|{int(s.is_leak)}" for s in specs])
    out = {"train": [], "val": [], "test": []}
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(round(fractions[1] * n))
        n_test = int(round(fractions[2] * n))
        out["test"].append(idx[:n_test])
        out["val"].append(idx[n_test : n_test + n_val])
        out["train"].append(idx[n_test + n_val :])
    return {k: np.sort(np.concatenate(v)) for k, v in out.items()}


def add_sensor_noise(
    out: SimulationOutput, cfg: DatasetConfig, specs: list[ScenarioSpec]
) -> np.ndarray:
    """Noisy sensor array (S, T, 4) in the order ``SENSOR_NAMES``.

    sigma_p = sensor_noise_p_frac * p_ref; sigma_q = sensor_noise_q_frac * nominal Q of the
    scenario. Noise seed is ``cfg.seed + 2``.
    """
    rng = np.random.default_rng(cfg.seed + 2)
    q_nom = np.array([cfg.pipe.area_m2 * s.nominal_velocity for s in specs])[:, None]
    sig_p = cfg.sim.sensor_noise_p_frac * cfg.pipe.p_ref
    sig_q = cfg.sim.sensor_noise_q_frac * q_nom
    clean = np.stack([out.p_in, out.p_out, out.q_in, out.q_out], axis=-1)
    noise = rng.standard_normal(clean.shape)
    noise[..., 0:2] *= sig_p
    noise[..., 2:4] *= sig_q[..., None]
    return clean + noise


def specs_to_arrays(specs: list[ScenarioSpec]) -> dict[str, np.ndarray]:
    d = {k: np.array([getattr(s, k) for s in specs]) for k in asdict(specs[0])}
    d["regime_id"] = np.array([REGIME_NAMES.index(s.regime) for s in specs])
    return d


def generate_dataset(cfg: DatasetConfig | None = None) -> dict[str, np.ndarray]:
    """Sample, simulate, add noise, split. Returns a flat dict ready for ``np.savez``."""
    cfg = cfg or DatasetConfig()
    specs = sample_scenarios(cfg)
    out = simulate(specs, cfg.pipe, cfg.sim)
    sensors = add_sensor_noise(out, cfg, specs)
    split = event_split(specs, cfg.split_fractions, cfg.seed)
    d: dict[str, np.ndarray] = {
        "t_sensor": out.t_sensor,
        "sensors": sensors.astype(np.float32),
        "sensors_clean": np.stack([out.p_in, out.p_out, out.q_in, out.q_out], -1).astype(
            np.float32
        ),
        "sensor_names": np.array(SENSOR_NAMES),
        "t_snap": out.t_snap,
        "x_centers": out.x_centers,
        "snap_p": out.snap_p,
        "snap_alpha_g": out.snap_alpha_g,
        "snap_u_g": out.snap_u_g,
        "snap_u_l": out.snap_u_l,
        "mass_initial": out.mass_initial,
        "mass_final": out.mass_final,
        "mass_in_total": out.mass_in_total,
        "mass_out_total": out.mass_out_total,
        "mass_leaked_total": out.mass_leaked_total,
        "train_idx": split["train"],
        "val_idx": split["val"],
        "test_idx": split["test"],
        "seed": np.array(cfg.seed),
        "n_cells": np.array(cfg.sim.n_cells),
        "length_m": np.array(cfg.pipe.length_m),
    }
    d.update(specs_to_arrays(specs))
    return d


def save_dataset(d: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
