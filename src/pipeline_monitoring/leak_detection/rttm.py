"""Simplified Real-Time Transient Model (RTTM) baseline.

Paper (section II-B, III-D): RTTM "solves variations of the above conservation equations to
predict dynamic pipeline responses. Deviations between predicted and actual sensor signals
are then interpreted to detect and locate leaks"; used as the physics-only baseline.

Implementation here (RESULTS.md 1.4): the Phase 3 two-fluid solver is run for each
scenario under the no-leak assumption with the operator-known boundary conditions (regime,
nominal velocity, inlet flow step and outlet pressure ramp). Residuals between the measured
and the modelled sensors are averaged over the physics window [200, 300] s:

* detection: the volumetric outlet-flow residual dQ = Q_out,model - Q_out,meas, scaled by
  the nominal flow, is compared with a threshold chosen on the validation set;
* quantification: leak mass rate = dQ * mixture density at the inlet (mass imbalance);
* localisation: with the outlet pressure fixed, a leak at x_L removes friction loss over
  (x_L, L); for turbulent friction dp/dx ~ Q^2 the inlet-pressure drop is
  dp_in ~ 2 (dp_model/L) (dQ/Q) (L - x_L), so x_L = L - dp_in / (2 (dp_model/L) dQ/Q).

This is a deliberately simple RTTM; it uses no labels except the detection threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .data.config import REGIMES, PipelineConfig, ScenarioSpec, SimulationConfig
from .data.two_fluid_solver import simulate
from .features import PHYSICS_WINDOW_S


@dataclass
class RTTMOutput:
    dq_rel: np.ndarray  # (n,) relative outlet-flow residual, positive when flow is missing
    rate_kg_s: np.ndarray  # (n,) estimated leak mass rate
    loc_m: np.ndarray  # (n,) estimated leak location
    dp_in: np.ndarray  # (n,) inlet pressure residual (Pa)


def specs_from_dataset(d: dict, idx: np.ndarray, no_leak: bool = True) -> list[ScenarioSpec]:
    specs = []
    for i in idx:
        s = ScenarioSpec(
            regime=str(d["regime"][i]),
            nominal_velocity=float(d["nominal_velocity"][i]),
            is_leak=bool(d["is_leak"][i]),
            leak_x_m=float(d["leak_x_m"][i]),
            leak_fraction=float(d["leak_fraction"][i]),
            leak_t_s=float(d["leak_t_s"][i]),
            inlet_step_t_s=float(d["inlet_step_t_s"][i]),
            inlet_step_factor=float(d["inlet_step_factor"][i]),
            outlet_dp_pa=float(d["outlet_dp_pa"][i]),
            outlet_ramp_t_s=float(d["outlet_ramp_t_s"][i]),
            outlet_ramp_dur_s=float(d["outlet_ramp_dur_s"][i]),
        )
        if no_leak:
            s = replace(s, is_leak=False, leak_x_m=0.0, leak_fraction=0.0, leak_t_s=0.0)
        specs.append(s)
    return specs


def run_rttm(
    d: dict,
    idx: np.ndarray,
    pipe: PipelineConfig | None = None,
    sim: SimulationConfig | None = None,
) -> RTTMOutput:
    pipe = pipe or PipelineConfig()
    sim = sim or SimulationConfig()
    specs = specs_from_dataset(d, idx, no_leak=True)
    model = simulate(specs, pipe, sim)
    t = d["t_sensor"]
    w = (t >= PHYSICS_WINDOW_S[0]) & (t <= PHYSICS_WINDOW_S[1])
    meas = d["sensors"][idx].astype(np.float64)
    q_out_meas = meas[:, w, 3].mean(axis=1)
    p_in_meas = meas[:, w, 0].mean(axis=1)
    q_out_mod = model.q_out[:, w].mean(axis=1)
    p_in_mod = model.p_in[:, w].mean(axis=1)
    p_out_mod = model.p_out[:, w].mean(axis=1)
    q_in_mod = model.q_in[:, w].mean(axis=1)

    dq = q_out_mod - q_out_meas  # m3/s missing at the outlet
    dq_rel = dq / q_in_mod
    ag = np.array([REGIMES[s.regime].inlet_gas_fraction for s in specs])
    rho_mix = ag * pipe.rho_gas_ref + (1 - ag) * pipe.rho_liquid
    rate = np.maximum(dq, 0.0) * rho_mix

    dp_in = p_in_mod - p_in_meas  # positive when the measured inlet pressure is lower
    grad = np.maximum(p_in_mod - p_out_mod, 1.0) / pipe.length_m
    with np.errstate(divide="ignore", invalid="ignore"):
        loc = pipe.length_m - dp_in / (2.0 * grad * np.maximum(dq_rel, 1e-6))
    loc = np.clip(np.nan_to_num(loc, nan=pipe.length_m / 2), 0.0, pipe.length_m)
    return RTTMOutput(dq_rel=dq_rel, rate_kg_s=rate, loc_m=loc, dp_in=dp_in)


def choose_threshold(dq_rel: np.ndarray, is_leak: np.ndarray) -> float:
    """Threshold on dq_rel maximising accuracy on the given (validation) set."""
    cands = np.sort(dq_rel)
    best_t, best_acc = 0.0, -1.0
    for t in np.concatenate([[cands[0] - 1e-9], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-9]]):
        acc = float(((dq_rel > t).astype(int) == is_leak.astype(int)).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t
