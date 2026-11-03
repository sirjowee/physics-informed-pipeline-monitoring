"""Geometry, fluid and simulation constants for the synthetic leak-detection generator.

The paper (Adegboye, Komolafe & Joseph, IJSRED 7(6) 2024, section III-A) specifies only:
a horizontal pipeline segment with pressure and flow sensors at inlet and outlet; leak
locations uniform on [0, L]; leak rates 0.1-5 % of nominal flow; stratified, slug and
annular regimes; Gaussian sensor noise and transient events. Every number below is this
re-implementation's choice (RESULTS.md 1.4). Geometry and fluid constants were confirmed
with the author on 2026-11-03.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineConfig:
    """Horizontal two-phase pipeline segment (gas G + liquid L)."""

    length_m: float = 2000.0
    diameter_m: float = 0.3
    rho_liquid: float = 850.0  # kg/m3, crude oil, treated as incompressible
    rho_gas_ref: float = 50.0  # kg/m3, natural gas at the reference pressure
    p_ref: float = 60.0e5  # Pa, reference (nominal inlet) pressure for the gas EOS
    p_outlet: float = 55.0e5  # Pa, nominal outlet pressure boundary condition
    wall_friction_factor: float = 0.02  # Darcy friction factor, both phases
    nominal_velocities: tuple[float, ...] = (1.5, 3.0, 4.5)  # m/s mixture velocity

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m**2 / 4.0

    @property
    def gas_sound_speed(self) -> float:
        """Isothermal ideal gas: p = c^2 rho_G, c = sqrt(p_ref / rho_gas_ref) (~346 m/s)."""
        return math.sqrt(self.p_ref / self.rho_gas_ref)


@dataclass(frozen=True)
class RegimeParams:
    """Regime approximation through inlet gas fraction and interfacial drag strength.

    ``drag_coeff`` multiplies rho_G * alpha_G * alpha_L * |u_G - u_L| (u_G - u_L) to give the
    interfacial drag per unit volume (units 1/m). Values are chosen so the steady slip
    velocity is of order 0.5-2 m/s, the usual range for these regimes in horizontal pipes
    (Brennen, Fundamentals of Multiphase Flow, 2005, ch. 7-8). Slug flow is approximated
    by a periodic oscillation of the inlet liquid fraction.
    """

    name: str
    inlet_gas_fraction: float
    drag_coeff: float
    slug_amplitude: float = 0.0
    slug_period_s: float = 20.0


REGIMES: dict[str, RegimeParams] = {
    "stratified": RegimeParams("stratified", inlet_gas_fraction=0.40, drag_coeff=2.0),
    "slug": RegimeParams(
        "slug", inlet_gas_fraction=0.55, drag_coeff=20.0, slug_amplitude=0.15, slug_period_s=20.0
    ),
    "annular": RegimeParams("annular", inlet_gas_fraction=0.85, drag_coeff=8.0),
}
REGIME_NAMES: tuple[str, ...] = tuple(REGIMES)


@dataclass(frozen=True)
class SimulationConfig:
    """Discretisation and sampling. Default CFL with c ~ 346 m/s, u <= 5 m/s, dx = 40 m is ~0.44."""

    n_cells: int = 50
    dt_s: float = 0.05
    duration_s: float = 300.0
    sample_dt_s: float = 0.5  # sensor sampling interval
    snapshot_dt_s: float = 2.0  # full-field snapshot interval (for residual checks / RTTM)
    leak_ramp_s: float = 5.0  # leak opens linearly over this time
    sensor_noise_p_frac: float = 0.001  # sigma = 0.1 % of p_ref (6 kPa)
    sensor_noise_q_frac: float = 0.005  # sigma = 0.5 % of nominal volumetric flow

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_s / self.dt_s))


@dataclass(frozen=True)
class ScenarioSpec:
    """One simulated event. Leak fields are ignored when ``is_leak`` is False."""

    regime: str
    nominal_velocity: float
    is_leak: bool
    leak_x_m: float = 0.0
    leak_fraction: float = 0.0  # of nominal mass flow, 0.001-0.05
    leak_t_s: float = 0.0  # onset time
    inlet_step_t_s: float = -1.0  # < 0 means no inlet flow step
    inlet_step_factor: float = 1.0
    outlet_dp_pa: float = 0.0  # 0 means no outlet pressure ramp
    outlet_ramp_t_s: float = -1.0
    outlet_ramp_dur_s: float = 20.0


@dataclass(frozen=True)
class DatasetConfig:
    """How many scenarios and how they are drawn (RESULTS.md 1.4)."""

    n_scenarios: int = 240  # 80 per regime; --full uses more
    leak_probability: float = 0.5
    leak_fraction_range: tuple[float, float] = (0.001, 0.05)
    leak_onset_range_s: tuple[float, float] = (100.0, 180.0)
    disturbance_probability: float = 0.5  # each of inlet step / outlet ramp, independently
    inlet_step_range: tuple[float, float] = (0.8, 1.2)
    outlet_dp_range_pa: tuple[float, float] = (-2.0e5, 2.0e5)
    disturbance_time_range_s: tuple[float, float] = (60.0, 240.0)
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)  # train / val / test, by event
    seed: int = 0
    pipe: PipelineConfig = field(default_factory=PipelineConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
