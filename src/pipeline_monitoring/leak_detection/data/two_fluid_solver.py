"""Batched 1-D isothermal two-fluid transient solver (finite volume, NumPy).

Governing equations (paper, section III-B), per phase k in {G, L}, with constant area A:

    d(alpha_k rho_k)/dt + d(alpha_k rho_k u_k)/dx                     = S_k,mass
    d(alpha_k rho_k u_k)/dt + d(alpha_k rho_k u_k^2)/dx + alpha_k dp/dx = F_k + S_k,momentum

    S_k,mass = -Q_leak,k(t) delta(x - x_L)      (leak as a localised sink)

Closures chosen here (none are given in the paper; RESULTS.md 1.4):

* gas: isothermal ideal gas, p = c^2 rho_G with c = sqrt(p_ref / rho_G,ref);
  liquid: incompressible. Volume fractions sum to one. This closes the system
  algebraically: alpha_L = m_L / rho_L, alpha_G = 1 - alpha_L, rho_G = m_G / alpha_G.
* F_k = wall friction + interfacial drag (horizontal pipe, no gravity):
    wall:   -(f / 2D) alpha_k rho_k u_k |u_k|                        (Darcy, f = 0.02)
    drag:   -/+ C_i alpha_G alpha_L rho_G (u_G - u_L) |u_G - u_L|   (regime-dependent C_i)
* the leak sink is spread over the cell containing x_L; the leaking mass is split
  between phases by their local mass fraction; the momentum sink is Q_leak,k u_k.
* the Dirac delta in S_k,mass is represented as 1/dx on the leak cell.

Numerics: conserved variables per unit volume U = (m_G, m_L, I_G, I_L) with
m_k = alpha_k rho_k and I_k = m_k u_k. Convective fluxes plus the alpha_k p part of the
pressure term are evaluated with the Rusanov (local Lax-Friedrichs) flux; the
non-conservative remainder + p d(alpha_k)/dx is a central-difference source. Forward Euler
in time at a fixed CFL-safe step. Ghost cells impose an inlet mass-flow condition
(given velocities and fractions, zero pressure gradient) and an outlet pressure condition
(given p, extrapolated fractions and velocities). All scenarios in a batch advance
together as (n_scenarios, n_cells) arrays, which is what keeps the CPU budget small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import REGIMES, PipelineConfig, ScenarioSpec, SimulationConfig

EPS = 1e-9
ALPHA_MIN = 1e-3


@dataclass
class Primitives:
    alpha_g: np.ndarray
    alpha_l: np.ndarray
    rho_g: np.ndarray
    p: np.ndarray
    u_g: np.ndarray
    u_l: np.ndarray


def primitives(
    m_g: np.ndarray, m_l: np.ndarray, i_g: np.ndarray, i_l: np.ndarray, pipe: PipelineConfig
) -> Primitives:
    """Recover (alpha, rho_G, p, u) from conserved variables."""
    alpha_l = np.clip(m_l / pipe.rho_liquid, ALPHA_MIN, 1.0 - ALPHA_MIN)
    alpha_g = 1.0 - alpha_l
    rho_g = np.maximum(m_g, EPS) / alpha_g
    p = pipe.gas_sound_speed**2 * rho_g
    u_g = i_g / np.maximum(m_g, EPS)
    u_l = i_l / np.maximum(m_l, EPS)
    return Primitives(alpha_g, alpha_l, rho_g, p, u_g, u_l)


def _ramp(t: float, t0: float, dur: float) -> float:
    if dur <= 0.0:
        return 1.0 if t >= t0 else 0.0
    return float(np.clip((t - t0) / dur, 0.0, 1.0))


@dataclass
class BatchForcing:
    """Per-scenario time-dependent boundary and leak forcing, vectorised over the batch."""

    regime_gas_fraction: np.ndarray  # (S,)
    drag_coeff: np.ndarray  # (S,)
    slug_amp: np.ndarray  # (S,)
    slug_period: np.ndarray  # (S,)
    u_nominal: np.ndarray  # (S,)
    inlet_step_t: np.ndarray  # (S,)
    inlet_step_factor: np.ndarray  # (S,)
    outlet_dp: np.ndarray  # (S,)
    outlet_ramp_t: np.ndarray  # (S,)
    outlet_ramp_dur: np.ndarray  # (S,)
    is_leak: np.ndarray  # (S,) bool
    leak_cell: np.ndarray  # (S,) int
    leak_t: np.ndarray  # (S,)
    leak_mass_rate: np.ndarray  # (S,) kg/s at full opening

    @classmethod
    def from_specs(
        cls, specs: list[ScenarioSpec], pipe: PipelineConfig, sim: SimulationConfig
    ) -> BatchForcing:
        dx = pipe.length_m / sim.n_cells
        reg = [REGIMES[s.regime] for s in specs]
        u_nom = np.array([s.nominal_velocity for s in specs])
        ag = np.array([r.inlet_gas_fraction for r in reg])
        mdot_nom = pipe.area_m2 * u_nom * (ag * pipe.rho_gas_ref + (1 - ag) * pipe.rho_liquid)
        return cls(
            regime_gas_fraction=ag,
            drag_coeff=np.array([r.drag_coeff for r in reg]),
            slug_amp=np.array([r.slug_amplitude for r in reg]),
            slug_period=np.array([r.slug_period_s for r in reg]),
            u_nominal=u_nom,
            inlet_step_t=np.array([s.inlet_step_t_s for s in specs]),
            inlet_step_factor=np.array([s.inlet_step_factor for s in specs]),
            outlet_dp=np.array([s.outlet_dp_pa for s in specs]),
            outlet_ramp_t=np.array([s.outlet_ramp_t_s for s in specs]),
            outlet_ramp_dur=np.array([s.outlet_ramp_dur_s for s in specs]),
            is_leak=np.array([s.is_leak for s in specs]),
            leak_cell=np.array(
                [min(int(s.leak_x_m / dx), sim.n_cells - 1) if s.is_leak else 0 for s in specs]
            ),
            leak_t=np.array([s.leak_t_s for s in specs]),
            leak_mass_rate=np.array([s.leak_fraction for s in specs]) * mdot_nom,
        )

    def inlet(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Inlet gas fraction and mixture velocity at time t, shape (S,) each."""
        ag = self.regime_gas_fraction - self.slug_amp * np.sin(2 * np.pi * t / self.slug_period)
        ag = np.clip(ag, ALPHA_MIN, 1 - ALPHA_MIN)
        step = np.where(
            (self.inlet_step_t >= 0) & (t >= self.inlet_step_t), self.inlet_step_factor, 1.0
        )
        return ag, self.u_nominal * step

    def outlet_pressure(self, t: float, p_out_nominal: float) -> np.ndarray:
        r = np.clip((t - self.outlet_ramp_t) / np.maximum(self.outlet_ramp_dur, EPS), 0.0, 1.0)
        r = np.where(self.outlet_ramp_t >= 0, r, 0.0)
        return p_out_nominal + self.outlet_dp * r

    def leak_rate(self, t: float, ramp_s: float) -> np.ndarray:
        """Total leaking mass rate (kg/s) at time t, shape (S,)."""
        r = np.clip((t - self.leak_t) / max(ramp_s, EPS), 0.0, 1.0)
        return np.where(self.is_leak, self.leak_mass_rate * r, 0.0)


@dataclass
class SimulationOutput:
    """Sensor series and field snapshots for a batch of scenarios."""

    t_sensor: np.ndarray  # (T,)
    p_in: np.ndarray  # (S, T) Pa, clean
    p_out: np.ndarray  # (S, T)
    q_in: np.ndarray  # (S, T) m3/s mixture volumetric flow, clean
    q_out: np.ndarray  # (S, T)
    t_snap: np.ndarray  # (K,)
    x_centers: np.ndarray  # (N,)
    snap_p: np.ndarray  # (S, K, N) float32
    snap_alpha_g: np.ndarray  # (S, K, N)
    snap_u_g: np.ndarray  # (S, K, N)
    snap_u_l: np.ndarray  # (S, K, N)
    mass_in_total: np.ndarray  # (S,) kg entering over the run
    mass_out_total: np.ndarray  # (S,) kg leaving at the outlet
    mass_leaked_total: np.ndarray  # (S,) kg leaked
    mass_initial: np.ndarray  # (S,) kg in the pipe at t = 0
    mass_final: np.ndarray  # (S,) kg in the pipe at the end


def _rusanov(
    u_l_state: np.ndarray, f_l: np.ndarray, u_r_state: np.ndarray, f_r: np.ndarray, lam: np.ndarray
) -> np.ndarray:
    return 0.5 * (f_l + f_r) - 0.5 * lam * (u_r_state - u_l_state)


def simulate(
    specs: list[ScenarioSpec],
    pipe: PipelineConfig | None = None,
    sim: SimulationConfig | None = None,
) -> SimulationOutput:
    """Run every scenario in ``specs`` together. Deterministic (no randomness inside)."""
    pipe = pipe or PipelineConfig()
    sim = sim or SimulationConfig()
    S, N = len(specs), sim.n_cells
    dx = pipe.length_m / N
    dt = sim.dt_s
    A = pipe.area_m2
    c = pipe.gas_sound_speed
    c2 = c * c
    f_wall = pipe.wall_friction_factor / (2.0 * pipe.diameter_m)
    forcing = BatchForcing.from_specs(specs, pipe, sim)
    x_centers = (np.arange(N) + 0.5) * dx

    # Initial condition: inlet fractions and velocity everywhere, linear pressure profile.
    ag0, u0 = forcing.inlet(0.0)
    p_lin = pipe.p_ref + (pipe.p_outlet - pipe.p_ref) * (x_centers / pipe.length_m)[None, :]
    alpha_g = np.repeat(ag0[:, None], N, axis=1)
    alpha_l = 1.0 - alpha_g
    rho_g = p_lin / c2
    m_g = alpha_g * rho_g
    m_l = alpha_l * pipe.rho_liquid
    i_g = m_g * u0[:, None]
    i_l = m_l * u0[:, None]

    n_steps = sim.n_steps
    sample_every = max(int(round(sim.sample_dt_s / dt)), 1)
    snap_every = max(int(round(sim.snapshot_dt_s / dt)), 1)
    T = n_steps // sample_every + 1
    K = n_steps // snap_every + 1
    t_sensor = np.arange(T) * sample_every * dt
    t_snap = np.arange(K) * snap_every * dt
    p_in = np.zeros((S, T))
    p_out = np.zeros((S, T))
    q_in = np.zeros((S, T))
    q_out = np.zeros((S, T))
    snap_p = np.zeros((S, K, N), np.float32)
    snap_ag = np.zeros((S, K, N), np.float32)
    snap_ug = np.zeros((S, K, N), np.float32)
    snap_ul = np.zeros((S, K, N), np.float32)
    mass_in = np.zeros(S)
    mass_out = np.zeros(S)
    mass_leak = np.zeros(S)
    mass_initial = (m_g + m_l).sum(axis=1) * A * dx

    rows = np.arange(S)
    leak_cell = forcing.leak_cell

    def record(k_sample: int | None, k_snap: int | None) -> None:
        pr = primitives(m_g, m_l, i_g, i_l, pipe)
        if k_sample is not None:
            q = A * (pr.alpha_g * pr.u_g + pr.alpha_l * pr.u_l)
            p_in[:, k_sample] = pr.p[:, 0]
            p_out[:, k_sample] = pr.p[:, -1]
            q_in[:, k_sample] = q[:, 0]
            q_out[:, k_sample] = q[:, -1]
        if k_snap is not None:
            snap_p[:, k_snap] = pr.p
            snap_ag[:, k_snap] = pr.alpha_g
            snap_ug[:, k_snap] = pr.u_g
            snap_ul[:, k_snap] = pr.u_l

    record(0, 0)
    for n in range(1, n_steps + 1):
        t = (n - 1) * dt
        pr = primitives(m_g, m_l, i_g, i_l, pipe)

        # Ghost cells. Inlet: imposed fractions/velocity, zero pressure gradient.
        ag_in, u_in = forcing.inlet(t)
        al_in = 1.0 - ag_in
        rho_g_in = pr.rho_g[:, 0]
        mg_in = ag_in * rho_g_in
        ml_in = al_in * pipe.rho_liquid
        # Outlet: imposed pressure, extrapolated fractions and velocities.
        p_bc = forcing.outlet_pressure(t, pipe.p_outlet)
        rho_g_out = p_bc / c2
        mg_out = pr.alpha_g[:, -1] * rho_g_out
        ml_out = pr.alpha_l[:, -1] * pipe.rho_liquid

        # Extended arrays with ghosts: shape (S, N + 2).
        def ext(a_in, a_mid, a_out):
            return np.concatenate([a_in[:, None], a_mid, a_out[:, None]], axis=1)

        MG = ext(mg_in, m_g, mg_out)
        ML = ext(ml_in, m_l, ml_out)
        UG = ext(u_in, pr.u_g, pr.u_g[:, -1])
        UL = ext(u_in, pr.u_l, pr.u_l[:, -1])
        AG = ext(ag_in, pr.alpha_g, pr.alpha_g[:, -1])
        AL = 1.0 - AG
        P = ext(pr.p[:, 0], pr.p, p_bc)
        IG = MG * UG
        IL = ML * UL

        # Rusanov fluxes at the N + 1 interfaces.
        lam = np.maximum(np.abs(UG), np.abs(UL))
        lam = np.maximum(lam[:, :-1], lam[:, 1:]) + c
        F_mg = _rusanov(MG[:, :-1], (MG * UG)[:, :-1], MG[:, 1:], (MG * UG)[:, 1:], lam)
        F_ml = _rusanov(ML[:, :-1], (ML * UL)[:, :-1], ML[:, 1:], (ML * UL)[:, 1:], lam)
        FIG = IG * UG + AG * P
        FIL = IL * UL + AL * P
        F_ig = _rusanov(IG[:, :-1], FIG[:, :-1], IG[:, 1:], FIG[:, 1:], lam)
        F_il = _rusanov(IL[:, :-1], FIL[:, :-1], IL[:, 1:], FIL[:, 1:], lam)

        # Non-conservative pressure remainder: + p d(alpha_k)/dx (central).
        dag_dx = (AG[:, 2:] - AG[:, :-2]) / (2 * dx)
        src_ig = pr.p * dag_dx
        src_il = -pr.p * dag_dx

        # Wall friction and interfacial drag.
        src_ig -= f_wall * m_g * pr.u_g * np.abs(pr.u_g)
        src_il -= f_wall * m_l * pr.u_l * np.abs(pr.u_l)
        slip = pr.u_g - pr.u_l
        drag = (
            forcing.drag_coeff[:, None] * pr.alpha_g * pr.alpha_l * pr.rho_g * slip * np.abs(slip)
        )
        src_ig -= drag
        src_il += drag

        # Leak sink on the leak cell, split by local mass fraction.
        q_leak = forcing.leak_rate(t, sim.leak_ramp_s)  # kg/s, (S,)
        src_mg = np.zeros_like(m_g)
        src_ml = np.zeros_like(m_l)
        if q_leak.any():
            mg_c = m_g[rows, leak_cell]
            ml_c = m_l[rows, leak_cell]
            frac_g = mg_c / np.maximum(mg_c + ml_c, EPS)
            per_vol = q_leak / (A * dx)
            src_mg[rows, leak_cell] -= per_vol * frac_g
            src_ml[rows, leak_cell] -= per_vol * (1.0 - frac_g)
            src_ig[rows, leak_cell] -= per_vol * frac_g * pr.u_g[rows, leak_cell]
            src_il[rows, leak_cell] -= per_vol * (1.0 - frac_g) * pr.u_l[rows, leak_cell]
            mass_leak += q_leak * dt

        m_g = m_g + dt * (-(F_mg[:, 1:] - F_mg[:, :-1]) / dx + src_mg)
        m_l = m_l + dt * (-(F_ml[:, 1:] - F_ml[:, :-1]) / dx + src_ml)
        i_g = i_g + dt * (-(F_ig[:, 1:] - F_ig[:, :-1]) / dx + src_ig)
        i_l = i_l + dt * (-(F_il[:, 1:] - F_il[:, :-1]) / dx + src_il)
        m_g = np.maximum(m_g, EPS)
        m_l = np.clip(m_l, ALPHA_MIN * pipe.rho_liquid, (1 - ALPHA_MIN) * pipe.rho_liquid)

        mass_in += (F_mg[:, 0] + F_ml[:, 0]) * A * dt
        mass_out += (F_mg[:, -1] + F_ml[:, -1]) * A * dt

        if not np.isfinite(m_g).all():
            raise FloatingPointError(f"two-fluid solver diverged at step {n} (t = {t:.2f} s)")

        record(
            n // sample_every if n % sample_every == 0 else None,
            n // snap_every if n % snap_every == 0 else None,
        )

    mass_final = (m_g + m_l).sum(axis=1) * A * dx
    return SimulationOutput(
        t_sensor=t_sensor,
        p_in=p_in,
        p_out=p_out,
        q_in=q_in,
        q_out=q_out,
        t_snap=t_snap,
        x_centers=x_centers,
        snap_p=snap_p,
        snap_alpha_g=snap_ag,
        snap_u_g=snap_ug,
        snap_u_l=snap_ul,
        mass_in_total=mass_in,
        mass_out_total=mass_out,
        mass_leaked_total=mass_leak,
        mass_initial=mass_initial,
        mass_final=mass_final,
    )


def mass_residual_steady(
    alpha_g: np.ndarray,
    u_g: np.ndarray,
    u_l: np.ndarray,
    p: np.ndarray,
    pipe: PipelineConfig,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Steady-state mass residuals d(alpha_k rho_k u_k)/dx for a single snapshot (N,) each.

    Interior central differences; used by the tests to check that a converged no-leak
    steady state satisfies the continuity equations.
    """
    rho_g = p / pipe.gas_sound_speed**2
    flux_g = alpha_g * rho_g * u_g
    flux_l = (1 - alpha_g) * pipe.rho_liquid * u_l
    r_g = (flux_g[2:] - flux_g[:-2]) / (2 * dx)
    r_l = (flux_l[2:] - flux_l[:-2]) / (2 * dx)
    return r_g, r_l
