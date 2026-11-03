"""Two-fluid residuals for the physics-informed loss (TensorFlow), shared with the tests.

The residuals are the paper's (section III-C.2):

    R_mass,k     = d(alpha_k rho_k)/dt + d(alpha_k rho_k u_k)/dx - S_k,mass
    R_momentum,k = d(alpha_k rho_k u_k)/dt + d(alpha_k rho_k u_k^2)/dx + alpha_k dp/dx - F_k - S_k,mom

with the same closures as the data generator (isothermal ideal gas, incompressible liquid,
Darcy wall friction, regime-dependent interfacial drag, horizontal pipe) and the leak sink
S_k,mass = -Q_leak,k delta(x - x_L). The Dirac delta is smoothed to a Gaussian of width
``DELTA_WIDTH_M`` so that the location head receives gradients. Each residual is divided
by a reference scale so the four terms are O(1) (RESULTS.md 1.4).

``residuals_from_fields`` takes fields and their derivatives as plain tensors, so it can be
evaluated on solver snapshots with finite differences (tests) or on network outputs with
automatic differentiation (training).
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from .data.config import PipelineConfig

DELTA_WIDTH_M = 40.0  # one solver cell
P_SCALE = 5.0e5  # Pa, pressure scale for the network output
U_SCALE = 5.0  # m/s


def gaussian_delta(x: tf.Tensor, x_leak: tf.Tensor, width: float = DELTA_WIDTH_M) -> tf.Tensor:
    """Smoothed Dirac delta (1/m) centred at ``x_leak``; integrates to one over x."""
    return tf.exp(-0.5 * ((x - x_leak) / width) ** 2) / (width * math.sqrt(2.0 * math.pi))


def residuals_from_fields(
    p: tf.Tensor,
    alpha_g: tf.Tensor,
    u_g: tf.Tensor,
    u_l: tf.Tensor,
    dp_dt: tf.Tensor,
    dp_dx: tf.Tensor,
    dag_dt: tf.Tensor,
    dag_dx: tf.Tensor,
    dug_dt: tf.Tensor,
    dug_dx: tf.Tensor,
    dul_dt: tf.Tensor,
    dul_dx: tf.Tensor,
    x: tf.Tensor,
    leak_rate_kg_s: tf.Tensor,
    leak_x_m: tf.Tensor,
    drag_coeff: tf.Tensor,
    pipe: PipelineConfig | None = None,
) -> dict[str, tf.Tensor]:
    """Scaled residuals, each the same shape as ``p``. All inputs in SI units.

    ``leak_rate_kg_s`` is the total leaking mass rate at the point's time (zero for no leak).
    """
    pipe = pipe or PipelineConfig()
    c2 = pipe.gas_sound_speed**2
    rho_l = pipe.rho_liquid
    A = pipe.area_m2
    f_wall = pipe.wall_friction_factor / (2.0 * pipe.diameter_m)

    rho_g = p / c2
    drho_g_dt = dp_dt / c2
    drho_g_dx = dp_dx / c2
    alpha_l = 1.0 - alpha_g
    m_g = alpha_g * rho_g
    m_l = alpha_l * rho_l

    dm_g_dt = dag_dt * rho_g + alpha_g * drho_g_dt
    dm_g_dx = dag_dx * rho_g + alpha_g * drho_g_dx
    dm_l_dt = -dag_dt * rho_l
    dm_l_dx = -dag_dx * rho_l

    # Leak sink per unit volume, split by local mass fraction.
    delta = gaussian_delta(x, leak_x_m)
    frac_g = m_g / (m_g + m_l)
    sink_g = leak_rate_kg_s * delta / A * frac_g
    sink_l = leak_rate_kg_s * delta / A * (1.0 - frac_g)

    r_mass_g = dm_g_dt + dm_g_dx * u_g + m_g * dug_dx + sink_g
    r_mass_l = dm_l_dt + dm_l_dx * u_l + m_l * dul_dx + sink_l

    slip = u_g - u_l
    drag = drag_coeff * alpha_g * alpha_l * rho_g * slip * tf.abs(slip)
    f_g = -f_wall * m_g * u_g * tf.abs(u_g) - drag
    f_l = -f_wall * m_l * u_l * tf.abs(u_l) + drag

    # d(m u)/dt + d(m u^2)/dx = u * R_mass_conv + m * (du/dt + u du/dx) where
    # R_mass_conv = dm/dt + d(m u)/dx (without the sink); expand directly instead.
    d_ig_dt = dm_g_dt * u_g + m_g * dug_dt
    d_ig_dx_u = (dm_g_dx * u_g + m_g * dug_dx) * u_g + m_g * u_g * dug_dx
    d_il_dt = dm_l_dt * u_l + m_l * dul_dt
    d_il_dx_u = (dm_l_dx * u_l + m_l * dul_dx) * u_l + m_l * u_l * dul_dx
    r_mom_g = d_ig_dt + d_ig_dx_u + alpha_g * dp_dx - f_g + sink_g * u_g
    r_mom_l = d_il_dt + d_il_dx_u + alpha_l * dp_dx - f_l + sink_l * u_l

    # Reference scales: rho U / L for mass, P / L for momentum.
    L = pipe.length_m
    s_mass_g = pipe.rho_gas_ref * U_SCALE / L
    s_mass_l = rho_l * U_SCALE / L
    s_mom = P_SCALE / L
    return {
        "mass_g": r_mass_g / s_mass_g,
        "mass_l": r_mass_l / s_mass_l,
        "mom_g": r_mom_g / s_mom,
        "mom_l": r_mom_l / s_mom,
    }


def residuals_on_snapshots(
    snap_p: np.ndarray,
    snap_ag: np.ndarray,
    snap_ug: np.ndarray,
    snap_ul: np.ndarray,
    t_snap: np.ndarray,
    x_centers: np.ndarray,
    leak_rate_kg_s: float,
    leak_x_m: float,
    drag_coeff: float,
    pipe: PipelineConfig | None = None,
) -> dict[str, np.ndarray]:
    """Finite-difference residuals on solver snapshots of one scenario, shape (K-2, N-2).

    Used by the tests: on a converged no-leak steady state the residuals should be small
    relative to the individual terms.
    """
    dt = float(t_snap[1] - t_snap[0])
    dx = float(x_centers[1] - x_centers[0])
    fields = [a.astype(np.float64) for a in (snap_p, snap_ag, snap_ug, snap_ul)]
    inner = [f[1:-1, 1:-1] for f in fields]
    d_dt = [(f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dt) for f in fields]
    d_dx = [(f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dx) for f in fields]
    x = np.broadcast_to(x_centers[1:-1][None, :], inner[0].shape)
    args = [tf.constant(a) for a in inner]
    derivs = []
    for i in range(4):
        derivs += [tf.constant(d_dt[i]), tf.constant(d_dx[i])]
    res = residuals_from_fields(
        *args,
        *derivs,
        tf.constant(x),
        tf.constant(float(leak_rate_kg_s), tf.float64),
        tf.constant(float(leak_x_m), tf.float64),
        tf.constant(float(drag_coeff), tf.float64),
        pipe,
    )
    return {k: v.numpy() for k, v in res.items()}
