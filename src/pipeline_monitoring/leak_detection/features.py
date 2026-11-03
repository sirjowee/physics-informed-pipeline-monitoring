"""Feature construction from the inlet/outlet sensor series, and training targets.

Paper (section III-C): "Inputs: sensor-derived features (location x, measured P, Q)";
"Data normalization: inputs and outputs scaled to improve stability". The exact feature
construction is not given. Chosen here (RESULTS.md 1.4):

* analysis window t in [60, 300] s (after the start-up transient, which is identical for
  leak and no-leak scenarios), averaged into 4 s bins -> 60 bins;
* five channels per bin: p_in, p_out, q_in, q_out and the volumetric imbalance
  q_in - q_out; each channel is standardised with the training-set mean and standard
  deviation of that channel (one scalar pair per channel, not per bin);
* the flow regime one-hot (3 values) and the nominal mixture velocity (scaled by 5 m/s)
  are appended because they are operating conditions known to the operator and the
  physics residual needs the regime's drag closure;
* targets: leak indicator, leak location scaled by L (defined for leak scenarios only),
  leak rate as a fraction of nominal mass flow scaled by 0.05 (zero for no-leak scenarios).

A variant that appended each channel's deviation from the scenario's own pre-leak
baseline (mean over [60, 100) s) was tried on 2026-11-03 and rejected because it scored
worse on the validation set (RESULTS.md 1.4); the test set was not consulted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.config import REGIME_NAMES

WINDOW_START_S = 60.0
BIN_S = 4.0
U_SCALE = 5.0  # m/s
RATE_SCALE = 0.05  # leak fraction scale
PHYSICS_WINDOW_S = (200.0, 300.0)  # where the field network and physics loss operate
N_CHANNELS = 5


@dataclass
class Normaliser:
    mean: np.ndarray  # (5,)
    std: np.ndarray  # (5,)

    @classmethod
    def fit(cls, channels: np.ndarray) -> Normaliser:
        """``channels``: (n, bins, 5) training-set binned channels."""
        flat = channels.reshape(-1, channels.shape[-1])
        return cls(flat.mean(axis=0), flat.std(axis=0) + 1e-12)


def binned_channels(d: dict, idx: np.ndarray) -> np.ndarray:
    """(len(idx), n_bins, 5) binned noisy sensor channels plus imbalance."""
    t = d["t_sensor"]
    s = d["sensors"][idx].astype(np.float64)  # (n, T, 4)
    imb = (s[..., 2] - s[..., 3])[..., None]
    s = np.concatenate([s, imb], axis=-1)  # (n, T, 5)
    mask = t >= WINDOW_START_S
    s = s[:, mask]
    tw = t[mask]
    n_bins = int(round((tw[-1] - tw[0] + (tw[1] - tw[0])) / BIN_S))
    per_bin = s.shape[1] // n_bins
    return s[:, : per_bin * n_bins].reshape(s.shape[0], n_bins, per_bin, N_CHANNELS).mean(axis=2)


def operating_conditions(d: dict, idx: np.ndarray) -> np.ndarray:
    """(n, 4): regime one-hot and nominal velocity / U_SCALE."""
    onehot = np.eye(len(REGIME_NAMES))[d["regime_id"][idx]]
    u = (d["nominal_velocity"][idx] / U_SCALE)[:, None]
    return np.concatenate([onehot, u], axis=1)


def build_features(d: dict, idx: np.ndarray, norm: Normaliser) -> np.ndarray:
    """Flattened standardised feature vector per scenario, (n, 60 * 5 + 4)."""
    ch = (binned_channels(d, idx) - norm.mean) / norm.std
    return np.concatenate([ch.reshape(len(idx), -1), operating_conditions(d, idx)], axis=1).astype(
        np.float32
    )


def build_targets(d: dict, idx: np.ndarray) -> dict[str, np.ndarray]:
    L = float(d["length_m"])
    return {
        "is_leak": d["is_leak"][idx].astype(np.float32),
        "loc": (d["leak_x_m"][idx] / L).astype(np.float32),
        "rate": (d["leak_fraction"][idx] / RATE_SCALE).astype(np.float32),
    }


def sensor_window(d: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Noisy sensor samples inside the physics window: times (T_w,) and values (n, T_w, 4)."""
    t = d["t_sensor"]
    m = (t >= PHYSICS_WINDOW_S[0]) & (t <= PHYSICS_WINDOW_S[1])
    return t[m], d["sensors"][idx][:, m].astype(np.float32)
