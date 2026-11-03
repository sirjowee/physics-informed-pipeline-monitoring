"""Synthetic |S11| trace simulator for the four pipeline conditions.

What the paper specifies (Adegboye, Komolafe, Joseph & Kajero, IJSRED 7(6) 2024,
section III-A/B/C):

* frequency sweep 1-10 GHz at 0.1 GHz resolution, 91 points per trace;
* four classes: normal, crack, corrosion, leak; 100 samples per class, 400 total;
* each defect changes the trace through (i) a resonant-frequency shift, (ii) amplitude
  damping of the resonance and (iii) a broadening or narrowing of the resonance bandwidth
  (section II, "In practical implementations, these variations manifest as ...");
* Gaussian noise ``x_noisy = x + N(0, sigma^2)`` with sigma in {0.1, 0.3, 1.0};
* per-trace normalisation to zero mean and unit variance.

What the paper does NOT specify, and what is chosen here (recorded in RESULTS.md 2.4):

* the paper generated traces with CST Microwave Studio plus "custom signal generation
  scripts"; no full-wave solver is used here. Each trace is an analytic |S11| curve in
  dB: a gently sloped mismatch baseline minus a Lorentzian resonance dip, plus a weaker
  second-harmonic dip. This is the standard single-resonance approximation of a patch
  antenna's return loss;
* the per-class mapping of the three qualitative effects to numbers (see
  ``CLASS_SIGNATURES``), confirmed with the author on 2026-11-03;
* the scale on which noise is added: the per-trace z-scored trace, so that sigma is in
  units of the normalised trace and sigma = 1.0 corresponds to a signal-to-noise ratio
  of 0 dB. This is the only scale on which the paper's pattern (100 % at 0.1 and 0.3,
  a drop at 1.0) is plausible: in raw dB units sigma = 1.0 is negligible against a
  25 dB dip, and in linear |S11| units sigma = 0.3 already swamps the signal.

Everything is deterministic given the seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_NAMES: tuple[str, ...] = ("normal", "crack", "corrosion", "leak")
"""Class order used for integer labels everywhere in this subpackage."""

F_START_GHZ = 1.0
F_STOP_GHZ = 10.0
F_STEP_GHZ = 0.1
N_POINTS = 91


def frequency_grid() -> np.ndarray:
    """Return the 91-point frequency grid 1.0, 1.1, ..., 10.0 GHz (paper: 1-10 GHz, 0.1 GHz)."""
    return np.round(F_START_GHZ + F_STEP_GHZ * np.arange(N_POINTS), 6)


@dataclass(frozen=True)
class ClassSignature:
    """Parameter ranges for one class. All ranges are sampled uniformly.

    ``shift_ghz`` is the resonant-frequency shift relative to the nominal normal
    resonance, ``depth_db`` the depth of the main dip below the baseline, and
    ``bw_scale`` the multiplier on the normal bandwidth (FWHM).
    """

    shift_ghz: tuple[float, float]
    depth_db: tuple[float, float]
    bw_scale: tuple[float, float]


# Nominal "normal" resonance; each trace draws its own values from these ranges.
NORMAL_F0_GHZ: tuple[float, float] = (4.9, 5.1)
NORMAL_DEPTH_DB: tuple[float, float] = (23.0, 27.0)
NORMAL_BW_GHZ: tuple[float, float] = (0.55, 0.65)

CLASS_SIGNATURES: dict[str, ClassSignature] = {
    # Intact pipe: no shift, full depth, nominal bandwidth.
    "normal": ClassSignature(shift_ghz=(0.0, 0.0), depth_db=(1.0, 1.0), bw_scale=(1.0, 1.0)),
    # Crack: a geometric discontinuity lowers the effective permittivity seen by the
    # patch -> resonance moves UP; mild extra loss -> slight damping; slightly narrower.
    "crack": ClassSignature(shift_ghz=(0.3, 0.8), depth_db=(18.0, 22.0), bw_scale=(0.8, 0.95)),
    # Corrosion: lossy oxide layer -> resonance moves DOWN a little, strong damping of the
    # dip (more power reflected), markedly broader resonance (lower Q).
    "corrosion": ClassSignature(shift_ghz=(-0.6, -0.2), depth_db=(10.0, 16.0), bw_scale=(1.5, 2.5)),
    # Leak: fluid near the antenna raises the effective permittivity -> larger DOWNWARD
    # shift, moderate damping, moderately broader.
    "leak": ClassSignature(shift_ghz=(-1.5, -0.8), depth_db=(15.0, 20.0), bw_scale=(1.2, 1.8)),
}
"""Qualitative-to-quantitative mapping of defect effects on |S11| (chosen here, not from the paper).

For ``normal`` the ``depth_db`` field is a multiplier (1.0) on the sampled normal depth;
for the defect classes it is the absolute dip depth in dB.
"""

# Baseline mismatch level (dB) off resonance and its slope across the band; chosen so the
# off-resonance return loss sits in the -2 to -4 dB range typical of a mismatched patch.
BASELINE_DB: tuple[float, float] = (-4.0, -2.0)
BASELINE_SLOPE_DB_PER_GHZ: tuple[float, float] = (-0.15, 0.05)
# Second-harmonic dip: fraction of the main depth and bandwidth multiplier. Gives the
# trace a second feature so the classifier cannot key on a single point.
HARMONIC_DEPTH_FRACTION = 0.35
HARMONIC_BW_SCALE = 1.6


@dataclass(frozen=True)
class TraceParams:
    """Resolved parameters of one trace (useful for tests and figures)."""

    label: int
    f0_ghz: float
    depth_db: float
    bw_ghz: float
    baseline_db: float
    slope_db_per_ghz: float


def _lorentzian_dip(f: np.ndarray, f0: float, depth_db: float, bw_ghz: float) -> np.ndarray:
    """Lorentzian dip of given depth (dB) and full width at half maximum (GHz)."""
    half = bw_ghz / 2.0
    return depth_db * half**2 / ((f - f0) ** 2 + half**2)


def s11_trace_db(f: np.ndarray, p: TraceParams) -> np.ndarray:
    """|S11| in dB on grid ``f`` for parameters ``p`` (clean, deterministic)."""
    base = p.baseline_db + p.slope_db_per_ghz * (f - F_START_GHZ)
    main = _lorentzian_dip(f, p.f0_ghz, p.depth_db, p.bw_ghz)
    harm = _lorentzian_dip(
        f, 2.0 * p.f0_ghz, HARMONIC_DEPTH_FRACTION * p.depth_db, HARMONIC_BW_SCALE * p.bw_ghz
    )
    return base - main - harm


def sample_trace_params(rng: np.random.Generator, label: int) -> TraceParams:
    """Draw one set of trace parameters for class ``label``."""
    name = CLASS_NAMES[label]
    sig = CLASS_SIGNATURES[name]
    f0_normal = rng.uniform(*NORMAL_F0_GHZ)
    depth_normal = rng.uniform(*NORMAL_DEPTH_DB)
    bw_normal = rng.uniform(*NORMAL_BW_GHZ)
    shift = rng.uniform(*sig.shift_ghz)
    depth = (
        depth_normal * rng.uniform(*sig.depth_db)
        if name == "normal"
        else rng.uniform(*sig.depth_db)
    )
    bw = bw_normal * rng.uniform(*sig.bw_scale)
    return TraceParams(
        label=label,
        f0_ghz=f0_normal + shift,
        depth_db=depth,
        bw_ghz=bw,
        baseline_db=rng.uniform(*BASELINE_DB),
        slope_db_per_ghz=rng.uniform(*BASELINE_SLOPE_DB_PER_GHZ),
    )


def generate_dataset(
    n_per_class: int = 100, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, list[TraceParams]]:
    """Generate the clean dataset.

    Returns ``X`` of shape ``(4 * n_per_class, 91)`` in dB, integer labels ``y`` in class
    order ``CLASS_NAMES``, and the per-trace parameters. Paper: 100 per class, 400 total.
    """
    rng = np.random.default_rng(seed)
    f = frequency_grid()
    traces, labels, params = [], [], []
    for label in range(len(CLASS_NAMES)):
        for _ in range(n_per_class):
            p = sample_trace_params(rng, label)
            traces.append(s11_trace_db(f, p))
            labels.append(label)
            params.append(p)
    return np.asarray(traces, dtype=np.float64), np.asarray(labels, dtype=np.int64), params


def zscore_per_trace(X: np.ndarray) -> np.ndarray:
    """Per-trace normalisation to zero mean and unit variance (paper, section III-C)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / sd


def add_gaussian_noise(X: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """``x_noisy = x + N(0, sigma^2)`` elementwise (paper, section III-B).

    Applied to the normalised trace; see module docstring for why. ``sigma = 0`` returns a
    copy of ``X`` unchanged.
    """
    if sigma == 0.0:
        return X.copy()
    rng = np.random.default_rng(seed)
    return X + rng.normal(0.0, sigma, size=X.shape)


def stratified_split(
    y: np.ndarray, test_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded stratified index split. Paper: 80/20 stratified.

    With a balanced 400-sample dataset this yields exactly 20 test samples per class
    (the paper's Table 1 prints supports of 18/18/18/26; see RESULTS.md 2.4).
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = int(round(test_fraction * len(idx)))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train = np.concatenate(train_idx)
    test = np.concatenate(test_idx)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
