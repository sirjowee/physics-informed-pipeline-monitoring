# RESULTS.md — Reproduction log

**Status:** in progress. Re-implementation started 2026-10-26 by Ibrahim M. Joseph (third author).
All numbers in the "Reproduced" columns are produced by the code in this repository on synthetic/simulated data, exactly as in the original papers. Where a reproduced number differs from the published one, the difference is recorded here on the day it is observed and is not tuned away.

Published figures are transcribed from the papers as printed. Nothing in this file is a claim of field validation.

---

## Paper 1 — Physics-Aware ML Models for Pipeline Leak Detection and Quantification
Adegboye, Komolafe & Joseph, *IJSRED* 7(6), Nov–Dec 2024, pp. 1311–1316.
https://www.ijsred.com/volume7/issue6/IJSRED-V7I6P142.pdf

### 1.1 Published results — PINN (Table 1 of the paper)

| Metric | Published | Reproduced | Run / commit | Notes |
|---|---|---|---|---|
| Detection accuracy | 0.95 | | | |
| Precision | ~0.91 | | | |
| Recall | ~0.91 | | | |
| ROC-AUC | not reported | | | report anyway |
| Localization MAE (m) | ~5 | | | |
| Leak-rate MAPE (%) | ~1 | | | |

### 1.2 Published comparative results (Table 2 of the paper)

| Model | Acc. (pub.) | Prec. (pub.) | Rec. (pub.) | Loc. err. m (pub.) | Rate err. % (pub.) | Acc. (repro) | Prec. (repro) | Rec. (repro) | Loc. err. m (repro) | Rate err. % (repro) |
|---|---|---|---|---|---|---|---|---|---|---|
| PINN | 0.95 | 0.91 | 0.91 | ~5 | ~1 | | | | | |
| Black-box NN (no physics loss) | 0.93 | 0.89 | 0.87 | ~11 | ~1.5 | | | | | |
| RTTM (physics-only) | 0.88 | 0.81 | 0.79 | ~20 | ~2.4 | | | | | |

### 1.3 Experimental setup as described in the paper

| Element | Paper says | Implemented here |
|---|---|---|
| Geometry | horizontal pipeline segment; pressure and flow sensors at inlet and outlet | |
| Leak locations | uniform on [0, L] | |
| Leak rates | 0.1 % – 5 % of nominal flow | |
| Flow regimes | stratified, slug, annular | |
| Disturbances | Gaussian sensor noise; transient events | |
| Physics | 1-D isothermal two-fluid model; per-phase mass + momentum conservation; leak as Dirac-delta sink at x_L | |
| Network | feed-forward MLP, tanh; inputs (x, P, Q); outputs leak prob., leak location, leak rate | |
| Loss | BCE (detection) + MSE (location) + MSE (rate) + λ_phys · Σ‖R_mass‖² + ‖R_momentum‖² at collocation points | |
| Derivatives | automatic differentiation | |
| Split | event-based, disjoint train / val / test | |
| Optimizer | Adam, "tuned" learning rate and batch size | |
| Framework | TensorFlow 2.x | |

### 1.4 Parameters the paper does NOT specify (chosen here — record every choice)

The paper omits these. Each value below is this re-implementation's choice, not the paper's. Changing them will change the numbers; they are logged so the reproduction is honest and repeatable.

| Parameter | Value used | Rationale |
|---|---|---|
| Pipeline length L (m) | | |
| Pipe diameter (m) | | |
| Phase densities ρ_G, ρ_L (kg/m³) | | |
| Nominal flow rate(s) | | |
| Friction / drag closure (F_k) | | |
| Number of scenarios generated (leak / no-leak) | | |
| Time steps per scenario, Δt | | |
| Sensor noise σ | | |
| MLP depth × width | | |
| λ_phys | | |
| Learning rate, batch size, epochs | | |
| Number of collocation points | | |
| Random seed(s) | | |
| Black-box baseline architecture | | same MLP, λ_phys = 0 |
| RTTM baseline implementation | | mass-balance / transient residual thresholding — describe |

### 1.5 Reproduction notes (dated)

- 2026-10-27 — Repository scaffolded; published figures transcribed above.
-

### 1.6 Known gaps between this re-implementation and the paper

- The paper reports results on a dataset whose size and generator parameters are not given; this reproduction uses the parameters in §1.4.
- The paper's statement "code and synthetic datasets available upon request" referred to the original development repository, which is no longer available (see README). This repository is a from-scratch re-implementation.
- Uncertainty quantification (ensembles / MC-dropout) was not explored in the paper and is not implemented here.

---

## Paper 2 — Electromagnetic Signal Classification Using Deep Learning for Pipeline Structural Health Monitoring
Adegboye, Komolafe & Joseph, *IJSRED* 7(6), Nov–Dec 2024, pp. 1317–1322 (approx.).

### 2.1 Published results — clean test data (Table 1 of the paper)

| Class | Precision (pub.) | Recall (pub.) | F1 (pub.) | Support (pub.) | Precision (repro) | Recall (repro) | F1 (repro) |
|---|---|---|---|---|---|---|---|
| Corrosion | 1.00 | 1.00 | 1.00 | 18 | | | |
| Crack | 1.00 | 1.00 | 1.00 | 18 | | | |
| Leak | 1.00 | 1.00 | 1.00 | 18 | | | |
| Normal | 1.00 | 1.00 | 1.00 | 26 | | | |
| **Overall accuracy** | **1.00** | | | **80** | | | |

### 2.2 Published results — noise robustness (Table 2 of the paper)

| Noise σ | Accuracy (pub.) | Loss (pub.) | Precision (pub.) | Recall (pub.) | F1 (pub.) | Accuracy (repro) | Precision (repro) | Recall (repro) | F1 (repro) |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.0000 | 0.2369 | 1.0000 | 1.0000 | 1.0000 | | | | |
| 0.1 | 1.0000 | 0.3226 | 1.0000 | 1.0000 | 1.0000 | | | | |
| 0.3 | 1.0000 | 0.2374 | 1.0000 | 1.0000 | 1.0000 | | | | |
| 1.0 | 0.9375 | 0.2980 | 0.9056 | 0.9375 | 0.9320 | | | | |

Paper also reports at σ = 1.0: normal and leak classes retained 100 % recall; corrosion recall fell to 72 %; confusion was primarily corrosion ↔ other classes. Macro-F1 reported as 0.93.

### 2.3 Experimental setup as described in the paper

| Element | Paper says | Implemented here |
|---|---|---|
| Sensing | patch antenna mounted on pipeline; reflection coefficient \|S11\| | analytic \|S11\|(dB) model: sloped mismatch baseline minus a Lorentzian resonance dip plus a weaker second-harmonic dip (no full-wave solver) |
| Frequency sweep | 1–10 GHz, 0.1 GHz resolution → 91 points per trace | same: 1.0, 1.1, …, 10.0 GHz, 91 points |
| Classes | normal, crack, corrosion, leak | same, integer labels in that order |
| Dataset | 100 samples per class, 400 total, balanced, simulation-generated | same: 100 per class, 400 total, `generate_dataset(seed=0)` |
| Noise model | x̃(f) = x(f) + N(0, σ²), σ ∈ {0.1, 0.3, 1.0} | same formula, added to the per-trace z-scored trace (see §2.4) |
| Preprocessing | per-trace z-score normalization | same |
| Model | Conv1D(32, k=5, ReLU) → MaxPool1D → Conv1D(64, k=3, ReLU) → MaxPool1D → Flatten → Dense(64) → Dropout(0.5) → Softmax(4) | |
| Training | categorical cross-entropy; Adam lr = 0.001; batch 16; 50 epochs; early stopping | |
| Split | 80/20 stratified | same, seeded; exactly 20 test samples per class |

### 2.4 Parameters the paper does NOT specify (chosen here)

| Parameter | Value used | Rationale |
|---|---|---|
| S11 signature model per class (resonance shift, damping, bandwidth change) | Normal: resonance f0 ~ U(4.9, 5.1) GHz, dip depth U(23, 27) dB, FWHM U(0.55, 0.65) GHz. Crack: shift U(+0.3, +0.8) GHz, depth U(18, 22) dB, FWHM × U(0.8, 0.95). Corrosion: shift U(−0.6, −0.2) GHz, depth U(10, 16) dB, FWHM × U(1.5, 2.5). Leak: shift U(−1.5, −0.8) GHz, depth U(15, 20) dB, FWHM × U(1.2, 1.8). Baseline U(−4, −2) dB with slope U(−0.15, 0.05) dB/GHz; second-harmonic dip at 2·f0 with 0.35 × depth and 1.6 × FWHM. | paper describes qualitative effects only (section II); directions follow the paper's own physical reasoning: a crack lowers effective permittivity (shift up), corrosion products and leaked fluid are lossy and raise permittivity (shift down, damping, broadening); corrosion made the subtlest class, matching the paper's finding that it is the one confused under noise. Mapping confirmed with the author 2026-11-03. |
| Within-class variation (defect size / material property ranges) | the uniform ranges above, drawn independently per trace | gives overlap between corrosion and leak without making classes inseparable |
| Random seed(s) | data seed 0 (generator and split); model seeds recorded in §2.5 | one fixed seed per role |
| Scale on which noise is added | per-trace z-scored trace (unit variance), so σ = 1.0 is 0 dB SNR | the only scale consistent with the paper's pattern: σ = 1.0 is negligible against a 25 dB dip in raw dB units, and σ = 0.3 already swamps the signal in linear \|S11\| units |
| Early-stopping criterion | | |

Note: the published clean-test support (18/18/18/26 = 80) is not perfectly balanced despite a balanced 400-sample dataset and a stratified split; this reproduction uses a true stratified split (20/20/20/20) and records the difference.

### 2.5 Reproduction notes (dated)

- 2026-10-27 — Published figures transcribed.
- 2026-11-03 — Phase 1: synthetic S11 generator, noise injection, normalisation and seeded stratified split implemented (`shm_classification/data/s11_simulator.py`); example figure `figures/shm_classification/final_s11_examples.png`. No model trained yet.

---

## How to read this file

- A reproduced number within a few points of the published one on a different synthetic generator is consistent with the paper.
- A reproduced number that is materially worse is reported as such, with the most likely cause in the notes.
- No number in this file was produced on field or laboratory data.
- Every run is tied to a commit hash so it can be re-executed.