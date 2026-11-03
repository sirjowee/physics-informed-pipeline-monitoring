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
| Geometry | horizontal pipeline segment; pressure and flow sensors at inlet and outlet | horizontal, L = 2000 m, D = 0.3 m; p and mixture volumetric Q sampled at the first and last cell every 0.5 s |
| Leak locations | uniform on [0, L] | same, `rng.uniform(0, L)`; sink placed on the cell containing x_L |
| Leak rates | 0.1 % – 5 % of nominal flow | same, uniform in [0.001, 0.05] of the nominal mass flow, opened linearly over 5 s at an onset time uniform in [100, 180] s |
| Flow regimes | stratified, slug, annular | approximated by inlet gas fraction + interfacial drag coefficient (stratified 0.40 / 2 m⁻¹, slug 0.55 / 20 m⁻¹ with a ±0.15, 20 s inlet-fraction oscillation, annular 0.85 / 8 m⁻¹) |
| Disturbances | Gaussian sensor noise; transient events | σ_p = 0.1 % of 60 bar, σ_Q = 0.5 % of nominal Q; inlet flow step ×U(0.8, 1.2) and/or outlet pressure ramp U(−2, +2) bar over 20 s, each with probability 0.5 at a time uniform in [60, 240] s, in leak and no-leak scenarios alike |
| Physics | 1-D isothermal two-fluid model; per-phase mass + momentum conservation; leak as Dirac-delta sink at x_L | same equations solved by a batched finite-volume scheme (Rusanov flux, forward Euler, 50 cells, Δt = 0.05 s); isothermal ideal gas + incompressible liquid; delta represented as 1/Δx on the leak cell; leaking mass split between phases by local mass fraction (`leak_detection/data/two_fluid_solver.py`) |
| Network | feed-forward MLP, tanh; inputs (x, P, Q); outputs leak prob., leak location, leak rate | tanh MLPs: encoder (features → 32-d latent h), heads h → (leak logit, x_L/L, rate), field network (x/L, t/T, h) → (p, α_G, u_G, u_L); features are binned measured P and Q (§1.4) and x enters the field network (`leak_detection/pinn.py`) |
| Loss | BCE (detection) + MSE (location) + MSE (rate) + λ_phys · Σ‖R_mass‖² + ‖R_momentum‖² at collocation points | same three data terms + a sensor-fit MSE on the field network's p and Q at x = 0 and x = L (how measured P, Q tie to the fields) + λ_phys × mean squared scaled residuals of the four two-fluid equations at random (x, t) in [0, L] × [200, 300] s, with the leak sink at the predicted location and rate |
| Derivatives | automatic differentiation | `tf.GradientTape` on the field network w.r.t. x and t (`leak_detection/physics.py`) |
| Split | event-based, disjoint train / val / test | same: whole scenarios, 60/20/20, stratified by (regime, leak), seeded |
| Optimizer | Adam, "tuned" learning rate and batch size | Adam, lr 2e-3, 32 scenarios per step, fixed by reasoning (§1.4); only λ_phys swept on validation |
| Framework | TensorFlow 2.x | TensorFlow 2.18.1, CPU |

### 1.4 Parameters the paper does NOT specify (chosen here — record every choice)

The paper omits these. Each value below is this re-implementation's choice, not the paper's. Changing them will change the numbers; they are logged so the reproduction is honest and repeatable.

| Parameter | Value used | Rationale |
|---|---|---|
| Pipeline length L (m) | 2000 | short enough for a 300 s transient to settle; confirmed with the author 2026-11-03 |
| Pipe diameter (m) | 0.3 | typical gathering-line size; confirmed with the author |
| Phase densities ρ_G, ρ_L (kg/m³) | ρ_G = 50 at 60 bar (isothermal ideal gas, c = √(p/ρ) ≈ 346 m/s); ρ_L = 850 incompressible | natural gas near 60 bar / 300 K; crude oil; confirmed with the author |
| Nominal flow rate(s) | mixture velocities 1.5, 3.0, 4.5 m/s (Q ≈ 0.106, 0.212, 0.318 m³/s); inlet p ≈ 60 bar, outlet p = 55 bar boundary | three operating points cycled over scenarios; confirmed with the author |
| Friction / drag closure (F_k) | wall: Darcy f = 0.02 per phase, −(f/2D) α_k ρ_k u_k\|u_k\|; interfacial drag C_i α_G α_L ρ_G (u_G − u_L)\|u_G − u_L\| with C_i per regime (2, 20, 8 m⁻¹); no gravity (horizontal) | simplest textbook closures (Brennen 2005, ch. 7–8); C_i chosen so steady slip is 0.5–2 m/s |
| Number of scenarios generated (leak / no-leak) | 240 total, 114 leak / 126 no-leak (seed 0), 80 per regime; `--full` gives 720 | sized for the 20-minute CPU budget (generation takes about 5 s) |
| Time steps per scenario, Δt | 300 s at Δt = 0.05 s (6000 steps); sensors every 0.5 s (601 samples); field snapshots every 2 s (151 × 50 cells) | CFL ≈ 0.44; the initial linear pressure profile settles within ~100 s, before any leak opens |
| Feature construction | window [60, 300] s averaged in 4 s bins (60 bins) × 5 channels (p_in, p_out, q_in, q_out, q_in − q_out), each channel standardised by its training-set mean/std; plus regime one-hot and nominal velocity / 5 m/s → 304 features | regime and nominal velocity are operator-known settings and the residual needs the regime's drag closure. A variant appending each channel's deviation from the scenario's pre-leak baseline (604 features) was tried and rejected on validation (PINN score 1.079 vs 0.870; black-box 1.178 vs 1.133) |
| Residual scaling | mass residuals ÷ ρ_k U/L (U = 5 m/s), momentum residuals ÷ P/L (P = 5 bar); Dirac delta → Gaussian of width 40 m (one cell) | makes the four residuals O(1) and gives the location head a gradient |
| Sensor noise σ | σ_p = 6 kPa (0.1 % of 60 bar); σ_Q = 0.5 % of the scenario's nominal Q; noise seed = data seed + 2 | order of a good field transmitter |
| MLP depth × width | encoder 3 × 64 tanh → 32; heads 1 × 64 tanh; field network 3 × 64 tanh | smallest sizes that fit the 144-scenario training set in seconds; not tuned |
| λ_phys | 0.1, chosen from {0.01, 0.1, 1.0} on the validation set by the score (1 − accuracy) + MAE_loc/L + min(MAPE, 1). Run used for the committed weights (`results/leak_selection.json`): validation accuracy / MAE / MAPE 0.812 / 698 m / 44 % at 0.1; 0.792 / 778 m / 48 % at 0.01; 0.854 / 753 m / 47 % at 1.0; black-box (λ = 0): 0.854 / 838 m / 51 %. | the only swept hyperparameter. **Reproducibility limit:** with λ_phys > 0 the training is not bit-reproducible on this machine (TensorFlow 2.18.1 CPU, macOS arm64) even with every random draw made in NumPy, op determinism on, single-threaded execution and oneDNN disabled; repeated same-seed runs differ by about ±0.05 in validation accuracy and ±100 m in MAE, so the choice between λ values is within run-to-run noise. The black-box run (λ = 0) reproduces exactly. Evaluation from the committed weights is deterministic. |
| Learning rate, batch size, epochs | Adam 2e-3; 32 scenarios per step; 4000 steps (≈ 890 passes over the 144 training scenarios) | fixed before training; 13 ms per step on an M1 CPU |
| Number of collocation points | 256 per step (8 per scenario), x ~ U(0, L), t ~ U(200, 300) s; plus 64 sensor samples per scenario per step for the sensor-fit term | physics window after every leak has fully opened, so no onset time is predicted |
| Random seed(s) | data seed 0 (scenario draw, split uses seed + 1, noise seed + 2); model seeds recorded in §1.5 | one fixed seed per role |
| Black-box baseline architecture | identical network and training, λ_phys = 0 (sensor-fit term kept) | same MLP, λ_phys = 0 |
| RTTM detection threshold | relative outlet-flow residual > 0.00154, chosen on the validation set (validation accuracy 1.000) | leak and no-leak residuals do not overlap on validation (noise std 0.0003 vs smallest leak 0.001) |
| RTTM baseline implementation | the Phase 3 solver re-run per scenario under the no-leak assumption with the known boundary conditions; residuals averaged over [200, 300] s; detection by a threshold on the relative outlet-flow residual chosen on the validation set; rate = volumetric imbalance × inlet mixture density; location from the inlet-pressure residual via the friction argument x_L = L − Δp_in / (2 (Δp_model/L) ΔQ/Q) (`leak_detection/rttm.py`) | mass-balance / transient residual thresholding, simplified |

### 1.5 Reproduction notes (dated)

- 2026-10-27 — Repository scaffolded; published figures transcribed above.
- 2026-11-03 — Phase 3: batched 1-D two-fluid finite-volume generator implemented with the closures in §1.4; default dataset of 240 scenarios generated in about 5 s (`data/leak/leak_dataset.npz`, not committed; regenerate with `python scripts/leak_generate_data.py`). Checks: mass balance closes to machine precision with and without a leak; no-leak steady-state continuity residual below 5 % of the phase flux per cell; a 3 % leak at 900 m lowers outlet flow after onset while the pre-onset series are identical. Figure `figures/leak_detection/final_sensor_examples.png`. No model trained yet.
- 2026-11-03 — Phase 4: PINN, black-box and simplified RTTM implemented (`leak_detection/pinn.py`, `physics.py`, `rttm.py`, `features.py`); `python scripts/leak_train.py` trains four networks (λ_phys ∈ {0, 0.01, 0.1, 1}) in about 1.6 min on an M1 CPU and selects λ_phys = 0.1 and the RTTM threshold on the validation set (`results/leak_selection.json`, weights in `results/leak_models/`). Validation numbers are in §1.4; they are far from the published figures already at this stage (networks: 79–85 % accuracy, 700–840 m localisation MAE, 44–51 % rate MAPE; RTTM: 100 %, 167 m, 9.6 %). Training with the physics term is not bit-reproducible on this machine (see the λ_phys row); the committed weights are the run reported. The test set has not been evaluated.

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
| Corrosion | 1.00 | 1.00 | 1.00 | 18 | 1.00 | 1.00 | 1.00 (support 20) |
| Crack | 1.00 | 1.00 | 1.00 | 18 | 1.00 | 1.00 | 1.00 (support 20) |
| Leak | 1.00 | 1.00 | 1.00 | 18 | 1.00 | 1.00 | 1.00 (support 20) |
| Normal | 1.00 | 1.00 | 1.00 | 26 | 1.00 | 1.00 | 1.00 (support 20) |
| **Overall accuracy** | **1.00** | | | **80** | **1.00** (80 test traces; run ccc31d6, data seed 0, model seed 0) | | |

### 2.2 Published results — noise robustness (Table 2 of the paper)

| Noise σ | Accuracy (pub.) | Loss (pub.) | Precision (pub.) | Recall (pub.) | F1 (pub.) | Accuracy (repro) | Precision (repro) | Recall (repro) | F1 (repro) |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.0000 | 0.2469 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.1 | 1.0000 | 0.3226 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.3 | 1.0000 | 0.2374 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1.0 | 0.8375 | 0.2980 | 0.9056 | 0.8375 | 0.8320 | 0.8625 | 0.8881 | 0.8625 | 0.8435 |

Transcription note (2026-11-03): the paper's printed Table 2 gives accuracy 0.8375, recall 0.8375 and F1 0.8320 at σ = 1.0 and a clean loss of 0.2469; the earlier transcription here (0.9375 / 0.9320 / 0.2369) was wrong and has been corrected to the printed table. The paper's text, however, says 94 % accuracy and macro-F1 0.93 at σ = 1.0, so the paper is internally inconsistent on this row. The text also reports at σ = 1.0: normal and leak retained 100 % recall; corrosion recall fell to 0.72 (F1 0.84); crack recall 1.00 with precision 0.78; confusion primarily corrosion ↔ other classes. Reproduced losses are far lower than the published ones (0.0001 vs 0.2469 clean) because this generator's clean traces are separable with near-certain softmax outputs; the published loss values suggest the original generator produced harder, more overlapping traces.

Reproduced per-class at σ = 1.0 (run ccc31d6): normal P 1.00 / R 0.45 / F1 0.62; crack 0.80 / 1.00 / 0.89; corrosion 0.80 / 1.00 / 0.89; leak 0.95 / 1.00 / 0.98. Confusion (rows true, cols predicted, order normal/crack/corrosion/leak): [9 5 5 1], [0 20 0 0], [0 0 20 0], [0 0 0 20]. The confused class is **normal**, not corrosion as in the paper: see §2.5.

### 2.3 Experimental setup as described in the paper

| Element | Paper says | Implemented here |
|---|---|---|
| Sensing | patch antenna mounted on pipeline; reflection coefficient \|S11\| | analytic \|S11\|(dB) model: sloped mismatch baseline minus a Lorentzian resonance dip plus a weaker second-harmonic dip (no full-wave solver) |
| Frequency sweep | 1–10 GHz, 0.1 GHz resolution → 91 points per trace | same: 1.0, 1.1, …, 10.0 GHz, 91 points |
| Classes | normal, crack, corrosion, leak | same, integer labels in that order |
| Dataset | 100 samples per class, 400 total, balanced, simulation-generated | same: 100 per class, 400 total, `generate_dataset(seed=0)` |
| Noise model | x̃(f) = x(f) + N(0, σ²), σ ∈ {0.1, 0.3, 1.0} | same formula, added to the per-trace z-scored trace (see §2.4) |
| Preprocessing | per-trace z-score normalization | same |
| Model | Conv1D(32, k=5, ReLU) → MaxPool1D → Conv1D(64, k=3, ReLU) → MaxPool1D → Flatten → Dense(64) → Dropout(0.5) → Softmax(4) | exactly this (`shm_classification/model.py`); pool size 2, valid padding, ReLU on Dense(64) (see §2.4) |
| Training | categorical cross-entropy; Adam lr = 0.001; batch 16; 50 epochs; early stopping | same; early stopping on validation loss, patience 10, best weights restored; run stopped at epoch 36 |
| Split | 80/20 stratified | same, seeded; exactly 20 test samples per class |

### 2.4 Parameters the paper does NOT specify (chosen here)

| Parameter | Value used | Rationale |
|---|---|---|
| S11 signature model per class (resonance shift, damping, bandwidth change) | Normal: resonance f0 ~ U(4.9, 5.1) GHz, dip depth U(23, 27) dB, FWHM U(0.55, 0.65) GHz. Crack: shift U(+0.3, +0.8) GHz, depth U(18, 22) dB, FWHM × U(0.8, 0.95). Corrosion: shift U(−0.6, −0.2) GHz, depth U(10, 16) dB, FWHM × U(1.5, 2.5). Leak: shift U(−1.5, −0.8) GHz, depth U(15, 20) dB, FWHM × U(1.2, 1.8). Baseline U(−4, −2) dB with slope U(−0.15, 0.05) dB/GHz; second-harmonic dip at 2·f0 with 0.35 × depth and 1.6 × FWHM. | paper describes qualitative effects only (section II); directions follow the paper's own physical reasoning: a crack lowers effective permittivity (shift up), corrosion products and leaked fluid are lossy and raise permittivity (shift down, damping, broadening); corrosion made the subtlest class, matching the paper's finding that it is the one confused under noise. Mapping confirmed with the author 2026-11-03. |
| Within-class variation (defect size / material property ranges) | the uniform ranges above, drawn independently per trace | gives overlap between corrosion and leak without making classes inseparable |
| Random seed(s) | data seed 0 (generator and split); model seeds recorded in §2.5 | one fixed seed per role |
| Scale on which noise is added | per-trace z-scored trace (unit variance), so σ = 1.0 is 0 dB SNR | the only scale consistent with the paper's pattern: σ = 1.0 is negligible against a 25 dB dip in raw dB units, and σ = 0.3 already swamps the signal in linear \|S11\| units |
| Early-stopping criterion | validation loss, patience 10, restore best weights; validation set = 20 % of the training portion, stratified, seeded | the paper mentions early stopping without a criterion; the test set is never used for stopping |
| Pooling size, padding, Dense activation | MaxPooling1D pool 2; "valid" padding; ReLU on Dense(64) | Keras defaults where the paper is silent |
| Model seed | 0 (Glorot init, dropout, shuffling via `tf.keras.utils.set_random_seed`, deterministic ops on) | single seed; CPU run takes about 2 s |
| Noise seeds at evaluation | 1000 + index of σ in (0.0, 0.1, 0.3, 1.0) | fixed per noise level |

Note: the published clean-test support (18/18/18/26 = 80) is not perfectly balanced despite a balanced 400-sample dataset and a stratified split; this reproduction uses a true stratified split (20/20/20/20) and records the difference.

### 2.5 Reproduction notes (dated)

- 2026-10-27 — Published figures transcribed.
- 2026-11-03 — Phase 1: synthetic S11 generator, noise injection, normalisation and seeded stratified split implemented (`shm_classification/data/s11_simulator.py`); example figure `figures/shm_classification/final_s11_examples.png`. No model trained yet.
- 2026-11-03 — Phase 2: 1D-CNN trained and evaluated once on the held-out test set (commit ccc31d6 for the data and model code, run via `python scripts/shm_train_eval.py --seed 0`, 36 epochs, 2.3 s on an M1 CPU; metrics in `results/shm_metrics.json`). Clean and σ = 0.1, 0.3: accuracy 1.0000 (paper: 1.0000). σ = 1.0: accuracy 0.8625, macro precision 0.8881, recall 0.8625, F1 0.8435 (paper table: 0.8375 / 0.9056 / 0.8375 / 0.8320; paper text: 0.94 / F1 0.93). Per-class corrosion recall at σ = 1.0 is **1.00 here versus 0.72 in the paper**; the class that degrades here is normal (recall 0.45), which is split between crack and corrosion. Likely cause: in this generator the normal resonance sits between the crack (shifted up) and corrosion (shifted down) resonances, so at 0 dB SNR it is the class with neighbours on both sides, whereas the original generator evidently made corrosion the subtlest signature. This is a property of the chosen signature mapping (§2.4), not of the network, and was not adjusted after seeing the test result.

---

## How to read this file

- A reproduced number within a few points of the published one on a different synthetic generator is consistent with the paper.
- A reproduced number that is materially worse is reported as such, with the most likely cause in the notes.
- No number in this file was produced on field or laboratory data.
- Every run is tied to a commit hash so it can be re-executed.