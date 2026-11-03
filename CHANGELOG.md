# Changelog

All notable changes to this re-implementation are recorded here. Dates are the dates the work was done; nothing is backdated.

## Unreleased

### 2026-11-03 — Phase 2, SHM 1D-CNN
- Added `shm_classification/model.py` (published architecture), `train.py` (Adam 0.001, batch 16, 50 epochs, early stopping on an internal validation split), `evaluate.py` (clean and noisy test metrics, confusion matrices, training curves), `scripts/shm_train_eval.py`, tests.
- RESULTS.md §2.1, §2.2, §2.3, §2.4 and §2.5 filled; Table 2 transcription corrected to the paper's printed values and the paper's text/table inconsistency noted.
- Committed `results/shm_metrics.json` and the two final figures.

### 2026-11-03 — Phase 1, SHM S11 dataset generator
- Added `shm_classification/data/s11_simulator.py`: 91-point 1–10 GHz grid, four-class analytic |S11| model (resonance shift, damping, bandwidth change), Gaussian noise injection, per-trace z-score, seeded stratified 80/20 split.
- Added `scripts/shm_generate_data.py` (writes `data/shm/s11_dataset.npz` and the example-trace figure) and tests.
- RESULTS.md §2.3 "Implemented here" and §2.4 parameter table filled for the data side.

### 2026-11-03 — Phase 0, housekeeping
- Renamed `requirments.txt` to `requirements.txt` and pinned versions.
- Rewrote README with title, provenance paragraph, author line, paper links, status and reproduction sections.
- Added `pyproject.toml`, `.gitignore`, `src/pipeline_monitoring/` package skeleton, `tests/`, `scripts/`, `figures/`, and GitHub Actions CI (ruff + pytest).
