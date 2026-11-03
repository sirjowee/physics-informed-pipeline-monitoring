# Changelog

All notable changes to this re-implementation are recorded here. Dates are the dates the work was done; nothing is backdated.

## Unreleased

### 2026-11-03 — Phase 1, SHM S11 dataset generator
- Added `shm_classification/data/s11_simulator.py`: 91-point 1–10 GHz grid, four-class analytic |S11| model (resonance shift, damping, bandwidth change), Gaussian noise injection, per-trace z-score, seeded stratified 80/20 split.
- Added `scripts/shm_generate_data.py` (writes `data/shm/s11_dataset.npz` and the example-trace figure) and tests.
- RESULTS.md §2.3 "Implemented here" and §2.4 parameter table filled for the data side.

### 2026-11-03 — Phase 0, housekeeping
- Renamed `requirments.txt` to `requirements.txt` and pinned versions.
- Rewrote README with title, provenance paragraph, author line, paper links, status and reproduction sections.
- Added `pyproject.toml`, `.gitignore`, `src/pipeline_monitoring/` package skeleton, `tests/`, `scripts/`, `figures/`, and GitHub Actions CI (ruff + pytest).
