"""Train the S11 1D-CNN and evaluate on clean and noisy test sets.

Usage:
    python scripts/shm_train_eval.py [--seed 0] [--data data/shm/s11_dataset.npz]

Regenerates the dataset if the .npz is missing, trains with early stopping on an internal
validation split, evaluates once on the held-out test set at sigma in {0, 0.1, 0.3, 1.0},
writes ``results/shm_metrics.json``, the figures under ``figures/shm_classification/`` and
prints the RESULTS.md tables.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from pipeline_monitoring.shm_classification.data import zscore_per_trace
from pipeline_monitoring.shm_classification.evaluate import (
    evaluate_all,
    results_markdown,
    save_figures,
)
from pipeline_monitoring.shm_classification.train import train_cnn1d

REPO = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0, help="model seed (data seed is in the .npz)")
    ap.add_argument("--data", type=Path, default=REPO / "data" / "shm" / "s11_dataset.npz")
    ap.add_argument("--results", type=Path, default=REPO / "results" / "shm_metrics.json")
    ap.add_argument("--figures", type=Path, default=REPO / "figures" / "shm_classification")
    args = ap.parse_args()

    if not args.data.exists():
        subprocess.check_call(
            [
                "python",
                str(REPO / "scripts" / "shm_generate_data.py"),
                "--out",
                str(args.data.parent),
            ]
        )
    d = np.load(args.data)
    X = zscore_per_trace(d["X_db"])
    y = d["y"]
    tr, te = d["train_idx"], d["test_idx"]

    t0 = time.time()
    res = train_cnn1d(X[tr], y[tr], seed=args.seed)
    train_s = time.time() - t0
    evals = evaluate_all(res.model, X[te], y[te])
    md = results_markdown(evals)
    print(md)
    print(f"epochs run: {res.epochs_run}, training time {train_s:.1f} s, commit {git_commit()}")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(
            {
                "commit": git_commit(),
                "model_seed": args.seed,
                "data_seed": int(d["seed"]),
                "epochs_run": res.epochs_run,
                "train_seconds": train_s,
                "history": res.history,
                "evaluations": [e.to_dict() for e in evals],
            },
            indent=2,
        )
    )
    save_figures(evals, res.history, args.figures)
    print(f"wrote {args.results} and figures in {args.figures}")


if __name__ == "__main__":
    main()
