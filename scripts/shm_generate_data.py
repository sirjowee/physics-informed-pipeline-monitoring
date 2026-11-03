"""Generate the synthetic |S11| dataset and the example-trace figure.

Usage:
    python scripts/shm_generate_data.py [--seed 0] [--out data/shm] [--no-figure]

Writes ``data/shm/s11_dataset.npz`` (clean traces in dB, labels, train/test indices) and
``figures/shm_classification/final_s11_examples.png`` showing one clean trace per class and the
same traces after normalisation and sigma = 1.0 noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline_monitoring.shm_classification.data import (  # noqa: E402
    CLASS_NAMES,
    add_gaussian_noise,
    frequency_grid,
    generate_dataset,
    stratified_split,
    zscore_per_trace,
)

REPO = Path(__file__).resolve().parents[1]


def make_figure(X: np.ndarray, y: np.ndarray, seed: int, path: Path) -> None:
    f = frequency_grid()
    Z = zscore_per_trace(X)
    Zn = add_gaussian_noise(Z, 1.0, seed=seed)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, name in enumerate(CLASS_NAMES):
        i = int(np.flatnonzero(y == label)[0])
        axes[0].plot(f, X[i], label=name)
        axes[1].plot(f, Z[i], label=name)
        axes[2].plot(f, Zn[i], label=name, lw=0.8)
    axes[0].set(title="Clean |S11| (dB)", xlabel="Frequency (GHz)", ylabel="|S11| (dB)")
    axes[1].set(title="Per-trace z-scored", xlabel="Frequency (GHz)", ylabel="normalised")
    axes[2].set(title="z-scored + Gaussian noise, sigma = 1.0", xlabel="Frequency (GHz)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Synthetic S11 traces, one example per class (seed {seed})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "shm")
    ap.add_argument(
        "--figure",
        type=Path,
        default=REPO / "figures" / "shm_classification" / "final_s11_examples.png",
    )
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    X, y, _ = generate_dataset(n_per_class=args.n_per_class, seed=args.seed)
    train_idx, test_idx = stratified_split(y, test_fraction=0.2, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "s11_dataset.npz"
    np.savez(
        out,
        X_db=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        freq_ghz=frequency_grid(),
        class_names=np.array(CLASS_NAMES),
        seed=args.seed,
    )
    print(f"wrote {out}: X {X.shape}, train {len(train_idx)}, test {len(test_idx)}")
    if not args.no_figure:
        make_figure(X, y, args.seed, args.figure)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
