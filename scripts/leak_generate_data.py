"""Generate the synthetic two-fluid leak dataset and the example sensor figure.

Usage:
    python scripts/leak_generate_data.py [--seed 0] [--n-scenarios 240] [--full]

``--full`` uses 720 scenarios (240 per regime); the default is sized for the 20-minute
CPU budget. The 300 s run length is kept in both modes because the feature and physics
windows are defined on it. Writes ``data/leak/leak_dataset.npz`` and
``figures/leak_detection/final_sensor_examples.png`` (inlet/outlet pressure and flow for
one leak and one no-leak scenario per regime).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline_monitoring.leak_detection.data import (  # noqa: E402
    REGIME_NAMES,
    DatasetConfig,
    generate_dataset,
    save_dataset,
)

REPO = Path(__file__).resolve().parents[1]


def make_figure(d: dict, path: Path) -> None:
    t = d["t_sensor"]
    fig, axes = plt.subplots(3, 4, figsize=(17, 9), sharex=True)
    for r, regime in enumerate(REGIME_NAMES):
        for is_leak, style in ((False, "-"), (True, "--")):
            cand = np.flatnonzero((d["regime"] == regime) & (d["is_leak"] == is_leak))
            if cand.size == 0:
                continue
            i = int(cand[0])
            label = (
                f"leak x={d['leak_x_m'][i]:.0f} m, {100 * d['leak_fraction'][i]:.1f} %, "
                f"t={d['leak_t_s'][i]:.0f} s"
                if is_leak
                else "no leak"
            )
            for c, name in enumerate(("p_in", "p_out", "q_in", "q_out")):
                y = d["sensors"][i, :, c]
                y = y / 1e5 if name.startswith("p") else y
                axes[r, c].plot(t, y, style, lw=0.9, label=label)
                if is_leak:
                    axes[r, c].axvline(d["leak_t_s"][i], color="k", lw=0.5, alpha=0.4)
        for c, name in enumerate(("p_in (bar)", "p_out (bar)", "q_in (m3/s)", "q_out (m3/s)")):
            axes[r, c].set_title(f"{regime}: {name}", fontsize=10)
            axes[r, c].grid(alpha=0.3)
        axes[r, 0].legend(fontsize=7)
    for c in range(4):
        axes[-1, c].set_xlabel("time (s)")
    fig.suptitle(
        "Noisy inlet/outlet sensors, one no-leak (solid) and one leak (dashed) scenario per regime"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-scenarios", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "leak" / "leak_dataset.npz")
    ap.add_argument(
        "--figure",
        type=Path,
        default=REPO / "figures" / "leak_detection" / "final_sensor_examples.png",
    )
    args = ap.parse_args()

    cfg = DatasetConfig(seed=args.seed)
    if args.full:
        cfg = replace(cfg, n_scenarios=720)
    if args.n_scenarios is not None:
        cfg = replace(cfg, n_scenarios=args.n_scenarios)

    t0 = time.time()
    d = generate_dataset(cfg)
    el = time.time() - t0
    save_dataset(d, args.out)
    n_leak = int(d["is_leak"].sum())
    print(
        f"wrote {args.out}: {cfg.n_scenarios} scenarios ({n_leak} leak), "
        f"{d['sensors'].shape[1]} samples each, train/val/test = "
        f"{len(d['train_idx'])}/{len(d['val_idx'])}/{len(d['test_idx'])}, {el:.1f} s"
    )
    make_figure(d, args.figure)
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
