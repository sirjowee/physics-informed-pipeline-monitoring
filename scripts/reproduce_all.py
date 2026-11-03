"""Regenerate every number and figure in RESULTS.md from the fixed seeds.

Usage:
    python scripts/reproduce_all.py [--full] [--seed 0]

Runs, in order: SHM data generation, SHM training and evaluation, leak data generation,
leak model training with validation-based selection, leak test evaluation. Each step is a
subprocess of the corresponding script in ``scripts/``; their printed tables are the
RESULTS.md tables. Timings and the commit hash are written to
``results/reproduce_all_log.json``.

``--full`` uses 720 leak scenarios and 8000 training steps per network instead of 240 and
4000. The default completes in a few minutes on a laptop CPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

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


def run(label: str, args: list[str]) -> dict:
    print(f"\n{'=' * 80}\n{label}\n{'=' * 80}", flush=True)
    t0 = time.time()
    subprocess.check_call([sys.executable, *args], cwd=REPO)
    el = time.time() - t0
    print(f"[{label}] {el:.1f} s", flush=True)
    return {"step": label, "seconds": el, "command": " ".join(args)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed = str(args.seed)
    steps = "8000" if args.full else "4000"
    leak_gen = ["scripts/leak_generate_data.py", "--seed", seed] + (["--full"] if args.full else [])

    t0 = time.time()
    log = [
        run("1/5 SHM: generate S11 dataset", ["scripts/shm_generate_data.py", "--seed", seed]),
        run("2/5 SHM: train and evaluate 1D-CNN", ["scripts/shm_train_eval.py", "--seed", seed]),
        run("3/5 Leak: generate two-fluid dataset", leak_gen),
        run(
            "4/5 Leak: train PINN / black-box, select on validation, RTTM threshold",
            ["scripts/leak_train.py", "--seed", seed, "--steps", steps],
        ),
        run("5/5 Leak: evaluate on the test set", ["scripts/leak_evaluate.py"]),
    ]
    total = time.time() - t0
    out = {
        "commit": git_commit(),
        "seed": args.seed,
        "full": args.full,
        "total_seconds": total,
        "steps": log,
    }
    (REPO / "results" / "reproduce_all_log.json").write_text(json.dumps(out, indent=2))
    print(
        f"\nTotal {total / 60:.1f} min; log written to results/reproduce_all_log.json (commit {out['commit']})"
    )


if __name__ == "__main__":
    main()
