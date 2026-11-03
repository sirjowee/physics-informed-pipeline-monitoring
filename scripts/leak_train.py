"""Train the leak-detection models and select hyperparameters on the validation set.

Usage:
    python scripts/leak_train.py [--data data/leak/leak_dataset.npz] [--steps 4000]

Protocol (RESULTS.md 1.4): architecture, learning rate, batch size and step count are
fixed by stated reasoning before any training; only lambda_phys is swept, over
{0.01, 0.1, 1.0}, and chosen by the validation score
    (1 - accuracy) + MAE_loc / L + min(MAPE_rate, 1).
The black-box baseline is the same network with lambda_phys = 0. The RTTM detection
threshold is chosen on the validation set. The test set is not touched here.

Writes ``results/leak_selection.json`` and the selected weights under ``results/leak_models/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from pipeline_monitoring.leak_detection.data import load_dataset
from pipeline_monitoring.leak_detection.features import (
    Normaliser,
    binned_channels,
    build_features,
    build_targets,
    sensor_window,
)
from pipeline_monitoring.leak_detection.pinn import PINNConfig, predict, save_model, train_pinn
from pipeline_monitoring.leak_detection.rttm import choose_threshold, run_rttm

REPO = Path(__file__).resolve().parents[1]
LAMBDAS = (0.01, 0.1, 1.0)


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def val_metrics(out: dict, y: dict, L: float) -> dict[str, float]:
    m = y["is_leak"] > 0.5
    acc = float(((out["prob"] > 0.5) == m).mean())
    mae = float(np.abs(out["loc"][m] - y["loc"][m]).mean() * L)
    mape = float(
        (np.abs(out["rate"][m] - y["rate"][m]) / np.maximum(y["rate"][m], 1e-6)).mean() * 100
    )
    return {
        "accuracy": acc,
        "loc_mae_m": mae,
        "rate_mape_pct": mape,
        "score": (1 - acc) + mae / L + min(mape / 100, 1.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=REPO / "data" / "leak" / "leak_dataset.npz")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "results")
    args = ap.parse_args()

    if not args.data.exists():
        subprocess.check_call(["python", str(REPO / "scripts" / "leak_generate_data.py")])
    d = load_dataset(args.data)
    L = float(d["length_m"])
    tr, va = d["train_idx"], d["val_idx"]
    norm = Normaliser.fit(binned_channels(d, tr))
    Xtr, Xva = build_features(d, tr, norm), build_features(d, va, norm)
    Ytr, Yva = build_targets(d, tr), build_targets(d, va)
    t_w, Str = sensor_window(d, tr)

    runs = []
    models = {}
    for lam in (0.0, *LAMBDAS):
        cfg = PINNConfig(lambda_phys=lam, steps=args.steps, seed=args.seed)
        t0 = time.time()
        model, log = train_pinn(d, tr, Xtr, Ytr, t_w, Str, cfg)
        el = time.time() - t0
        vm = val_metrics(predict(model, Xva), Yva, L)
        name = "blackbox" if lam == 0.0 else f"pinn_lambda_{lam:g}"
        runs.append(
            {"name": name, "lambda_phys": lam, "train_seconds": el, "final_losses": log[-1], **vm}
        )
        models[name] = model
        print(
            f"{name:18s} {el:6.1f} s  val acc {vm['accuracy']:.3f}  loc MAE {vm['loc_mae_m']:7.1f} m  rate MAPE {vm['rate_mape_pct']:6.1f} %  score {vm['score']:.3f}"
        )

    pinn_runs = [r for r in runs if r["lambda_phys"] > 0]
    best = min(pinn_runs, key=lambda r: r["score"])
    print(f"selected lambda_phys = {best['lambda_phys']} (validation score {best['score']:.3f})")

    t0 = time.time()
    rt = run_rttm(d, va)
    thr = choose_threshold(rt.dq_rel, d["is_leak"][va])
    rttm_acc = float(((rt.dq_rel > thr).astype(int) == d["is_leak"][va].astype(int)).mean())
    print(
        f"RTTM threshold on dq_rel = {thr:.5f} (validation accuracy {rttm_acc:.3f}, {time.time() - t0:.1f} s)"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    save_model(models[best["name"]], args.out / "leak_models" / "pinn.weights.h5")
    save_model(models["blackbox"], args.out / "leak_models" / "blackbox.weights.h5")
    np.savez(args.out / "leak_models" / "normaliser.npz", mean=norm.mean, std=norm.std)
    sel = {
        "commit": git_commit(),
        "seed": args.seed,
        "steps": args.steps,
        "config": {
            k: v
            for k, v in PINNConfig(steps=args.steps, seed=args.seed).__dict__.items()
            if k != "pipe"
        },
        "runs": runs,
        "selected_lambda_phys": best["lambda_phys"],
        "rttm_threshold_dq_rel": thr,
        "rttm_val_accuracy": rttm_acc,
    }
    (args.out / "leak_selection.json").write_text(json.dumps(sel, indent=2))
    print(f"wrote {args.out / 'leak_selection.json'}")


if __name__ == "__main__":
    main()
