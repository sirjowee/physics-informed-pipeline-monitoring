"""Evaluate the selected PINN, the black-box network and the RTTM baseline on the test set.

Usage:
    python scripts/leak_evaluate.py

Reads ``results/leak_selection.json`` and the weights written by ``scripts/leak_train.py``,
evaluates once on the held-out test scenarios (plus per-regime breakdown and a noise
sweep), writes ``results/leak_metrics.json`` and the figures under
``figures/leak_detection/``, and prints the RESULTS.md tables.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from pipeline_monitoring.leak_detection.data import PipelineConfig, load_dataset
from pipeline_monitoring.leak_detection.evaluate import (
    compute_metrics,
    network_predictions,
    per_regime,
    results_markdown,
    rttm_predictions,
    save_figures,
    with_extra_noise,
)
from pipeline_monitoring.leak_detection.features import RATE_SCALE, Normaliser, build_targets
from pipeline_monitoring.leak_detection.pinn import PINNConfig, load_model

REPO = Path(__file__).resolve().parents[1]
NOISE_FACTORS = (1.0, 2.0, 4.0, 8.0)


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
    ap.add_argument("--data", type=Path, default=REPO / "data" / "leak" / "leak_dataset.npz")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--figures", type=Path, default=REPO / "figures" / "leak_detection")
    args = ap.parse_args()

    sel = json.loads((args.results / "leak_selection.json").read_text())
    d = load_dataset(args.data)
    pipe = PipelineConfig()
    te = d["test_idx"]
    nz = np.load(args.results / "leak_models" / "normaliser.npz")
    norm = Normaliser(nz["mean"], nz["std"])
    n_features = 60 * 5 + 4
    cfg_kw = {k: v for k, v in sel["config"].items() if k != "lambda_phys"}
    pinn = load_model(
        args.results / "leak_models" / "pinn.weights.h5",
        n_features,
        PINNConfig(lambda_phys=sel["selected_lambda_phys"], **cfg_kw),
    )
    bbox = load_model(
        args.results / "leak_models" / "blackbox.weights.h5",
        n_features,
        PINNConfig(lambda_phys=0.0, **cfg_kw),
    )

    y = build_targets(d, te)
    L = float(d["length_m"])
    preds = {
        "PINN": network_predictions(pinn, d, te, norm),
        "Black-box NN": network_predictions(bbox, d, te, norm),
        "RTTM": rttm_predictions(d, te, sel["rttm_threshold_dq_rel"], pipe),
    }
    metrics = {
        k: compute_metrics(p, y["is_leak"], y["loc"] * L, y["rate"] * RATE_SCALE)
        for k, p in preds.items()
    }
    print(results_markdown(metrics))
    regimes = {k: per_regime(p, d, te) for k, p in preds.items()}
    print("\nPer-regime accuracy / loc MAE (m) / rate MAPE (%):")
    for k, r in regimes.items():
        print(
            "  "
            + k
            + ": "
            + "; ".join(
                f"{reg} {v['accuracy']:.3f} / {v['loc_mae_m']:.0f} / {v['rate_mape_pct']:.0f}"
                for reg, v in r.items()
            )
        )

    robustness: dict[str, list[dict]] = {k: [] for k in preds}
    for i, f in enumerate(NOISE_FACTORS):
        dn = with_extra_noise(d, te, f, pipe, seed=500 + i)
        pn = {
            "PINN": network_predictions(pinn, dn, te, norm),
            "Black-box NN": network_predictions(bbox, dn, te, norm),
            "RTTM": rttm_predictions(dn, te, sel["rttm_threshold_dq_rel"], pipe),
        }
        for k, p in pn.items():
            mm = compute_metrics(p, y["is_leak"], y["loc"] * L, y["rate"] * RATE_SCALE)
            robustness[k].append(
                {
                    "noise_factor": f,
                    "accuracy": mm.accuracy,
                    "loc_mae_m": mm.loc_mae_m,
                    "rate_mape_pct": mm.rate_mape_pct,
                }
            )
    print(f"\nNoise robustness (accuracy at noise multipliers {NOISE_FACTORS}):")
    for k, c in robustness.items():
        print("  " + k + ": " + ", ".join(f"{x['accuracy']:.3f}" for x in c))

    out = {
        "commit": git_commit(),
        "selection_commit": sel["commit"],
        "selected_lambda_phys": sel["selected_lambda_phys"],
        "rttm_threshold_dq_rel": sel["rttm_threshold_dq_rel"],
        "n_test": int(len(te)),
        "metrics": {k: m.to_dict() for k, m in metrics.items()},
        "per_regime": regimes,
        "noise_robustness": robustness,
    }
    (args.results / "leak_metrics.json").write_text(json.dumps(out, indent=2))
    save_figures(preds, d, te, robustness, args.figures)
    print(f"\nwrote {args.results / 'leak_metrics.json'} and figures in {args.figures}")


if __name__ == "__main__":
    main()
