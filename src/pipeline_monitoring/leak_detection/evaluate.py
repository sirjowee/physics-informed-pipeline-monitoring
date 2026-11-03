"""Test-set evaluation of the PINN, the black-box network and the RTTM baseline.

Metrics follow the paper (section III-E): detection accuracy, precision, recall and
ROC-AUC; localisation mean absolute error in metres; leak-rate mean absolute percentage
error. Localisation and rate errors are computed over the true leak scenarios (the paper
does not say whether they are restricted to detected leaks; using all true leaks is the
stricter choice). Per-regime breakdowns and a noise-robustness sweep are added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from .data.config import REGIME_NAMES, PipelineConfig
from .features import RATE_SCALE, Normaliser, build_features, build_targets
from .pinn import LeakPINN, predict
from .rttm import run_rttm


@dataclass
class Predictions:
    """Per-scenario outputs in common units: probability/score, x_L (m), rate fraction."""

    score: np.ndarray  # higher = more leak-like (probability for networks, dq_rel for RTTM)
    is_leak_pred: np.ndarray  # bool
    loc_m: np.ndarray
    rate_frac: np.ndarray


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    loc_mae_m: float
    rate_mape_pct: float
    n: int
    n_leak: int
    confusion: list[list[int]] = field(default_factory=list)  # rows true (no-leak, leak)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    p: Predictions, is_leak: np.ndarray, loc_m: np.ndarray, rate_frac: np.ndarray
) -> Metrics:
    y = is_leak.astype(int)
    yp = p.is_leak_pred.astype(int)
    m = y == 1
    auc = float(roc_auc_score(y, p.score)) if 0 < y.sum() < len(y) else float("nan")
    cm = [[int(((y == a) & (yp == b)).sum()) for b in (0, 1)] for a in (0, 1)]
    return Metrics(
        accuracy=float(accuracy_score(y, yp)),
        precision=float(precision_score(y, yp, zero_division=0)),
        recall=float(recall_score(y, yp, zero_division=0)),
        roc_auc=auc,
        loc_mae_m=float(np.abs(p.loc_m[m] - loc_m[m]).mean()) if m.any() else float("nan"),
        rate_mape_pct=float((np.abs(p.rate_frac[m] - rate_frac[m]) / rate_frac[m]).mean() * 100)
        if m.any()
        else float("nan"),
        n=int(len(y)),
        n_leak=int(m.sum()),
        confusion=cm,
    )


def network_predictions(model: LeakPINN, d: dict, idx: np.ndarray, norm: Normaliser) -> Predictions:
    out = predict(model, build_features(d, idx, norm))
    L = float(d["length_m"])
    return Predictions(
        score=out["prob"],
        is_leak_pred=out["prob"] > 0.5,
        loc_m=out["loc"] * L,
        rate_frac=out["rate"] * RATE_SCALE,
    )


def rttm_predictions(
    d: dict, idx: np.ndarray, threshold: float, pipe: PipelineConfig
) -> Predictions:
    r = run_rttm(d, idx, pipe)
    from .data.config import REGIMES

    ag = np.array([REGIMES[s].inlet_gas_fraction for s in d["regime"][idx]])
    u = d["nominal_velocity"][idx]
    mdot = pipe.area_m2 * u * (ag * pipe.rho_gas_ref + (1 - ag) * pipe.rho_liquid)
    return Predictions(
        score=r.dq_rel,
        is_leak_pred=r.dq_rel > threshold,
        loc_m=r.loc_m,
        rate_frac=r.rate_kg_s / mdot,
    )


def per_regime(p: Predictions, d: dict, idx: np.ndarray) -> dict[str, dict]:
    y = build_targets(d, idx)
    L = float(d["length_m"])
    out = {}
    for name in REGIME_NAMES:
        m = d["regime"][idx] == name
        sub = Predictions(p.score[m], p.is_leak_pred[m], p.loc_m[m], p.rate_frac[m])
        out[name] = compute_metrics(
            sub, y["is_leak"][m], y["loc"][m] * L, y["rate"][m] * RATE_SCALE
        ).to_dict()
    return out


def with_extra_noise(
    d: dict, idx: np.ndarray, factor: float, pipe: PipelineConfig, seed: int
) -> dict:
    """Copy of ``d`` whose sensors for ``idx`` carry additional Gaussian noise.

    ``factor`` multiplies the generator's sigma (0.1 % of 60 bar for p, 0.5 % of nominal Q);
    the added noise has std sqrt(factor^2 - 1) * sigma so the total std is factor * sigma.
    """
    if factor <= 1.0:
        return d
    rng = np.random.default_rng(seed)
    dd = dict(d)
    s = d["sensors"].copy()
    extra = np.sqrt(factor**2 - 1.0)
    sig_p = 0.001 * pipe.p_ref * extra
    sig_q = (0.005 * pipe.area_m2 * d["nominal_velocity"][idx] * extra)[:, None]
    n = rng.standard_normal(s[idx].shape)
    n[..., 0:2] *= sig_p
    n[..., 2:4] *= sig_q[..., None]
    s[idx] = s[idx] + n.astype(np.float32)
    dd["sensors"] = s
    return dd


def results_markdown(all_metrics: dict[str, Metrics]) -> str:
    lines = [
        "| Model | Accuracy | Precision | Recall | ROC-AUC | Loc. MAE (m) | Rate MAPE (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in all_metrics.items():
        lines.append(
            f"| {name} | {m.accuracy:.3f} | {m.precision:.3f} | {m.recall:.3f} | {m.roc_auc:.3f} | {m.loc_mae_m:.1f} | {m.rate_mape_pct:.1f} |"
        )
    return "\n".join(lines)


def save_figures(
    preds: dict[str, Predictions], d: dict, idx: np.ndarray, robustness: dict, out_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    y = build_targets(d, idx)
    L = float(d["length_m"])
    m = y["is_leak"] > 0.5
    names = list(preds)

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        cm = compute_metrics(
            preds[name], y["is_leak"], y["loc"] * L, y["rate"] * RATE_SCALE
        ).confusion
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center")
        ax.set_xticks([0, 1], ["no leak", "leak"])
        ax.set_yticks([0, 1], ["no leak", "leak"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name)
    fig.suptitle("Detection confusion matrices (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "final_confusion_matrices.png", dpi=130)
    plt.close(fig)

    for key, true_v, label, fname in (
        ("loc_m", y["loc"] * L, "leak location (m)", "final_location_scatter.png"),
        (
            "rate_frac",
            y["rate"] * RATE_SCALE * 100,
            "leak rate (% of nominal flow)",
            "final_rate_scatter.png",
        ),
    ):
        fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
        for ax, name in zip(np.atleast_1d(axes), names, strict=True):
            pv = getattr(preds[name], key)[m]
            pv = pv * 100 if key == "rate_frac" else pv
            ax.scatter(true_v[m], pv, s=14, alpha=0.7)
            lim = [0, max(true_v[m].max(), pv.max()) * 1.05]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xlabel(f"true {label}")
            ax.set_ylabel(f"predicted {label}")
            ax.set_title(name)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Predicted versus true {label}, true leak scenarios in the test set")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in robustness.items():
        ax.plot(
            [c["noise_factor"] for c in curve], [c["accuracy"] for c in curve], "o-", label=name
        )
    ax.set_xlabel("sensor noise multiplier (1 = generator sigma)")
    ax.set_ylabel("detection accuracy (test)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Robustness to additional sensor noise")
    fig.tight_layout()
    fig.savefig(out_dir / "final_noise_robustness.png", dpi=130)
    plt.close(fig)
