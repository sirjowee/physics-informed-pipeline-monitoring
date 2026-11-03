"""Evaluation of the S11 1D-CNN on clean and noisy test sets.

Metrics follow the paper (section III-E and Tables 1-2): accuracy, precision, recall,
F1-score (macro-averaged, as the paper's Table 2 row-level values are consistent with
macro averages), per-class metrics, confusion matrices, plus the categorical cross-entropy
loss the paper lists in Table 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .data import CLASS_NAMES, add_gaussian_noise
from .train import to_model_input

NOISE_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0)


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalResult:
    sigma: float
    accuracy: float
    loss: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def evaluate_at_noise(
    model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, sigma: float, noise_seed: int
) -> EvalResult:
    """Evaluate on ``X_test`` (normalised) after adding N(0, sigma^2) noise with a fixed seed."""
    Xn = add_gaussian_noise(X_test, sigma, seed=noise_seed)
    probs = model.predict(to_model_input(Xn), verbose=0)
    y_pred = probs.argmax(axis=1)
    Y = tf.keras.utils.to_categorical(y_test, len(CLASS_NAMES))
    loss = float(tf.keras.losses.categorical_crossentropy(Y, probs).numpy().mean())
    p, r, f, s = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASS_NAMES)))
    return EvalResult(
        sigma=sigma,
        accuracy=float(accuracy_score(y_test, y_pred)),
        loss=loss,
        precision_macro=float(pm),
        recall_macro=float(rm),
        f1_macro=float(fm),
        per_class={
            name: ClassMetrics(float(p[i]), float(r[i]), float(f[i]), int(s[i]))
            for i, name in enumerate(CLASS_NAMES)
        },
        confusion=cm.tolist(),
    )


def evaluate_all(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: tuple[float, ...] = NOISE_LEVELS,
    noise_seed: int = 1000,
) -> list[EvalResult]:
    """Evaluate at every noise level; noise seed is ``noise_seed + index`` per level."""
    return [
        evaluate_at_noise(model, X_test, y_test, s, noise_seed + i)
        for i, s in enumerate(noise_levels)
    ]


def results_markdown(results: list[EvalResult]) -> str:
    """Markdown tables in the layout of RESULTS.md 2.1 and 2.2."""
    lines = ["| Noise σ | Accuracy | Loss | Precision (macro) | Recall (macro) | F1 (macro) |"]
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.sigma:.1f} | {r.accuracy:.4f} | {r.loss:.4f} | {r.precision_macro:.4f} | "
            f"{r.recall_macro:.4f} | {r.f1_macro:.4f} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"Per-class at σ = {r.sigma:.1f}:")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for name, m in r.per_class.items():
            lines.append(
                f"| {name} | {m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} | {m.support} |"
            )
        lines.append(f"Confusion (rows = true, cols = predicted, order {list(CLASS_NAMES)}):")
        for row in r.confusion:
            lines.append("    " + " ".join(f"{v:3d}" for v in row))
        lines.append("")
    return "\n".join(lines)


def save_figures(results: list[EvalResult], history: dict[str, list[float]], out_dir: Path) -> None:
    """Confusion matrices (one panel per noise level) and training curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4))
    for ax, r in zip(np.atleast_1d(axes), results, strict=True):
        cm = np.asarray(r.confusion)
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_title(f"σ = {r.sigma:.1f}, acc {r.accuracy:.3f}")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out_dir / "final_confusion_matrices.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ep = range(1, len(history["loss"]) + 1)
    axes[0].plot(ep, history["accuracy"], label="train")
    axes[0].plot(ep, history["val_accuracy"], label="validation")
    axes[0].set(title="Model accuracy", xlabel="epoch", ylabel="accuracy")
    axes[1].plot(ep, history["loss"], label="train")
    axes[1].plot(ep, history["val_loss"], label="validation")
    axes[1].set(title="Model loss", xlabel="epoch", ylabel="categorical cross-entropy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "final_training_curves.png", dpi=130)
    plt.close(fig)
