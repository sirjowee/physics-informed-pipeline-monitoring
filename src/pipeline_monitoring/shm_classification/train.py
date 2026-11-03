"""Training loop for the S11 1D-CNN.

Paper (section III-E): categorical cross-entropy, Adam lr = 0.001, batch 16, 50 epochs,
80/20 stratified split, early stopping (section IV-C mentions it; no criterion given).

Chosen here (RESULTS.md 2.4): a validation set of 20 % of the training portion (stratified,
seeded) for early stopping on validation loss, patience 10, best weights restored. The test
set is never used for stopping or model selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .data import CLASS_NAMES, stratified_split
from .model import build_cnn1d

EPOCHS = 50
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 10
VALIDATION_FRACTION = 0.2


def set_determinism(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow and enable deterministic ops."""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def to_model_input(X: np.ndarray) -> np.ndarray:
    """(n, 91) float array -> (n, 91, 1) float32 tensor input."""
    return X.astype(np.float32)[..., None]


@dataclass
class TrainResult:
    model: tf.keras.Model
    history: dict[str, list[float]]
    train_idx: np.ndarray
    val_idx: np.ndarray
    epochs_run: int


def train_cnn1d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 0,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOPPING_PATIENCE,
    verbose: int = 0,
) -> TrainResult:
    """Train on ``X_train`` (normalised traces) with an internal validation split."""
    set_determinism(seed)
    fit_idx, val_idx = stratified_split(y_train, test_fraction=VALIDATION_FRACTION, seed=seed)
    n_classes = len(CLASS_NAMES)
    Y = tf.keras.utils.to_categorical(y_train, n_classes)
    model = build_cnn1d(seed=seed)
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    hist = model.fit(
        to_model_input(X_train[fit_idx]),
        Y[fit_idx],
        validation_data=(to_model_input(X_train[val_idx]), Y[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[stop],
        verbose=verbose,
    )
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return TrainResult(
        model=model,
        history=history,
        train_idx=fit_idx,
        val_idx=val_idx,
        epochs_run=len(history["loss"]),
    )
