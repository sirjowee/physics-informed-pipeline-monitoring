"""Tests for the S11 1D-CNN model and training loop (small and fast)."""

import numpy as np

from pipeline_monitoring.shm_classification.data import generate_dataset, zscore_per_trace
from pipeline_monitoring.shm_classification.evaluate import evaluate_at_noise
from pipeline_monitoring.shm_classification.model import build_cnn1d
from pipeline_monitoring.shm_classification.train import (
    set_determinism,
    to_model_input,
    train_cnn1d,
)


def _small_data(n_per_class=12, seed=0):
    X, y, _ = generate_dataset(n_per_class=n_per_class, seed=seed)
    return zscore_per_trace(X), y


def test_model_architecture_matches_paper():
    m = build_cnn1d()
    names = [layer.__class__.__name__ for layer in m.layers]
    assert names == [
        "Conv1D",
        "MaxPooling1D",
        "Conv1D",
        "MaxPooling1D",
        "Flatten",
        "Dense",
        "Dropout",
        "Dense",
    ]
    assert m.layers[0].filters == 32 and m.layers[0].kernel_size == (5,)
    assert m.layers[2].filters == 64 and m.layers[2].kernel_size == (3,)
    assert m.layers[5].units == 64 and m.layers[6].rate == 0.5
    assert m.output_shape == (None, 4)


def test_model_is_deterministic_given_seed():
    X, _ = _small_data()
    set_determinism(3)
    p1 = build_cnn1d(seed=3).predict(to_model_input(X), verbose=0)
    set_determinism(3)
    p2 = build_cnn1d(seed=3).predict(to_model_input(X), verbose=0)
    assert np.allclose(p1, p2)


def test_training_reduces_loss():
    X, y = _small_data()
    res = train_cnn1d(X, y, seed=0, epochs=5, batch_size=16, patience=5)
    assert res.history["loss"][-1] < res.history["loss"][0]
    assert res.epochs_run <= 5


def test_evaluate_returns_consistent_metrics():
    X, y = _small_data()
    res = train_cnn1d(X, y, seed=0, epochs=3, patience=3)
    r = evaluate_at_noise(res.model, X, y, sigma=0.0, noise_seed=0)
    assert 0.0 <= r.accuracy <= 1.0
    assert sum(m.support for m in r.per_class.values()) == len(y)
    assert np.asarray(r.confusion).sum() == len(y)
