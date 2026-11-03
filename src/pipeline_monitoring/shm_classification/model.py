"""1D-CNN for |S11| trace classification, exactly as the paper lists it.

Paper (section III-D): Input (91) -> Conv1D(32, k=5, ReLU) -> MaxPooling1D -> Conv1D(64, k=3,
ReLU) -> MaxPooling1D -> Flatten -> Dense(64) -> Dropout(0.5) -> Softmax(4).

Not specified by the paper and chosen here (RESULTS.md 2.4): pooling size 2 (Keras default),
"valid" padding (Keras default), ReLU on the Dense(64) layer, Glorot-uniform initialisation
with a fixed seed.
"""

from __future__ import annotations

import tensorflow as tf

from .data import CLASS_NAMES, N_POINTS


def build_cnn1d(n_points: int = N_POINTS, n_classes: int = len(CLASS_NAMES), seed: int = 0):
    """Return the compiled Keras model (Adam, lr 0.001, categorical cross-entropy)."""
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_points, 1)),
            tf.keras.layers.Conv1D(32, 5, activation="relu", kernel_initializer=init),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, 3, activation="relu", kernel_initializer=init),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init),
            tf.keras.layers.Dropout(0.5, seed=seed),
            tf.keras.layers.Dense(n_classes, activation="softmax", kernel_initializer=init),
        ],
        name="s11_cnn1d",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
