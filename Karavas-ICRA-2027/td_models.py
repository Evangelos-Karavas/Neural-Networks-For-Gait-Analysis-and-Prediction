#!/usr/bin/env python3
"""The four 18-channel models compared in this study.

Two backbones (LSTM, CNN) crossed with two temporal representations
(timestamp, phase variable). All four take a 51-sample window and emit the
18 joint angles at the next sample, so the prediction horizon is identical
across models and the comparison is between backbone and conditioning only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, Flatten, LSTM, MaxPooling1D, Reshape,
)
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from td_pipeline import HORIZON, N_ANGLES, WINDOW


def build_lstm(input_dim: int, window: int = WINDOW, output_dim: int = N_ANGLES,
               lr: float = 1e-3, dropout: float = 0.2,
               horizon: int = HORIZON) -> Sequential:
    model = Sequential([
        LSTM(192, activation="tanh", return_sequences=True, input_shape=(window, input_dim)),
        Dropout(dropout),
        LSTM(192, activation="tanh", return_sequences=False),
        Dropout(dropout),
        Dense(256, activation="relu"),
        Dense(horizon * output_dim, activation="linear"),
        Reshape((horizon, output_dim)),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse",
        metrics=["mae", RootMeanSquaredError(name="rmse")],
    )
    return model


def build_cnn(input_dim: int, window: int = WINDOW, output_dim: int = N_ANGLES,
              lr: float = 1e-4, dropout: float = 0.2,
              horizon: int = HORIZON) -> Sequential:
    model = Sequential([
        Conv1D(32, 3, strides=2, padding="same", activation="relu",
               input_shape=(window, input_dim)),
        Conv1D(48, 3, strides=2, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(dropout),

        Conv1D(256, 3, strides=2, padding="same", activation="relu"),
        Conv1D(256, 3, strides=2, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(dropout),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(dropout),
        Dense(horizon * output_dim, activation="linear"),
        Reshape((horizon, output_dim)),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse",
        metrics=["mae", RootMeanSquaredError(name="rmse")],
    )
    return model


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str                 # "timestamp" (18 inputs) or "pv" (20 inputs)
    builder: Callable[..., Sequential]
    epochs: int
    lr: float

    @property
    def input_dim(self) -> int:
        return N_ANGLES + 2 if self.kind == "pv" else N_ANGLES

    def build(self, horizon: int = HORIZON) -> Sequential:
        return self.builder(input_dim=self.input_dim, lr=self.lr, horizon=horizon)


MODEL_SPECS: dict[str, ModelSpec] = {
    "timestamp_lstm": ModelSpec("timestamp_lstm", "Timestamp LSTM", "timestamp", build_lstm, 150, 1e-3),
    "pv_lstm":        ModelSpec("pv_lstm",        "PV LSTM",        "pv",        build_lstm, 150, 1e-3),
    "timestamp_cnn":  ModelSpec("timestamp_cnn",  "Timestamp CNN",  "timestamp", build_cnn,  200, 1e-4),
    "pv_cnn":         ModelSpec("pv_cnn",         "PV CNN",         "pv",        build_cnn,  200, 1e-4),
}

MODEL_ORDER = ["timestamp_lstm", "pv_lstm", "timestamp_cnn", "pv_cnn"]


def training_callbacks(patience: int = 25):
    """Early stopping matters here: a fold trains on 8 subjects, so the models
    reach their best validation loss well before the epoch budget runs out."""
    return [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0),
    ]
