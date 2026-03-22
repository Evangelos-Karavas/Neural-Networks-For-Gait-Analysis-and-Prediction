#!/usr/bin/env python3
"""
CNN modality ablation for healthy gait prediction

Trains three next-step CNN models on the SAME healthy dataset:
  1) CNN + kinematics only
  2) CNN + kinetics only
  3) CNN + both

Default task:
  predict next-timestep sagittal-plane joint angles
  [LHip, RHip, LKnee, RKnee, LAnkle, RAnkle]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = Path("Ablation_Study/dynamics_total_augmented.xlsx")
SHEET_NAME = 0

OUTPUT_DIR = Path("/Ablation_Study/data/cnn_modality_outputs")
MODELS_DIR = OUTPUT_DIR / "models"
SCALERS_DIR = OUTPUT_DIR / "scalers"
PLOTS_DIR = OUTPUT_DIR / "plots"

WINDOW = 51
TEST_RATIO = 0.15
VAL_RATIO = 0.15
EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42

# Predict sagittal-plane angles at the next timestep
TARGET_COLS = [
    "LHipAngles (1)",
    "RHipAngles (1)",
    "LKneeAngles (1)",
    "RKneeAngles (1)",
    "LAnkleAngles (1)",
    "RAnkleAngles (1)",
]

# Use all forces + moments as kinetics
KINETIC_KEYWORDS = ("Moment", "Force")
# Use the same 6 sagittal-plane angles as kinematic inputs
KINEMATIC_COLS = TARGET_COLS.copy()

JOINT_LABELS = [
    "L Hip", "R Hip", "L Knee", "R Knee", "L Ankle", "R Ankle"
]


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def load_dataset(path: Path, sheet_name=0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna().reset_index(drop=True)
    return df


def infer_kinetic_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if any(k in c for k in KINETIC_KEYWORDS)]
    if not cols:
        raise ValueError("No kinetic columns found. Check column names in the Excel file.")
    return cols



def build_windows(
    data_x: np.ndarray,
    data_y: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds rolling windows for next-step prediction.

    X[i] = data_x[i : i+window]
    y[i] = data_y[i+window]
    """
    n = len(data_x)
    if n <= window:
        raise ValueError(f"Not enough rows ({n}) for window size {window}.")

    X, y = [], []
    for i in range(n - window):
        X.append(data_x[i:i + window])
        y.append(data_y[i + window])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)



def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, np.ndarray]:
    n = len(X)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split sizes are invalid. Reduce val/test ratios or use more data.")

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val": X[n_train:n_train + n_val],
        "y_val": y[n_train:n_train + n_val],
        "X_test": X[n_train + n_val:],
        "y_test": y[n_train + n_val:],
    }



def fit_feature_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    return scaler



def transform_window_data(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = X.shape
    flat = X.reshape(-1, shape[-1])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(shape).astype(np.float32)



def fit_target_scaler(y_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(y_train)
    return scaler


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def build_cnn_model(window: int, n_features: int, n_outputs: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_outputs),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


# -----------------------------------------------------------------------------
# Evaluation / plots
# -----------------------------------------------------------------------------
def per_joint_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for i, joint in enumerate(JOINT_LABELS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        metrics[joint] = {"MAE": float(mae), "RMSE": float(rmse)}
    return metrics



def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }



def save_history_plot(history: tf.keras.callbacks.History, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def save_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str,
    out_path: Path,
    n_points: int = 200,
) -> None:
    n_points = min(n_points, len(y_true))
    t = np.arange(n_points)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(t, y_true[:n_points, i], label="ground truth")
        ax.plot(t, y_pred[:n_points, i], label="prediction")
        ax.set_title(JOINT_LABELS[i])
        ax.set_ylabel("Angle (deg)")
        ax.grid(True, alpha=0.3)

    axes[-2].set_xlabel("Test timestep")
    axes[-1].set_xlabel("Test timestep")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(title_prefix)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Training runner
# -----------------------------------------------------------------------------
def train_modality_model(
    df: pd.DataFrame,
    modality_name: str,
    feature_cols: List[str],
    target_cols: List[str],
) -> Dict:
    print(f"\n{'=' * 80}")
    print(f"Training: {modality_name}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Targets  ({len(target_cols)}): {target_cols}")
    print(f"{'=' * 80}")

    x_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df[target_cols].to_numpy(dtype=np.float32)

    X, y = build_windows(x_raw, y_raw, window=WINDOW)
    split = chronological_split(X, y, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)

    x_scaler = fit_feature_scaler(split["X_train"])
    y_scaler = fit_target_scaler(split["y_train"])

    X_train = transform_window_data(split["X_train"], x_scaler)
    X_val = transform_window_data(split["X_val"], x_scaler)
    X_test = transform_window_data(split["X_test"], x_scaler)

    y_train = y_scaler.transform(split["y_train"]).astype(np.float32)
    y_val = y_scaler.transform(split["y_val"]).astype(np.float32)

    model = build_cnn_model(WINDOW, len(feature_cols), len(target_cols))

    cb = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=1,
        callbacks=cb,
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = split["y_test"]

    metrics = {
        "modality": modality_name,
        "n_features": len(feature_cols),
        "window": WINDOW,
        "overall": overall_metrics(y_true, y_pred),
        "per_joint": per_joint_metrics(y_true, y_pred),
    }

    # Save artifacts
    modality_slug = modality_name.lower().replace(" + ", "_").replace(" ", "_")
    model_path = MODELS_DIR / f"cnn_{modality_slug}.keras"
    x_scaler_path = SCALERS_DIR / f"x_scaler_{modality_slug}.joblib"
    y_scaler_path = SCALERS_DIR / f"y_scaler_{modality_slug}.joblib"
    history_plot = PLOTS_DIR / f"history_{modality_slug}.png"
    pred_plot = PLOTS_DIR / f"prediction_{modality_slug}.png"
    metrics_path = OUTPUT_DIR / f"metrics_{modality_slug}.json"

    model.save(model_path)
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    save_history_plot(history, f"{modality_name} - training history", history_plot)
    save_prediction_plot(y_true, y_pred, f"{modality_name} - test predictions", pred_plot)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to:   {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Overall MAE:  {metrics['overall']['MAE']:.4f}")
    print(f"Overall RMSE: {metrics['overall']['RMSE']:.4f}")

    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH, sheet_name=SHEET_NAME)

    kinetic_cols = infer_kinetic_columns(df)
    kinematic_cols = [c for c in KINEMATIC_COLS if c in df.columns]
    if len(kinematic_cols) != len(KINEMATIC_COLS):
        missing = sorted(set(KINEMATIC_COLS) - set(kinematic_cols))
        raise ValueError(f"Missing expected kinematic columns: {missing}")

    experiment_definitions = {
        "CNN + kinematics only": kinematic_cols,
        "CNN + kinetics only": kinetic_cols,
        "CNN + both": kinetic_cols + kinematic_cols,
    }

    all_results = []
    for name, feature_cols in experiment_definitions.items():
        result = train_modality_model(df, name, feature_cols, TARGET_COLS)
        row = {
            "Model": name,
            "#Features": result["n_features"],
            "Window": result["window"],
            "Overall MAE": result["overall"]["MAE"],
            "Overall RMSE": result["overall"]["RMSE"],
        }
        for joint in JOINT_LABELS:
            row[f"{joint} MAE"] = result["per_joint"][joint]["MAE"]
            row[f"{joint} RMSE"] = result["per_joint"][joint]["RMSE"]
        all_results.append(row)

    summary_df = pd.DataFrame(all_results)
    summary_csv = OUTPUT_DIR / "cnn_modality_summary.csv"
    summary_xlsx = OUTPUT_DIR / "cnn_modality_summary.xlsx"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_excel(summary_xlsx, index=False)

    print("\nDone.")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary XLSX: {summary_xlsx}")
    print(summary_df)


if __name__ == "__main__":
    main()
