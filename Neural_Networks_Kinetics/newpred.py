"""
TD-trained LSTM that predicts the NEXT stride angles (51x18) from the CURRENT stride (51x36),
then applies it to CP strides to generate "corrected / adaptive reference trajectories".

Matches the idea: train on TD only, feed CP, use predictions as reference.  :contentReference[oaicite:3]{index=3}

Assumptions:
- TD merged file contains time-contiguous strides (51 rows per stride, repeated).
- CP folder contains one stride per .xlsx (51 rows).
- Columns match your FORCE_COLS + ANGLE_COLS for inputs (36), and ANGLE_COLS for outputs (18).
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler  # you can swap to MinMaxScaler if you prefer
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
STRIDE = 51

TD_MERGED_FILE = "Data_Normal/dynamics_total_augmented.xlsx"  # your merged TD file (augmented ok)
CP_FOLDER      = "Data_CP"                                   # each file = one stride

OUT_DIR  = "Predictions"
PLOT_DIR = os.path.join(OUT_DIR, "Plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_SAVE    = os.path.join(OUT_DIR, "td_nextstride_model.keras")
SCALER_X_SAVE = os.path.join(OUT_DIR, "scaler_x36.save")
SCALER_Y_SAVE = os.path.join(OUT_DIR, "scaler_y18.save")

# =========================================================
# COLUMNS (same as yours)
# =========================================================
FORCE_COLS = [
    'LHipForce (1)','RHipForce (1)','LHipForce (2)','RHipForce (2)','LHipForce (3)','RHipForce (3)',
    'LKneeForce (1)','RKneeForce (1)','LKneeForce (2)','RKneeForce (2)','LKneeForce (3)','RKneeForce (3)',
    'LAnkleForce (1)','RAnkleForce (1)','LAnkleForce (2)','RAnkleForce (2)','LAnkleForce (3)','RAnkleForce (3)'
]
ANGLE_COLS = [
    'LHipAngles (1)','RHipAngles (1)','LHipAngles (2)','RHipAngles (2)','LHipAngles (3)','RHipAngles (3)',
    'LKneeAngles (1)','RKneeAngles (1)','LKneeAngles (2)','RKneeAngles (2)','LKneeAngles (3)','RKneeAngles (3)',
    'LAnkleAngles (1)','RAnkleAngles (1)','LAnkleAngles (2)','RAnkleAngles (2)','LAnkleAngles (3)','RAnkleAngles (3)'
]

INPUT_COLS  = FORCE_COLS + ANGLE_COLS  # 36
OUTPUT_COLS = ANGLE_COLS               # 18

# sagittal indices (for plotting)
SAG_NAMES = [
    "LHipAngles (1)", "RHipAngles (1)",
    "LKneeAngles (1)", "RKneeAngles (1)",
    "LAnkleAngles (1)", "RAnkleAngles (1)"
]
SAG_IDX = [ANGLE_COLS.index(n) for n in SAG_NAMES]

# =========================================================
# HELPERS
# =========================================================
def load_td_merged(path: str) -> np.ndarray:
    """Load TD merged file, return float32 array (T, 36)."""
    df = pd.read_excel(path, sheet_name=0, usecols=INPUT_COLS)
    df.columns = df.columns.str.strip()
    df = df.dropna().reset_index(drop=True)
    return df.to_numpy(np.float32)

def segment_strides(A_2d: np.ndarray, stride_len: int) -> np.ndarray:
    """(T,F) -> (N,stride_len,F), truncating to full strides."""
    n = len(A_2d) // stride_len
    A_2d = A_2d[: n * stride_len]
    return A_2d.reshape(n, stride_len, A_2d.shape[-1])

def next_stride_pairs(strides_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given contiguous TD strides: (N,51,36)
    X_k = stride k (all 36)
    Y_k = angles of stride k+1 (18)
    """
    if strides_3d.shape[0] < 2:
        return np.empty((0, STRIDE, 36), np.float32), np.empty((0, STRIDE, 18), np.float32)
    X = strides_3d[:-1]                 # (N-1,51,36)
    Y = strides_3d[1:, :, -18:]         # (N-1,51,18)  angles only
    return X, Y

def scale_3d(scaler, A):
    shp = A.shape
    return scaler.transform(A.reshape(-1, shp[-1])).reshape(shp)

def inv_scale_3d(scaler, A):
    shp = A.shape
    return scaler.inverse_transform(A.reshape(-1, shp[-1])).reshape(shp)

def load_cp_stride_file(fp: str) -> np.ndarray | None:
    """
    Reads a CP stride file expected to contain 51 rows in sheet 'Data'
    with your INPUT_COLS. Returns (51,36) or None.
    """
    try:
        df = pd.read_excel(fp, sheet_name="Data", header=0, skiprows=[1,2], usecols=INPUT_COLS)
        df.columns = df.columns.str.strip()
        df = df.dropna(how="all").reset_index(drop=True)
        if len(df) < STRIDE:
            return None
        df = df.iloc[:STRIDE].copy()
        if len(df) != STRIDE:
            return None
        return df.to_numpy(np.float32)
    except Exception as e:
        print(f"[WARN] Skipping {os.path.basename(fp)}: {e}")
        return None

# =========================================================
# LOAD + BUILD TD TRAINING SET
# =========================================================
td_2d = load_td_merged(TD_MERGED_FILE)
td_strides = segment_strides(td_2d, STRIDE)    # (Ntd,51,36)
print("TD strides:", td_strides.shape)

X_td, Y_td = next_stride_pairs(td_strides)     # (Ntd-1,51,36), (Ntd-1,51,18)
print("TD next-stride pairs:", X_td.shape, Y_td.shape)

if len(X_td) < 10:
    raise RuntimeError("Not enough TD stride pairs to train. Check your merged file / STRIDE value.")

# =========================================================
# SPLIT (stride-level shuffle split)
# NOTE: subject-level split is better if you have subject IDs.  :contentReference[oaicite:4]{index=4}
# =========================================================
idx = np.arange(len(X_td))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, Y_tr = X_td[tr_idx], Y_td[tr_idx]
X_va, Y_va = X_td[va_idx], Y_td[va_idx]

# =========================================================
# SCALE (fit on TD train only)
# =========================================================
sc_x = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))  # 36
sc_y = StandardScaler().fit(Y_tr.reshape(-1, Y_tr.shape[-1]))  # 18
joblib.dump(sc_x, SCALER_X_SAVE)
joblib.dump(sc_y, SCALER_Y_SAVE)

X_tr_s = scale_3d(sc_x, X_tr)
X_va_s = scale_3d(sc_x, X_va)
Y_tr_s = scale_3d(sc_y, Y_tr)
Y_va_s = scale_3d(sc_y, Y_va)

# =========================================================
# MODEL (2-layer LSTM similar spirit to thesis LSTM) :contentReference[oaicite:5]{index=5}
# Sequence-to-sequence output: (51,18)
# =========================================================
model = Sequential([
    Input(shape=(STRIDE, 36)),
    LSTM(128, return_sequences=True),
    LayerNormalization(),
    Dropout(0.2),

    LSTM(128, return_sequences=True),
    LayerNormalization(),
    Dropout(0.2),

    Dense(64, activation="tanh"),
    Dropout(0.2),
    Dense(18)  # linear
])

model.compile(
    optimizer=Adam(1e-3),
    loss="mse",
    metrics=["mae", tf.keras.metrics.RootMeanSquaredError()]
)
print(model.summary())

# =========================================================
# TRAIN
# =========================================================
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
]

history = model.fit(
    X_tr_s, Y_tr_s,
    validation_data=(X_va_s, Y_va_s),
    epochs=300,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save(MODEL_SAVE, include_optimizer=True)
print("Saved model:", MODEL_SAVE)




import matplotlib.pyplot as plt
import numpy as np
import os

def plot_all_cp_predictions_overlay(
    cp_folder,
    model,
    sc_x,
    sc_y,
    stride_len=51,
    sag_idx=None,
    sag_names=None,
    show_cp_current=True,
    max_files=None,
    save_path=None
):
    """
    Overlays all CP predicted next-stride sagittal angles on the same 6-subplot figure.
    Optionally overlays CP current sagittal angles too (fainter).
    """
    if sag_idx is None or sag_names is None:
        raise ValueError("Provide sag_idx and sag_names.")

    cp_files = [f for f in sorted(os.listdir(cp_folder)) if f.endswith(".xlsx")]
    if not cp_files:
        raise RuntimeError(f"No CP .xlsx files found in {cp_folder}")

    if max_files is not None:
        cp_files = cp_files[:max_files]

    t = np.arange(stride_len)

    plt.figure(figsize=(14, 10))

    used = 0
    skipped = 0

    for f in cp_files:
        fp = os.path.join(cp_folder, f)
        cp_stride = load_cp_stride_file(fp)  # (51,36) or None
        if cp_stride is None:
            skipped += 1
            continue

        # predict next stride angles from CP input
        cp_in_s = scale_3d(sc_x, cp_stride[None, ...])          # (1,51,36)
        pred_next_s = model.predict(cp_in_s, verbose=0)[0]      # (51,18) scaled
        pred_next = sc_y.inverse_transform(pred_next_s)         # (51,18) degrees
        pred_sag6 = pred_next[:, sag_idx]                       # (51,6)

        # optionally also show CP current (from input stride)
        if show_cp_current:
            cp_now = cp_stride[:, -18:]                         # (51,18)
            cp_now_sag6 = cp_now[:, sag_idx]                    # (51,6)

        # plot all 6 sagittal traces
        for i, name in enumerate(sag_names):
            plt.subplot(3, 2, i + 1)

            if show_cp_current:
                # CP current in thin line (no legend entry spam)
                plt.plot(t, cp_now_sag6[:, i], linewidth=1, alpha=0.15)

            # predicted next stride overlay
            plt.plot(t, pred_sag6[:, i], linewidth=1.5, alpha=0.35)

            plt.title(name)
            plt.grid(True)

        used += 1

    # Add a single legend handle (avoid 200 legend entries)
    # We'll add dummy lines for legend meaning.
    plt.subplot(3, 2, 1)
    if show_cp_current:
        plt.plot([], [], linewidth=1, alpha=0.15, label="CP current (faint)")
    plt.plot([], [], linewidth=1.5, alpha=0.35, label="Predicted next stride (overlay)")
    plt.legend(loc="best")

    plt.suptitle(f"All CP predicted next-stride sagittal angles overlaid (n={used}, skipped={skipped})",
                 y=1.02, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")

    plt.show()
    plt.close()


# ---- call it ----
plot_all_cp_predictions_overlay(
    cp_folder=CP_FOLDER,
    model=model,
    sc_x=sc_x,
    sc_y=sc_y,
    stride_len=STRIDE,
    sag_idx=SAG_IDX,
    sag_names=SAG_NAMES,
    show_cp_current=True,              # set True if you also want CP current faintly
    max_files=None,                     # or set e.g. 50 to limit
    save_path=os.path.join(PLOT_DIR, "all_cp_pred_nextstride_sagittal_overlay.png")
)
