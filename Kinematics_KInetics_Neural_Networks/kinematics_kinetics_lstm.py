import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ===================================================
# SETTINGS
# ===================================================
HEALTHY_FILE = "Data_Normal/dynamics_total_augmented.xlsx"   # TD merged/augmented
CP_FOLDER    = "Data_CP/"                                   # CP stride files (each one stride)
STRIDE       = 51

MODEL_SAVE       = "Saved_Models/td_nextstride_36to18.keras"
SCALER_X_SAVE    = "Scaler/td_x36_scaler.save"
SCALER_Y_SAVE    = "Scaler/td_y18_scaler.save"

PLOT_DIR = "Predictions/Plots"
OUT_DIR  = "Predictions"
os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_X_SAVE), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ===================================================
# COLUMNS
# ===================================================
MOMENT_COLS = [
    'LHipMoment (1)','RHipMoment (1)','LHipMoment (2)','RHipMoment (2)','LHipMoment (3)','RHipMoment (3)',
    'LKneeMoment (1)','RKneeMoment (1)','LKneeMoment (2)','RKneeMoment (2)','LKneeMoment (3)','RKneeMoment (3)',
    'LAnkleMoment (1)','RAnkleMoment (1)','LAnkleMoment (2)','RAnkleMoment (2)','LAnkleMoment (3)','RAnkleMoment (3)'
]
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

INPUT_COLS  = FORCE_COLS + ANGLE_COLS   # 36
OUTPUT_COLS = ANGLE_COLS                               # 18

# Sagittal plot indices inside the 18-angle output
SAG_NAMES = [
    "LHipAngles (1)", "RHipAngles (1)",
    "LKneeAngles (1)", "RKneeAngles (1)",
    "LAnkleAngles (1)", "RAnkleAngles (1)"
]
SAG_IDX = [ANGLE_COLS.index(n) for n in SAG_NAMES]  # robust

# ===================================================
# HELPERS
# ===================================================
def load_td_excel(path):
    df = pd.read_excel(path, sheet_name=0, usecols=INPUT_COLS)
    df.columns = df.columns.str.strip()
    df = df.dropna().reset_index(drop=True)
    return df

def load_cp_folder(folder):
    """Each CP file is one stride (51 rows). Returns (N,51,36) and names."""
    strides = []
    names = []
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".xlsx"):
            continue
        fp = os.path.join(folder, file)
        try:
            df = pd.read_excel(fp, sheet_name="Data", header=0, skiprows=[1,2], usecols=INPUT_COLS)
            df.columns = df.columns.str.strip()
            df = df.dropna(how="all").reset_index(drop=True)
            if len(df) < STRIDE:
                continue
            df = df.iloc[:STRIDE].copy()
            if len(df) != STRIDE:
                continue
            strides.append(df.to_numpy(np.float32))
            names.append(file)
        except Exception as e:
            print(f"[WARN] Skipping {file}: {e}")

    if len(strides) == 0:
        return np.empty((0, STRIDE, len(INPUT_COLS)), dtype=np.float32), []
    return np.stack(strides, axis=0), names

def to_next_stride_pairs(strides_3d):
    """
    TD only: strides are time-contiguous in the merged file.
    strides_3d: (N,51,36)
    returns X:(N-1,51,36), Y:(N-1,51,18)
    """
    if len(strides_3d) < 2:
        return np.empty((0, STRIDE, 36), dtype=np.float32), np.empty((0, STRIDE, 18), dtype=np.float32)
    X = strides_3d[:-1]
    Y = strides_3d[1:, :, -18:]
    return X, Y

def scale_3d(scaler, A):
    shp = A.shape
    return scaler.transform(A.reshape(-1, shp[-1])).reshape(shp)

def inv_scale_3d(scaler, A):
    shp = A.shape
    return scaler.inverse_transform(A.reshape(-1, shp[-1])).reshape(shp)

def phase_align_by_knee(cp6, ref6):
    """
    Align CP stride to reference stride by matching knee angles (sagittal).
    cp6/ref6: (51,6) in DEGREES.
    Returns a circularly-shifted cp6 so that LKnee matches best.
    """
    lknee_cp = cp6[:, 2]
    lknee_ref = ref6[:, 2]
    best_shift = 0
    best_err = 1e18
    for s in range(STRIDE):
        rolled = np.roll(lknee_cp, -s)
        err = np.mean((rolled - lknee_ref) ** 2)
        if err < best_err:
            best_err = err
            best_shift = s
    return np.roll(cp6, -best_shift, axis=0), best_shift

# ===================================================
# LOAD TD (TRAIN DATA)
# ===================================================
df_td = load_td_excel(HEALTHY_FILE)
data_td = df_td.to_numpy(np.float32)

n_td = len(data_td) // STRIDE
data_td = data_td[:n_td * STRIDE]
td_strides = data_td.reshape(n_td, STRIDE, -1)  # (Ntd,51,36)
print("TD strides:", td_strides.shape)

X_td, Y_td = to_next_stride_pairs(td_strides)
print("TD pairs:", X_td.shape, Y_td.shape)

# ===================================================
# TRAIN/VAL SPLIT (TD ONLY)
# ===================================================
idx = np.arange(len(X_td))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, X_va = X_td[tr_idx], X_td[va_idx]
Y_tr, Y_va = Y_td[tr_idx], Y_td[va_idx]

# ===================================================
# SCALERS (FIT ON TD TRAIN ONLY)
# ===================================================
sc_x = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))   # 36
sc_y = StandardScaler().fit(Y_tr.reshape(-1, Y_tr.shape[-1]))   # 18
joblib.dump(sc_x, SCALER_X_SAVE)
joblib.dump(sc_y, SCALER_Y_SAVE)
print("Saved scalers:", SCALER_X_SAVE, SCALER_Y_SAVE)

X_tr_s = scale_3d(sc_x, X_tr)
X_va_s = scale_3d(sc_x, X_va)
Y_tr_s = scale_3d(sc_y, Y_tr)
Y_va_s = scale_3d(sc_y, Y_va)

# ===================================================
# WEIGHTED LOSS: weight axis (1) > axis (2) > axis (3)
# Indices for axis (1)/(2)/(3) within ANGLE_COLS:
#   for each joint-side we have (1),(2),(3)
# Here: (1) are SAG_IDX + also hip/knee/ankle only exist, so that's correct.
# But we generalize: find indices containing "(1)", "(2)", "(3)".
# ===================================================
w = np.ones((18,), dtype=np.float32)
idx1 = [i for i, c in enumerate(ANGLE_COLS) if "(1)" in c]
idx2 = [i for i, c in enumerate(ANGLE_COLS) if "(2)" in c]
idx3 = [i for i, c in enumerate(ANGLE_COLS) if "(3)" in c]
w[idx1] = 3.0
w[idx2] = 1.5
w[idx3] = 1.0
w_tf = tf.constant(w, dtype=tf.float32)

@tf.keras.utils.register_keras_serializable()
def weighted_mse(y_true, y_pred):
    err2 = tf.square(y_true - y_pred)         # (B,51,18)
    return tf.reduce_mean(err2 * w_tf)

# ===================================================
# MODEL
# ===================================================
model = Sequential([
    Input(shape=(STRIDE, 36)),
    LSTM(256, return_sequences=True),
    LayerNormalization(),
    Dropout(0.2),

    LSTM(256, return_sequences=True),
    LayerNormalization(),
    Dropout(0.2),

    Dense(128, activation="tanh"),
    Dropout(0.2),
    Dense(18)
])

model.compile(
    optimizer=Adam(1e-3),
    loss=weighted_mse,
    metrics=["mae", tf.keras.metrics.RootMeanSquaredError()]
)
print(model.summary())

# ===================================================
# TRAIN
# ===================================================
cb = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5)
]
history = model.fit(
    X_tr_s, Y_tr_s,
    validation_data=(X_va_s, Y_va_s),
    epochs=200,
    batch_size=32,
    callbacks=cb,
    verbose=1
)

model.save(MODEL_SAVE, include_optimizer=True)
print("Saved model:", MODEL_SAVE)

# ===================================================
# TD VALIDATION PLOT (pred next stride vs real next stride)
# ===================================================
pred_va_s = model.predict(X_va_s, verbose=0)
pred_va = inv_scale_3d(sc_y, pred_va_s)
true_va = inv_scale_3d(sc_y, Y_va_s)

ex = 0
t = np.arange(STRIDE)
plt.figure(figsize=(12, 10))
for i, (idx_ang, name) in enumerate(zip(SAG_IDX, SAG_NAMES)):
    plt.subplot(3, 2, i+1)
    plt.plot(t, true_va[ex][:, idx_ang], label="TD real (k+1)")
    plt.plot(t, pred_va[ex][:, idx_ang], "--", label="TD predicted")
    plt.title(f"TD VAL: {name}")
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "td_val_sagittal_pred_vs_real.png"), dpi=150)
plt.show()

# ===================================================
# CP: "CORRECTION" EVALUATION (NO pairing)
# We do NOT compare to CP next stride.
# We compare CP stride to a healthy reference stride produced by the TD model.
# ===================================================
cp_strides, cp_names = load_cp_folder(CP_FOLDER)
print("CP strides loaded:", cp_strides.shape)

if len(cp_strides) == 0:
    print("[WARN] No CP strides found.")
    raise SystemExit(0)

# --- Build a healthy reference stride using TD model in "generator mode" ---
# Seed = a random TD stride from your dataset
seed_td = td_strides[np.random.randint(0, len(td_strides))]          # (51,36)
seed_td_s = scale_3d(sc_x, seed_td[None, ...])                       # (1,51,36)

ref_angles_s = model.predict(seed_td_s, verbose=0)[0]                # (51,18) scaled
ref_angles = sc_y.inverse_transform(ref_angles_s)                    # (51,18) degrees
ref_sag6 = ref_angles[:, SAG_IDX]                                    # (51,6)

# Save the full healthy reference angles (18) so you can load it in ROS2 as reference
pd.DataFrame(ref_angles, columns=OUTPUT_COLS).to_excel(
    os.path.join(OUT_DIR, "healthy_reference_stride_from_td_model.xlsx"), index=False
)
print("Saved healthy reference stride (18): Predictions/healthy_reference_stride_from_td_model.xlsx")

# --- Take one CP file and align/plot vs reference ---
cp0 = cp_strides[0]                 # (51,36)
cp0_angles = cp0[:, -18:]           # (51,18) degrees
cp0_sag6 = cp0_angles[:, SAG_IDX]   # (51,6)

cp0_sag6_aligned, shift = phase_align_by_knee(cp0_sag6, ref_sag6)
print(f"Phase aligned CP[0] by shift={shift} samples (circular).")

plt.figure(figsize=(12, 10))
for i, name in enumerate(SAG_NAMES):
    plt.subplot(3, 2, i+1)
    plt.plot(t, cp0_sag6_aligned[:, i], label="CP stride (aligned)")
    plt.plot(t, ref_sag6[:, i], "--", label="Healthy ref (TD model)")
    plt.title(f"CP vs HealthyRef: {name}")
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "cp_vs_healthyref_sagittal.png"), dpi=150)
plt.show()

# Optional: save an 18-angle "corrected reference" for this CP file (just the ref)
np.save(os.path.join(OUT_DIR, "corrected_cp_reference_gait.npy"), ref_sag6.astype(np.float32))
print("Saved corrected_cp_reference_gait.npy (51x6 sagittal reference).")
