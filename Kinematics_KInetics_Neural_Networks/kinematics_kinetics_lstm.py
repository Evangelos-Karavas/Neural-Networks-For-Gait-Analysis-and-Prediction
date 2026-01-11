import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, LayerNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ===================================================
# SETTINGS
# ===================================================
HEALTHY_FILE = "Data_Normal/dynamics_total_augmented.xlsx"  # augmented healthy file (single sheet)
CP_FOLDER = "Data_CP/"                                     # CP stride files (sheet "Data")
OUT_DIR_ASSIST = "Predictions/Assistance_Sagittal/"
OUT_DIR_ANGLEPRED = "Predictions/Angle_Predictions/"

MODEL_TYPE = "cnn"   # "cnn" or "lstm"
STRIDE = 51

# --- saved artifacts ---
MODEL_INV_SAVE = "Saved_Models/invdyn_angles_to_sagmom.keras"
SCALER_INV_X_SAVE = "Scaler/invdyn_x_scaler.save"
SCALER_INV_Y_SAVE = "Scaler/invdyn_y_scaler.save"

MODEL_ANG_SAVE = "Saved_Models/kinetics_to_angles.keras"
SCALER_ANG_X_SAVE = "Scaler/kin2ang_x_scaler.save"
SCALER_ANG_Y_SAVE = "Scaler/kin2ang_y_scaler.save"

# Create folders
for p in [
    os.path.dirname(MODEL_INV_SAVE),
    os.path.dirname(MODEL_ANG_SAVE),
    os.path.dirname(SCALER_INV_X_SAVE),
    os.path.dirname(SCALER_ANG_X_SAVE),
    OUT_DIR_ASSIST,
    OUT_DIR_ANGLEPRED,
]:
    os.makedirs(p, exist_ok=True)

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

# Sagittal moments only (1) = sagittal
MOMENT_SAG_COLS = [
    "LHipMoment (1)", "RHipMoment (1)",
    "LKneeMoment (1)", "RKneeMoment (1)",
    "LAnkleMoment (1)", "RAnkleMoment (1)"
]

# ===================================================
# HELPERS
# ===================================================
def reshape_strides(A_flat, stride_len=51):
    n = len(A_flat) // stride_len
    A_flat = A_flat[:n * stride_len]
    return A_flat.reshape(n, stride_len, -1)

def add_derivatives(angles_3d):
    vel = np.gradient(angles_3d, axis=1)
    acc = np.gradient(vel, axis=1)
    return np.concatenate([angles_3d, vel, acc], axis=-1)  # (N,51,54)

def scale3d(sc, A):
    shp = A.shape
    return sc.transform(A.reshape(-1, shp[-1])).reshape(shp)

def build_model(model_type, T, xin, yout):
    if model_type.lower() == "cnn":
        model = Sequential([
            Input(shape=(T, xin)),
            Conv1D(128, 5, padding="same", activation="relu"),
            Dropout(0.2),
            Conv1D(128, 5, padding="same", activation="relu"),
            Dropout(0.2),
            Conv1D(64, 3, padding="same", activation="relu"),
            Dense(yout)
        ])
    elif model_type.lower() == "lstm":
        model = Sequential([
            Input(shape=(T, xin)),
            LSTM(96, return_sequences=True),
            LayerNormalization(),
            Dropout(0.2),
            LSTM(96, return_sequences=True),
            LayerNormalization(),
            Dropout(0.2),
            Dense(64, activation="tanh"),
            Dense(yout)
        ])
    else:
        raise ValueError("MODEL_TYPE must be 'cnn' or 'lstm'")

    model.compile(
        optimizer=Adam(1e-3),
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError()]
    )
    return model

def load_cp_stride_file(path, usecols):
    # CP format: row0 headers, row1 Left/Right, row2 units, then 51 numeric rows
    df = pd.read_excel(
        path,
        sheet_name="Data",
        header=0,
        skiprows=[1, 2],
        usecols=usecols
    )
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) >= STRIDE:
        df = df.iloc[:STRIDE].copy()

    if len(df) != STRIDE:
        raise ValueError(f"{os.path.basename(path)} has {len(df)} rows, expected {STRIDE}")
    return df

# ===================================================
# LOAD HEALTHY (augmented excel) - first sheet
# ===================================================
# Your augmented file was saved with to_excel(index=False), so it’s typically sheet 0.
df_h = pd.read_excel(HEALTHY_FILE, sheet_name=0, usecols=ANGLE_COLS + MOMENT_SAG_COLS + MOMENT_COLS + FORCE_COLS)
df_h.columns = df_h.columns.str.strip()
df_h = df_h.dropna().reset_index(drop=True)

# ===================================================
# LOAD CP (all files)
# ===================================================
cp_names = []
cp_angles_list = []
cp_moms_sag_list = []
cp_moms_all_list = []
cp_forces_list = []

for file in sorted(os.listdir(CP_FOLDER)):
    if not file.endswith(".xlsx"):
        continue
    path = os.path.join(CP_FOLDER, file)
    try:
        df_cp = load_cp_stride_file(path, usecols=ANGLE_COLS + MOMENT_SAG_COLS + MOMENT_COLS + FORCE_COLS)

        cp_names.append(file)
        cp_angles_list.append(df_cp[ANGLE_COLS].to_numpy(np.float32))           # (51,18)
        cp_moms_sag_list.append(df_cp[MOMENT_SAG_COLS].to_numpy(np.float32))   # (51,6)
        cp_moms_all_list.append(df_cp[MOMENT_COLS].to_numpy(np.float32))       # (51,18)
        cp_forces_list.append(df_cp[FORCE_COLS].to_numpy(np.float32))          # (51,18)

    except Exception as e:
        print(f"[WARN] Skipping {file}: {e}")

Xcp_ang = np.stack(cp_angles_list, axis=0)     # (Ncp,51,18)
Ycp_mom_sag = np.stack(cp_moms_sag_list, axis=0)  # (Ncp,51,6)
Xcp_mom_all = np.stack(cp_moms_all_list, axis=0)  # (Ncp,51,18)
Xcp_force = np.stack(cp_forces_list, axis=0)      # (Ncp,51,18)

print("CP loaded:", Xcp_ang.shape, Ycp_mom_sag.shape)

# ===================================================
# (A) INVERSE DYNAMICS SURROGATE
# Healthy: angles(+derivatives) -> sagittal moments (6)
# ===================================================
Xh_ang_flat = df_h[ANGLE_COLS].to_numpy(np.float32)
Yh_sag_flat = df_h[MOMENT_SAG_COLS].to_numpy(np.float32)

Xh_ang = reshape_strides(Xh_ang_flat, STRIDE)   # (Nh,51,18)
Yh_sag = reshape_strides(Yh_sag_flat, STRIDE)   # (Nh,51,6)

Xh_in = add_derivatives(Xh_ang)                 # (Nh,51,54)

# Shuffle split by stride
idx = np.arange(len(Xh_in))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, X_va = Xh_in[tr_idx], Xh_in[va_idx]
Y_tr, Y_va = Yh_sag[tr_idx], Yh_sag[va_idx]

sc_inv_x = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
sc_inv_y = StandardScaler().fit(Y_tr.reshape(-1, Y_tr.shape[-1]))

joblib.dump(sc_inv_x, SCALER_INV_X_SAVE)
joblib.dump(sc_inv_y, SCALER_INV_Y_SAVE)

X_tr_s = scale3d(sc_inv_x, X_tr)
X_va_s = scale3d(sc_inv_x, X_va)
Y_tr_s = scale3d(sc_inv_y, Y_tr)
Y_va_s = scale3d(sc_inv_y, Y_va)

model_inv = build_model(MODEL_TYPE, STRIDE, X_tr_s.shape[-1], Y_tr_s.shape[-1])
print(model_inv.summary())

history_inv = model_inv.fit(
    X_tr_s, Y_tr_s,
    validation_data=(X_va_s, Y_va_s),
    epochs=200,          # you can raise; early stopping optional
    batch_size=16,
    verbose=1
)

model_inv.save(MODEL_INV_SAVE, include_optimizer=True)
print("Saved inverse model:", MODEL_INV_SAVE)

# ---- Assistance computation ----
target_angles = Xh_ang.mean(axis=0)  # (51,18) typical template
target_angles_all = np.repeat(target_angles[None, :, :], repeats=len(Xcp_ang), axis=0)

Xtarget_in = add_derivatives(target_angles_all)
Xtarget_in_s = scale3d(sc_inv_x, Xtarget_in)

Mreq_s = model_inv.predict(Xtarget_in_s, verbose=0)  # scaled required sag moments
Mreq = sc_inv_y.inverse_transform(Mreq_s.reshape(-1, 6)).reshape(Mreq_s.shape)
Mexo = Mreq - Ycp_mom_sag

# Save assistance files
for i, name in enumerate(cp_names):
    base = os.path.splitext(name)[0]
    out_path = os.path.join(OUT_DIR_ASSIST, f"{base}_assist_sag.xlsx")

    df_out = pd.DataFrame({
        "t": np.arange(STRIDE),
        "CP_LHipM1": Ycp_mom_sag[i,:,0], "CP_RHipM1": Ycp_mom_sag[i,:,1],
        "CP_LKneeM1": Ycp_mom_sag[i,:,2], "CP_RKneeM1": Ycp_mom_sag[i,:,3],
        "CP_LAnkM1": Ycp_mom_sag[i,:,4], "CP_RAnkM1": Ycp_mom_sag[i,:,5],

        "REQ_LHipM1": Mreq[i,:,0], "REQ_RHipM1": Mreq[i,:,1],
        "REQ_LKneeM1": Mreq[i,:,2], "REQ_RKneeM1": Mreq[i,:,3],
        "REQ_LAnkM1": Mreq[i,:,4], "REQ_RAnkM1": Mreq[i,:,5],

        "EXO_LHipM1": Mexo[i,:,0], "EXO_RHipM1": Mexo[i,:,1],
        "EXO_LKneeM1": Mexo[i,:,2], "EXO_RKneeM1": Mexo[i,:,3],
        "EXO_LAnkM1": Mexo[i,:,4], "EXO_RAnkM1": Mexo[i,:,5],
    })
    df_out.to_excel(out_path, index=False)

print("Saved assistance files to:", OUT_DIR_ASSIST)

# Plot one assistance example
if len(cp_names) > 0:
    j = 0
    t = np.arange(STRIDE)
    labels = ["LHipM1","RHipM1","LKneeM1","RKneeM1","LAnkM1","RAnkM1"]

    plt.figure(figsize=(14, 18))
    for k in range(6):
        plt.subplot(3, 2, k+1)
        plt.plot(t, Ycp_mom_sag[j,:,k], label="CP measured")
        plt.plot(t, Mreq[j,:,k], label="Required (typical target)")
        plt.plot(t, Mexo[j,:,k], "--", label="Assistance (exo)")
        plt.title(labels[k])
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

# ===================================================
# (B) ANGLE PREDICTION MODEL
# Healthy: (moments + forces) -> angles
# ===================================================
INPUT_COLS_ANG = MOMENT_COLS + FORCE_COLS   # 36 inputs
OUTPUT_COLS_ANG = ANGLE_COLS                # 18 outputs

Xh_k_flat = df_h[INPUT_COLS_ANG].to_numpy(np.float32)
Yh_a_flat = df_h[OUTPUT_COLS_ANG].to_numpy(np.float32)

Xh_k = reshape_strides(Xh_k_flat, STRIDE)   # (Nh,51,36)
Yh_a = reshape_strides(Yh_a_flat, STRIDE)   # (Nh,51,18)

# Build CP kinetics input (moments+forces)
Xcp_k = np.concatenate([Xcp_mom_all, Xcp_force], axis=-1)  # (Ncp,51,36)
Ycp_a = Xcp_ang                                           # measured angles (Ncp,51,18)

# Split healthy
idx = np.arange(len(Xh_k))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr, va = idx[:split], idx[split:]

X_tr, X_va = Xh_k[tr], Xh_k[va]
Y_tr, Y_va = Yh_a[tr], Yh_a[va]

sc_ang_x = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
sc_ang_y = StandardScaler().fit(Y_tr.reshape(-1, Y_tr.shape[-1]))

joblib.dump(sc_ang_x, SCALER_ANG_X_SAVE)
joblib.dump(sc_ang_y, SCALER_ANG_Y_SAVE)

X_tr_s = scale3d(sc_ang_x, X_tr)
X_va_s = scale3d(sc_ang_x, X_va)
Y_tr_s = scale3d(sc_ang_y, Y_tr)
Y_va_s = scale3d(sc_ang_y, Y_va)

model_ang = build_model(MODEL_TYPE, STRIDE, X_tr_s.shape[-1], Y_tr_s.shape[-1])
print(model_ang.summary())

history_ang = model_ang.fit(
    X_tr_s, Y_tr_s,
    validation_data=(X_va_s, Y_va_s),
    epochs=200,
    batch_size=16,
    verbose=1 
)

model_ang.save(MODEL_ANG_SAVE, include_optimizer=True)
print("Saved angle model:", MODEL_ANG_SAVE)

# ---- Predict angles for one CP stride and visualize ----
if len(cp_names) > 0:
    j = 0
    cp_pred_s = model_ang.predict(scale3d(sc_ang_x, Xcp_k[j:j+1]), verbose=0)[0]  # (51,18) scaled
    cp_pred = sc_ang_y.inverse_transform(cp_pred_s)                               # (51,18) unscaled

    # Save predicted stride
    out_path = os.path.join(OUT_DIR_ANGLEPRED, f"{os.path.splitext(cp_names[j])[0]}_pred_angles.xlsx")
    pd.DataFrame(cp_pred, columns=OUTPUT_COLS_ANG).to_excel(out_path, index=False)
    print("Saved predicted angles example to:", out_path)

    t = np.arange(STRIDE)
    plt.figure(figsize=(15, 26))
    for i, col in enumerate(OUTPUT_COLS_ANG):
        plt.subplot(9, 2, i+1)
        plt.plot(t, Ycp_a[j, :, i], label="CP measured angles")
        plt.plot(t, cp_pred[:, i], "--", label="Predicted angles (from kinetics)")
        plt.title(col)
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()
