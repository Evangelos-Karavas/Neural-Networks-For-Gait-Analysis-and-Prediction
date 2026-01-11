import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# =========================================================
# SETTINGS
# =========================================================
HEALTHY_FILE = "Data_Normal/dynamics_total_augmented.xlsx"  # augmented healthy file (sheet 0)
CP_FOLDER = "Data_CP/"                                     # CP stride files (sheet "Data" + 2 extra rows)
MODEL_SAVE = "Saved_Models/lstm_nextstep_angles.keras"
SCALER_X_SAVE = "Scaler/nextstep_x_scaler.save"
SCALER_Y_SAVE = "Scaler/nextstep_y_scaler.save"

STRIDE = 51
PAST_W = 10
EPOCHS = 200
BATCH_SIZE = 64
MODEL_UNITS = 128

# If True: angles+moms+forces as input (54). If False: moms+forces only (36).
USE_ANGLES_IN_INPUT = True

SEED = 7
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_X_SAVE), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_Y_SAVE), exist_ok=True)

# =========================================================
# COLUMNS
# =========================================================
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

if USE_ANGLES_IN_INPUT:
    INPUT_COLS = ANGLE_COLS + MOMENT_COLS + FORCE_COLS   # 54
else:
    INPUT_COLS = MOMENT_COLS + FORCE_COLS                # 36

OUTPUT_COLS = ANGLE_COLS  # 18

# =========================================================
# HELPERS
# =========================================================
def reshape_strides(A_flat, stride_len=51):
    n = len(A_flat) // stride_len
    A_flat = A_flat[:n * stride_len]
    return A_flat.reshape(n, stride_len, -1)

def build_nextstep_windows(strides_X, strides_Y, past_w):
    Xw, Yw = [], []
    N, T, _ = strides_X.shape
    for s in range(N):
        for t in range(past_w - 1, T - 1):
            Xw.append(strides_X[s, t - past_w + 1:t + 1, :])
            Yw.append(strides_Y[s, t + 1, :])  # next step
    return np.asarray(Xw, np.float32), np.asarray(Yw, np.float32)

def load_cp_stride(path, usecols):
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

def make_model(past_w, xin, yout=18):
    model = models.Sequential([
        layers.Input(shape=(past_w, xin)),
        layers.LSTM(MODEL_UNITS, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(128, activation="tanh"),
        layers.Dense(yout)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError()]
    )
    return model

def closed_loop_correct_angles(df_cp, model, sc_x, sc_y, past_w, use_angles_in_input=True):
    """
    Closed-loop correction:
    - Use predicted/corrected angles inside the next input windows
    - Moments/forces always come from the measured CP stride (as exo hasn't acted yet)
    Returns corrected angles: (51,18)
    """
    cp_angles = df_cp[ANGLE_COLS].to_numpy(np.float32)      # (51,18)
    cp_mom    = df_cp[MOMENT_COLS].to_numpy(np.float32)     # (51,18)
    cp_force  = df_cp[FORCE_COLS].to_numpy(np.float32)      # (51,18)

    corr = cp_angles.copy()

    xin_local = (54 if use_angles_in_input else 36)

    for t in range(past_w - 1, STRIDE - 1):
        win = []
        for k in range(t - past_w + 1, t + 1):
            if use_angles_in_input:
                feat = np.concatenate([corr[k], cp_mom[k], cp_force[k]], axis=0)  # <-- feedback corr[k]
            else:
                feat = np.concatenate([cp_mom[k], cp_force[k]], axis=0)
            win.append(feat)

        xwin = np.stack(win, axis=0)[None, :, :]  # (1,past_w,xin)
        xwin_s = sc_x.transform(xwin.reshape(-1, xin_local)).reshape(xwin.shape)

        yhat_s = model.predict(xwin_s, verbose=0)[0]        # (18,) scaled
        yhat = sc_y.inverse_transform(yhat_s.reshape(1,-1))[0]
        corr[t + 1] = yhat

    return corr

# =========================================================
# LOAD HEALTHY + TRAIN
# =========================================================
df_h = pd.read_excel(HEALTHY_FILE, sheet_name=0, usecols=list(set(INPUT_COLS + OUTPUT_COLS)))
df_h.columns = df_h.columns.str.strip()
df_h = df_h.dropna().reset_index(drop=True)

Xh_flat = df_h[INPUT_COLS].to_numpy(np.float32)
Yh_flat = df_h[OUTPUT_COLS].to_numpy(np.float32)

Xh = reshape_strides(Xh_flat, STRIDE)  # (Nh,51,Xin)
Yh = reshape_strides(Yh_flat, STRIDE)  # (Nh,51,18)

print("Healthy strides:", Xh.shape, Yh.shape)

# typical template for plotting
typical_template = Yh.mean(axis=0)  # (51,18)

# windowed training
Xw, Yw = build_nextstep_windows(Xh, Yh, PAST_W)
print("Healthy windows:", Xw.shape, Yw.shape)

idx = np.arange(len(Xw))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, X_va = Xw[tr_idx], Xw[va_idx]
Y_tr, Y_va = Yw[tr_idx], Yw[va_idx]

xin = X_tr.shape[-1]

sc_x = tf.keras.utils.serialize_keras_object  # placeholder? no

# Scale
sc_x = StandardScaler().fit(X_tr.reshape(-1, xin))
sc_y = StandardScaler().fit(Y_tr)

joblib.dump(sc_x, SCALER_X_SAVE)
joblib.dump(sc_y, SCALER_Y_SAVE)

X_tr_s = sc_x.transform(X_tr.reshape(-1, xin)).reshape(X_tr.shape)
X_va_s = sc_x.transform(X_va.reshape(-1, xin)).reshape(X_va.shape)
Y_tr_s = sc_y.transform(Y_tr)
Y_va_s = sc_y.transform(Y_va)

# Train model
model = make_model(PAST_W, xin, yout=18)
print(model.summary())

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=12, factor=0.5, min_lr=1e-5),
]

model.fit(
    X_tr_s, Y_tr_s,
    validation_data=(X_va_s, Y_va_s),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

model.save(MODEL_SAVE, include_optimizer=True)
print("Saved model:", MODEL_SAVE)
print("Saved scalers:", SCALER_X_SAVE, SCALER_Y_SAVE)

# =========================================================
# CLOSED-LOOP INFERENCE ON ONE CP FILE + PLOT
# =========================================================
cp_files = [f for f in sorted(os.listdir(CP_FOLDER)) if f.endswith(".xlsx")]
if len(cp_files) == 0:
    raise RuntimeError(f"No CP .xlsx files found in {CP_FOLDER}")

# pick the first CP file
cp_file = cp_files[0]
df_cp = load_cp_stride(os.path.join(CP_FOLDER, cp_file), usecols=ANGLE_COLS + MOMENT_COLS + FORCE_COLS)

cp_angles = df_cp[ANGLE_COLS].to_numpy(np.float32)  # measured CP angles
corr = closed_loop_correct_angles(df_cp, model, sc_x, sc_y, PAST_W, use_angles_in_input=USE_ANGLES_IN_INPUT)

t = np.arange(STRIDE)
plt.figure(figsize=(15, 26))
for i, col in enumerate(ANGLE_COLS):
    plt.subplot(9, 2, i + 1)
    plt.plot(t, cp_angles[:, i], label="CP measured")
    plt.plot(t, typical_template[:, i], label="Typical (healthy mean)")
    plt.plot(t, corr[:, i], "--", label="Corrected (closed-loop LSTM)")
    plt.title(col)
    plt.grid(True)
    plt.legend()

plt.suptitle(f"Closed-loop corrected CP angles (file: {cp_file})", y=1.01)
plt.tight_layout()
plt.show()
