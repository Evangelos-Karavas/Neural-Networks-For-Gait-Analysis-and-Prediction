import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ===================================================
# SETTINGS
# ===================================================
HEALTHY_FILE = "Data_Normal/dynamics_total_augmented.xlsx"
CP_FOLDER = "Data_CP/"
MODEL_SAVE = "Saved_Models/dynamics_lstm.keras"
SCALER_DYN_SAVE = "Scaler/dyn_scaler.save"       # scales inputs (54)
SCALER_ANG_SAVE = "Scaler/ang_scaler.save"       # scales outputs (18)
STRIDE = 51

# ---------------------------------------------------
# COLUMNS
# ---------------------------------------------------
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

INPUT_COLS = MOMENT_COLS + FORCE_COLS + ANGLE_COLS   # 54 inputs
OUTPUT_COLS = ANGLE_COLS                             # 18 outputs

# ===================================================
# LOAD HEALTHY DATA
# ===================================================
df_h = pd.read_excel(HEALTHY_FILE, usecols=INPUT_COLS)
df_h = df_h.dropna().reset_index(drop=True)

# ===================================================
# LOAD CP DATA
# ===================================================
all_cp = []
for file in os.listdir(CP_FOLDER):
    if file.endswith(".xlsx"):
        df_cp = pd.read_excel(os.path.join(CP_FOLDER, file),
                              sheet_name="Data",
                              usecols=INPUT_COLS,
                              skiprows=[1,2])
        df_cp = df_cp.dropna().reset_index(drop=True)
        all_cp.append(df_cp)

df_cp = pd.concat(all_cp, ignore_index=True)
print("Loaded CP data:", df_cp.shape)

# ===================================================
# SEPARATE INTO INPUT GROUPS
# ===================================================
X_h_dyn = df_h[MOMENT_COLS + FORCE_COLS + ANGLE_COLS].to_numpy()   # (N,54)
Y_h_ang = df_h[ANGLE_COLS].to_numpy()                              # (N,18)

X_cp_dyn = df_cp[MOMENT_COLS + FORCE_COLS + ANGLE_COLS].to_numpy()
Y_cp_ang = df_cp[ANGLE_COLS].to_numpy()

# ===================================================
# FIT SCALERS ONLY ON HEALTHY DATA
# ===================================================
sc_dyn = StandardScaler().fit(X_h_dyn)
sc_ang = StandardScaler().fit(Y_h_ang)

joblib.dump(sc_dyn, SCALER_DYN_SAVE)
joblib.dump(sc_ang, SCALER_ANG_SAVE)
print("Saved scalers.")

# ===================================================
# SCALE BOTH DATASETS
# ===================================================
X_h_scaled = sc_dyn.transform(X_h_dyn)
Y_h_scaled = sc_ang.transform(Y_h_ang)

X_cp_scaled = sc_dyn.transform(X_cp_dyn)
Y_cp_scaled = sc_ang.transform(Y_cp_ang)

# ===================================================
# RESHAPE INTO STRIDES
# ===================================================
def reshape_strides(X_flat, Y_flat):
    num_strides = len(X_flat) // STRIDE
    X_flat = X_flat[:num_strides * STRIDE]
    Y_flat = Y_flat[:num_strides * STRIDE]
    Xs = X_flat.reshape(num_strides, STRIDE, -1)
    Ys = Y_flat.reshape(num_strides, STRIDE, -1)
    return Xs, Ys

Xh, Yh = reshape_strides(X_h_scaled, Y_h_scaled)
Xcp, Ycp = reshape_strides(X_cp_scaled, Y_cp_scaled)

print("Healthy strides:", Xh.shape)
print("CP strides:", Xcp.shape)

# NEXT-STRIDE PREDICTION FORMAT
X_train = Xh[:-1]     # stride i
Y_train = Yh[1:]      # stride i+1

# CP evaluation dataset
X_test_cp = Xcp[:-1]  # stride i
Y_test_cp = Ycp[1:]   # stride i+1

# ===================================================
# TRAIN/VAL SPLIT
# ===================================================
split = int(0.8 * len(X_train))
X_tr, X_val = X_train[:split], X_train[split:]
Y_tr, Y_val = Y_train[:split], Y_train[split:]

# ===================================================
# WEIGHTED LOSS
# ===================================================
joint_weights = tf.constant([3,3,2, 3,3,2, 3,3,2, 3,3,2, 3,3,2, 3,3,2], dtype=tf.float32)

def weighted_mse(y_true, y_pred):
    return tf.reduce_mean(joint_weights * tf.square(y_true - y_pred))

# ===================================================
# BUILD LSTM MODEL
# ===================================================
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(STRIDE, 54)),
    tf.keras.layers.LayerNormalization(),

    LSTM(256, return_sequences=True),
    tf.keras.layers.LayerNormalization(),

    Dense(128, activation='tanh'),
    Dropout(0.2),

    Dense(18)     # output angles (scaled)
])

model.compile(
    optimizer=Adam(1e-3),
    loss=weighted_mse,
    metrics=["mae", RootMeanSquaredError()]
)

print(model.summary())

# ===================================================
# TRAIN
# ===================================================
history = model.fit(
    X_tr, Y_tr,
    epochs=120,
    batch_size=32,
    validation_data=(X_val, Y_val),
    verbose=1
)

model.save(MODEL_SAVE, include_optimizer=True)
print("Model saved:", MODEL_SAVE)

# ===================================================
# TEST ON CP DATA (CORRECTION TEST)
# ===================================================
cp_pred_scaled = model.predict(X_test_cp)
cp_pred = sc_ang.inverse_transform(cp_pred_scaled.reshape(-1,18)).reshape(cp_pred_scaled.shape)

# Save first CP prediction
pd.DataFrame(cp_pred[0], columns=OUTPUT_COLS).to_excel(
    "Predictions/cp_corrected_stride.xlsx", index=False)

print("Saved corrected CP stride → Predictions/cp_corrected_stride.xlsx")

# ===================================================
# PLOT CP CORRECTION VS CP INPUT
# ===================================================
t = np.arange(STRIDE)
actual_cp = sc_ang.inverse_transform(Y_test_cp[0])  # CP angles (unscaled)
pred_cp = cp_pred[0]                                # corrected (healthy-like)

plt.figure(figsize=(15, 26))

for i, col in enumerate(OUTPUT_COLS):
    plt.subplot(9, 2, i+1)
    plt.plot(t, actual_cp[:, i], label="CP Input")
    plt.plot(t, pred_cp[:, i], "--", label="Corrected (LSTM)")
    plt.title(col)
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.show()
