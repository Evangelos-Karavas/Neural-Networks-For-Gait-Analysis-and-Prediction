import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
# ===================================================
# SETTINGS
# ===================================================
DATA_FILE = "Data_Normal/dynamics_total_augmented.xlsx"
MODEL_SAVE = "Saved_Models/stride_lstm_54to18.keras"
SCALER_DYN_SAVE = "Scaler/dyn_scaler.save"
SCALER_ANG_SAVE = "Scaler/ang_scaler.save"
S_OUT_SAVE = "Scaler/output_angle_scaler.save"
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

INPUT_COLS = MOMENT_COLS + FORCE_COLS + ANGLE_COLS
OUTPUT_COLS = ANGLE_COLS   # predict next 18 angle axes

# ===================================================
# LOAD DATA
# ===================================================
df = pd.read_excel(DATA_FILE, usecols=INPUT_COLS)
df = df.dropna().reset_index(drop=True)

# ===================================================
# SEPARATE INPUT GROUPS
# ===================================================
mom_force = df[MOMENT_COLS + FORCE_COLS].to_numpy()
angles    = df[ANGLE_COLS].to_numpy()

# ===================================================
# CREATE SCALERS
# ===================================================
sc_dyn = StandardScaler()
sc_ang = StandardScaler()

mom_force_scaled = sc_dyn.fit_transform(mom_force)
angles_scaled    = sc_ang.fit_transform(angles)

# full scaled input
df_scaled = np.concatenate([mom_force_scaled, angles_scaled], axis=1)

joblib.dump(sc_dyn, SCALER_DYN_SAVE)
joblib.dump(sc_ang, SCALER_ANG_SAVE)
print("Saved scalers.")

# ===================================================
# STRIDE SHAPES
# ===================================================
num_strides = len(df_scaled) // STRIDE
df_scaled = df_scaled[:num_strides * STRIDE]
data = df_scaled.reshape(num_strides, STRIDE, len(INPUT_COLS))

angles_original = angles.reshape(num_strides, STRIDE, 18)

# X = stride i inputs
# Y = stride i+1 angles (scaled with sc_ang)
X = data[:-1]
Y_raw = angles_original[1:]

Y_scaled = sc_ang.transform(Y_raw.reshape(-1, 18)).reshape(Y_raw.shape)

print("X:", X.shape)
print("Y:", Y_scaled.shape)

# ===================================================
# TRAIN/VAL SPLIT
# ===================================================
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y_scaled[:split], Y_scaled[split:]

# ===================================================
# WEIGHTED LOSS (improves axes 2 & 3)
# ===================================================
joint_weights = tf.constant(
    [3,3,1, 3,3,1, 3,3,1, 3,3,1, 3,3,1, 3,3,1], dtype=tf.float32
)

def weighted_mse(y_true, y_pred):
    return tf.reduce_mean(joint_weights * tf.square(y_true - y_pred))

# ===================================================
# BUILD MODEL
# ===================================================
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(STRIDE, len(INPUT_COLS))),
    tf.keras.layers.LayerNormalization(),
    LSTM(256, return_sequences=True),
    tf.keras.layers.LayerNormalization(),
    Dense(128, activation='tanh'),
    Dropout(0.2),
    Dense(18)
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
    X_train, Y_train,
    epochs=60,
    batch_size=32,
    validation_data=(X_val, Y_val),
    verbose=1
)

model.save(MODEL_SAVE)
print("Model saved.")

# ===================================================
# PREDICT NEXT STRIDE
# ===================================================
last_stride = X[-1].reshape(1, STRIDE, len(INPUT_COLS))
pred_scaled = model.predict(last_stride)[0]   # shape (51,18)

# inverse-scale angles
pred = sc_ang.inverse_transform(pred_scaled)

pd.DataFrame(pred, columns=ANGLE_COLS).to_excel(
    "Predictions/next_stride_angles.xlsx", index=False)

print("Saved prediction.")

# ===================================================
# PLOT ALL ANGLES
# ===================================================
t = np.arange(STRIDE)
actual = angles_original[-1]   # true
predicted = pred               # predicted inverse-scaled

plt.figure(figsize=(14, 22))

for i, col in enumerate(ANGLE_COLS):
    plt.subplot(9, 2, i+1)
    plt.plot(t, actual[:, i], label="Actual")
    plt.plot(t, predicted[:, i], "--", label="Predicted")
    plt.title(col)
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.show()