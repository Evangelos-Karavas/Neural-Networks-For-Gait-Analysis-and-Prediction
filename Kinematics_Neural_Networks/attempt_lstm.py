import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.metrics import RootMeanSquaredError
from tensorflow.python.keras.optimizer_v2 import adam
import tensorflow.python.keras as tf_keras

import matplotlib.pyplot as plt

# --- keep your save_model version workaround ---
from keras import __version__
tf_keras.__version__ = __version__

# =====================================================================================
# CONFIG
# =====================================================================================
STRIDE_LEN = 51
IN_LEN     = 51
OUT_LEN    = 51
EPOCHS     = 150
BATCH_SIZE = 102
USE_CONTINUITY_LOSS = True  # set False if you want plain MSE

os.makedirs("Saved_Models", exist_ok=True)
os.makedirs("Predictions",  exist_ok=True)
os.makedirs("Scaler",       exist_ok=True)

# =====================================================================================
# Load data
# =====================================================================================
data_typical = "Data_Normal/randomized_data_healthy.xlsx"
columns_to_read = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                   'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']

data = pd.read_excel(data_typical, usecols=columns_to_read).copy()
data.fillna(0, inplace=True)

# ----- Load CP folder -----
folder_path = 'Data_CP/'
file_counter = 0
data_cerebral_palsy = pd.DataFrame()

for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_counter += 1
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path, sheet_name='Data',
                           usecols=columns_to_read, skiprows=[1, 2])
        df.fillna(0, inplace=True)
        data_cerebral_palsy = pd.concat([data_cerebral_palsy, df], ignore_index=True)
        if file_counter >= 500:
            break

data_cerebral_palsy.fillna(0, inplace=True)

# =====================================================================================
# Angle hygiene: unwrap hips to remove artificial ±180° jumps
# =====================================================================================
def unwrap_deg(series: pd.Series) -> pd.Series:
    rad = np.deg2rad(series.to_numpy())
    return pd.Series(np.rad2deg(np.unwrap(rad)), index=series.index)

for c in ['LHipAngles (1)', 'RHipAngles (1)']:
    if c in data.columns:
        data[c] = unwrap_deg(data[c])
    if c in data_cerebral_palsy.columns:
        data_cerebral_palsy[c] = unwrap_deg(data_cerebral_palsy[c])

# =====================================================================================
# Scale (fit once on combined to be robust across domains)
# =====================================================================================
scaler = StandardScaler()
_ = scaler.fit(pd.concat([data, data_cerebral_palsy], axis=0, ignore_index=True))
typ_scaled = scaler.transform(data)
cp_scaled  = scaler.transform(data_cerebral_palsy)

joblib.dump(scaler, "Scaler/standard_scaler_lstm.save")

# =====================================================================================
# Windowing: input = previous 51 steps, target = immediate next 51 steps
# =====================================================================================
def make_windows(X: np.ndarray, in_len=51, out_len=51, step=1):
    """
    X: (T, F) continuous time-series
    returns:
      Xin:  (N, in_len,  F)
      Yout: (N, out_len, F) where Yout[n,0] == X[start+in_len]
    """
    T, F = X.shape
    if T < in_len + out_len:
        raise ValueError("Series too short for the chosen in_len/out_len.")
    starts = range(0, T - (in_len + out_len) + 1, step)
    Xin  = np.stack([X[s:s+in_len]           for s in starts], axis=0).astype(np.float32)
    Yout = np.stack([X[s+in_len:s+in_len+out_len] for s in starts], axis=0).astype(np.float32)
    return Xin, Yout

X_typ, Y_typ = make_windows(typ_scaled, IN_LEN, OUT_LEN, step=1)
X_cp,  Y_cp  = make_windows(cp_scaled,  IN_LEN, OUT_LEN, step=1)

# =====================================================================================
# Split: train on typical (80%), validate on typical tail + small CP head, test on CP
# =====================================================================================
split_idx_typ = int(0.8 * len(X_typ))
split_idx_cp  = int(0.2 * len(X_cp)) if len(X_cp) > 5 else max(1, len(X_cp)//5)

X_train_input, Y_train_output = X_typ[:split_idx_typ], Y_typ[:split_idx_typ]
X_val_input  = np.vstack((X_typ[split_idx_typ:], X_cp[:split_idx_cp])) if split_idx_cp > 0 else X_typ[split_idx_typ:]
Y_val_output = np.vstack((Y_typ[split_idx_typ:], Y_cp[:split_idx_cp])) if split_idx_cp > 0 else Y_typ[split_idx_typ:]
X_test_input, Y_test_output  = X_cp, Y_cp

# =====================================================================================
# Model
# =====================================================================================
model = Sequential([
    LSTM(100, activation='tanh', return_sequences=True, input_shape=(IN_LEN, 6)),
    LSTM(100, activation='tanh', return_sequences=True),
    Dropout(0.4),
    Dense(6, activation='linear')
])

# Optional continuity loss to match velocities (reduces subtle edge artifacts)
import tensorflow as tf
def continuity_loss(y_true, y_pred):
    v_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    v_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    return tf.reduce_mean(tf.square(v_true - v_pred))

def composite_loss(y_true, y_pred, alpha=1.0, beta=0.3):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return alpha*mse + beta*continuity_loss(y_true, y_pred)

loss_fn = composite_loss if USE_CONTINUITY_LOSS else 'mse'

model.compile(
    optimizer=adam.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=['mae', RootMeanSquaredError()]
)

model.summary()

history = model.fit(
    X_train_input, Y_train_output,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_input, Y_val_output),
    verbose=1
)

# =====================================================================================
# Evaluation
# =====================================================================================
def evaluate_model(model, X, Y, label):
    loss, mae, rmse = model.evaluate(X, Y, verbose=0)
    print(f"🔹 {label} Evaluation:")
    print(f"Loss: {loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    print("=" * 90)

evaluate_model(model, X_train_input, Y_train_output, "Training")
evaluate_model(model, X_val_input,   Y_val_output,   "Validation")
evaluate_model(model, X_test_input,  Y_test_output,  "Testing (CP)")

model.save("Saved_Models/Timestamp_lstm_model.keras", include_optimizer=True)

# =====================================================================================
# Autoregressive rollout with SLIDING WINDOW (no discontinuities between blocks)
# =====================================================================================
def predict_future_steps(model, last_window_51x6, num_blocks, scaler):
    """
    last_window_51x6: scaled array (51,6) to start from
    num_blocks: how many 51-length blocks to roll out
    returns: (num_blocks*51, 6) in ORIGINAL scale
    """
    preds_scaled = []
    current = last_window_51x6.reshape(1, IN_LEN, 6)
    for _ in range(num_blocks):
        next_block = model.predict(current, verbose=0)  # (1,51,6) immediate continuation
        preds_scaled.append(next_block[0])              # (51,6)
        # slide the input window to end at the last predicted step
        current = np.concatenate([current[0], next_block[0]], axis=0)[-IN_LEN:].reshape(1, IN_LEN, 6)
    preds_scaled = np.vstack(preds_scaled)
    return scaler.inverse_transform(preds_scaled)

# Choose starting windows (scaled)
last_window_typ = X_train_input[-1] if len(X_train_input) else X_typ[-1]
last_window_cp  = X_test_input[-1]  if len(X_test_input)  else X_cp[-1]

num_blocks = 4
next_steps_prediction    = predict_future_steps(model, last_window_typ, num_blocks, scaler)
next_steps_prediction_cp = predict_future_steps(model, last_window_cp,  num_blocks, scaler)

# =====================================================================================
# Build ground truth for plots from ORIGINAL continuous arrays (no stride reset)
# =====================================================================================
def build_gt(original_df: pd.DataFrame, start_after_window_idx: int, num_blocks: int, block_len: int) -> np.ndarray:
    """
    original_df: unscaled, continuous dataframe (T,F)
    start_after_window_idx: index of the sample *after* the 51-step input window ends
    returns: (num_blocks*block_len, F)
    """
    total = num_blocks * block_len
    arr = original_df.to_numpy()
    end = start_after_window_idx + total
    return arr[start_after_window_idx:end, :]

# For typical: take the window that produced last_window_typ from typ_scaled
# Find its position in the continuous series (the very end of the training range)
typ_window_end = IN_LEN + (len(X_train_input) - 1)  # window index ends here inside the sliding set
typ_start_idx_for_gt = typ_window_end  # first gt sample right after the input window
gt_typ = build_gt(data.iloc[:], typ_start_idx_for_gt, num_blocks, OUT_LEN)

# For CP: similarly from CP series
cp_window_end = IN_LEN + (len(X_test_input) - 1) if len(X_test_input) else IN_LEN + (len(X_cp) - 1)
cp_start_idx_for_gt = cp_window_end
gt_cp = build_gt(data_cerebral_palsy.iloc[:], cp_start_idx_for_gt, num_blocks, OUT_LEN)

# =====================================================================================
# Save predictions
# =====================================================================================
column_names = columns_to_read
pd.DataFrame(next_steps_prediction,    columns=column_names)\
  .to_excel("Predictions/timestamps_typical_lstm.xlsx", index=False)
pd.DataFrame(next_steps_prediction_cp, columns=column_names)\
  .to_excel("Predictions/timestamps_cp_lstm.xlsx", index=False)

# =====================================================================================
# Plotting
# =====================================================================================
def plot_comparison(predicted, actual):
    """Plots actual vs predicted joint angles for 6 channels."""
    time = np.arange(actual.shape[0])
    labels_left  = ['LHipAngles', 'LKneeAngles', 'LAnkleAngles']
    labels_right = ['RHipAngles', 'RKneeAngles', 'RAnkleAngles']
    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

    # Left leg
    for i, name in enumerate(labels_left):
        axes[i].plot(time, actual[:, i],   label=f"Actual {name}",   color='blue')
        axes[i].plot(time, predicted[:, i],label=f"Predicted {name}",linestyle='dashed', color='red')
        axes[i].set_ylabel("Angle"); axes[i].legend(); axes[i].set_title(f"Comparison: {name}")

    # Right leg
    for i, name in enumerate(labels_right):
        j = i + 3
        axes[j].plot(time, actual[:, j],   label=f"Actual {name}",   color='blue')
        axes[j].plot(time, predicted[:, j],label=f"Predicted {name}",linestyle='dashed', color='red')
        axes[j].set_ylabel("Angle"); axes[j].legend(); axes[j].set_title(f"Comparison: {name}")

    axes[-1].set_xlabel("Time (Phase Progression)")
    plt.tight_layout()
    plt.show()

plot_comparison(next_steps_prediction,    gt_typ)
plot_comparison(next_steps_prediction_cp, gt_cp)
