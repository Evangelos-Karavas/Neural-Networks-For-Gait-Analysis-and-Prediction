import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import tensorflow.python.keras as tf_keras
import matplotlib.pyplot as plt
from matplotlib import pyplot
#This is for saving the model (There were issues with __version__ when calling the function save_model)
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

# ================================
# Load Typical Data
# ================================
data_typical = "Data_Normal/randomized_data_healthy.xlsx"
columns_to_read = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                   'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
data = pd.read_excel(data_typical, usecols=columns_to_read)
data.fillna(0, inplace=True)  # Fill any missing values with 0

# ==============================================
# Load and Process Cerebral Palsy Data
# ==============================================
folder_path = 'Data_CP/'
file_counter = 0
data_cerebral_palsy = pd.DataFrame()

for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_counter += 1
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path, 'Data', usecols=columns_to_read, skiprows=[1, 2])
        df.fillna(0, inplace=True)

    data_cerebral_palsy = pd.concat([data_cerebral_palsy, df], ignore_index=True)
    if file_counter >= 500:
        break

# ==========================
# Divergence Fixing 
# ==========================
if not data_cerebral_palsy.empty:
    for col in columns_to_read:
        last_value = data_cerebral_palsy[col].values[-1]
        first_value = data_cerebral_palsy[col].values[0]
        divergence = np.abs(last_value - first_value)
        if divergence > 5:
            mean_value = (last_value + first_value) / 2
            data_cerebral_palsy.loc[data_cerebral_palsy.index[-1], col] = mean_value
            data_cerebral_palsy.loc[data_cerebral_palsy.index[0], col] = mean_value
        if divergence > 2:
            mean_value = (last_value + first_value) / 2
            data_cerebral_palsy.loc[data_cerebral_palsy.index[-1], col] = mean_value
            data_cerebral_palsy.loc[data_cerebral_palsy.index[0], col] = mean_value

data_cerebral_palsy.fillna(0, inplace=True)

# Wrap-around right leg by 25 steps (cyclic shift)
# delay = 25
# right_leg_columns = [
#     'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
# for col in right_leg_columns:
#     original = data_cerebral_palsy[col].values
#     shifted = np.concatenate([original[-delay:], original[:-delay]])
#     data_cerebral_palsy[col] = shifted

# Properly Scale the data to input in Neural Network and save for ROS Node
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)   
joblib.dump(scaler, "Scaler/standard_scaler_typical_lstm.save")
data_cp_scaled = scaler.transform(data_cerebral_palsy)
joblib.dump(scaler, "Scaler/standard_scaler_cp_lstm.save")

# ==========================
# Reshape Data for LSTM
# ==========================
num_samples_typical = len(data_scaled) // 51  
num_samples_cp = len(data_cp_scaled) // 51

data_lstm_typical = data_scaled[:num_samples_typical * 51].reshape(num_samples_typical, 51, 6)
data_lstm_cp = data_cp_scaled[:num_samples_cp * 51].reshape(num_samples_cp, 51, 6)

# === NEW/CHANGED: build (current_stride -> next_stride) pairs
def make_next_stride_pairs(arr):
    # X: stride i, Y: stride i+1
    if len(arr) < 2:
        raise ValueError("Need at least 2 strides to build next-stride pairs.")
    return arr[:-1], arr[1:]

X_typ, Y_typ = make_next_stride_pairs(data_lstm_typical)
X_cp,  Y_cp  = make_next_stride_pairs(data_lstm_cp)

# Split indices on these pairs (not the raw strides)
split_idx_typical = max(1, int(0.2 * len(X_typ)))
split_idx_cp      = max(1, int(0.2 * len(X_cp)))

# ==========================
# Split for Train / Val / Test
# ==========================
# Train on typical (holdout the first chunk for val), validate on mix, test on CP
X_train_input, Y_train_output = X_typ[split_idx_typical:], Y_typ[split_idx_typical:]
X_val_input  = np.vstack((X_typ[:split_idx_typical], X_cp[:split_idx_cp]))
Y_val_output = np.vstack((Y_typ[:split_idx_typical], Y_cp[:split_idx_cp]))
X_test_input, Y_test_output   = X_cp, Y_cp

# ==============================================
# Build LSTM Model  (seq -> next-seq)
# ==============================================
model = Sequential([
    LSTM(100, activation='tanh', return_sequences=True, input_shape=(51, 6)),
    LSTM(100, activation='tanh', return_sequences=True),
    Dropout(0.4),
    Dense(6, activation='linear')
])

# === NEW/CHANGED: accuracy is meaningless for regression
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae', RootMeanSquaredError()])

model.summary()

history = model.fit(
    X_train_input, Y_train_output,
    epochs=150,
    batch_size=102,
    validation_data=(X_val_input, Y_val_output),
    verbose=1
)
# ==============================================
# Evaluate Model
# ==============================================
def evaluate_model(model, X, Y, label):
    results = model.evaluate(X, Y, verbose=0)
    # order: [loss, mae, rmse]
    loss, mae, rmse = results[0], results[1], results[2]
    print(f"🔹 {label} Evaluation:")
    print(f"Loss (MSE): {loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print("="*90)

evaluate_model(model, X_train_input, Y_train_output, "Training Data")
evaluate_model(model, X_val_input,   Y_val_output,   "Validation Data")
evaluate_model(model, X_test_input,  Y_test_output,  "Testing Data")

model.save("Saved_Models/Timestamp_lstm_model.keras", include_optimizer=True)

# ==============================================
# Predict Next N Strides (autoregressive rollout)
# ==============================================
def predict_future_strides(model, last_stride_51x6, num_strides, scaler):
    """
    last_stride_51x6: scaled array (51,6) to start from
    returns: (num_strides*51, 6) in ORIGINAL scale
    """
    preds_scaled = []
    current = last_stride_51x6.reshape(1, 51, 6)
    for _ in range(num_strides):
        next_stride = model.predict(current, verbose=0)  # (1,51,6)
        preds_scaled.append(next_stride[0])              # (51,6)
        current = next_stride                            # feed prediction as next input
    preds_scaled = np.vstack(preds_scaled)               # (num_strides*51, 6)
    return scaler.inverse_transform(preds_scaled)

# Choose starting strides (already scaled)
last_known_typical = X_train_input[-1]      # (51,6)
last_known_cp      = X_test_input[-1]       # (51,6)

num_steps = 4
next_steps_prediction     = predict_future_strides(model, last_known_typical, num_steps, scaler)
next_steps_prediction_cp  = predict_future_strides(model, last_known_cp,     num_steps, scaler)

def rollout_from_index(data_lstm, i_start, num_strides, model, scaler):
    """
    data_lstm: (N_strides, 51, 6)  scaled array
    i_start: index of the current stride to start from (must satisfy i_start + num_strides <= N_strides - 1)
    returns:
        pred_orig: (num_strides*51, 6) predictions in ORIGINAL scale
        gt_orig:   (num_strides*51, 6) ground-truth in ORIGINAL scale
    """
    stride = 51
    # 1) Starting stride (scaled)
    start_stride = data_lstm[i_start]  # (51,6)

    # 2) Predict future strides (scaled -> orig)
    pred_orig = predict_future_strides(model, start_stride, num_strides, scaler)  # (num_strides*51, 6)

    # 3) Build ground truth directly from the original (unscaled) data
    #    Rebuild per-stride view from the original dataframe that matches 'data_lstm'
    #    NOTE: use the same original dataframe you scaled from (data for typical, data_cerebral_palsy for CP)
    return pred_orig

# ----- TYPICAL: choose a safe start index -----
stride = 51
num_strides = 4

# Rebuild stride-wise ORIGINAL arrays for ground truth
num_samples_typical = len(data) // stride
typ_orig_strides = data.iloc[:num_samples_typical*stride, :].to_numpy().reshape(num_samples_typical, stride, 6)

# data_lstm_typical is the *scaled* version with same number of strides
N_typ = data_lstm_typical.shape[0]
i_start_typ = N_typ - (num_strides + 1)          # leaves exactly num_strides future strides: i+1 ... i+num_strides
assert i_start_typ >= 0, "Not enough strides to leave room for ground truth."

# Predict from this start
next_steps_prediction = predict_future_strides(model, data_lstm_typical[i_start_typ], num_strides, scaler)

# Ground truth next strides in ORIGINAL scale
gt_typ = typ_orig_strides[i_start_typ+1 : i_start_typ+1+num_strides].reshape(num_strides*stride, 6)

assert next_steps_prediction.shape == gt_typ.shape

# ----- CP: same logic -----
num_samples_cp = len(data_cerebral_palsy) // stride
cp_orig_strides = data_cerebral_palsy.iloc[:num_samples_cp*stride, :].to_numpy().reshape(num_samples_cp, stride, 6)

N_cp = data_lstm_cp.shape[0]
i_start_cp = N_cp - (num_strides + 1)
assert i_start_cp >= 0, "Not enough CP strides to leave room for ground truth."

next_steps_prediction_cp = predict_future_strides(model, data_lstm_cp[i_start_cp], num_strides, scaler)
gt_cp = cp_orig_strides[i_start_cp+1 : i_start_cp+1+num_strides].reshape(num_strides*stride, 6)

assert next_steps_prediction_cp.shape == gt_cp.shape

# ----- Save & plot -----
column_names = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']

next_steps_prediction_df    = pd.DataFrame(next_steps_prediction,    columns=column_names)
next_steps_prediction_cp_df = pd.DataFrame(next_steps_prediction_cp, columns=column_names)

os.makedirs("Predictions", exist_ok=True)
next_steps_prediction_df.to_excel("Predictions/timestamps_typical_lstm.xlsx", index=False)
next_steps_prediction_cp_df.to_excel("Predictions/timestamps_cp_lstm.xlsx", index=False)



# # Get actual next 4 steps from dataset for plotting
# actual_next_4_steps = data.iloc[len(data) - (num_steps * 51):, :].values  
# actual_next_4_steps_cp = data_cerebral_palsy.iloc[len(data_cerebral_palsy) - (num_steps * 51):, :].values  

# # Convert the numpy array to a pandas DataFrame
# column_names = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
#                 'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
# next_steps_prediction_df = pd.DataFrame(next_steps_prediction, columns=column_names)
# next_steps_prediction_cp_df = pd.DataFrame(next_steps_prediction_cp, columns=column_names)


# next_steps_prediction_df = next_steps_prediction_df[column_names]
# next_steps_prediction_cp_df = next_steps_prediction_cp_df[column_names]

# # ==========================
# # Save Predictions
# # ==========================
# predicted_df = pd.DataFrame(next_steps_prediction_df)
# predicted_df.to_excel("Predictions/timestamps_typical_lstm.xlsx", index=False)

# predicted_df_cp = pd.DataFrame(next_steps_prediction_cp_df)
# predicted_df_cp.to_excel("Predictions/timestamps_cp_lstm.xlsx", index=False)
# print("Prediction complete. Data saved")



# # ====================================================
# # Plot for Data (Typical_Data  ||  CP_Data)
# # ====================================================
def plot_comparison(predicted, actual):
    """Plots actual vs predicted joint angles."""
    time = np.arange(actual.shape[0])  # Time index for the dataset

    labels_left = ['LHipAngles', 'LKneeAngles', 'LAnkleAngles']
    labels_right = ['RHipAngles', 'RKneeAngles', 'RAnkleAngles']

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

    # Left Leg Joint Angles (Columns 0,1,2)
    for i in range(len(labels_left)):
        axes[i].plot(time, actual[:, i], label=f"Actual {labels_left[i]}", color='blue')
        axes[i].plot(time, predicted[:, i], label=f"Predicted {labels_left[i]}", linestyle='dashed', color='red')
        axes[i].set_ylabel("Angle")
        axes[i].legend()
        axes[i].set_title(f"Comparison: {labels_left[i]}")

    # Right Leg Joint Angles (Columns 3,4,5)
    for i in range(len(labels_right)):
        axes[i + 3].plot(time, actual[:, i + 3], label=f"Actual {labels_right[i]}", color='blue')
        axes[i + 3].plot(time, predicted[:, i + 3], label=f"Predicted {labels_right[i]}", linestyle='dashed', color='red')
        axes[i + 3].set_ylabel("Angle")
        axes[i + 3].legend()
        axes[i + 3].set_title(f"Comparison: {labels_right[i]}")
        
    axes[-1].set_xlabel("Time (Phase Progression)")

    plt.tight_layout()
    plt.show()

# plot_comparison(next_steps_prediction_df.values, actual_next_4_steps)
# plot_comparison(next_steps_prediction_cp_df.values, actual_next_4_steps_cp)



# Plot against the correct ground truth
plot_comparison(next_steps_prediction_df.values,    gt_typ)
plot_comparison(next_steps_prediction_cp_df.values, gt_cp)



# def plot_multiple_knee_predictions(actual_data, predicted_steps_list, label="Typical"):
#     """
#     Plots actual knee angles for multiple strides and overlays multiple predicted strides.
    
#     Parameters:
#     - actual_data: shape (N, 8)
#     - predicted_steps_list: list of predicted arrays (each of shape (51, 8))
#     """
#     stride_length = 51
#     time = np.arange(stride_length)

#     # Define color map for predictions
#     colors = pyplot.get_cmap('tab10', len(predicted_steps_list))

#     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#     axs[0].set_title(f"{label} - Left Knee Angles")
#     axs[1].set_title(f"{label} - Right Knee Angles")

#     # Plot actual left and right knee angles for each stride
#     num_actual_strides = len(actual_data) // stride_length
#     for i in range(num_actual_strides):
#         start = i * stride_length
#         end = start + stride_length
#         axs[0].plot(time, actual_data[start:end, 2], color='lightgray', alpha=0.5)
#         axs[1].plot(time, actual_data[start:end, 3], color='lightgray', alpha=0.5)

#     # Plot each predicted stride
#     for idx, pred in enumerate(predicted_steps_list):
#         axs[0].plot(time, pred[:, 2], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)
#         axs[1].plot(time, pred[:, 3], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)

#     axs[0].set_xlabel("Timestep")
#     axs[0].set_ylabel("Left Knee Angle")
#     axs[0].legend()

#     axs[1].set_xlabel("Timestep")
#     axs[1].set_ylabel("Right Knee Angle")
#     axs[1].legend()

#     plt.tight_layout()
#     plt.show()

#     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#     axs[0].set_title(f"{label} - Left Hip Angles")
#     axs[1].set_title(f"{label} - Right Hip Angles")

#     # Plot actual left and right knee angles for each stride
#     num_actual_strides = len(actual_data) // stride_length
#     for i in range(num_actual_strides):
#         start = i * stride_length
#         end = start + stride_length
#         axs[0].plot(time, actual_data[start:end, 0], color='lightgray', alpha=0.5)
#         axs[1].plot(time, actual_data[start:end, 1], color='lightgray', alpha=0.5)

#     # Plot each predicted stride
#     for idx, pred in enumerate(predicted_steps_list):
#         axs[0].plot(time, pred[:, 0], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)
#         axs[1].plot(time, pred[:, 1], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)

#     axs[0].set_xlabel("Timestep")
#     axs[0].set_ylabel("Left Hip Angle")
#     axs[0].legend()

#     axs[1].set_xlabel("Timestep")
#     axs[1].set_ylabel("Right Hip Angle")
#     axs[1].legend()

#     plt.tight_layout()
#     plt.show()

#     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#     axs[0].set_title(f"{label} - Left Ankle Angles")
#     axs[1].set_title(f"{label} - Right Ankle Angles")

#     # Plot actual left and right knee angles for each stride
#     num_actual_strides = len(actual_data) // stride_length
#     for i in range(num_actual_strides):
#         start = i * stride_length
#         end = start + stride_length
#         axs[0].plot(time, actual_data[start:end, 4], color='lightgray', alpha=0.5)
#         axs[1].plot(time, actual_data[start:end, 5], color='lightgray', alpha=0.5)

#     # Plot each predicted stride
#     for idx, pred in enumerate(predicted_steps_list):
#         axs[0].plot(time, pred[:, 4], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)
#         axs[1].plot(time, pred[:, 5], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)

#     axs[0].set_xlabel("Timestep")
#     axs[0].set_ylabel("Left Ankle Angle")
#     axs[0].legend()

#     axs[1].set_xlabel("Timestep")
#     axs[1].set_ylabel("Right Ankle Angle")
#     axs[1].legend()

#     plt.tight_layout()
#     plt.show()


# actual_next_20_steps = data.iloc[len(data) - 1020:, :].values
# actual_next_20_steps_cp = data_cerebral_palsy.iloc[len(data_cerebral_palsy) - 1020:, :].values

# predicted_strides = []
# current_input = last_known_step.reshape(1, 51, 6)

# for _ in range(5):
#     pred = model.predict(current_input)
#     predicted_strides.append(scaler.inverse_transform(pred[0]))
#     current_input = pred  # Use last prediction as next input

# predicted_strides_cp = []
# last_known_step_cp = last_known_step_cp.reshape(1, 51, 6)

# for _ in range(5):
#     pred = model.predict(last_known_step_cp)
#     predicted_strides_cp.append(scaler.inverse_transform(pred[0]))
#     last_known_step_cp = pred  # Use last prediction as next input

# plot_multiple_knee_predictions(actual_next_20_steps, predicted_strides, label="Typical")
# plot_multiple_knee_predictions(actual_next_20_steps_cp, predicted_strides_cp, label="CP")

# # Plot mixed loss over epochs
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title("Combined Loss over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.show()


# # Plot MAE and RMSE over epochs
# plt.figure(figsize=(12, 5))

# # Plot MAE
# plt.subplot(1, 2, 1)
# plt.plot(history.history['mae'], label='Training MAE')
# plt.plot(history.history['val_mae'], label='Validation MAE')
# plt.title("MAE over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Mean Absolute Error")
# plt.legend()
# plt.grid(True)

# # Plot RMSE
# plt.subplot(1, 2, 2)
# plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
# plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
# plt.title("RMSE over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Root Mean Squared Error")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()