import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__
import matplotlib.pyplot as plt
from matplotlib import pyplot

#This is for saving the model (There were issues with __version__ when calling the function save_model)
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

# ==============================================================================
# Create directories used for saving Predictions - Keras Models - and Scaler
# ==============================================================================
directory_predictions = "Predictions"
try:
    os.mkdir(directory_predictions)
    print(f"Directory '{directory_predictions}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_predictions}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_predictions}'.")
except Exception as e:
    print(f"An error occurred: {e}")
directory_models = "Saved_Models"
try:
    os.mkdir(directory_models)
    print(f"Directory '{directory_models}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_models}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_models}'.")
except Exception as e:
    print(f"An error occurred: {e}")
directory_scaler = "Scaler"
try:
    os.mkdir(directory_scaler)
    print(f"Directory '{directory_scaler}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_scaler}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_scaler}'.")
except Exception as e:
    print(f"An error occurred: {e}")
# ================================
# Load Typical Data
# ================================
columns_to_read = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                   'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']

# Load Typical Data
data = pd.read_excel('Data_Normal/randomized_data_healthy.xlsx', usecols=columns_to_read)
data.fillna(0, inplace=True)


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
delay = 25
right_leg_columns = [
    'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
for col in right_leg_columns:
    original = data_cerebral_palsy[col].values
    shifted = np.concatenate([original[-delay:], original[:-delay]])
    data_cerebral_palsy[col] = shifted

# Properly Scale the data to input in Neural Network
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_cp_scaled = scaler.fit_transform(data_cerebral_palsy)

#Save scaler for ROS node
joblib.dump(scaler, "Scaler/standard_scaler_cnn.save")

# ==========================
# Reshape Data for CNN
# ==========================
num_samples_typical = len(data_scaled) // 51  
num_samples_cp = len(data_cp_scaled) // 51

data_cnn_typical = data_scaled[:num_samples_typical * 51].reshape(num_samples_typical, 51, 6)
data_cnn_cp = data_cp_scaled[:num_samples_cp * 51].reshape(num_samples_cp, 51, 6)

# ==========================
# Split for Train / Val / Test
# ==========================
X_train = data_cnn_typical[:450]
X_test = data_cnn_cp[:450]
X_val = np.vstack((data_cnn_typical[450:500], data_cnn_cp[450:500]))  

# ==========================
# Separate Inputs (X) and Outputs (Y)
# ==========================
X_train_input, Y_train_output = X_train[:, :, :6], X_train[:, :, :6]  
X_val_input, Y_val_output = X_val[:, :, :6], X_val[:, :, :6]  
X_test_input, Y_test_output = X_test[:, :, :6], X_test[:, :, :6]  

# ==============================================
# Build CNN Model
# ==============================================
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(51, 6)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    Flatten(),
    Dense(64, activation='relu'),  # bottleneck
    Dense(51 * 6, activation='linear'),
    Reshape((51, 6))
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy', 'mae'])

# Check model summary
model.summary()

# Train model
history = model.fit(X_train_input, Y_train_output, epochs=200, batch_size=51, validation_data=(X_val_input, Y_val_output))

# ==============================================
# Evaluate Model
# ==============================================
def evaluate_model(model, X, Y, label):
    loss, accuracy, mae = model.evaluate(X, Y, verbose=1)
    print(f"🔹 {label} Evaluation:")
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, MAE: {mae:.4f}")
    print("="*90)

evaluate_model(model, X_train_input, Y_train_output, "Training Data")
evaluate_model(model, X_val_input, Y_val_output, "Validation Data")
evaluate_model(model, X_test_input, Y_test_output, "Testing Data")

model.save("Saved_Models/Timestamp_cnn_model.keras", include_optimizer=True)

# ==============================================
# Predict Next 4 Steps
# ==============================================
def predict_next_steps(model, last_51_rows, num_steps):
    """Predicts the next N steps using the trained CNN model."""
    predicted_steps = []
    current_input = last_51_rows.reshape(1, 51, 6)  # Ensure shape is correct

    for _ in range(num_steps):  
        predicted_step = model.predict(current_input)  # Predict full 6 features
        predicted_steps.append(predicted_step.reshape(51, 6))  
        current_input = predicted_step.reshape(1, 51, 6)  # Use predicted values as new input

    predicted_steps = np.vstack(predicted_steps)  # Stack predictions into a sequence
    return scaler.inverse_transform(predicted_steps)  # Convert back to original scale

# Get last known step for prediction
last_known_step = X_train_input[-1]
last_known_step_cp = X_test_input[-1] 
num_steps=4
next_steps_prediction = predict_next_steps(model, last_known_step, num_steps)
next_steps_prediction_cp = predict_next_steps(model, last_known_step_cp, num_steps=4)
# Get actual next 4 steps from dataset for plotting
actual_next_4_steps = data.iloc[len(data) - (num_steps * 51):, :].values  
actual_next_4_steps_cp = data_cerebral_palsy.iloc[len(data_cerebral_palsy) - (num_steps * 51):, :].values  

# Convert the numpy array to a pandas DataFrame
column_names = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
next_steps_prediction_df = pd.DataFrame(next_steps_prediction, columns=column_names)
next_steps_prediction_cp_df = pd.DataFrame(next_steps_prediction_cp, columns=column_names)


next_steps_prediction_df = next_steps_prediction_df[column_names]
next_steps_prediction_cp_df = next_steps_prediction_cp_df[column_names]

for col in columns_to_read:
    last_value = next_steps_prediction_cp_df[col].values[-1]
    first_value = next_steps_prediction_cp_df[col].values[0]
    divergence = np.abs(last_value - first_value)
    if divergence > 5:
        mean_value = (last_value + first_value) / 2
        next_steps_prediction_cp_df.loc[next_steps_prediction_cp_df.index[-1], col] = mean_value
        next_steps_prediction_cp_df.loc[next_steps_prediction_cp_df.index[0], col] = mean_value
    for col in columns_to_read:
        if divergence > 2:
            mean_value = (last_value + first_value) / 2
            next_steps_prediction_cp_df.loc[next_steps_prediction_cp_df.index[-1], col] = mean_value
            next_steps_prediction_cp_df.loc[next_steps_prediction_cp_df.index[0], col] = mean_value


# ==========================
# Save Predictions
# ==========================
predicted_df = pd.DataFrame(next_steps_prediction_df, columns=column_names)
predicted_df.to_excel("Predictions/timestamps_typical_cnn.xlsx", index=False)

predicted_df_cp = pd.DataFrame(next_steps_prediction_cp_df, columns=column_names)
predicted_df_cp.to_excel("Predictions/timestamps_cp_cnn.xlsx", index=False)

print("Prediction complete. Data saved")



# ====================================================
# Plot for Data (Typical_Data  ||  CP_Data)
# ====================================================
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

plot_comparison(next_steps_prediction_df.values, actual_next_4_steps)
plot_comparison(next_steps_prediction_cp_df.values, actual_next_4_steps_cp)

def plot_multiple_knee_predictions(actual_data, predicted_steps_list, label="Typical"):
    """
    Plots actual knee angles for multiple strides and overlays multiple predicted strides.
    
    Parameters:
    - actual_data: shape (N, 8)
    - predicted_steps_list: list of predicted arrays (each of shape (51, 8))
    """
    stride_length = 51
    time = np.arange(stride_length)

    # Define color map for predictions
    colors = pyplot.get_cmap('tab10', len(predicted_steps_list))

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].set_title(f"{label} - Left Knee Angles")
    axs[1].set_title(f"{label} - Right Knee Angles")

    # Plot actual left and right knee angles for each stride
    num_actual_strides = len(actual_data) // stride_length
    for i in range(num_actual_strides):
        start = i * stride_length
        end = start + stride_length
        axs[0].plot(time, actual_data[start:end, 2], color='lightgray', alpha=0.5)
        axs[1].plot(time, actual_data[start:end, 3], color='lightgray', alpha=0.5)

    # Plot each predicted stride
    for idx, pred in enumerate(predicted_steps_list):
        axs[0].plot(time, pred[:, 2], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)
        axs[1].plot(time, pred[:, 3], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)

    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Left Knee Angle")
    axs[0].legend()

    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Right Knee Angle")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].set_title(f"{label} - Left Hip Angles")
    axs[1].set_title(f"{label} - Right Hip Angles")

    # Plot actual left and right knee angles for each stride
    num_actual_strides = len(actual_data) // stride_length
    for i in range(num_actual_strides):
        start = i * stride_length
        end = start + stride_length
        axs[0].plot(time, actual_data[start:end, 0], color='lightgray', alpha=0.5)
        axs[1].plot(time, actual_data[start:end, 1], color='lightgray', alpha=0.5)

    # Plot each predicted stride
    for idx, pred in enumerate(predicted_steps_list):
        axs[0].plot(time, pred[:, 0], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)
        axs[1].plot(time, pred[:, 1], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)

    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Left Hip Angle")
    axs[0].legend()

    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Right Hip Angle")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].set_title(f"{label} - Left Ankle Angles")
    axs[1].set_title(f"{label} - Right Ankle Angles")

    # Plot actual left and right knee angles for each stride
    num_actual_strides = len(actual_data) // stride_length
    for i in range(num_actual_strides):
        start = i * stride_length
        end = start + stride_length
        axs[0].plot(time, actual_data[start:end, 4], color='lightgray', alpha=0.5)
        axs[1].plot(time, actual_data[start:end, 5], color='lightgray', alpha=0.5)

    # Plot each predicted stride
    for idx, pred in enumerate(predicted_steps_list):
        axs[0].plot(time, pred[:, 4], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)
        axs[1].plot(time, pred[:, 5], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)

    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Left Ankle Angle")
    axs[0].legend()

    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Right Ankle Angle")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


actual_next_20_steps = data.iloc[len(data) - 1020:, :].values
actual_next_20_steps_cp = data_cerebral_palsy.iloc[len(data_cerebral_palsy) - 1020:, :].values

predicted_strides = []
current_input = last_known_step.reshape(1, 51, 6)

for _ in range(5):
    pred = model.predict(current_input)
    predicted_strides.append(scaler.inverse_transform(pred[0]))
    current_input = pred  # Use last prediction as next input

predicted_strides_cp = []
last_known_step_cp = last_known_step_cp.reshape(1, 51, 6)

for _ in range(5):
    pred = model.predict(last_known_step_cp)
    predicted_strides_cp.append(scaler.inverse_transform(pred[0]))
    last_known_step_cp = pred  # Use last prediction as next input

plot_multiple_knee_predictions(actual_next_20_steps, predicted_strides, label="Typical")
plot_multiple_knee_predictions(actual_next_20_steps_cp, predicted_strides_cp, label="CP")


# # Plot mixed loss over epochs
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title("Combined Loss over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.show()


# # Plot MAE, MSE, and optionally RMSE over epochs
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

# # Plot MSE (Loss)
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training MSE')
# plt.plot(history.history['val_loss'], label='Validation MSE')
# plt.title("MSE (Loss) over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Mean Squared Error")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()