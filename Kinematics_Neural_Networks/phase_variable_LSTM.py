import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, savgol_filter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers.recurrent import LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
#This is for saving the model (There were issues with __version__ when calling the function save_model)
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__


def combined_loss(y_true, y_pred):
    # Mean Squared Error
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # If output is 2D (no sequence), skip temporal terms
    if len(y_pred.shape) < 3:
        return mse

    # Smoothness Penalty (encourages temporal consistency)
    smoothness = tf.reduce_mean(tf.square(y_pred[:, 1:, :] - y_pred[:, :-1, :]))

    # Swing-phase Mutual Exclusion Penalty
    phase_left = y_pred[:, :, 0]   # PhaseVariable_Left
    phase_right = y_pred[:, :, 1]  # PhaseVariable_Right

    # Boolean masks: 1 if in swing phase (> 0.5), else 0
    left_swing = tf.cast(phase_left > 0.5, tf.float32)
    right_swing = tf.cast(phase_right > 0.5, tf.float32)

    # Penalize overlap (both in swing at the same time)
    simultaneous_swing = tf.reduce_mean(left_swing * right_swing)

    # Combine all terms with respective weights
    total_loss = (
        mse +
        0.01 * smoothness +
        1.0 * simultaneous_swing  # Adjust weight (e.g., 0.5 if too aggressive)
    )
    return total_loss

# =============================================
# Compute the Phase Variable for Processed Data
# =============================================
def compute_phase_variable(df, thigh_col_left, thigh_col_right, foot_contact_col_left, foot_contact_col_right):
    """Computes the phase variable for both legs based on thigh angles and foot contact, ensuring codependence."""
    theta_touchdown_left = df[thigh_col_left].iloc[0]  # Thigh angle at touchdown (left)
    theta_min_left = df[thigh_col_left].min()  # Minimum thigh angle (left)
    theta_touchdown_right = df[thigh_col_right].iloc[0]  # Thigh angle at touchdown (right)
    theta_min_right = df[thigh_col_right].min()  # Minimum thigh angle (right)
    c = 0.53  # Default scaling constant
    s_left = np.zeros(len(df))
    s_right = np.zeros(len(df))
    
    foot_contact_binary_left = np.zeros(len(df))  # Initialize stance phase indicator (left)
    foot_contact_binary_right = np.zeros(len(df))  # Initialize stance phase indicator (right)
    for i in range(0, len(df), 51):
        foot_contact_percent_left = df[foot_contact_col_left].iloc[i]  
        stance_rows_left = int(51 * (foot_contact_percent_left / 100))  
        foot_contact_binary_left[i:i + stance_rows_left] = 1  
        
        foot_contact_percent_right = df[foot_contact_col_right].iloc[i]  
        stance_rows_right = int(51 * (foot_contact_percent_right / 100))  
        foot_contact_binary_right[i:i + stance_rows_right] = 1  

    stride_count_left = 0
    stride_count_right = 0

    for i in range(len(df)):
        if foot_contact_binary_left[i] == 1 and foot_contact_binary_right[i] == 0:  # Stance phase (left) and swing phase (right)
            s_left[i] = stride_count_left + ((theta_touchdown_left - df[thigh_col_left].iloc[i]) / (theta_touchdown_left - theta_min_left)) * c
            theta_m_right = df[thigh_col_right].min()
            if i > 0:
                s_right[i] = max(s_right[i-1], stride_count_right + 1 + ((1 - (s_right[i-1] % 1)) / (theta_touchdown_right - theta_m_right)) * (df[thigh_col_right].iloc[i] - theta_touchdown_right))
            else:
                s_right[i] = stride_count_right + 1 + ((1 - (s_right[i-1] % 1)) / (theta_touchdown_right - theta_m_right)) * (df[thigh_col_right].iloc[i] - theta_touchdown_right)
        elif foot_contact_binary_right[i] == 1 and foot_contact_binary_left[i] == 0:  # Stance phase (right) and swing phase (left)
            s_right[i] = stride_count_right + ((theta_touchdown_right - df[thigh_col_right].iloc[i]) / (theta_touchdown_right - theta_min_right)) * c
            theta_m_left = df[thigh_col_left].min()
            if i > 0:
                s_left[i] = max(s_left[i-1], stride_count_left + 1 + ((1 - (s_left[i-1] % 1)) / (theta_touchdown_left - theta_m_left)) * (df[thigh_col_left].iloc[i] - theta_touchdown_left))
            else:
                s_left[i] = stride_count_left + 1 + ((1 - (s_left[i-1] % 1)) / (theta_touchdown_left - theta_m_left)) * (df[thigh_col_left].iloc[i] - theta_touchdown_left)
        elif foot_contact_binary_left[i] == 1 and foot_contact_binary_right[i] == 1:  # Both legs in stance phase
            s_left[i] = stride_count_left + ((theta_touchdown_left - df[thigh_col_left].iloc[i]) / (theta_touchdown_left - theta_min_left)) * c
            s_right[i] = stride_count_right + ((theta_touchdown_right - df[thigh_col_right].iloc[i]) / (theta_touchdown_right - theta_min_right)) * c
        elif foot_contact_binary_left[i] == 0 and foot_contact_binary_right[i] == 0:  # Both legs in swing phase
            theta_m_left = df[thigh_col_left].min()
            if i > 0:
                s_left[i] = max(s_left[i-1], stride_count_left + 1 + ((1 - (s_left[i-1] % 1)) / (theta_touchdown_left - theta_m_left)) * (df[thigh_col_left].iloc[i] - theta_touchdown_left))
            else:
                s_left[i] = stride_count_left + 1 + ((1 - (s_left[i-1] % 1)) / (theta_touchdown_left - theta_m_left)) * (df[thigh_col_left].iloc[i] - theta_touchdown_left)
            theta_m_right = df[thigh_col_right].min()
            if i > 0:
                s_right[i] = max(s_right[i-1], stride_count_right + 1 + ((1 - (s_right[i-1] % 1)) / (theta_touchdown_right - theta_m_right)) * (df[thigh_col_right].iloc[i] - theta_touchdown_right))
            else:
                s_right[i] = stride_count_right + 1 + ((1 - (s_right[i-1] % 1)) / (theta_touchdown_right - theta_m_right)) * (df[thigh_col_right].iloc[i] - theta_touchdown_right)
        if s_left[i] <= s_left[i-1] - 0.5:
            s_left[i] = s_left[i-1]
        if s_right[i] <= s_right[i-1] - 0.5:
            s_right[i] = s_right[i-1]

        # Update stride count
        if i > 0:
            if foot_contact_binary_left[i] == 1 and foot_contact_binary_left[i-1] == 0:
                stride_count_left += 1
            if foot_contact_binary_right[i] == 1 and foot_contact_binary_right[i-1] == 0:
                stride_count_right += 1
    
    return s_left, s_right
# =============================================
# Compute the Phase Variable for Predicted Data
# =============================================
def compute_phase_variable_prediction(df, thigh_col_left, thigh_col_right):
    """Computes the phase variable for predicted data"""
    theta_touchdown_left = df[thigh_col_left].iloc[0]  # Thigh angle at touchdown (left)
    theta_min_left = df[thigh_col_left].min()  # Minimum thigh angle (left)
    theta_touchdown_right = df[thigh_col_right].iloc[0]  # Thigh angle at touchdown (right)
    theta_min_right = df[thigh_col_right].min()  # Minimum thigh angle (right)
    c = 0.53  # Default scaling constant
    s_left = np.zeros(len(df))
    s_right = np.zeros(len(df))
    
    foot_contact_binary_left = np.zeros(len(df))  # Initialize stance phase indicator (left)
    foot_contact_binary_right = np.zeros(len(df))  # Initialize stance phase indicator (right)
    for i in range(0, len(df), 51):
        foot_contact_percent_left = 66.19
        stance_rows_left = int(51 * (foot_contact_percent_left / 100))  
        foot_contact_binary_left[i:i + stance_rows_left] = 1  
        foot_contact_percent_right = 64.03
        stance_rows_right = int(51 * (foot_contact_percent_right / 100))  
        foot_contact_binary_right[i:i + stance_rows_right] = 1  
    for i in range(len(df)):
        if foot_contact_binary_left[i] == 1 and foot_contact_binary_right[i] == 0:  # Stance phase (left) and swing phase (right)
            s_left[i] = ((theta_touchdown_left - df[thigh_col_left].iloc[i]) / (theta_touchdown_left - theta_min_left)) * c
            theta_m_right = df[thigh_col_right].min()
            s_right[i] = 1 + ((1 - s_right[i-1]) / (theta_touchdown_right - theta_m_right)) * (df[thigh_col_right].iloc[i] - theta_touchdown_right)
        elif foot_contact_binary_right[i] == 1 and foot_contact_binary_left[i] == 0:  # Stance phase (right) and swing phase (left)
            s_right[i] = ((theta_touchdown_right - df[thigh_col_right].iloc[i]) / (theta_touchdown_right - theta_min_right)) * c
            theta_m_left = df[thigh_col_left].min()
            s_left[i] = 1 + ((1 - s_left[i-1]) / (theta_touchdown_left - theta_m_left)) * (df[thigh_col_left].iloc[i] - theta_touchdown_left)
        elif foot_contact_binary_left[i] == 1 and foot_contact_binary_right[i] == 1:  # Both legs in stance phase
            s_left[i] = ((theta_touchdown_left - df[thigh_col_left].iloc[i]) / (theta_touchdown_left - theta_min_left)) * c
            s_right[i] = ((theta_touchdown_right - df[thigh_col_right].iloc[i]) / (theta_touchdown_right - theta_min_right)) * c
        elif foot_contact_binary_left[i] == 0 and foot_contact_binary_right[i] == 0:  # Both legs in swing phase
            theta_m_left = df[thigh_col_left].min()
            s_left[i] = 1 + ((1 - s_left[i-1]) / (theta_touchdown_left - theta_m_left)) * (df[thigh_col_left].iloc[i] - theta_touchdown_left)
            theta_m_right = df[thigh_col_right].min()
            s_right[i] = 1 + ((1 - s_right[i-1]) / (theta_touchdown_right - theta_m_right)) * (df[thigh_col_right].iloc[i] - theta_touchdown_right)
    return s_left, s_right

# =====================================================================================================
# Prepare Data for Neural Network (Both Left and Right Legs) for Typical and CP Data
# =====================================================================================================
def prepare_neural_network_data(df):
    """Computes phase variables for both left and right legs and formats the data for LSTM."""
    df['PhaseVariable_Left'], df['PhaseVariable_Right'] = compute_phase_variable(df, 'LHipAngles (1)', 'RHipAngles (1)', 'Left Foot Off', 'Right Foot Off')

    df = df[['PhaseVariable_Left', 'PhaseVariable_Right', 
             'LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
             'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']]
    return df

# ==============================================
#  Butter-worth Filtering - Smoothing
# ==============================================
def butterworth_filter(data, cutoff, fs, order, filter_type):
    """
    Applies a Butterworth filter (low-pass or high-pass).
    - cutoff: Cutoff frequency (Hz)
    - fs: Sampling frequency (Hz)
    - order: Filter order
    - filter_type: 'low' for low-pass, 'high' for high-pass
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

def savitzky_golay_filter(data, window_size=7, poly_order=3):
    """
    Applies Savitzky-Golay filter for trajectory smoothing.
    - window_size: Must be an odd number
    - poly_order: Degree of polynomial to fit
    """
    return savgol_filter(data, window_length=window_size, polyorder=poly_order)



# ==============================================
# Load and Process Typical Data
# ==============================================
columns_to_read = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)', 'Left Foot Off',
                   'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)', 'Right Foot Off']

# Load Typical Data
data = pd.read_excel('Data_Normal/randomized_data_healthy.xlsx', usecols=columns_to_read)
if not data.empty:
    for col in columns_to_read:
        last_value = data[col].values[-1]
        first_value = data[col].values[0]
        divergence = np.abs(last_value - first_value)
        if divergence > 5:
            mean_value = (last_value + first_value) / 2
            data.loc[data.index[-1], col] = mean_value
            data.loc[data.index[0], col] = mean_value
            for col in columns_to_read:
                last_value = data[col].values[-1]
                first_value = data[col].values[0]
                divergence = np.abs(last_value - first_value)
                if divergence > 2:
                    mean_value = (last_value + first_value) / 2
                    data.loc[data.index[-1], col] = mean_value
                    data.loc[data.index[0], col] = mean_value
data.fillna(0, inplace=True)  # Fill any missing values with 0

folder_path = 'Data_CP/'
file_counter = 0
data_cerebral_palsy = pd.DataFrame()
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_counter += 1
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path, 'Data', usecols=columns_to_read, skiprows=[1, 2])
        data_cerebral_palsy = pd.concat([data_cerebral_palsy, df], ignore_index=True)
        if file_counter >= 500:
            break
# Check for divergence and fix by averaging first and last values
if not data_cerebral_palsy.empty:
    for col in columns_to_read:
        last_value = data_cerebral_palsy[col].values[-1]
        first_value = data_cerebral_palsy[col].values[0]
        divergence = np.abs(last_value - first_value)
        if divergence > 5:
            mean_value = (last_value + first_value) / 2
            data_cerebral_palsy.loc[data_cerebral_palsy.index[-1], col] = mean_value
            data_cerebral_palsy.loc[data_cerebral_palsy.index[0], col] = mean_value
            for col in columns_to_read:
                last_value = data_cerebral_palsy[col].values[-1]
                first_value = data_cerebral_palsy[col].values[0]
                divergence = np.abs(last_value - first_value)
                if divergence > 2:
                    mean_value = (last_value + first_value) / 2
                    data_cerebral_palsy.loc[data_cerebral_palsy.index[-1], col] = mean_value
                    data_cerebral_palsy.loc[data_cerebral_palsy.index[0], col] = mean_value
data_cerebral_palsy.fillna(0, inplace=True)  # Fill any missing values with 0

processed_data = prepare_neural_network_data(data)
processed_cp_data = prepare_neural_network_data(data_cerebral_palsy)
# ==========================================
# Normalize Data and Train-Test-Val Split
# ==========================================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(processed_data)
data_cp_scaled = scaler.fit_transform(processed_cp_data)

# ==========================
# Reshape Data for LSTM
# ==========================
num_samples_typical = len(data_scaled) // 51  
num_samples_cp = len(data_cp_scaled) // 51

data_lstm_typical = data_scaled[:num_samples_typical * 51].reshape(num_samples_typical, 51, 8)
data_lstm_cp = data_cp_scaled[:num_samples_cp * 51].reshape(num_samples_cp, 51, 8)

# ==========================
# Split for Train / Val / Test
# ==========================
X_train = data_lstm_typical[:450]
X_test = data_lstm_cp[:450]
X_val = np.vstack((data_lstm_typical[450:500], data_lstm_cp[450:500]))  

# ==========================
# Separate Inputs (X) and Outputs (Y)
# ==========================
X_train_input, Y_train_output = X_train[:, :, :8], X_train[:, :, :8]  
X_val_input, Y_val_output = X_val[:, :, :8], X_val[:, :, :8]  
X_test_input, Y_test_output = X_test[:, :, :8], X_test[:, :, :8]  
Y_train_output = Y_train_output[:, -1, :]
Y_val_output = Y_val_output[:, -1, :]
Y_test_output = Y_test_output[:, -1, :]

def thresholded_accuracy(y_true, y_pred, threshold=0.2):
    """
    Computes thresholded accuracy: the percentage of predicted values within a specified degree threshold.
    """
    error = np.abs(y_true - y_pred)
    within_threshold = np.mean(error <= threshold)
    return within_threshold * 100

# ====================================================
# Build LSTM Model
# ====================================================
model = Sequential([
    LSTM(100, activation='tanh', return_sequences=True, input_shape=(51, 8)), # Activation: tanh, sigmoid, relu
    LSTM(100, activation='tanh'),
    Dropout(0.2),
    Dense(8, activation='linear')])

model.compile(optimizer=Adam(learning_rate=0.003), loss='mse', metrics=['mae', 'Accuracy'])

# Train Model
history = model.fit(X_train_input, Y_train_output, epochs=150, batch_size=102, validation_data=(X_val_input, Y_val_output))

# ==============================================
# Evaluate Model
# ==============================================
def evaluate_model(model, X, Y, label, threshold=0.2):
    loss, mae, accuracy = model.evaluate(X, Y, verbose=1)
    Y_pred = model.predict(X)

    thresh_acc = thresholded_accuracy(Y, Y_pred, threshold=threshold)

    print(f"🔹 {label} Evaluation:")
    print(f"LSTM Loss (MSE): {loss:.4f}, MAE: {mae:.4f}, Accuracy (Keras): {accuracy:.4f}")
    print(f"LSTM Thresholded Accuracy (±{threshold}°): {thresh_acc:.2f}%")
    print("=" * 90)

evaluate_model(model, X_train_input, Y_train_output, "Training Data", threshold=0.2)
evaluate_model(model, X_val_input, Y_val_output, "Validation Data", threshold=0.2)
evaluate_model(model, X_test_input, Y_test_output, "Testing Data", threshold=0.2)

model.save("Saved_Models/PV_lstm_model.keras")

# ==============================================
# Predict Next 4 Steps
# ==============================================
def predict_next_steps(model, last_51_rows, num_steps=408):
    """
    Predicts the next N timesteps by feeding in 51-timestep input windows and getting 1 timestep output at a time.
    Assumes model outputs shape (1, 8).
    """
    predicted_sequence = []
    current_window = last_51_rows.copy()  # shape (51, 8)

    for _ in range(num_steps * 1):  # Change this if you want more or fewer steps
        input_batch = current_window.reshape(1, 51, 8)  # shape (1, 51, 8)
        predicted_step = model.predict(input_batch, verbose=0)  # shape (1, 8)

        next_step = predicted_step[0]  # shape (8,)
        predicted_sequence.append(next_step)

        # Update input window by sliding and appending the new prediction
        current_window = np.vstack([current_window[1:], next_step])

    predicted_sequence = np.array(predicted_sequence)  # shape (num_steps, 8)
    return scaler.inverse_transform(predicted_sequence)

# Get last known step for prediction
last_known_step = X_train_input[-1]
last_known_step_cp = X_test_input[-1] 
num_steps=204
next_steps_prediction = predict_next_steps(model, last_known_step, num_steps)
next_steps_prediction_cp = predict_next_steps(model, last_known_step_cp, num_steps)

# Get actual next 4 steps from dataset for plotting
actual_next_4_steps = processed_data.iloc[len(processed_data) - 204:, :].values  
actual_next_4_steps_cp = processed_cp_data.iloc[len(processed_cp_data) - 204:, :].values  

# Convert the numpy array to a pandas DataFrame
column_names = ['PhaseVariable_Left', 'PhaseVariable_Right', 
                'LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
column_names_no_pv = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                      'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']

next_steps_prediction_df = pd.DataFrame(next_steps_prediction, columns=column_names)
next_steps_prediction_cp_df = pd.DataFrame(next_steps_prediction_cp, columns=column_names)

# Compute phase variables for the predicted data
next_steps_prediction_df['PhaseVariable_Left'], next_steps_prediction_df['PhaseVariable_Right'] = compute_phase_variable_prediction(next_steps_prediction_df, 'LHipAngles (1)', 'RHipAngles (1)')
next_steps_prediction_cp_df['PhaseVariable_Left'], next_steps_prediction_cp_df['PhaseVariable_Right'] = compute_phase_variable_prediction(next_steps_prediction_cp_df, 'LHipAngles (1)', 'RHipAngles (1)')

next_steps_prediction_df = next_steps_prediction_df[column_names]
next_steps_prediction_cp_df = next_steps_prediction_cp_df[column_names]


for col in column_names:
    for i in range(204):
        last_value = next_steps_prediction_df[col].values[i-1]
        first_value = next_steps_prediction_df[col].values[i]
        divergence = np.abs(last_value - first_value)
        if divergence > 2:
            mean_value = (last_value + first_value) / 2
            next_steps_prediction_df[col].values[i] = mean_value
            next_steps_prediction_df[col].values[i-1] = mean_value
        last_value = next_steps_prediction_cp_df[col].values[i-1]
        first_value = next_steps_prediction_cp_df[col].values[i]
        divergence = np.abs(last_value - first_value)
        if divergence > 2:
            mean_value = (last_value + first_value) / 2
            next_steps_prediction_cp_df[col].values[i] = mean_value
            next_steps_prediction_cp_df[col].values[i-1] = mean_value


# ==========================
# Save Predictions
# ==========================
predicted_df = pd.DataFrame(next_steps_prediction_df, columns=column_names)
predicted_df.to_excel("Predictions/PV_typical_lstm.xlsx", index=False)

predicted_df_cp = pd.DataFrame(next_steps_prediction_cp_df, columns=column_names)
predicted_df_cp.to_excel("Predictions/PV_cp_lstm.xlsx", index=False)
print("Prediction complete. Data saved")

# ==============================================================================
# All Plots for Phase Variable || Predicted Angles || Loss - Accuracy of Model
# ==============================================================================


def plot_comparison(predicted, actual):
    """
    Plots actual vs predicted joint angles and phase variables.
    Assumes both predicted and actual are of shape (N, 8)
    """
    time = np.arange(actual.shape[0])  # Time index for the dataset
    labels_left = ['LHipAngles', 'LKneeAngles', 'LAnkleAngles']
    labels_right = ['RHipAngles', 'RKneeAngles', 'RAnkleAngles']

    fig, axes = plt.subplots(8, 1, figsize=(12, 16), sharex=True)

    # Phase Variable - Left
    axes[0].plot(time, actual[:, 0], label="Actual PV Left", color='blue')
    axes[0].plot(time, predicted[:, 0], label="Predicted PV Left", linestyle='--', color='red')
    axes[0].set_ylabel("PV Left")
    axes[0].set_ylim([0, 1.2])
    axes[0].legend()

    # Phase Variable - Right
    axes[1].plot(time, actual[:, 1], label="Actual PV Right", color='blue')
    axes[1].plot(time, predicted[:, 1], label="Predicted PV Right", linestyle='--', color='red')
    axes[1].set_ylabel("PV Right")
    axes[1].set_ylim([0, 1.2])
    axes[1].legend()

    # Joint angles: Left side
    for i in range(len(labels_left)):
        axes[i + 2].plot(time, actual[:, i + 2], label=f"Actual {labels_left[i]}", color='blue')
        axes[i + 2].plot(time, predicted[:, i + 2], label=f"Predicted {labels_left[i]}", linestyle='--', color='red')
        axes[i + 2].set_ylabel("Angle")
        axes[i + 2].set_title(f"{labels_left[i]}")
        axes[i + 2].legend()

    # Joint angles: Right side
    for i in range(len(labels_right)):
        axes[i + 5].plot(time, actual[:, i + 5], label=f"Actual {labels_right[i]}", color='blue')
        axes[i + 5].plot(time, predicted[:, i + 5], label=f"Predicted {labels_right[i]}", linestyle='--', color='red')
        axes[i + 5].set_ylabel("Angle")
        axes[i + 5].set_title(f"{labels_right[i]}")
        axes[i + 5].legend()

    axes[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.show()

def plot_multiple_knee_predictions(actual_data, predicted_steps_list, label="Typical"):
    """
    Plots actual hip and ankle angles per stride and overlays single-timestep predictions.
    Each item in predicted_steps_list must be a (1, 8) array.
    """
    stride_length = 51
    time = np.arange(stride_length)

    # Plot Hip Angles
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].set_title(f"{label} - Left Hip Angle")
    axs[1].set_title(f"{label} - Right Hip Angle")

    num_strides = len(actual_data) // stride_length
    for i in range(num_strides):
        start = i * stride_length
        end = start + stride_length
        axs[0].plot(time, actual_data[start:end, 0], color='gray', alpha=0.4)
        axs[1].plot(time, actual_data[start:end, 1], color='gray', alpha=0.4)

    # Overlay predicted points
    for idx, pred in enumerate(predicted_steps_list):
        axs[0].plot(idx, pred[0, 2], 'ro')
        axs[1].plot(idx, pred[0, 5], 'bo')

    axs[0].set_xlabel("Stride")
    axs[0].set_ylabel("LHip Angle")

    axs[1].set_xlabel("Stride")
    axs[1].set_ylabel("RHip Angle")

    plt.tight_layout()
    plt.show()


predicted_strides = []
current_input = last_known_step.reshape(1, 51, 8)

for _ in range(204):  # For 204 timesteps (8 strides × 51)
    pred = model.predict(current_input)  # (1, 1, 8) or (1, 8)
    pred = pred.reshape(1, -1)
    predicted_strides.append(pred)
    current_input = np.concatenate([current_input[:, 1:, :], pred.reshape(1, 1, 8)], axis=1)

predicted_array = np.vstack(predicted_strides)
plot_comparison(predicted_array, actual_next_4_steps)