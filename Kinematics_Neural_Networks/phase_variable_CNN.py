import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
from matplotlib import pyplot
#This is for saving the model (There were issues with __version__ when calling the function save_model)
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

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
        foot_contact_percent_right = 66.03
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
# Prepare Data for CNN (Both Left and Right Legs) for Typical and CP Data
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
data.fillna(0, inplace=True)

# Define Sampling Frequency
fs = 1000  # Adjust if needed (depends on dataset)

# Apply Filters to Each Column
for col in columns_to_read:
    data[col] = butterworth_filter(data[col].values, cutoff=250, fs=fs, order=2, filter_type='low')
    data[col] = butterworth_filter(data[col].values, cutoff=15, fs=fs, order=2, filter_type='high')
    data[col] = savitzky_golay_filter(data[col].values)

# ==============================================
# Load and Process Cerebral Palsy Data
# ==============================================
folder_path = "Data_CP/"
all_cp_data = []
cp_data_counter = 0
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        df1 = pd.read_excel(file_path, 'Data', usecols=columns_to_read, skiprows=[1, 2])
        df1.fillna(0, inplace=True)

        # Apply Filters
        for col in columns_to_read:
            df1[col] = butterworth_filter(df1[col].values, cutoff=250, fs=fs, order=2, filter_type='low')
            df1[col] = butterworth_filter(df1[col].values, cutoff=15, fs=fs, order=2, filter_type='high')
            df1[col] = savitzky_golay_filter(df1[col].values)
            if cp_data_counter > 500:
                break
            cp_data_counter =+ 1
        all_cp_data.append(df1)

# Merge CP Data
data_cerebral_palsy = pd.concat(all_cp_data, ignore_index=True)

processed_data = prepare_neural_network_data(data)
processed_cp_data = prepare_neural_network_data(data_cerebral_palsy)
# ==========================================
# Normalize Data and Train-Test-Val Split
# ==========================================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(processed_data)
data_cp_scaled = scaler.fit_transform(processed_cp_data)

#Save scaler for ROS node
joblib.dump(scaler, "Scaler/pv_standard_scaler.save")

# ==========================
# Reshape Data for CNN
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
# Y_train_output = Y_train_output[:, -1, :]
# Y_val_output = Y_val_output[:, -1, :]
# Y_test_output = Y_test_output[:, -1, :]
def thresholded_accuracy(y_true, y_pred, threshold=0.2):
    """
    Computes thresholded accuracy: the percentage of predicted values within a specified degree threshold.
    """
    error = np.abs(y_true - y_pred)
    within_threshold = np.mean(error <= threshold)
    return within_threshold * 100

# ==============================================
# Build CNN Model
# ==============================================
def build_cnn():
    model = Sequential([
        # First convolutional block
        Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(51, 8)),
        Conv1D(48, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, strides=2),

        # Second convolutional block
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, strides=2),

        # Reduce time steps
        GlobalMaxPooling1D(),

        # Output layer for 51 time steps with 8 features
        Dense(51 * 8, activation='linear'),
        Reshape((51, 8)),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'Accuracy']
    )
    
    return model
model = build_cnn()

# Check model summary
model.summary()

# Train model
history = model.fit(X_train_input, Y_train_output,epochs=200, batch_size=102,validation_data=(X_val_input, Y_val_output))
# ==============================================
# Evaluate Model
# ==============================================
def evaluate_model(model, X, Y, label, threshold=0.2):
    loss, mae, accuracy = model.evaluate(X, Y, verbose=1)
    Y_pred = model.predict(X)

    thresh_acc = thresholded_accuracy(Y, Y_pred, threshold=threshold)

    print(f"🔹 {label} Evaluation:")
    print(f"CNN Loss (MSE): {loss:.4f}, MAE: {mae:.4f}, Accuracy (Keras): {accuracy:.4f}")
    print(f"CNN Thresholded Accuracy (±{threshold}°): {thresh_acc:.2f}%")
    print("=" * 90)

evaluate_model(model, X_train_input, Y_train_output, "Training Data", threshold=0.2)
evaluate_model(model, X_val_input, Y_val_output, "Validation Data", threshold=0.2)
evaluate_model(model, X_test_input, Y_test_output, "Testing Data", threshold=0.2)


model.save("Saved_Models/PV_cnn_model.keras", include_optimizer=True)

# ==============================================
# Predict Next 4 Steps
# ==============================================
def predict_next_steps(model, last_51_rows, num_steps):
    """
    Predicts the next N timesteps by feeding in 51-timestep input windows and getting 1 timestep output at a time.
    Assumes model outputs shape (1, 8).
    """
    input_batch = last_51_rows.reshape(1, 51, 8)
    predicted_stride = model.predict(input_batch, verbose=0)  # shape (1, 51, 8)
    return scaler.inverse_transform(predicted_stride.reshape(-1, 8))


# Get last known step for prediction
last_known_step = X_train_input[-1]
last_known_step_cp = X_test_input[-1] 
num_steps=4
next_steps_prediction = predict_next_steps(model, last_known_step, num_steps)
next_steps_prediction_cp = predict_next_steps(model, last_known_step_cp, num_steps)
# Get actual next 4 steps from dataset for plotting
actual_next_4_steps = processed_data.iloc[len(processed_data) - 51:, :].values  
actual_next_4_steps_cp = processed_cp_data.iloc[len(processed_cp_data) - 51:, :].values  

# Convert the numpy array to a pandas DataFrame
column_names = ['PhaseVariable_Left', 'PhaseVariable_Right', 
                'LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
next_steps_prediction_df = pd.DataFrame(next_steps_prediction, columns=column_names)
next_steps_prediction_cp_df = pd.DataFrame(next_steps_prediction_cp, columns=column_names)

# Compute phase variables for the predicted data
next_steps_prediction_df['PhaseVariable_Left'], next_steps_prediction_df['PhaseVariable_Right'] = compute_phase_variable_prediction(next_steps_prediction_df, 'LHipAngles (1)', 'RHipAngles (1)')
next_steps_prediction_cp_df['PhaseVariable_Left'], next_steps_prediction_cp_df['PhaseVariable_Right'] = compute_phase_variable_prediction(next_steps_prediction_cp_df, 'LHipAngles (1)', 'RHipAngles (1)')

next_steps_prediction_df = next_steps_prediction_df[column_names]
next_steps_prediction_cp_df = next_steps_prediction_cp_df[column_names]


# ==========================
# Save Predictions
# ==========================
predicted_df = pd.DataFrame(next_steps_prediction_df, columns=column_names)
predicted_df.to_excel("Predictions/PV_typical_cnn.xlsx", index=False)

predicted_df_cp = pd.DataFrame(next_steps_prediction_cp_df, columns=column_names)
predicted_df_cp.to_excel("Predictions/PV_cp_cnn.xlsx", index=False)
print("Prediction complete. Data saved")



# ==============================================================================
# All Plots for Phase Variable || Predicted Angles || Loss - Accuracy of Model
# ==============================================================================


def plot_phase_variables(predicted_df, actual_array, title_suffix="Typical"):
    """
    Plots predicted and actual phase variables for left and right legs.
    """
    time = np.arange(predicted_df.shape[0])

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Convert to NumPy arrays to avoid multi-indexing errors
    pv_left = predicted_df['PhaseVariable_Left'].to_numpy()
    pv_right = predicted_df['PhaseVariable_Right'].to_numpy()

    # Plot Phase Variable - Left
    ax[0].plot(time, pv_left, label='Predicted Left', color='blue')
    ax[0].set_ylabel("Phase Variable Left")
    ax[0].set_ylim(-0.2, 1.2)
    ax[0].set_title(f"Phase Variable (Left) - {title_suffix}")
    ax[0].legend()

    # Plot Phase Variable - Right
    ax[1].plot(time, pv_right, label='Predicted Right', color='red')
    ax[1].set_ylabel("Phase Variable Right")
    ax[1].set_ylim(-0.2, 1.2)
    ax[1].set_xlabel("Time (Samples)")
    ax[1].set_title(f"Phase Variable (Right) - {title_suffix}")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

plot_phase_variables(next_steps_prediction_df, actual_next_4_steps, title_suffix="Typical")
plot_phase_variables(next_steps_prediction_cp_df, actual_next_4_steps_cp, title_suffix="CP")


def plot_comparison(predicted, actual):
    """Plots actual vs predicted joint angles and marks swing phase when the predicted phase variable changes sharply."""
    time = np.arange(actual.shape[0])  # Time index for the dataset

    labels_left = ['LHipAngles', 'LKneeAngles', 'LAnkleAngles']
    labels_right = ['RHipAngles', 'RKneeAngles', 'RAnkleAngles']

    fig, axes = plt.subplots(8, 1, figsize=(12, 16), sharex=True)

    # Compute the change in predicted phase variables over time
    delta_pred_left = np.abs(np.diff(predicted[:, 0]))  # Change in predicted left phase variable
    delta_pred_right = np.abs(np.diff(predicted[:, 1]))  # Change in predicted right phase variable

    # Detect swing phase start indices where |predicted[t] - predicted[t-1]| > 0.2
    swing_left = np.where(delta_pred_left > 0.15)[0] + 1  # Offset by +1 due to np.diff
    swing_right = np.where(delta_pred_right > 0.15)[0] + 1  # Offset by +1 due to np.diff

    # Function to filter unique swing phase start indices (avoid consecutive detections)
    def filter_swing_starts(indices):
        """Filters consecutive indices, keeping only the first occurrence in a swing phase event."""
        filtered = []
        for i in range(len(indices)):
            if i == 0 or (indices[i] - indices[i - 1] > 5):  # Ensures at least 5 time steps apart
                filtered.append(indices[i])
        return np.array(filtered)

    # Apply filtering to remove consecutive detections
    swing_left = filter_swing_starts(swing_left)
    swing_right = filter_swing_starts(swing_right)

    # Function to add swing phase vertical lines
    def add_swing_phase_lines(ax):
        """Adds red vertical lines at swing phase start indices."""
        for t in np.unique(np.concatenate((swing_left, swing_right))):  # Combine both swing phases
            ax.axvline(x=t, color='gray', linestyle='dashed', alpha=0.8, label="Swing Phase Start" if t == swing_left[0] else "")

    # Phase Variable Plot (Left)
    axes[0].plot(time, actual[:, 0], label="Actual Phase Variable (Left)", color='blue')
    axes[0].plot(time, predicted[:, 0], label="Predicted Phase Variable (Left)", color='red')
    axes[0].set_ylabel("Phase Variable (Left)")
    axes[0].set_ylim([0, 1.2])
    add_swing_phase_lines(axes[0])
    axes[0].legend()

    # Phase Variable Plot (Right)
    axes[1].plot(time, actual[:, 1], label="Actual Phase Variable (Right)", color='blue')
    axes[1].plot(time, predicted[:, 1], label="Predicted Phase Variable (Right)", color='red')
    axes[1].set_ylabel("Phase Variable (Right)")
    axes[1].set_ylim([0, 1.2])
    add_swing_phase_lines(axes[1])
    axes[1].legend()

    # Left Leg Joint Angles (Columns 2,3,4)
    for i in range(len(labels_left)):
        axes[i + 2].plot(time, actual[:, i + 2], label=f"Actual {labels_left[i]}", color='blue')
        axes[i + 2].plot(time, predicted[:, i + 2], label=f"Predicted {labels_left[i]}", linestyle='dashed', color='red')
        axes[i + 2].set_ylabel("Angle")
        axes[i + 2].legend()
        axes[i + 2].set_title(f"Comparison: {labels_left[i]}")
        add_swing_phase_lines(axes[i + 2])  # Add swing phase lines

    # Right Leg Joint Angles (Columns 5,6,7)
    for i in range(len(labels_right)):
        axes[i + 5].plot(time, actual[:, i + 5], label=f"Actual {labels_right[i]}", color='blue')
        axes[i + 5].plot(time, predicted[:, i + 5], label=f"Predicted {labels_right[i]}", linestyle='dashed', color='red')
        axes[i + 5].set_ylabel("Angle")
        axes[i + 5].legend()
        axes[i + 5].set_title(f"Comparison: {labels_right[i]}")
        add_swing_phase_lines(axes[i + 5])  # Add swing phase lines

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
    for idx in range(num_actual_strides):
        start = idx * stride_length
        end = start + stride_length
        axs[0].plot(time, actual_data[start:end, 3], color=colors(idx), alpha=0.5)
        axs[1].plot(time, actual_data[start:end, 6], color=colors(idx), alpha=0.5)

    # Plot each predicted stride
    for idx, pred in enumerate(predicted_steps_list):
        axs[0].plot(time, pred[:, 3], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)
        axs[1].plot(time, pred[:, 6], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)

    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Left Knee Angle")
    axs[0].legend()

    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Right Knee Angle")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


actual_next_20_steps = processed_data.iloc[len(processed_data) - 1020:, :].values
actual_next_20_steps_cp = processed_cp_data.iloc[len(processed_cp_data) - 1020:, :].values

predicted_strides = []
current_input = last_known_step.reshape(1, 51, 8)

for _ in range(5):
    pred = model.predict(current_input)  # shape (1, 8)
    predicted_strides.append(scaler.inverse_transform(pred[0].reshape(1, -1))[0])
    current_input = np.vstack([current_input[0][1:], pred[0]])  # shape (51, 8)

predicted_strides_cp = []
last_known_step_cp = last_known_step_cp.reshape(1, 51, 8)

for _ in range(5):
    pred = model.predict(last_known_step_cp)  # shape (1, 8)
    predicted_strides.append(scaler.inverse_transform(pred[0].reshape(1, -1))[0])
    last_known_step_cp = np.vstack([last_known_step_cp[0][1:], pred[0]])  # shape (51, 8)


plot_multiple_knee_predictions(actual_next_20_steps, predicted_strides, label="Typical")
plot_multiple_knee_predictions(actual_next_20_steps, predicted_strides_cp, label="Typical")



def plot_multiple_knee_predictions(actual_data, label="Typical"):
    """
    Plots actual knee angles for multiple strides and overlays multiple predicted strides.
    
    Parameters:
    - actual_data: shape (N, 8)
    - predicted_steps_list: list of predicted arrays (each of shape (51, 8))
    """
    stride_length = 51
    time = np.arange(stride_length)


    fig, axs = plt.subplots(1, 2, figsize=(14, 6))


    # Plot actual left and right knee angles for each stride
    num_actual_strides = len(actual_data) // stride_length
    for idx in range(num_actual_strides):
        start = idx * stride_length
        end = start + stride_length
        axs[0].plot(time, actual_data[start:end, 2], color='red', alpha=0.5)
        axs[1].plot(time, actual_data[start:end, 5], color='blue', alpha=0.5)



    fig, axs = plt.subplots(1, 2, figsize=(14, 6))


    # Plot actual left and right knee angles for each stride
    num_actual_strides = len(actual_data) // stride_length
    for idx in range(num_actual_strides):
        start = idx * stride_length
        end = start + stride_length
        axs[0].plot(time, actual_data[start:end, 3], color='red', alpha=0.5)
        axs[1].plot(time, actual_data[start:end, 6], color='blue', alpha=0.5)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))


    # Plot actual left and right ankle angles for each stride
    num_actual_strides = len(actual_data) // stride_length
    for idx in range(num_actual_strides):
        start = idx * stride_length
        end = start + stride_length
        axs[0].plot(time, actual_data[start:end, 4], color='red', alpha=0.5)
        axs[1].plot(time, actual_data[start:end, 7], color='blue', alpha=0.5)

    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Left Knee Angle")
    axs[0].legend()

    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Right Knee Angle")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


actual_next_20_steps = processed_data.iloc[len(processed_data) - 2040:, :].values
actual_next_20_steps_cp = processed_cp_data.iloc[len(processed_cp_data) - 2040:, :].values



plot_multiple_knee_predictions(actual_next_20_steps, label="Typical")
plot_multiple_knee_predictions(actual_next_20_steps_cp, label="CP")


def plot_phase_with_contact(df, title="Phase Variables with Swing/Stance", label_prefix="Typical"):
    """
    Plots the phase variable with swing and stance phases highlighted.
    """
    num_samples = 1020
    time = np.arange(num_samples)

    left_phase = df['PhaseVariable_Left'].iloc[-num_samples:].values
    right_phase = df['PhaseVariable_Right'].iloc[-num_samples:].values
    left_contact = df['Left Foot Off'].iloc[-num_samples:].values
    right_contact = df['Right Foot Off'].iloc[-num_samples:].values

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for ax, phase, contact, side in zip(axes, [left_phase, right_phase], [left_contact, right_contact], ['Left', 'Right']):
        ax.plot(time, phase, label=f'{label_prefix} - {side}', color='black')
        ax.set_ylabel(f'Phase Variable ({side})')
        ax.set_ylim(0, 2.5)

        # Shade swing (foot off ground) and stance (foot on ground) regions
        for i in range(len(contact)):
            if contact[i] < 50:  # Consider <50% as swing, >=50% as stance
                ax.axvspan(i, i + 1, color='mistyrose', alpha=0.4)  # Swing
            else:
                ax.axvspan(i, i + 1, color='lightblue', alpha=0.3)   # Stance

        ax.legend(loc="upper right")
        ax.grid(True)

    axes[1].set_xlabel("Time (samples)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# Run plotting
plot_phase_with_contact(data.iloc[-1020:], title="Typical Phase Variable with Swing/Stance", label_prefix="Typical")
plot_phase_with_contact(data_cerebral_palsy.iloc[-1020:], title="CP Phase Variable with Swing/Stance", label_prefix="CP")