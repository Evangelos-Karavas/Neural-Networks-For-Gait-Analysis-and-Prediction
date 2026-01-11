import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Set the folder path
folder_path = "Data_Normal"

# Define columns to read
columns_to_read = ['LHipAngles (1)', 'RHipAngles (1)', 'LKneeAngles (1)', 'RKneeAngles (1)', 
                   'LAnkleAngles (1)', 'RAnkleAngles (1)', 'Left Foot Off', 'Right Foot Off']

# Collect all data
all_data = []

for file in os.listdir(folder_path):
    if file.endswith(".xlsx"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path, sheet_name="Data", usecols=columns_to_read, skiprows=[1, 2])
        all_data.append(df)

merged_data = pd.concat(all_data, ignore_index=True)
column_std = merged_data.std()

num_total_cycles = 500
num_original_cycles = 60
num_noisy_cycles = num_total_cycles - num_original_cycles
randomized_data = []
base_noise = 0.1

# Helper function to broadcast foot-off values
def broadcast_foot_off(cycle):
    lfo_value = cycle['Left Foot Off'].dropna().iloc[0]
    rfo_value = cycle['Right Foot Off'].dropna().iloc[0]
    cycle['Left Foot Off'] = lfo_value
    cycle['Right Foot Off'] = rfo_value
    return cycle

# Keep 60 original gait cycles with foot-off broadcasting
for i in range(num_original_cycles):
    start = np.random.randint(0, len(merged_data) // 51) * 51
    cycle = merged_data.iloc[start:start + 51].copy()
    cycle = broadcast_foot_off(cycle)
    randomized_data.append(cycle)

# Generate 440 noisy versions
for i in range(num_noisy_cycles):
    base_cycle = randomized_data[i % num_original_cycles].copy()

    noise = np.random.normal(loc=0, scale=base_noise * column_std.values, size=base_cycle.shape)
    noise = np.clip(noise, -0.5, 0.5)

    noisy_cycle = base_cycle + noise

    # Restore foot-off values (we don't want to add noise to these)
    noisy_cycle['Left Foot Off'] = base_cycle['Left Foot Off']
    noisy_cycle['Right Foot Off'] = base_cycle['Right Foot Off']

    randomized_data.append(noisy_cycle)

final_df = pd.concat(randomized_data, ignore_index=True)

def moving_average(data, window_size=7):
    """
    Applies a centered moving average to the data.
    :param data: 1D NumPy array or Pandas Series
    :param window_size: must be odd
    :return: smoothed array
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


# === Smoothing the entire dataset ===
columns_to_smooth = ['LHipAngles (1)', 'RHipAngles (1)', 'LKneeAngles (1)', 'RKneeAngles (1)', 
                   'LAnkleAngles (1)', 'RAnkleAngles (1)']

# Apply filter
# for col in columns_to_smooth:
#     final_df[col] = moving_average(final_df[col].values, window_size=7)

# Wrap-around right leg by 25 steps (cyclic shift)
delay = 25
right_leg_columns = [
    'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)', 'Right Foot Off'
]

# Perform cyclic shift for each right leg column
for col in right_leg_columns:
    original = final_df[col].values
    shifted = np.concatenate([original[-delay:], original[:-delay]])
    final_df[col] = shifted

def sg_smooth_per_cycle(df, cols, cycle_len=51, window=9, polyorder=3):
    assert window % 2 == 1 and window <= cycle_len
    out = df.copy()
    n = len(df)
    for start in range(0, n, cycle_len):
        stop = min(start + cycle_len, n)
        for col in cols:
            segment = out[col].iloc[start:stop].to_numpy()
            # circular pad to avoid edge artifacts within a cycle
            pad = window // 2
            seg_padded = np.r_[segment[-pad:], segment, segment[:pad]]
            seg_smooth = savgol_filter(seg_padded, window_length=window, polyorder=polyorder, mode='interp')
            out[col].iloc[start:stop] = seg_smooth[pad:-pad]
    return out
final_df = sg_smooth_per_cycle(final_df, columns_to_smooth, cycle_len=51, window=9, polyorder=3)

# Combine all cycles and export
output_path = "Data_Normal/randomized_data_healthy.xlsx"
final_df.to_excel(output_path, index=False)

print("Shape of df:", final_df.shape)
print(f"Saved randomized data to {output_path}")