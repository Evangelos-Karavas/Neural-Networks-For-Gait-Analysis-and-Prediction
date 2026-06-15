import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ==========================================
# SETTINGS
# ==========================================
DATA_FOLDER = "Data_Normal"
OUTPUT_FILE = "Data_Normal/dynamics_total_augmented.xlsx"
STRIDE = 51

NUM_TOTAL_CYCLES    = 500
NUM_ORIGINAL_CYCLES = 60
NUM_NOISY_CYCLES    = NUM_TOTAL_CYCLES - NUM_ORIGINAL_CYCLES  # 440
BASE_NOISE = 0.1

# ------------------------------------------
# Columns to use
# ------------------------------------------
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
FOOT_OFF_COLS = ['Left Foot Off', 'Right Foot Off']

ALL_COLS  = MOMENT_COLS + FORCE_COLS + ANGLE_COLS
FULL_COLS = ALL_COLS + FOOT_OFF_COLS

RIGHT_COLS = [c for c in ALL_COLS if c.startswith(('RHip', 'RKnee', 'RAnkle'))]
SHIFT = STRIDE // 2  # 25

# ==========================================
# Load all data from Data_Normal
# ==========================================
all_data = []
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".xlsx"):
        df = pd.read_excel(
            os.path.join(DATA_FOLDER, file),
            sheet_name="Data",
            usecols=FULL_COLS,
            skiprows=[1, 2]
        )
        df = df[FULL_COLS]  # enforce column order
        df[FOOT_OFF_COLS] = df[FOOT_OFF_COLS].ffill().bfill()
        df = df.dropna().reset_index(drop=True)
        all_data.append(df)

merged = pd.concat(all_data, ignore_index=True)

# ==========================================
# Apply right-leg phase shift per stride
# (avoids cross-boundary artifacts)
# ==========================================
n_strides_total = len(merged) // STRIDE
merged = merged.iloc[:n_strides_total * STRIDE].reset_index(drop=True)

arr      = merged.values.copy()
right_idx = [FULL_COLS.index(c) for c in RIGHT_COLS]

for s in range(n_strides_total):
    a, b = s * STRIDE, (s + 1) * STRIDE
    arr[a:b, right_idx] = np.roll(arr[a:b, right_idx], SHIFT, axis=0)

merged = pd.DataFrame(arr, columns=FULL_COLS)

# ==========================================
# Compute column std for noise scaling
# ==========================================
column_std = merged.std()

# ==========================================
# Helper: keep foot-off constant within stride
# ==========================================
def broadcast_foot_off(cycle):
    cycle = cycle.copy()
    for col in FOOT_OFF_COLS:
        vals = cycle[col].dropna()
        if len(vals) > 0:
            cycle[col] = vals.iloc[0]
    return cycle

# ==========================================
# Select 60 original strides randomly
# ==========================================
np.random.seed(42)
randomized_data = []

for i in range(NUM_ORIGINAL_CYCLES):
    stride_idx = np.random.randint(0, n_strides_total)
    start = stride_idx * STRIDE
    cycle = merged.iloc[start:start + STRIDE].copy()
    cycle = broadcast_foot_off(cycle)
    randomized_data.append(cycle)

# ==========================================
# Generate 440 noisy copies
# ==========================================
for i in range(NUM_NOISY_CYCLES):
    base_cycle = randomized_data[i % NUM_ORIGINAL_CYCLES].copy()

    noise = np.random.normal(loc=0, scale=BASE_NOISE * column_std.values, size=base_cycle.shape)
    noise = np.clip(noise, -0.5, 0.5)

    noisy_cycle = base_cycle + noise

    # Restore foot-off values (no noise on event markers)
    noisy_cycle['Left Foot Off']  = base_cycle['Left Foot Off']
    noisy_cycle['Right Foot Off'] = base_cycle['Right Foot Off']

    randomized_data.append(noisy_cycle)

final_df = pd.concat(randomized_data, ignore_index=True)

# ==========================================
# Savitzky-Golay smooth angles per cycle
# ==========================================
def sg_smooth_per_cycle(df, cols, cycle_len=51, window=9, polyorder=3):
    assert window % 2 == 1 and window <= cycle_len
    out = df.copy()
    n = len(df)
    for start in range(0, n, cycle_len):
        stop = min(start + cycle_len, n)
        for col in cols:
            segment = out[col].iloc[start:stop].to_numpy()
            pad = window // 2
            seg_padded = np.r_[segment[-pad:], segment, segment[:pad]]
            seg_smooth = savgol_filter(seg_padded, window_length=window, polyorder=polyorder, mode='interp')
            out[col].iloc[start:stop] = seg_smooth[pad:-pad]
    return out

final_df = sg_smooth_per_cycle(final_df, ANGLE_COLS)

# ==========================================
# Save
# ==========================================
final_df.to_excel(OUTPUT_FILE, index=False)
print("Shape:", final_df.shape)
print(f"Saved {NUM_TOTAL_CYCLES} cycles ({NUM_ORIGINAL_CYCLES} original + {NUM_NOISY_CYCLES} noisy) to {OUTPUT_FILE}")
