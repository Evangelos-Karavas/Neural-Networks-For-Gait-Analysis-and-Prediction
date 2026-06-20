import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Set the folder path
folder_path = "Data_Normal"

STRIDE_LEN = 51

# Define columns to read
columns_to_read = ['LHipAngles (1)', 'RHipAngles (1)', 'LKneeAngles (1)', 'RKneeAngles (1)',
                   'LAnkleAngles (1)', 'RAnkleAngles (1)', 'Left Foot Off', 'Right Foot Off']

RIGHT_ANGLE_COLS = ['RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
LHIP_COL = 'LHipAngles (1)'
RHIP_COL = 'RHipAngles (1)'
LFO_COL = 'Left Foot Off'
RFO_COL = 'Right Foot Off'
PV_COLS = ['PhaseVariable_Left', 'PhaseVariable_Right']

# Collect all data
all_data = []

for file in os.listdir(folder_path):
    if file.endswith(".xlsx"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path, sheet_name="Data", usecols=columns_to_read, skiprows=[1, 2])
        all_data.append(df)

merged_data = pd.concat(all_data, ignore_index=True)


# ============================================================
# Phase variable + right-leg half-stride offset
# (single source of truth: networks load these precomputed columns)
# ============================================================
def compute_pv_stride(q: np.ndarray, c: float, enforce_monotonic: bool = True) -> np.ndarray:
    q = q.astype(np.float64)
    N = q.shape[0]
    c = float(np.clip(c, 0.05, 0.95))

    q0 = float(q[0])
    idx_min = int(np.argmin(q))
    qmin = float(q[idx_min])

    denom_stance = q0 - qmin
    s = np.zeros(N, dtype=np.float64)
    s[:idx_min + 1] = ((q0 - q[:idx_min + 1]) / denom_stance) * c
    sm = s[idx_min]
    qh_m = qmin
    denom_swing = q0 - qh_m
    s[idx_min:] = 1.0 + ((1.0 - sm) / denom_swing) * (q[idx_min:] - q0)
    s = np.clip(s, 0.0, 1.0)

    if enforce_monotonic:
        for i in range(1, N):
            if s[i] < s[i - 1]:
                s[i] = s[i - 1]

    return s


def compute_phase_variables(df, stride_len, lhip_col, rhip_col, lfo_col, rfo_col, enforce_monotonic=True):
    n_strides = len(df) // stride_len
    out = df.iloc[:n_strides * stride_len].copy()

    pvL = np.zeros(len(out), dtype=np.float32)
    pvR = np.zeros(len(out), dtype=np.float32)

    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        cL = float(out[lfo_col].iloc[a]) / 100.0
        cR = float(out[rfo_col].iloc[a]) / 100.0
        qL = out[lhip_col].values[a:b]
        qR = out[rhip_col].values[a:b]
        pvL[a:b] = compute_pv_stride(qL, c=cL, enforce_monotonic=enforce_monotonic)
        pvR[a:b] = compute_pv_stride(qR, c=cR, enforce_monotonic=enforce_monotonic)

    out["PhaseVariable_Left"] = pvL
    out["PhaseVariable_Right"] = pvR
    return out


def roll_stridewise_1d(x: np.ndarray, stride_len: int, shift: int) -> np.ndarray:
    x = np.asarray(x)
    n_strides = len(x) // stride_len
    x = x[:n_strides * stride_len].copy()
    out = x.copy()
    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        out[a:b] = np.roll(x[a:b], shift)
    return out


def apply_right_leg_half_stride_offset(df, stride_len, right_angle_cols, pv_right_col="PhaseVariable_Right"):
    shift = stride_len // 2
    out = df.copy()
    for col in right_angle_cols:
        out[col] = roll_stridewise_1d(out[col].to_numpy(), stride_len, shift)
    out[pv_right_col] = roll_stridewise_1d(out[pv_right_col].to_numpy(), stride_len, shift)
    return out


# PV must be computed on the unshifted data (it's anchored to the start of each
# stride block), then the right leg is rolled half a stride together with its PV.
merged_data = compute_phase_variables(merged_data, STRIDE_LEN, LHIP_COL, RHIP_COL, LFO_COL, RFO_COL)
merged_data = apply_right_leg_half_stride_offset(merged_data, STRIDE_LEN, RIGHT_ANGLE_COLS)

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

    # Restore foot-off values and phase variables (we don't want to add noise to these)
    noisy_cycle['Left Foot Off'] = base_cycle['Left Foot Off']
    noisy_cycle['Right Foot Off'] = base_cycle['Right Foot Off']
    noisy_cycle['PhaseVariable_Left'] = base_cycle['PhaseVariable_Left']
    noisy_cycle['PhaseVariable_Right'] = base_cycle['PhaseVariable_Right']

    randomized_data.append(noisy_cycle)

final_df = pd.concat(randomized_data, ignore_index=True)

columns_to_smooth = ['LHipAngles (1)', 'RHipAngles (1)', 'LKneeAngles (1)', 'RKneeAngles (1)',
                   'LAnkleAngles (1)', 'RAnkleAngles (1)']

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
