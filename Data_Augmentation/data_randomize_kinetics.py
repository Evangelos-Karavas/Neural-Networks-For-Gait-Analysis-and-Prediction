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

LHIP_COL = 'LHipAngles (1)'
RHIP_COL = 'RHipAngles (1)'
LFO_COL  = 'Left Foot Off'
RFO_COL  = 'Right Foot Off'
PV_COLS  = ['PhaseVariable_Left', 'PhaseVariable_Right']


# ------------------------------------------
# Phase variable computation (matches the phase-variable network scripts)
# ------------------------------------------
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
# Phase variable computation
# (must run on the unshifted data: it's anchored to the start of each stride)
# ==========================================
merged = compute_phase_variables(merged, STRIDE, LHIP_COL, RHIP_COL, LFO_COL, RFO_COL)
FULL_COLS = FULL_COLS + PV_COLS

# ==========================================
# Apply right-leg phase shift per stride
# (avoids cross-boundary artifacts; PhaseVariable_Right rides along with the
# right-leg signals it was computed from)
# ==========================================
n_strides_total = len(merged) // STRIDE
merged = merged.iloc[:n_strides_total * STRIDE].reset_index(drop=True)

arr      = merged[FULL_COLS].values.copy()
right_idx = [FULL_COLS.index(c) for c in RIGHT_COLS + ['PhaseVariable_Right']]

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

    # Restore foot-off values and phase variables (no noise on event markers/derived signals)
    noisy_cycle['Left Foot Off']  = base_cycle['Left Foot Off']
    noisy_cycle['Right Foot Off'] = base_cycle['Right Foot Off']
    noisy_cycle['PhaseVariable_Left']  = base_cycle['PhaseVariable_Left']
    noisy_cycle['PhaseVariable_Right'] = base_cycle['PhaseVariable_Right']

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
