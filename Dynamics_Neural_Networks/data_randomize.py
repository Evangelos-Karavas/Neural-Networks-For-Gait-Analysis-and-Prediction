import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, resample

# ==========================================
# SETTINGS
# ==========================================
DATA_FOLDER = "Data_Normal"
OUTPUT_FILE = "Data_Normal/dynamics_total_augmented.xlsx"
STRIDE = 51

APPLY_AUGMENTATION = True        # Toggle ON/OFF

# ------------------------------------------
# Columns to use
# ------------------------------------------
MOMENT_COLS = [
    'LHipMoment (1)','RHipMoment (1)','LHipMoment (2)','RHipMoment (2)','LHipMoment (3)','RHipMoment (3)',
    'LKneeMoment (1)','RKneeMoment (1)','LKneeMoment (2)','RKneeMoment (2)','LKneeMoment (3)','RKneeMoment (3)',
    'LAnkleMoment (1)','RAnkleMoment (1)','LAnkleMoment (2)','RAnkleMoment (2)','LAnkleMoment (3)','RAnkleMoment (3)'
]
# Forces (18)
FORCE_COLS = [
    'LHipForce (1)','RHipForce (1)','LHipForce (2)','RHipForce (2)','LHipForce (3)','RHipForce (3)',
    'LKneeForce (1)','RKneeForce (1)','LKneeForce (2)','RKneeForce (2)','LKneeForce (3)','RKneeForce (3)',
    'LAnkleForce (1)','RAnkleForce (1)','LAnkleForce (2)','RAnkleForce (2)','LAnkleForce (3)','RAnkleForce (3)'
]
# Angles (18) — TARGET OUTPUTS
ANGLE_COLS = [
    'LHipAngles (1)','RHipAngles (1)','LHipAngles (2)','RHipAngles (2)','LHipAngles (3)','RHipAngles (3)',
    'LKneeAngles (1)','RKneeAngles (1)','LKneeAngles (2)','RKneeAngles (2)','LKneeAngles (3)','RKneeAngles (3)',
    'LAnkleAngles (1)','RAnkleAngles (1)','LAnkleAngles (2)','RAnkleAngles (2)','LAnkleAngles (3)','RAnkleAngles (3)'
]
ALL_COLS = MOMENT_COLS + FORCE_COLS + ANGLE_COLS

# ==========================================
# Load data
# ==========================================
all_data = []
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".xlsx"):
        df = pd.read_excel(os.path.join(DATA_FOLDER, file),
                           sheet_name="Data",
                           usecols=ALL_COLS,
                           skiprows=[1,2])
        df = df.dropna().reset_index(drop=True)
        all_data.append(df)

merged = pd.concat(all_data, ignore_index=True)

# Smooth angles
for col in ANGLE_COLS:
    merged[col] = savgol_filter(merged[col], 9, 3)

# ==========================================
# Split into strides
# ==========================================
num_strides = len(merged) // STRIDE
merged = merged.iloc[:num_strides*STRIDE]
strides = merged.values.reshape(num_strides, STRIDE, -1)

augmented_strides = []

# ==========================================
# Augmentation Functions
# ==========================================
def augment_stride(stride):
    """Apply safe biomechanical augmentations."""
    new = stride.copy()

    # 1. amplitude scaling (±5%)
    scale = 1.0 + np.random.uniform(-0.05, 0.05, size=stride.shape[1:])
    new *= scale

    # 2. small Gaussian noise (1% of std)
    noise_std = np.std(stride, axis=0) * 0.01
    noise = np.random.normal(0, noise_std, stride.shape)
    new += noise

    # 3. time warp (resample)
    factor = np.random.uniform(0.95, 1.05)  # ±5%
    warped = resample(new, int(STRIDE * factor))
    warped = resample(warped, STRIDE)  # back to 51 samples

    return warped

# ==========================================
# Apply augmentations
# ==========================================
for s in strides:
    augmented_strides.append(s)  # original stride

    if APPLY_AUGMENTATION:
        for _ in range(3):  # generate 3 variants per stride
            augmented_strides.append(augment_stride(s))

augmented_strides = np.array(augmented_strides)

# Flatten back into dataframe
final_data = augmented_strides.reshape(-1, augmented_strides.shape[-1])
final_df = pd.DataFrame(final_data, columns=ALL_COLS)

final_df.to_excel(OUTPUT_FILE, index=False)
print("Saved augmented dataset:", OUTPUT_FILE)
