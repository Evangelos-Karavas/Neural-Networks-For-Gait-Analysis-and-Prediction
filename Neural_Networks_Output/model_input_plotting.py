#!/usr/bin/env python3
"""
plot_planes_from_folders_0to51.py

Separate overlay neural_networks_outputs/plots (NO resampling) for TD + CP using your REAL files:

- TD: reads many .xlsx from Data_Normal/
- CP: reads many .xlsx from Data_CP/
- Both use: sheet_name="Data", skiprows=[1,2]

For each plane (1=sagittal, 2=frontal, 3=transverse) it saves 3 figures:
  Hip / Knee / Ankle
Each figure overlays N_STRIDES_PLOT strides:
  - Left joint in red
  - Right joint in blue
X axis is sample index 0..(STRIDE_LEN-1) (default 0..50 for STRIDE_LEN=51)

Outputs (examples):
  Neural_Networks_Outputs/Plots/TD_sagittal_hip.png
  Neural_Networks_Outputs/Plots/TD_frontal_knee.png
  Neural_Networks_Outputs/Plots/CP_transverse_ankle.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# SETTINGS
# =========================
TD_FOLDER = "Data_Normal/"
CP_FOLDER = "Data_CP/"

SHEET_NAME = "Data"
SKIPROWS = [1, 2]

MAX_TD_FILES = 500
MAX_CP_FILES = 500

STRIDE_LEN = 51          # stored stride length
N_STRIDES_PLOT = 40      # how many strides to overlay per plot

OUT_DIR = "Neural_Networks_Outputs/Plots"

PLANE_NAME = {1: "sagittal", 2: "frontal", 3: "transverse"}


# =========================
# Column helpers
# =========================
def cols_for_plane(plane: int):
    if plane not in (1, 2, 3):
        raise ValueError("plane must be 1 (sagittal), 2 (frontal), or 3 (transverse)")
    return [
        f"LHipAngles ({plane})",
        f"LKneeAngles ({plane})",
        f"LAnkleAngles ({plane})",
        f"RHipAngles ({plane})",
        f"RKneeAngles ({plane})",
        f"RAnkleAngles ({plane})",
    ]


# =========================
# Data loading from folders
# =========================
def load_concat_from_folder(folder: str, plane: int, max_files: int) -> pd.DataFrame:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    usecols = cols_for_plane(plane)

    frames = []
    count = 0
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".xlsx"):
            continue
        fp = os.path.join(folder, fn)

        try:
            df = pd.read_excel(
                fp,
                sheet_name=SHEET_NAME,
                usecols=usecols,
                skiprows=SKIPROWS
            ).fillna(0)
        except Exception as e:
            print(f"Skipping {fp} (read error): {e}")
            continue

        # enforce exact column order
        df = df[usecols]
        frames.append(df)
        count += 1
        if count >= max_files:
            break

    if not frames:
        raise RuntimeError(f"No xlsx files loaded from {folder}. Check sheet/columns/skiprows.")
    return pd.concat(frames, ignore_index=True).fillna(0)


# =========================
# Stride slicing
# =========================
def to_strides(arr_Tx6: np.ndarray, stride_len: int) -> np.ndarray:
    """
    arr_Tx6 -> (N, stride_len, 6)
    Channels are [LHip, LKnee, LAnkle, RHip, RKnee, RAnkle]
    """
    T = arr_Tx6.shape[0]
    n = T // stride_len
    if n <= 0:
        raise ValueError("Not enough samples for a single stride.")
    return arr_Tx6[:n * stride_len].reshape(n, stride_len, 6)


# =========================
# Plotting
# =========================
def plot_joint_group(strides: np.ndarray, group: str, plane_name: str, dataset_name: str,
                     out_path: str, n_strides_plot: int = 40):
    """
    group: "hip" | "knee" | "ankle"
    """
    group = group.lower()
    idx_map = {
        "hip":   (0, 3),  # LHip, RHip
        "knee":  (1, 4),  # LKnee, RKnee
        "ankle": (2, 5),  # LAnkle, RAnkle
    }
    if group not in idx_map:
        raise ValueError("group must be hip/knee/ankle")

    li, ri = idx_map[group]
    N, S, _ = strides.shape
    useN = min(N, n_strides_plot)

    x = np.arange(S)  # 0..50 for STRIDE_LEN=51

    plt.figure(figsize=(10, 4.6))

    # overlay multiple strides
    for i in range(useN):
        plt.plot(x, strides[i, :, li], color="red",  alpha=0.25, linewidth=1.0)
        plt.plot(x, strides[i, :, ri], color="blue", alpha=0.25, linewidth=1.0)

    plt.title(f"{dataset_name} {group.capitalize()} angles — {plane_name} plane")
    plt.xlabel(f"Sample (0..{S-1})")
    plt.ylabel("Angle (deg)")
    plt.grid(True, alpha=0.35)

    # legend handles
    plt.plot([], [], color="red",  label="Left")
    plt.plot([], [], color="blue", label="Right")
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.show()


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for plane in (1, 2, 3):
        plane_name = PLANE_NAME[plane]
        print(f"\n--- Plane: {plane_name} ({plane}) ---")

        # ---- TD ----
        print("Loading TD...")
        td_df = load_concat_from_folder(TD_FOLDER, plane, MAX_TD_FILES)
        td_arr = td_df.to_numpy(dtype=np.float32)
        td_strides = to_strides(td_arr, STRIDE_LEN)
        print(f"TD strides: {td_strides.shape[0]}")

        plot_joint_group(td_strides, "hip",   plane_name, "TD",
                         os.path.join(OUT_DIR, f"TD_{plane_name}_hip.png"),   N_STRIDES_PLOT)
        plot_joint_group(td_strides, "knee",  plane_name, "TD",
                         os.path.join(OUT_DIR, f"TD_{plane_name}_knee.png"),  N_STRIDES_PLOT)
        plot_joint_group(td_strides, "ankle", plane_name, "TD",
                         os.path.join(OUT_DIR, f"TD_{plane_name}_ankle.png"), N_STRIDES_PLOT)

        # ---- CP ----
        print("Loading CP...")
        cp_df = load_concat_from_folder(CP_FOLDER, plane, MAX_CP_FILES)
        cp_arr = cp_df.to_numpy(dtype=np.float32)
        cp_strides = to_strides(cp_arr, STRIDE_LEN)
        print(f"CP strides: {cp_strides.shape[0]}")

        plot_joint_group(cp_strides, "hip",   plane_name, "CP",
                         os.path.join(OUT_DIR, f"CP_{plane_name}_hip.png"),   N_STRIDES_PLOT)
        plot_joint_group(cp_strides, "knee",  plane_name, "CP",
                         os.path.join(OUT_DIR, f"CP_{plane_name}_knee.png"),  N_STRIDES_PLOT)
        plot_joint_group(cp_strides, "ankle", plane_name, "CP",
                         os.path.join(OUT_DIR, f"CP_{plane_name}_ankle.png"), N_STRIDES_PLOT)

    print(f"\nDone. Saved neural_networks_outputs/plots to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
