# Neural Networks for Human Gait Analysis and Prediction

LSTM and CNN models that predict lower-limb joint angles for children with
cerebral palsy (CP), trained on the gait of typically developed (TD) children.
The code accompanies the Diploma Thesis *On the Use of Neural Networks in the
Analysis and Prediction of Human Gait* (NTUA, School of Mechanical Engineering).
The thesis itself is in
[ON_THE_USE_OF_NEURAL_NETWORKS_IN_THE_ANALYSIS_AND_PREDICTION_OF_HUMAN_GAIT.pdf](ON_THE_USE_OF_NEURAL_NETWORKS_IN_THE_ANALYSIS_AND_PREDICTION_OF_HUMAN_GAIT.pdf).

The models are meant to supply reference trajectories for a lower-limb
exoskeleton controller (ROS2).

**Data source:** all gait recordings were contributed by
[ELEPAP](https://elepap.gr/) (Hellenic Society for the Protection and
Rehabilitation of Disabled People).

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Data format](#data-format)
- [Installation](#installation)
- [Running the pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Ablation study](#ablation-study)

---

## How it works

### The task: next-tick prediction

Each recorded gait cycle (one stride, heel strike to heel strike) is
normalised to **51 samples**, i.e. 0 % to 100 % of the cycle in 2 % steps.
Every model sees a sliding **window of the last 51 samples** and predicts the
joint angles at the **next sample**:

```
X[i] = angles[i : i+51]      (51 × channels)
y[i] = angles[i+51]          (channels,)
```

At test time the models can also be run recursively (each prediction is fed
back as input) to generate a trajectory several steps ahead.

### Two input representations

| Approach | Input per time step | Folder |
|---|---|---|
| **Timestamp** | joint angles only | [Neural_Networks_Timestamps/](Neural_Networks_Timestamps/) |
| **Phase variable (PV)** | joint angles + a left and a right *phase variable* | [Neural_Networks_Phase_Variable/](Neural_Networks_Phase_Variable/) |

The **phase variable** is a monotonic 0 → 1 signal that tells the network
*where in the gait cycle* it is. It is computed per stride from the sagittal
hip angle, anchored at the measured toe-off (`Left Foot Off` / `Right Foot Off`)
event. Before training, the right leg is shifted by half a stride (25 samples)
so the two legs line up in phase.

Each approach is implemented as both an **LSTM** and a **1-D CNN**, giving four
models in total.

### Two channel sets

Each of the four models exists in two versions:

| Version | Channels | Script suffix | Typical-gait training file |
|---|---|---|---|
| Sagittal only | 6: L/R hip, knee, ankle in the sagittal plane | none, e.g. `timestamps_LSTM.py` | `Data_Normal/randomized_data_healthy.xlsx` |
| All planes | 18: L/R hip, knee, ankle in the sagittal, frontal and transverse planes | `_all_planes`, e.g. `timestamps_LSTM_all_planes.py` | `Data_Normal/dynamics_total_augmented.xlsx` |

The **all-planes** versions are the most recent ones and the ones used in the
final comparison.

### Training and evaluation protocol

1. **Data augmentation.** The TD recordings are expanded to 500 strides:
   60 real strides plus 440 copies with small Gaussian noise, then smoothed
   per stride with a Savitzky–Golay filter.
2. **Scaling.** A `StandardScaler` is fitted on the training data and saved
   next to the model, so inference uses the same normalisation.
3. **Training.** On typical gait (the all-planes scripts also add the CP
   training patients), with Adam, `ReduceLROnPlateau` and up to 150 epochs. The PV models also keep a
   checkpoint of the weights whose recursive rollout matches the ground truth
   best, measured with dynamic time warping (DTW).
4. **Held-out CP subjects.** In the all-planes scripts, the CP trials are
   split **by patient** (70 % train / 15 % validation / 15 % test, seed 42),
   so no patient appears in more than one split.
5. **Evaluation.** MAE, RMSE and R² in degrees, per channel and per held-out
   subject, plus plots of prediction vs. ground truth over the gait cycle.

---

## Repository layout

```
├── Data_Normal/                     TD recordings (62 trials, 13 children) + generated training files
├── Data_CP/                         CP recordings (514 trials, 106 children)
├── Data_Augmentation/
│   ├── data_randomize_kinematics.py   → Data_Normal/randomized_data_healthy.xlsx (sagittal angles)
│   └── data_randomize_kinetics.py     → Data_Normal/dynamics_total_augmented.xlsx (all angles, moments, forces)
├── Neural_Networks_Timestamps/
│   ├── timestamps_LSTM.py / timestamps_CNN.py                       (6 channels)
│   └── timestamps_LSTM_all_planes.py / timestamps_CNN_all_planes.py (18 channels)
├── Neural_Networks_Phase_Variable/
│   ├── phase_variable_lstm.py / phase_variable_cnn.py                       (6 channels)
│   └── phase_variable_lstm_all_planes.py / phase_variable_cnn_all_planes.py (18 channels)
├── Neural_Networks_Output/          trained models, scalers, predictions, plots (created by training)
│   ├── model_input_plotting.py        plots of the raw input strides, TD vs CP
│   ├── models_comparison.py           4-model comparison of the 6-channel models
│   └── models_comparison_all_planes.py 4-model comparison of the 18-channel models (metrics only)
├── models_complete_comparison.py    main evaluation: all 4 all-planes models on held-out CP subjects
├── Ablation_Study/                  kinematics vs. kinetics input ablation
└── requirements.txt
```

---

## Data format

Every trial is one Excel file (`.xlsx`) exported from the motion-capture
software:

- The data are on the sheet **`Data`**. Rows 2 and 3 (units / sub-headers)
  are skipped when reading.
- Each trial contains whole strides of **51 samples** each.
- Column names follow the pattern `<Side><Joint><Quantity> (<plane>)`, for
  example `LKneeAngles (1)`. Plane `(1)` is sagittal, `(2)` frontal and `(3)`
  transverse.
- `Left Foot Off` / `Right Foot Off` hold the toe-off event as a percentage of
  the gait cycle.
- CP file names start with the patient ID (`36686-20240711-1-05-01.xlsx` is
  patient `36686`). That ID is what the patient-level split groups by.

`Data_Normal/` also contains three generated or aggregate files that are
**not** single-trial recordings: `randomized_data_healthy.xlsx`,
`dynamics_total_augmented.xlsx` and `Healthy_Data_1_Person.xlsx`.

---

## Installation

The code was developed and tested on **Windows 10 with Python 3.10** and
TensorFlow 2.12 (CPU).

```bash
git clone https://github.com/Evangelos-Karavas/Neural-Networks-For-Gait-Analysis-and-Prediction.git
cd Neural-Networks-For-Gait-Analysis-and-Prediction

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# Linux / macOS:         source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

To check that TensorFlow works:

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

**Keep TensorFlow at 2.12.** It ships Keras 2. TensorFlow 2.16 and later
switch to Keras 3, which changes the saved-model format and some APIs used
here.

**GPU:** TensorFlow has no native GPU support on Windows after version 2.10.
To train on an NVIDIA GPU, use Linux or WSL2 with CUDA 11.8 and cuDNN 8.6, the
versions TensorFlow 2.12 is built against. Every script also runs on the CPU;
training just takes longer.

---

## Running the pipeline

> **Run every script from the repository root.** All data and output paths
> are relative to it, e.g. `Data_CP/` and `Neural_Networks_Output/`.

Trained models are **not** stored in the repository (`Neural_Networks_Output/`
is git-ignored), so on a fresh clone you train first and then evaluate.

### 1. Data augmentation (optional)

The augmented training files are already in `Data_Normal/`. Re-create them
only if you change the raw TD recordings:

```bash
python Data_Augmentation/data_randomize_kinematics.py   # randomized_data_healthy.xlsx
python Data_Augmentation/data_randomize_kinetics.py     # dynamics_total_augmented.xlsx
```

The scripts read every `.xlsx` trial in `Data_Normal/` (skipping their own
two output files) and use a fixed random seed, so re-runs are reproducible.

### 2. Train the models

All-planes (18-channel) models, used in the final comparison:

```bash
python Neural_Networks_Timestamps/timestamps_LSTM_all_planes.py
python Neural_Networks_Timestamps/timestamps_CNN_all_planes.py
python Neural_Networks_Phase_Variable/phase_variable_lstm_all_planes.py
python Neural_Networks_Phase_Variable/phase_variable_cnn_all_planes.py
```

Sagittal-only (6-channel) models:

```bash
python Neural_Networks_Timestamps/timestamps_LSTM.py
python Neural_Networks_Timestamps/timestamps_CNN.py
python Neural_Networks_Phase_Variable/phase_variable_lstm.py
python Neural_Networks_Phase_Variable/phase_variable_cnn.py
```

Each script prints per-channel metrics, opens its plots and saves the model,
scaler and predictions under `Neural_Networks_Output/`. Hyperparameters
(`WINDOW`, `EPOCHS`, learning rate, `MAX_CP_FILES`, …) are constants at the
top of each file.

### 3. Compare the four models

After training the four all-planes models:

```bash
python models_complete_comparison.py --list-subjects        # held-out CP patients and their index
python models_complete_comparison.py                        # metric table over all held-out patients
python models_complete_comparison.py --subject 3 --plot     # plots for one patient
python models_complete_comparison.py --demo                 # preview of the plots with synthetic data, no models needed
```

The flags and plots are described in
[README_comparison_plots.md](README_comparison_plots.md).

---

## Outputs

Training writes to these folders (created automatically):

| Folder | Contents |
|---|---|
| `Neural_Networks_Output/Saved_Models/` | Keras models (`.keras`) |
| `Neural_Networks_Output/Scaler/` | fitted scalers (`.save`, loaded with `joblib`) |
| `Neural_Networks_Output/Predictions/` | predicted and ground-truth trajectories (`.xlsx`) |
| `Neural_Networks_Output/Plots/` | training curves and prediction plots |

Model and scaler files of the all-planes models, as loaded by
`models_complete_comparison.py`:

| Model | Model file | Scaler file(s) |
|---|---|---|
| Timestamp LSTM | `Timestamp_lstm_next_tick_model_18.keras` | `standard_scaler_typical_lstm_next_tick_18.save` |
| Timestamp CNN | `Timestamp_cnn_next_tick_model_18.keras` | `standard_scaler_typical_cnn_next_tick_18.save` |
| PV LSTM | `PV_lstm_next_tick_18ch_final.keras` (best-DTW checkpoint: `PV_lstm_best_rollout_dtw_18ch.keras`) | `scaler_pv_lstm_18ch.save`, `scaler_angles_lstm_18ch.save` |
| PV CNN | `PV_cnn_next_tick_18ch_final.keras` (best-DTW checkpoint: `PV_cnn_best_rollout_dtw_18ch.keras`) | `scaler_pv_cnn_18ch.save`, `scaler_angles_cnn_18ch.save` |

---

## Ablation study

[Ablation_Study/](Ablation_Study/) tests which input modality matters by
training the same network on kinematics only, kinetics only (joint moments and
forces) and both:

```bash
python Ablation_Study/cnn_modality_ablation.py
python Ablation_Study/lstm_modality_ablation.py
python Ablation_Study/ablation_plots.py
```

Results (metrics as JSON/CSV/XLSX, models, scalers and plots) are already
committed under `Ablation_Study/data/` and `Ablation_Study/ablation_plots_output/`.

---

## Acknowledgements

- Gait data: ELEPAP (Hellenic Society for the Protection and Rehabilitation of
  Disabled People).
- Framework: TensorFlow / Keras.
