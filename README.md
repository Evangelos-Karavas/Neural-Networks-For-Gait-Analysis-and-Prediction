# Neural Networks for Human Gait Analysis and Prediction

A comprehensive collection of deep learning models for analyzing and predicting human gait patterns in both typically developed children and those with cerebral palsy. This project uses LSTM and CNN neural networks trained on kinematic and kinetic data from the lower limbs.

**Update:** Latest implementation uses joint kinematics and kinetics data for improved prediction accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Data](#data)
- [Models and Approaches](#models-and-approaches)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Applications](#applications)
- [References](#references)

---

## Project Overview

This repository contains neural network implementations for gait analysis developed as part of a Master's Thesis at NTUA (National Technical University of Athens) - Mechanical Engineering Department. The models predict joint angles and dynamics based on biomechanical measurements from children with and without cerebral palsy.

The trained models are designed to be integrated with ROS2 for real-time lower-limb exoskeleton control applications.

**Data Source:** All gait data were contributed by [ELEPAP](https://elepap.gr/) (Hellenic Society for the Protection and Rehabilitation of Disabled People).

---

## Features

- **Multiple Neural Network Architectures:** LSTM and CNN models for temporal sequence learning
- **Multi-Prediction Approaches:**
  - **Phase Variable-based:** Using normalized gait phases for prediction
  - **Timestamp-based:** Using time-series data directly
  - **Kinematics-Kinetics:** Joint forces, moments, and angles prediction
- **Data Preprocessing:** Automatic data augmentation and normalization pipelines
- **Model Persistence:** Saved Keras models and StandardScaler objects for deployment
- **Visualization Tools:** Prediction plotting and analysis capabilities
- **Support for Multiple Demographics:** Models trained on both typically developed and cerebral palsy populations

---

## Project Structure

```
Neural-Networks-For-Human-Gait-Prediction/
│
├── README.md                                    # This file
├── data_randomize_kinematics.py                # Data augmentation for kinematic data
├── data_randomize_kinetics.py                  # Data augmentation for kinetic data
│
├── Data_Normal/                                 # Typically developed children data
│   ├── *.xlsx                                   # Individual subject files
│   └── randomized_data_healthy.xlsx             # Augmented dataset
│
├── Data_CP/                                     # Cerebral palsy patient data
│   └── *.xlsx                                   # Individual patient files (1 stride per file)
│
├── Kinematics_Kinetics_Neural_Networks/        # Latest implementation
│   ├── kinematics_kinetics_lstm.py              # LSTM model (36→18 features)
│   ├── dynamics_timestamps_lstm.py              # Dynamics LSTM (54→18 features)
│   └── newpred.py                               # Prediction pipeline
│
├── Kinematics_Neural_Networks/                 # Kinematics-only models
│   ├── timestamps_CNN.py                        # CNN for kinematic prediction
│   └── timetamps_LSTM.py                        # LSTM for kinematic prediction
│
├── Phase_Variable_Neural_Networks/             # Phase variable approach
│   ├── phase_variable_lstm.py                   # Main phase variable LSTM
│   └── lstm_dtw.py                              # DTW-based LSTM variant
│
├── Saved_Models/                                # Trained Keras models
│   ├── td_nextstride_54to18.keras               # Top model: 54 inputs → 18 outputs
│   ├── best_dynamics_lstm_nextstride.keras
│   ├── PV_lstm_model.keras
│   ├── Timestamp_lstm_model.keras
│   └── ... (additional models)
│
├── Scaler/                                      # Fitted StandardScaler objects
│   ├── td_x36_scaler.save                       # Input scaler (36 features)
│   ├── td_y18_scaler.save                       # Output scaler (18 features)
│   ├── ang_scaler.save
│   ├── dyn_scaler.save
│   └── ... (additional scalers)
│
└── Predictions/                                 # Model predictions and plots
    ├── scaler_x36.save
    ├── scaler_y18.save
    ├── td_nextstride_model.keras
    └── Plots/                                   # Prediction visualization plots
```

---

## Data

### Data Format

All data files are in **Excel (.xlsx)** format containing biomechanical measurements from motion capture analysis:

**Kinematic Features (Joint Angles - 3D):**
- LHipAngles, RHipAngles
- LKneeAngles, RKneeAngles
- LAnkleAngles, RAnkleAngles

**Kinetic Features (Forces and Moments - 3D):**
- Joint Moments: LHipMoment, RHipMoment, LKneeMoment, RKneeMoment, LAnkleMoment, RAnkleMoment
- Joint Forces: LHipForce, RHipForce, LKneeForce, RKneeForce, LAnkleForce, RAnkleForce

**Event Markers:**
- Left Foot Off (timing marker)
- Right Foot Off (timing marker)

### Data Specifications

- **Stride Length:** 51 time samples per gait cycle
- **Typically Developed (TD):** Healthy children baseline data
- **Cerebral Palsy (CP):** Patient data with neurological conditions
- **Augmentation:** Training data expanded 8× through noise injection for improved generalization

---

## Models and Approaches

### 1. Kinematics-Kinetics LSTM (Recommended)

**File:** `Kinematics_Kinetics_Neural_Networks/kinematics_kinetics_lstm.py`

- **Input:** 36 features (18 joint forces/moments + 18 joint angles)
- **Output:** 18 features (next stride joint angles)
- **Architecture:** LSTM with LayerNormalization and Dropout
- **Task:** Next-stride angle prediction
- **Saved Model:** `td_nextstride_54to18.keras`

```python
# Quick usage
model = tf.keras.models.load_model('Saved_Models/td_nextstride_54to18.keras')
X_scaled = scaler_x.transform(X)  # Scale input
predictions = model.predict(X_scaled)
predictions = scaler_y.inverse_transform(predictions)  # Unscale output
```

### 2. Dynamics Timestamp LSTM

**File:** `Kinematics_Kinetics_Neural_Networks/dynamics_timestamps_lstm.py`

- **Input:** 54 features (18 moments + 18 forces + 18 angles)
- **Output:** 18 features (joint angles)
- **Task:** Inverse dynamics prediction
- **Approach:** Timestamp-based (no phase variable normalization)

### 3. Phase Variable LSTM

**File:** `Phase_Variable_Neural_Networks/phase_variable_lstm.py`

- **Input:** 8 features (2 phase variables + 6 sagittal plane angles)
- **Output:** 6 features (next stride angles - sagittal plane)
- **Task:** Phase-normalized gait prediction
- **Benefit:** Improved generalization across different gait speeds

**Phase Variable Computation:**
- Normalized time within gait cycle (0 to 1)
- Computed independently for left and right legs based on foot-off events
- Provides invariance to gait duration variations

### 4. Kinematics CNN & LSTM

**File:** `Kinematics_Neural_Networks/`

- CNN-based temporal feature extraction
- LSTM-based sequence modeling
- Limited to kinematic data (angles only)
- Suitable for real-time applications with reduced computational cost

---

## Installation

### Requirements

- Python 3.8+
- TensorFlow/Keras 2.10+
- NumPy, Pandas, Scikit-learn
- Matplotlib (for visualization)
- Joblib (for scaler persistence)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Evangelos-Karavas/Neural-Networks-For-Human-Gait-Prediction.git
cd Neural-Networks-For-Human-Gait-Prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib joblib openpyxl
```

---

## Usage

### 1. Data Preprocessing

Augment and randomize kinematic data:
```bash
python data_randomize_kinematics.py
python data_randomize_kinetics.py
```

These scripts generate augmented training datasets with controlled noise injection.

### 2. Training Models

Train the kinematics-kinetics LSTM model:
```bash
python Kinematics_Kinetics_Neural_Networks/kinematics_kinetics_lstm.py
```

The script will:
- Load and preprocess data
- Fit StandardScaler on training data
- Train LSTM model
- Save model and scaler for inference
- Generate predictions and plots

### 3. Making Predictions

Use the prediction pipeline:
```bash
python Kinematics_Kinetics_Neural_Networks/newpred.py
```

Or programmatically:
```python
import tensorflow as tf
import joblib
import numpy as np

# Load model and scalers
model = tf.keras.models.load_model('Saved_Models/td_nextstride_54to18.keras')
scaler_x = joblib.load('Scaler/td_x36_scaler.save')
scaler_y = joblib.load('Scaler/td_y18_scaler.save')

# Prepare input (shape: N × 36)
X_input = np.random.randn(10, 36)

# Scale, predict, unscale
X_scaled = scaler_x.transform(X_input)
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

print("Predicted joint angles:", y_pred)
```

### 4. Visualization

Models automatically generate prediction plots in `Predictions/Plots/`:
- Sagittal plane angle predictions vs. actual
- 3D joint moment/force comparisons
- Error distributions

---

## Model Training

### Architecture Details

**LSTM Network:**
```
Input Layer (36)
    ↓
LSTM (128 units, return_sequences=True)
    ↓
LayerNormalization
    ↓
Dropout (0.2)
    ↓
LSTM (64 units)
    ↓
Dense (32, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (18)
```

### Training Parameters

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Mean Squared Error
- **Metric:** Root Mean Squared Error
- **Epochs:** 100-150
- **Batch Size:** 512
- **Validation Split:** 20%
- **Early Stopping:** Yes (patience=15)

### Data Split

- **Training:** 80% (typically developed data)
- **Validation:** 20% (typically developed data)
- **Testing:** CP patient data (separate populations)

---

## Applications

### Primary Application: Exoskeleton Control

These models power the real-time joint angle prediction for a lower-limb exoskeleton designed to assist children with cerebral palsy. Integration with ROS2 enables:

- **Predictive Control:** Anticipatory angle predictions enable smoother, more natural movement
- **Adaptive Assistance:** Models adapt to individual gait patterns
- **Real-time Inference:** Scalers and models optimized for fast inference

**Related Project:** [ROS2-Lower-Limb-Exoskeleton-Control](https://github.com/Evangelos-Karavas/ROS2-Lower-Limb-Exoskeleton-Control)

### Other Applications

- Gait assessment and analysis
- Rehabilitation monitoring
- Prosthetic control
- Biomechanical research
- Motion prediction for animation

---

## Key Implementation Notes

### Scaler Management

All models use StandardScaler fitted **only on typically developed data** to maintain consistent baseline:
- `td_x36_scaler.save` - Input features (forces, moments, angles)
- `td_y18_scaler.save` - Output features (joint angles)
- Applied identically during training and inference for reproducibility

### Data Organization

- **Per-file structure:** Each Excel file represents continuous gait cycles (multiple strides)
- **CP data:** One stride per file for easier management
- **TD data:** Merged and augmented for statistical robustness

### Convergence Tips

If retraining models:
1. Ensure CP data is loaded with correct sheet names ("Data") and skiprows=[1,2]
2. Normalize all inputs/outputs using fitted scalers
3. Use batch normalization/layer normalization for stability
4. Monitor validation loss for early stopping

---

## References & Attribution

- **Data Source:** ELEPAP (Hellenic Society for the Protection and Rehabilitation of Disabled People)
- **Framework:** TensorFlow/Keras
- **Related Work:** Master Thesis on Neural Networks in Gait Analysis (NTUA)

For detailed methodology, please refer to the Master's Thesis chapters 3-4 on Phase Variables and data preprocessing.


## License

This project is provided as-is for research and educational purposes. The gait data from ELEPAP is proprietary and cannot be redistributed without permission.

---

## Contact & Support

For questions about the project architecture, model training, or integration with external systems, please refer to the individual script headers for specific implementation details.
