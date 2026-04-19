# Neural Networks for Human Gait Analysis and Prediction

A comprehensive collection of deep learning models for analyzing and predicting human gait patterns in both typically developed children and those with cerebral palsy. This project uses LSTM and CNN neural networks trained on kinematic and kinetic data from the lower limbs.

**Update:** Latest implementation uses joint kinematics and kinetics data for improved prediction accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Approaches](#models-and-approaches)
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
- **Data Preprocessing:** Automatic data augmentation and normalization pipelines
- **Model Persistence:** Saved Keras models and StandardScaler objects for deployment

---

## Project Structure

```
Neural-Networks-For-Gait-Analysis-and-Prediction/
│
├── README.md
├── Data_Augmentation/
│   ├── data_randomize_kinematics.py
│   └── data_randomize_kinetics.py
│
├── Neural_Networks_Phase_Variable/
│   ├── phase_variable_cnn.py
│   └── phase_variable_lstm.py
│
├── Neural_Networks_Timestamps/
│   ├── timestamps_CNN.py
│   └── timetamps_LSTM.py
│
├── Saved_Models/
│   ├── PV_best_rollout_cnn.keras
│   ├── PV_rolling_next_tick_cnn.keras
│   ├── PV_rolling_next_tick_lstm.keras
│   ├── Timestamp_cnn_model.keras
│   └── Timestamp_lstm_model.keras
│
├── Scaler/
│   ├── scaler_angles_cnn.save
│   ├── scaler_angles.save
│   ├── scaler_pv_cnn.save
│   ├── scaler_pv.save
│   ├── standard_scaler_cp_lstm.save
│   ├── standard_scaler_typical_cnn.save
│   └── standard_scaler_typical_lstm.save
│
└── Predictions/
```

---

## Installation

### Requirements

- Python 3.8+
- TensorFlow/Keras 2.10+
- NumPy, Pandas, Scikit-learn
- Matplotlib (for visualization)
- Joblib (for scaler persistence)
- Cuda Cores Libraries are ncie to have

### Setup

```bash
python -m venv venv
source venv/bin/activate        # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Data Augmentation

```bash
python Data_Augmentation/data_randomize_kinematics.py
python Data_Augmentation/data_randomize_kinetics.py
```

### Training Models

**Phase Variable LSTM:**
```bash
python Neural_Networks_Phase_Variable/phase_variable_lstm.py
```

**Phase Variable CNN:**
```bash
python Neural_Networks_Phase_Variable/phase_variable_cnn.py
```

**Timestamp LSTM:**
```bash
python Neural_Networks_Timestamps/timestamps_LSTM.py
```

**Timestamp CNN:**
```bash
python Neural_Networks_Timestamps/timestamps_CNN.py
```

---

## Models and Approaches

### Phase Variable Models
- Located in `Neural_Networks_Phase_Variable/`
- Supports both LSTM and CNN architectures
- Saved models: `PV_best_rollout_cnn.keras`, `PV_rolling_next_tick_lstm.keras`, `PV_rolling_next_tick_cnn.keras`

### Timestamp Models
- Located in `Neural_Networks_Timestamps/`
- LSTM and CNN implementations
- Saved models: `Timestamp_lstm_model.keras`, `Timestamp_cnn_model.keras`

---

## References & Attribution

- **Data Source:** ELEPAP (Hellenic Society for the Protection and Rehabilitation of Disabled People)
- **Framework:** TensorFlow/Keras
- **Related Work:** Master Thesis (NTUA) on the Use of Neural Networks in the Analysis and Prediction of Human GAIT.

