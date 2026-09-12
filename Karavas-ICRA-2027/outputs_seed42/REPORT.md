# TD-only model comparison

12 typically developed subjects, 59 strides. 6 random subject-level splits (8 train / 2 val / 2 test subjects), 7 augmented copies per training subject. Prediction horizon 10 samples (~200 ms).
Each model is therefore scored on 12 held-out subject evaluations.

## Teacher-forced accuracy at +10 samples, held-out TD subjects (18 channels)

| Model | MAE (deg) | MAE SD | RMSE (deg) | RMSE SD | R2 | R2 SD | R2>0 |
|---|---|---|---|---|---|---|---|
| Timestamp LSTM | 5.66 | 1.82 | 6.66 | 2.02 | -1.26 | 1.39 | 3/12 |
| PV LSTM | 5.48 | 1.60 | 6.49 | 1.78 | -1.17 | 1.26 | 2/12 |
| Timestamp CNN | 5.22 | 1.58 | 6.14 | 1.77 | -1.06 | 1.24 | 4/12 |
| PV CNN | 5.13 | 1.54 | 6.02 | 1.71 | -1.11 | 1.37 | 4/12 |
| Persistence (baseline) | 8.76 | 0.83 | 10.89 | 0.97 | -1.10 | 0.14 | 0/12 |
| Linear (baseline) | 14.56 | 2.27 | 19.60 | 3.07 | -10.85 | 2.16 | 0/12 |

## Per-joint teacher-forced error, sagittal plane (MAE/RMSE, degrees)

| Joint | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| L Hip | 5.25/6.25 | 5.39/6.39 | 5.43/6.47 | 5.00/5.94 |
| L Knee | 7.85/9.93 | 8.14/10.35 | 7.34/9.12 | 7.36/9.45 |
| L Ankle | 4.85/6.02 | 4.69/5.92 | 4.25/5.29 | 4.10/5.19 |
| R Hip | 6.44/7.84 | 6.08/7.34 | 6.32/7.41 | 5.39/6.23 |
| R Knee | 9.27/11.28 | 8.09/10.19 | 7.78/9.48 | 7.09/8.49 |
| R Ankle | 5.32/6.62 | 4.61/5.81 | 4.17/5.26 | 4.16/5.22 |
| Mean | 6.50/7.99 | 6.17/7.67 | 5.88/7.17 | 5.52/6.75 |

## Recursive rollout over 6 strides

| Model | Rollout MAE (deg) | Rollout MAE SD | Rollout R2 | Hip phase lag (samples) |
|---|---|---|---|---|
| Timestamp LSTM | 6.63 | 1.98 | -2.15 | 2.92 |
| PV LSTM | 6.21 | 1.85 | -2.06 | 1.17 |
| Timestamp CNN | 5.85 | 1.74 | -1.62 | 2.50 |
| PV CNN | 5.54 | 1.72 | -1.71 | 1.17 |

## Mean absolute error per gait phase (degrees)

| Phase | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| Loading Response | 5.07 | 5.00 | 4.78 | 4.68 |
| Mid Stance | 4.74 | 4.69 | 4.41 | 4.42 |
| Terminal Stance | 5.31 | 4.82 | 4.68 | 4.70 |
| Pre- Swing | 7.07 | 6.14 | 5.88 | 5.85 |
| Initial Swing | 7.02 | 6.85 | 6.65 | 6.44 |
| Mid Swing | 6.14 | 6.48 | 5.90 | 5.74 |
| Terminal Swing | 5.39 | 5.31 | 5.04 | 4.78 |
