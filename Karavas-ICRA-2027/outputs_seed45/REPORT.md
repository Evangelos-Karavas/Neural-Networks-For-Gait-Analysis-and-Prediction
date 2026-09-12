# TD-only model comparison

12 typically developed subjects, 59 strides. 6 random subject-level splits (8 train / 2 val / 2 test subjects), 7 augmented copies per training subject. Prediction horizon 10 samples (~200 ms).
Each model is therefore scored on 12 held-out subject evaluations.

## Teacher-forced accuracy at +10 samples, held-out TD subjects (18 channels)

| Model | MAE (deg) | MAE SD | RMSE (deg) | RMSE SD | R2 | R2 SD | R2>0 |
|---|---|---|---|---|---|---|---|
| Timestamp LSTM | 5.49 | 1.69 | 6.50 | 1.85 | -1.22 | 1.32 | 3/12 |
| PV LSTM | 5.31 | 1.57 | 6.38 | 1.76 | -1.06 | 1.18 | 3/12 |
| Timestamp CNN | 4.86 | 1.60 | 5.73 | 1.73 | -0.83 | 1.11 | 5/12 |
| PV CNN | 4.83 | 1.41 | 5.70 | 1.55 | -0.81 | 1.10 | 4/12 |
| Persistence (baseline) | 8.76 | 0.83 | 10.89 | 0.97 | -1.10 | 0.14 | 0/12 |
| Linear (baseline) | 14.56 | 2.27 | 19.60 | 3.07 | -10.85 | 2.16 | 0/12 |

## Per-joint teacher-forced error, sagittal plane (MAE/RMSE, degrees)

| Joint | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| L Hip | 4.90/5.93 | 4.84/5.92 | 4.85/5.88 | 4.16/5.00 |
| L Knee | 6.59/8.22 | 7.38/9.44 | 6.22/7.55 | 6.30/7.67 |
| L Ankle | 4.70/5.77 | 4.89/6.08 | 4.06/5.05 | 4.21/5.26 |
| R Hip | 6.31/7.89 | 6.27/7.95 | 6.65/7.78 | 5.80/6.69 |
| R Knee | 9.82/12.30 | 8.28/10.50 | 6.95/8.63 | 6.74/8.29 |
| R Ankle | 5.00/6.29 | 5.35/6.80 | 4.61/5.65 | 4.70/5.79 |
| Mean | 6.22/7.73 | 6.17/7.78 | 5.56/6.76 | 5.32/6.45 |

## Recursive rollout over 6 strides

| Model | Rollout MAE (deg) | Rollout MAE SD | Rollout R2 | Hip phase lag (samples) |
|---|---|---|---|---|
| Timestamp LSTM | 7.01 | 1.95 | -2.27 | 4.08 |
| PV LSTM | 5.85 | 1.50 | -1.80 | 1.17 |
| Timestamp CNN | 5.42 | 1.75 | -1.24 | 2.08 |
| PV CNN | 5.43 | 1.51 | -1.44 | 1.00 |

## Mean absolute error per gait phase (degrees)

| Phase | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| Loading Response | 4.80 | 4.54 | 4.29 | 4.45 |
| Mid Stance | 4.50 | 4.26 | 4.15 | 4.18 |
| Terminal Stance | 5.04 | 4.84 | 4.59 | 4.44 |
| Pre- Swing | 6.52 | 6.44 | 5.86 | 5.49 |
| Initial Swing | 6.84 | 6.95 | 5.84 | 5.91 |
| Mid Swing | 6.21 | 6.09 | 5.43 | 5.35 |
| Terminal Swing | 5.41 | 5.11 | 4.54 | 4.57 |
