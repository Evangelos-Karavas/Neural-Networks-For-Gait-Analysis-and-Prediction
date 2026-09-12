# TD-only model comparison

12 typically developed subjects, 59 strides. 6 random subject-level splits (8 train / 2 val / 2 test subjects), 7 augmented copies per training subject. Prediction horizon 10 samples (~200 ms).
Each model is therefore scored on 12 held-out subject evaluations.

## Teacher-forced accuracy at +10 samples, held-out TD subjects (18 channels)

| Model | MAE (deg) | MAE SD | RMSE (deg) | RMSE SD | R2 | R2 SD | R2>0 |
|---|---|---|---|---|---|---|---|
| Timestamp LSTM | 5.16 | 1.30 | 6.19 | 1.52 | -1.07 | 1.16 | 3/12 |
| PV LSTM | 5.15 | 1.56 | 6.18 | 1.77 | -1.09 | 1.28 | 4/12 |
| Timestamp CNN | 4.89 | 1.51 | 5.82 | 1.71 | -0.95 | 1.17 | 4/12 |
| PV CNN | 4.86 | 1.50 | 5.75 | 1.65 | -0.82 | 1.16 | 4/12 |
| Persistence (baseline) | 8.76 | 0.83 | 10.89 | 0.97 | -1.10 | 0.14 | 0/12 |
| Linear (baseline) | 14.56 | 2.27 | 19.60 | 3.07 | -10.85 | 2.16 | 0/12 |

## Per-joint teacher-forced error, sagittal plane (MAE/RMSE, degrees)

| Joint | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| L Hip | 4.76/5.89 | 5.50/6.60 | 4.74/5.71 | 4.58/5.52 |
| L Knee | 7.44/9.53 | 7.53/9.68 | 6.75/8.73 | 6.61/8.24 |
| L Ankle | 3.97/5.13 | 4.26/5.62 | 3.72/4.69 | 3.98/5.00 |
| R Hip | 5.75/7.07 | 5.10/6.23 | 5.16/6.10 | 4.93/5.82 |
| R Knee | 8.01/9.85 | 7.64/9.52 | 7.06/8.87 | 7.78/9.34 |
| R Ankle | 4.67/5.83 | 4.66/5.81 | 4.32/5.40 | 4.27/5.27 |
| Mean | 5.77/7.22 | 5.78/7.24 | 5.29/6.58 | 5.36/6.53 |

## Recursive rollout over 6 strides

| Model | Rollout MAE (deg) | Rollout MAE SD | Rollout R2 | Hip phase lag (samples) |
|---|---|---|---|---|
| Timestamp LSTM | 6.24 | 1.68 | -1.85 | 2.83 |
| PV LSTM | 5.95 | 1.57 | -1.92 | 1.50 |
| Timestamp CNN | 5.60 | 1.77 | -1.40 | 2.42 |
| PV CNN | 5.31 | 1.74 | -1.27 | 1.08 |

## Mean absolute error per gait phase (degrees)

| Phase | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| Loading Response | 4.57 | 4.70 | 4.42 | 4.66 |
| Mid Stance | 4.39 | 4.39 | 4.14 | 4.23 |
| Terminal Stance | 4.46 | 4.54 | 4.36 | 4.24 |
| Pre- Swing | 6.05 | 6.16 | 5.80 | 5.65 |
| Initial Swing | 6.34 | 6.17 | 5.98 | 5.87 |
| Mid Swing | 6.23 | 5.98 | 5.69 | 5.43 |
| Terminal Swing | 5.05 | 5.01 | 4.65 | 4.78 |
