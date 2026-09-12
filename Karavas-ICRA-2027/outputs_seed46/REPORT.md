# TD-only model comparison

12 typically developed subjects, 59 strides. 6 random subject-level splits (8 train / 2 val / 2 test subjects), 7 augmented copies per training subject. Prediction horizon 10 samples (~200 ms).
Each model is therefore scored on 12 held-out subject evaluations.

## Teacher-forced accuracy at +10 samples, held-out TD subjects (18 channels)

| Model | MAE (deg) | MAE SD | RMSE (deg) | RMSE SD | R2 | R2 SD | R2>0 |
|---|---|---|---|---|---|---|---|
| Timestamp LSTM | 5.16 | 1.34 | 6.16 | 1.46 | -1.04 | 1.06 | 2/12 |
| PV LSTM | 5.11 | 1.40 | 6.10 | 1.53 | -1.02 | 1.13 | 4/12 |
| Timestamp CNN | 4.87 | 1.64 | 5.75 | 1.77 | -0.80 | 1.23 | 4/12 |
| PV CNN | 4.84 | 1.51 | 5.74 | 1.67 | -0.93 | 1.17 | 4/12 |
| Persistence (baseline) | 8.76 | 0.83 | 10.89 | 0.97 | -1.10 | 0.14 | 0/12 |
| Linear (baseline) | 14.56 | 2.27 | 19.60 | 3.07 | -10.85 | 2.16 | 0/12 |

## Per-joint teacher-forced error, sagittal plane (MAE/RMSE, degrees)

| Joint | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| L Hip | 4.80/5.86 | 4.94/5.94 | 4.63/5.56 | 4.54/5.42 |
| L Knee | 6.73/8.59 | 7.45/9.35 | 6.74/8.50 | 6.25/7.83 |
| L Ankle | 4.39/5.51 | 4.11/5.15 | 3.77/4.78 | 4.07/5.14 |
| R Hip | 5.89/7.17 | 5.96/7.12 | 5.50/6.49 | 5.13/6.04 |
| R Knee | 7.61/9.61 | 7.57/9.42 | 7.22/8.86 | 6.75/8.41 |
| R Ankle | 4.49/5.67 | 4.31/5.49 | 3.69/4.80 | 4.15/5.19 |
| Mean | 5.65/7.07 | 5.72/7.08 | 5.26/6.50 | 5.15/6.34 |

## Recursive rollout over 6 strides

| Model | Rollout MAE (deg) | Rollout MAE SD | Rollout R2 | Hip phase lag (samples) |
|---|---|---|---|---|
| Timestamp LSTM | 6.18 | 1.40 | -1.91 | 2.58 |
| PV LSTM | 5.52 | 1.45 | -1.49 | 1.08 |
| Timestamp CNN | 5.58 | 1.60 | -1.19 | 2.08 |
| PV CNN | 5.39 | 1.53 | -1.38 | 0.83 |

## Mean absolute error per gait phase (degrees)

| Phase | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| Loading Response | 4.65 | 4.90 | 4.51 | 4.57 |
| Mid Stance | 4.19 | 4.23 | 4.08 | 4.01 |
| Terminal Stance | 4.26 | 4.23 | 4.30 | 4.20 |
| Pre- Swing | 6.20 | 5.79 | 5.50 | 5.46 |
| Initial Swing | 6.58 | 6.45 | 5.99 | 5.98 |
| Mid Swing | 6.18 | 6.08 | 5.73 | 5.67 |
| Terminal Swing | 5.24 | 5.22 | 4.74 | 4.82 |
