# TD-only model comparison

12 typically developed subjects, 60 strides. 5 random subject-level splits (8 train / 2 val / 2 test subjects), 7 augmented copies per training subject.
Each model is therefore scored on 10 held-out subject evaluations.

## Teacher-forced accuracy on held-out TD subjects (all 18 channels)

| Model | MAE (deg) | MAE SD | RMSE (deg) | RMSE SD | R2 | R2 SD | R2>0 |
|---|---|---|---|---|---|---|---|
| Timestamp LSTM | 3.35 | 0.54 | 4.04 | 0.59 | -0.05 | 0.70 | 6/10 |
| PV LSTM | 3.51 | 0.53 | 4.21 | 0.55 | -0.11 | 0.78 | 6/10 |
| Timestamp CNN | 4.21 | 0.52 | 5.00 | 0.58 | -0.51 | 0.91 | 5/10 |
| PV CNN | 4.18 | 0.56 | 4.94 | 0.59 | -0.53 | 1.09 | 6/10 |

## Per-joint teacher-forced error, sagittal plane (MAE/RMSE, degrees)

| Joint | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| L Hip | 3.15/4.01 | 3.61/4.65 | 4.18/5.06 | 4.51/5.50 |
| L Knee | 4.40/5.38 | 5.00/6.07 | 6.64/7.86 | 6.10/7.31 |
| L Ankle | 2.67/3.26 | 2.95/3.63 | 2.90/3.73 | 3.10/3.94 |
| R Hip | 3.73/4.48 | 4.42/5.22 | 5.00/5.88 | 5.19/5.89 |
| R Knee | 4.99/6.13 | 5.66/6.83 | 6.53/8.00 | 6.08/7.48 |
| R Ankle | 2.86/3.57 | 2.82/3.58 | 3.47/4.49 | 3.27/4.29 |
| Mean | 3.63/4.47 | 4.08/5.00 | 4.79/5.84 | 4.71/5.73 |

## Recursive rollout over 6 strides

| Model | Rollout MAE (deg) | Rollout MAE SD | Rollout R2 | Hip phase lag (samples) |
|---|---|---|---|---|
| Timestamp LSTM | 7.70 | 1.39 | -3.77 | 5.10 |
| PV LSTM | 6.04 | 1.98 | -1.89 | 0.70 |
| Timestamp CNN | 4.83 | 0.95 | -0.84 | 2.00 |
| PV CNN | 4.49 | 0.56 | -0.65 | 1.10 |

## Mean absolute error per gait phase (degrees)

| Phase | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| Loading Response | 3.48 | 3.47 | 3.83 | 3.89 |
| Mid Stance | 3.06 | 3.08 | 3.62 | 3.63 |
| Terminal Stance | 2.96 | 3.17 | 3.92 | 3.88 |
| Pre- Swing | 3.71 | 3.73 | 4.99 | 4.99 |
| Initial Swing | 3.66 | 4.01 | 4.96 | 4.85 |
| Mid Swing | 3.40 | 3.81 | 4.61 | 4.38 |
| Terminal Swing | 3.64 | 3.71 | 4.05 | 4.18 |
