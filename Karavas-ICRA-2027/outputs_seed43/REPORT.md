# TD-only model comparison

12 typically developed subjects, 59 strides. 6 random subject-level splits (8 train / 2 val / 2 test subjects), 7 augmented copies per training subject. Prediction horizon 10 samples (~200 ms).
Each model is therefore scored on 12 held-out subject evaluations.

## Teacher-forced accuracy at +10 samples, held-out TD subjects (18 channels)

| Model | MAE (deg) | MAE SD | RMSE (deg) | RMSE SD | R2 | R2 SD | R2>0 |
|---|---|---|---|---|---|---|---|
| Timestamp LSTM | 5.16 | 1.33 | 6.17 | 1.47 | -1.00 | 1.11 | 2/12 |
| PV LSTM | 5.06 | 1.23 | 6.07 | 1.31 | -0.98 | 0.97 | 2/12 |
| Timestamp CNN | 4.83 | 1.37 | 5.72 | 1.49 | -0.81 | 1.03 | 4/12 |
| PV CNN | 4.80 | 1.18 | 5.70 | 1.30 | -0.76 | 0.95 | 5/12 |
| Persistence (baseline) | 8.76 | 0.83 | 10.89 | 0.97 | -1.10 | 0.14 | 0/12 |
| Linear (baseline) | 14.56 | 2.27 | 19.60 | 3.07 | -10.85 | 2.16 | 0/12 |

## Per-joint teacher-forced error, sagittal plane (MAE/RMSE, degrees)

| Joint | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| L Hip | 5.09/6.12 | 4.86/5.76 | 4.63/5.51 | 4.44/5.34 |
| L Knee | 6.32/8.27 | 6.30/8.11 | 7.51/9.01 | 7.46/8.99 |
| L Ankle | 4.58/5.54 | 4.47/5.55 | 3.99/4.84 | 4.19/5.30 |
| R Hip | 6.53/7.87 | 6.40/7.59 | 5.34/6.27 | 5.23/6.09 |
| R Knee | 8.63/10.85 | 8.50/10.63 | 6.83/8.65 | 7.22/9.06 |
| R Ankle | 4.58/5.98 | 4.52/5.84 | 4.22/5.30 | 4.58/5.65 |
| Mean | 5.96/7.44 | 5.84/7.25 | 5.42/6.60 | 5.52/6.74 |

## Recursive rollout over 6 strides

| Model | Rollout MAE (deg) | Rollout MAE SD | Rollout R2 | Hip phase lag (samples) |
|---|---|---|---|---|
| Timestamp LSTM | 5.70 | 1.38 | -1.53 | 1.92 |
| PV LSTM | 5.67 | 1.14 | -1.52 | 1.50 |
| Timestamp CNN | 5.53 | 1.71 | -1.31 | 2.25 |
| PV CNN | 5.24 | 1.38 | -1.17 | 0.92 |

## Mean absolute error per gait phase (degrees)

| Phase | Timestamp LSTM | PV LSTM | Timestamp CNN | PV CNN |
|---|---|---|---|---|
| Loading Response | 4.62 | 4.61 | 4.37 | 4.49 |
| Mid Stance | 4.40 | 4.27 | 4.16 | 4.25 |
| Terminal Stance | 4.50 | 4.32 | 4.40 | 4.31 |
| Pre- Swing | 5.93 | 5.93 | 5.61 | 5.35 |
| Initial Swing | 6.83 | 6.24 | 5.93 | 5.87 |
| Mid Swing | 5.95 | 6.11 | 5.62 | 5.44 |
| Terminal Swing | 4.81 | 4.93 | 4.46 | 4.54 |
