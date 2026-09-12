# Karavas ICRA 2027 — TD-only gait prediction

Train on typically developed (TD) gait, predict held-out TD subjects, compare
four networks. No cerebral-palsy data, no exoskeleton deployment — one story:

> Four forward models of TD gait — **Timestamp LSTM**, **PV LSTM**,
> **Timestamp CNN**, **PV CNN** — trained under an identical pipeline on TD
> subjects and compared on TD subjects they have never seen.

Self-contained: nothing here imports from the older `Neural_Networks_*` folders,
which keep the CP-based study intact.

## Run it

```bash
python run_comparison.py --quick     # ~1 min smoke test: 1 split, 5 epochs
python run_comparison.py             # full run: 5 splits x 4 models
python run_comparison.py --repeats 10 --aug-rounds 9
```

Run from inside this folder. Everything lands in `outputs/`:

| File | What it is |
|---|---|
| `REPORT.md` | every table, formatted for the paper |
| `table_generalization.csv` | teacher-forced MAE/RMSE/R² per model over held-out subjects |
| `table_per_joint_sagittal.csv` | per-joint MAE/RMSE, sagittal plane |
| `table_rollout_stability.csv` | recursive-rollout error and hip phase lag |
| `table_per_phase.csv` | error per gait phase |
| `per_subject_metrics.csv`, `per_channel_metrics.csv` | raw per-evaluation rows |
| `run_config.json` | subjects, the exact splits, and every setting used |
| `figures/` | the five figures |

## The data

12 TD subjects (NV031–NV046), 60 strides, from `../Data_Normal/`. Each `.xlsx`
is one 51-sample gait cycle: 18 joint angles (hip/knee/ankle × sagittal/frontal/
transverse × left/right) plus the measured foot-off percentage per side.

Small enough that how subjects are split matters more than anything else in the
pipeline, which is why the evaluation is **repeated random subject-level
splits**: 5 independent draws of 8 train / 2 validation / 2 test subjects, each
with its own seed. A model is scored on every held-out subject of every split,
and the tables report mean ± SD across those evaluations.

## Pipeline

```
load 51-sample strides per subject   (endpoints averaged when they disagree >5°)
        ↓
subject-level split                  (whole subjects, never strides)
        ↓
augment TRAINING subjects only       (time warp → amplitude scale → noise → SG smooth)
        ↓
phase variable per stride            (from sagittal hip + that stride's foot-off)
        ↓
right leg rolled half a stride       (so each leg is expressed from its own heel strike)
        ↓
scalers fit on training data only
        ↓
rolling windows built within one subject's own trials
```

Three things this guarantees, and the earlier pipeline did not:

1. **No augmentation leakage.** `Data_Normal/randomized_data_healthy.xlsx` and
   `dynamics_total_augmented.xlsx` pool all 60 strides *before* augmenting, so a
   held-out subject's stride reappears — noised — in training. Those files are
   unused here; augmentation is generated inside each split from its training
   subjects only.
2. **No scaler leakage.** Both scalers are fit on the training split alone.
3. **No cross-subject windows.** Windows are built per subject, so no input ever
   spans two people.

## The prediction task

All four models take 51 consecutive samples and predict the 18 joint angles
**10 samples (~200 ms) ahead** — the order of the actuation and sensing delay a
wearable-robot controller has to cover. The horizon matters enormously:

| Horizon | Persistence | Linear extrap. | Timestamp CNN |
|---|---|---|---|
| 1 sample | **1.32°** | **0.61°** | 4.60° |
| 10 samples (~200 ms) | 8.40° | 14.16° | **4.32°** |
| 25 samples (~½ stride) | 9.40° | 36.19° | **4.25°** |

At one sample ahead the task is trivially solved by repeating the last sample,
and every network loses to it — such a benchmark cannot rank models. Both
baselines are therefore computed on every run and reported alongside the
networks. Change the horizon with `--horizon`.

## The four models

All four emit the full horizon in one forward pass, which is also what lets the
recursive rollout chain blocks of predictions. Identical horizon across models,
so the comparison isolates backbone and conditioning.

| | Timestamp (18 inputs) | Phase variable (20 inputs) |
|---|---|---|
| **LSTM** | LSTM(192) ×2 + Dense(256) | same, + s_L and s_R |
| **CNN** | Conv 32/48 → pool → Conv 256/256 → pool → Dense(256) | same, + s_L and s_R |

LSTM: lr 1e-3, 150 epochs. CNN: lr 1e-4, 200 epochs. Both with
`ReduceLROnPlateau` and early stopping (patience 25, best weights restored) —
with 8 training subjects the validation loss bottoms out well inside the budget.

## Three readouts

Distinct measurements, often conflated:

- **Teacher forced** — the model always sees measured data. Best-case per-step
  accuracy; no error can accumulate.
- **Recursive rollout** — the model is fed its own output for 6 strides. The only
  readout where drift and phase misalignment appear. A PV model still receives a
  correct phase value each step, because the phase variable is computed from
  measured hip kinematics rather than from the prediction; that self-correction
  is the property under test. Reported both as MAE and as **hip phase lag** —
  the signed sample shift that best realigns the predicted hip with the measured
  one over the final stride, which separates "wrong shape" from "right shape,
  wrong timing".
- **Phase binned** — teacher-forced error grouped into the seven gait phases,
  anchored to each subject's own measured toe-off.

## Notes for the paper text

Points where the current manuscript does not match what the code does:

- **No Butterworth filter.** The manuscript says a low-pass Butterworth filter
  was applied. No filtering exists anywhere in this repository. The only
  smoothing is Savitzky-Golay, and only inside augmentation. Either drop the
  claim or add the filter.
- **The phase-variable constant `c` is measured, not fixed at 0.53.** Each
  stride uses its own foot-off percentage — 59.6% on average across this
  cohort, ranging 54.5–62.4% between subjects — so `c` varies per stride and
  per subject.
- **Saturation is ~13%, not 20–25%.** Measured over all 120 stride–leg pairs,
  the phase variable holds at `s = 1` over 12.6% of the cycle on average
  (median 15.7%, up to ~25% in the longest cases). The manuscript's 20–25% is
  an upper bound quoted as the typical value. The terminal-swing argument
  still applies, but the number must change.
- **The right-leg half-stride roll is a modelling choice worth stating.** The
  Vicon export normalizes each leg to its *own* gait cycle, so in the source
  files both legs are already in phase. Rolling the right leg half a stride
  restores the bilateral relationship of real walking. The earlier pipeline did
  this without stating it; it is kept here for consistency, and it should be
  stated.
- **All four models now share the same multi-step horizon.** The old
  18-channel Timestamp CNN was the only one with a 10-sample head, and the
  manuscript carried a caveat about using only its first sample. Now every
  model emits the full horizon, so the special case and its caveat both go.
- **One-step-ahead results are not reportable.** At a one-sample horizon
  persistence (1.31°) and linear extrapolation (0.59°) beat every network
  (best 3.35°). Any table at that horizon measures nothing.

## Files

| File | |
|---|---|
| `td_pipeline.py` | loading, splitting, augmentation, phase variable, windows, scaling |
| `td_models.py` | the four architectures and their training callbacks |
| `td_eval.py` | the three readouts, metrics, figures |
| `run_comparison.py` | entry point: trains everything, writes every table and figure |
| `make_tables.py` | CSVs -> `outputs/tables.tex`, paste-ready IEEE tables |
| `diagnose.py` | why-is-it-underfitting harness (baselines vs augmentation vs batch size) |
| `HANDOFF.md` | state of the work, deadline, and what still needs writing |
