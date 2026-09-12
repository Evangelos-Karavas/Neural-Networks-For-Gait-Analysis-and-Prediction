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
python run_comparison.py             # full run: 6 splits x 4 models, ~9 min on GPU
python run_comparison.py --seed 43 --output outputs_seed43 --no-figures
python aggregate_seeds.py outputs_seed*   # does an effect hold in every seed?
```

On a GPU machine every command above needs the CUDA libraries on the loader
path, or TensorFlow silently falls back to CPU and the run takes 5.6 hours
instead of 9 minutes — see the GPU section of `HANDOFF.md`.

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
| `figures/` | the six figures |

## The data

12 TD subjects (NV031–NV046) from `../Data_Normal/`, 60 recorded strides of
which **59 are usable**. Each `.xlsx` is one 51-sample gait cycle: 18 joint
angles (hip/knee/ankle × sagittal/frontal/transverse × left/right) plus the
measured foot-off percentage per side.

One stride is dropped: `NV037-20181029-5-05-01.xlsx` is missing 12 samples from
every left-leg channel. Filling such gaps with `0.0` would be invisible — zero
is a physically plausible joint angle — and would also corrupt that stride's
phase variable, which is derived from the left hip. **Strides with any missing
joint angle are dropped, loudly, rather than filled.**

Small enough that how subjects are split matters more than anything else in the
pipeline, which is why the test sets **partition the cohort**: 6 splits of
8 train / 2 validation / 2 test, arranged so every subject is held out
**exactly once**. Drawing each test set independently at random — the more
common choice — left a third of this cohort never evaluated while testing others
twice, and the subjects it skipped were the hardest ones, which flattered every
model by about a degree. The tables report mean ± SD over the 12 held-out
evaluations.

Weight initialisation and dropout are seeded per (split, model), so a rerun
reproduces the published numbers exactly.

## Pipeline

```
load 51-sample strides per subject   (drop any with missing angles;
                                      endpoints averaged when they disagree >5°)
        ↓
partitioning subject-level split     (whole subjects, each held out exactly once)
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
   `dynamics_total_augmented.xlsx` pool every stride *before* augmenting, so a
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

Baseline MAE by horizon, measured on the current 59-stride data:

| Horizon | Persistence | Linear extrap. |
|---|---|---|
| 1 sample | **1.31°** | **0.58°** |
| 5 samples | 5.54° | 5.81° |
| 10 samples (~200 ms) | 8.66° | 13.84° |
| 25 samples (~½ stride) | 10.46° | 34.63° |

At one sample ahead the task is trivially solved by repeating the last sample —
no network here gets below 3.35° — so every network loses to a constant-output
rule and such a benchmark cannot rank models. By 10 samples the networks lead
both baselines comfortably (best model 5.13° against 8.66° and 14.56° on the
reported run). Both baselines are therefore computed on every run and reported
alongside the networks; if a change makes the networks lose to them again, that
is a bug, not a result. Change the horizon with `--horizon`.

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

Every disagreement between the manuscript and the code has now been reconciled
in the `.tex`. Listed here so they are not silently reintroduced:

- **No Butterworth filter.** The manuscript claimed one; no filtering exists
  anywhere in this repository. The only smoothing is Savitzky-Golay, inside
  augmentation. *Claim removed.*
- **No Optuna search.** The manuscript credited hyperparameters to an Optuna
  study; `grep -r optuna` returns nothing. *Claim removed*, replaced by what is
  true and more useful — settings are fixed per backbone and identical across
  the two conditionings, so the comparison is controlled.
- **`c` is measured, not fixed at 0.53.** Each stride uses its own foot-off
  percentage — 59.6% mean, 54.5–62.4% across subjects.
- **Saturation is ~13%, not 20–25%.** 12.6% mean, 15.7% median over all
  stride–leg pairs; ~25% is the upper tail, not the typical value.
- **The CNN description must match the code**: all four convolutions are
  stride 2, dropout follows each pooling stage, and the stack reduces the
  51-sample window to a single 256-dimensional descriptor.
- **The right-leg half-stride roll is a modelling choice, and is stated.** The
  Vicon export normalizes each leg to its own cycle, so both legs arrive already
  in phase; the roll restores the bilateral relationship of real walking.
- **All four models share the same multi-step horizon**, so the old
  Timestamp-CNN special case and its caveat are gone.
- **One-step-ahead results are not reportable**, and the paper says why.

One claim must *stay* in the paper: under rollout a PV model keeps receiving a
**measured** phase value (`td_eval.py:145`) while a timestamp model gets nothing
external. That asymmetry is the deployment assumption the representation
encodes, and the comparison looks rigged if it goes unstated.

## Files

| File | |
|---|---|
| `td_pipeline.py` | loading, splitting, augmentation, phase variable, windows, scaling |
| `td_models.py` | the four architectures and their training callbacks |
| `td_eval.py` | the three readouts, metrics, figures |
| `run_comparison.py` | entry point: trains everything, writes every table and figure |
| `make_tables.py` | CSVs -> `outputs/tables.tex`, paste-ready IEEE tables |
| `aggregate_seeds.py` | aggregates several `outputs_seed*` runs; reports whether an effect holds in every seed |
| `diagnose.py` | why-is-it-underfitting harness (baselines vs augmentation vs batch size) |
| `HANDOFF.md` | state of the work, deadline, venue rules, results, and what is still open |
