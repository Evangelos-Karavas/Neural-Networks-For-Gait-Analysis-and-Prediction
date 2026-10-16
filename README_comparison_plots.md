# Model comparison and gait-phase plots

[models_complete_comparison.py](models_complete_comparison.py) evaluates the
four **all-planes (18-channel)** models side by side: Timestamp CNN, Timestamp
LSTM, PV CNN and PV LSTM. It runs them on the held-out CP patients. See the
main [README](README.md) for how the models are trained.

## What it does

1. Finds the **held-out CP test patients**. It uses the same patient-level
   70 / 15 / 15 split (seed 42) as the `*_all_planes.py` training scripts, so
   these patients were never seen during training.
2. For each test patient, it rebuilds that patient's own scaled rolling
   windows, so no window crosses a patient boundary.
3. Runs **teacher-forced next-tick prediction** with each model: every
   prediction uses the real previous 51 samples as input.
4. Reports MAE, RMSE and R² in degrees, per patient, and the mean ± standard
   deviation across patients for each model.
5. Optionally draws plots for one chosen patient.

## Usage

Run from the repository root, after training the four `*_all_planes.py`
models:

```bash
# held-out CP test patients and their index
python models_complete_comparison.py --list-subjects

# metric table over all held-out patients
python models_complete_comparison.py

# metrics and every plot for one patient (index from --list-subjects)
python models_complete_comparison.py --subject 3 --plot

# same, compared against a chosen healthy subject
python models_complete_comparison.py --list-healthy-subjects
python models_complete_comparison.py --subject 3 --plot --healthy_subject 2

# preview of the plots with synthetic data; needs neither models, data nor TensorFlow
python models_complete_comparison.py --demo
```

| Flag | Meaning |
|---|---|
| `--list-subjects` | Print the held-out CP test patients and exit. |
| `--subject N` | Evaluate only patient `N` instead of all of them. |
| `--plot` | Draw the plots for `--subject` (and for `--healthy_subject`). |
| `--list-healthy-subjects` | Print the healthy subjects in `Data_Normal/` and exit. |
| `--healthy_subject N` | Healthy subject drawn next to the CP patient (default `0`). These subjects were part of the typical-gait training data, so they are **not** a held-out set. |
| `--no_save` | Show plots without saving PNGs. |
| `--demo` | Render the plots from synthetic data and exit. |

The models and scalers are loaded from `Neural_Networks_Output/Saved_Models/`
and `Neural_Networks_Output/Scaler/`. The file names are listed in the main
README under [Outputs](README.md#outputs).

## Plots

Plots are saved to the folder set by `PLOT_DIR` at the top of the script
(currently `Neural_Networks_Output/Subjects/SUBJECT_5`; change it per
patient). `--demo` writes to `Plots/`.

| File | What it shows |
|---|---|
| `compare4_{side}_subject_*.png` | Prediction vs. ground truth over time, all 4 models, all 9 channels of one leg. |
| `compare4_phase_error_{side}_subject_*.png` | Mean absolute error vs. gait phase, all 4 models. |
| `cycle_phases_{side}_subject_*.png` | Trajectory averaged over **one normalised gait cycle (0–100 %)**, ground truth ± 1 SD band, all 4 models, with the gait phases shaded. The main figure. |
| `phase_error_phases_{side}_subject_*.png` | Mean absolute error across the gait cycle, showing **where** in the stride each model struggles. |
| `per_phase_bars_{side}_subject_*.png` | Mean absolute error **inside each gait phase**, one bar group per model. |
| `hero_left_knee_p1_subject_*.png` | One large, fully labelled panel (left knee, sagittal plane), suitable for a slide. |

## Gait phases

Seven phases are shaded: warm colours for stance, amber for pre-swing
(push-off) and cool colours for swing, with a dashed line at toe-off.

| Stance | Swing |
|---|---|
| LR — Loading Response | ISw — Initial Swing |
| MSt — Mid Stance | MSw — Mid Swing |
| TSt — Terminal Stance | TSw — Terminal Swing |
| PSw — Pre-Swing | |

The phase boundaries are **anchored to each patient's measured toe-off**: the
mean `Left Foot Off` / `Right Foot Off` value, read by `mean_toe_off()`. Stance
and swing are then subdivided with the standard relative fractions. The
shading therefore reflects that patient, not a generic textbook cycle.

Colours and phase fractions (`PHASE_COLORS`, `_STANCE_FRAC`, `_SWING_FRAC`)
are defined under the `# ===== PHASE_PLOT_SECTION_START =====` banner in the
script.

### Drawing a single figure

```python
to_left, to_right = mean_toe_off(df_subject)
plot_single_joint_hero(preds, gt_ref, "S3", side="Right", joint="Ankle",
                       plane_col=0, toe_off_pct=to_right, save_dir="Plots")
```
