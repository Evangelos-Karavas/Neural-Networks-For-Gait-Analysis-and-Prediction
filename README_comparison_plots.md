# Gait-phase plots — single-file version

Everything now lives in **one file: `models_comparison_all_planes.py`**.
No separate `gait_phase_plots.py` import is needed — the plotting code, the
metrics, and a built-in synthetic preview are all inside this one script.

Keep it next to your `Saved_Models/`, `Scaler/`, and `Data_CP/` folders.

## Quick start

```bash
# preview the new plots WITHOUT any models or data (synthetic demo) ->
# writes to Plots/. This also works without TensorFlow installed.
python models_comparison_all_planes.py --demo

# real run: list held-out subjects, pick an index
python models_comparison_all_planes.py --list-subjects

# real run: every plot (old + new phase-shaded) for one subject
python models_comparison_all_planes.py --subject 3 --plot
```

PNGs go to `Plots/`. The new phase-shaded figures are:

| File | What it shows |
|---|---|
| `cycle_phases_{side}_subject_*.png` | Predicted vs ground-truth trajectory averaged over **one normalised gait cycle (0-100%)**, GT +/-1 SD band, all 4 models, 8 phases shaded. The main figure. |
| `phase_error_phases_{side}_subject_*.png` | Mean abs(error) across the gait cycle - shows **where** in the stride each model struggles. |
| `per_phase_bars_{side}_subject_*.png` | Grouped bar chart of mean abs(error) **inside each of the 8 phases**, one group per model - the quantitative summary. |
| `hero_left_knee_p1_subject_*.png` | One large, fully-labelled single-joint panel - a clean "hero" slide. |

## The gait phases

All 8 sub-phases from your thesis' Figure 1.1 are shaded (warm = stance,
amber = pre-swing/push-off, cool = swing), with a dashed line at toe-off:

LR Loading Response - MSt Mid Stance - TSt Terminal Stance -
PSw Pre-Swing - ISw Initial Swing - MSw Mid Swing - TSw Terminal Swing

Phase boundaries are **anchored to each subject's measured toe-off** (the
*Foot Off* % in your data, read by the `mean_toe_off()` helper) and then
sub-divided with the standard within-stance / within-swing fractions, so the
shading reflects that patient - not a generic textbook overlay.

## What changed inside the file

- The phase-plotting functions are pasted in under the
  `# ===== PHASE_PLOT_SECTION_START =====` banner.
- `main()` now also calls `make_all_phase_plots(...)` after your two original
  plots when you pass `--subject N --plot`.
- A `--demo` flag renders the synthetic preview and exits.
- The TensorFlow import is now **lazy** (done inside `load_all_models`), so
  `--demo` and the plotting code run even without TensorFlow.
- Two constants were renamed to avoid clashing with your originals:
  `GT_COLOR_DARK` (new GT colour for the phase plots) and `_PHASE_SIDE_BASE`
  (the {"Left":0,"Right":9} lookup). Your original `SIDE_BASE`/`GT_COLOR`
  and the original two plots are untouched.

### Calling a single figure yourself

```python
to_left, to_right = mean_toe_off(df_subject)
plot_single_joint_hero(preds, gt_ref, "S3", side="Right", joint="Ankle",
                       plane_col=0, toe_off_pct=to_right, save_dir="Plots")
```

Colours and phase fractions live near the top of the
`PHASE_PLOT_SECTION_START` block if you want to tweak them.

### One caveat
I can't run this against your actual `.keras` models / `Data_CP/` here, so do
one real run on your machine to confirm the real curves look right. The
`--demo` output confirms the plotting itself is correct.
