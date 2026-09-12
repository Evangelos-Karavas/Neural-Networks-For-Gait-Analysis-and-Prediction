# Handoff — ICRA 2027 gait-prediction paper

Written 2026-09-12 at the end of a Windows session. **Read this first.** It is
the full briefing: what the work is, what was decided and why, what was found,
what to run, and what still has to be written.

---

## 1. The deadline and the venue rules

**ICRA 2027 — submission 2026-09-15, 23:59 PST.** Three hard constraints,
checked against the official CFP:

- **8 pages including references.** Over-length papers are *returned without
  review*; there is no over-length fee.
- **Double-anonymous review.** No author names, affiliations or emails in the
  PDF. Already done in the `.tex` (real block kept commented out).
- Optional video, ≤20 MB, ≤180 s, submitted *with* the initial submission or
  never.

The paper class (`ieeeconf.cls`) is already correct for ICRA.

## 2. The story the paper must tell

Set by the supervisor, and it is a deliberate narrowing of the old EMBC draft:

> Train on typically developed (TD) gait, test on held-out TD subjects, and
> compare **timestamp** conditioning against **phase-variable** conditioning
> across an LSTM and a CNN backbone.

Specific instructions:
- **Longer introduction**, **shorter abstract**.
- **No cerebral-palsy material anywhere.** The old paper's entire results
  section was CP-based; it is gone.
- **No exoskeleton deployment claims.** One paragraph of robotics motivation is
  retained deliberately, because ICRA is a robotics venue and a paper with no
  robot connection risks a scope complaint. Remove it only if the supervisor
  insists.
- **Lead with temporal stability and long-horizon prediction.** This is the
  result, not an aside.

## 3. Decisions already taken (do not silently revisit)

| Decision | Choice | Why |
|---|---|---|
| Evaluation protocol | Repeated random subject-level splits, 5 draws of 8 train / 2 val / 2 test | Only 12 subjects; a single split would make the result an accident of who was held out |
| Augmentation | Regenerated per split from training subjects only | The pre-existing bulk files leak (see §5) |
| Code layout | Shared pipeline module + thin runners, in this folder | The old code duplicated the same loader across 8 near-identical scripts |
| Model scope | 18-channel (all three planes) only | Halves the training cost and still supports every evaluation |
| Prediction horizon | **10 samples (~200 ms)** | One sample ahead is trivially solved by persistence (see §5) |
| Framing if models tie | Make the tie the point | Short-horizon accuracy cannot rank these models; that is the argument |
| Paper delivery | Edit in place on `main`, renamed file | User's explicit choice |

## 3a. Two possible workflows — confirm which one you are in

The author has not fixed this yet. Ask, or infer from what is available:

- **Training only.** Run the sweep, commit `outputs/` (the CSVs, `REPORT.md`,
  `tables.tex` and `figures/`) and push. The paper is then finished in a
  separate session on the Windows side, which holds the full conversation
  context. In this mode, do *not* edit the `.tex`.
- **Training and writing.** Run the sweep, then write §IV–VI, the abstract and
  the contributions paragraph, insert the tables from `make_tables.py`, and
  swap in the new figures. Requires the paper repo checked out alongside.
  Sections 6, 7 and 8 below tell you what to say and what to avoid.

Either way the sweep is the first step, and `outputs/` is the deliverable.

## 4. What to run

```bash
cd Karavas-ICRA-2027
python run_comparison.py     # 5 splits x 4 models at the 200 ms horizon
python make_tables.py        # outputs/*.csv -> outputs/tables.tex (IEEE format)
```

**Check the GPU is actually being used before starting:**

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If that prints `[]`, stop. On CPU this sweep took **5.6 hours**. Context: the
Windows venv has `tensorflow-intel 2.12.0`, a CPU-only build — TensorFlow
dropped native-Windows GPU support after 2.10 — and the GPU is an RTX 5060 Ti
with **compute capability 12.0** (Blackwell), which needs CUDA 12.8+, so no
native-Windows TensorFlow can ever drive it. Linux with
`tensorflow[and-cuda]` is the only working path. That is why the run moved to
Ubuntu.

Useful flags: `--quick` (1-minute smoke test), `--horizon N`, `--repeats N`,
`--models ...`, `--no-figures`.

Everything lands in `outputs/`: `REPORT.md` (all tables in Markdown),
`tables.tex`, per-subject/per-channel/rollout/per-phase/baseline CSVs,
`run_config.json` (exact splits and settings), and `figures/`.

## 5. Two findings that invalidated the first run — do not undo these

**(a) One-sample-ahead prediction is not a real task.** The first full run used
a one-sample horizon. Parameter-free baselines crush every network there:

| Predictor | MAE | R² |
|---|---|---|
| Linear extrapolation | 0.59° | 0.96 |
| Persistence (repeat last sample) | 1.31° | 0.92 |
| Best network (Timestamp LSTM) | 3.35° | −0.05 |

A benchmark a constant-output rule wins cannot rank models. The horizon is now
10 samples, where the networks win properly:

| Horizon | Persistence | Linear | Timestamp CNN |
|---|---|---|---|
| 1 | **1.32°** | **0.61°** | 4.60° |
| 5 | 5.49° | 5.95° | **4.32°** |
| 10 (~200 ms) | 8.40° | 14.16° | **4.32°** |
| 25 (~½ stride) | 9.40° | 36.19° | **4.25°** |

**Both baselines are now computed on every run** and printed in `REPORT.md` and
in the LaTeX table. If a change makes the networks lose to them again, that is
a bug, not a result. `HORIZON` lives in `td_pipeline.py`.

**(b) The old paper's numbers were inflated by a data leak.** The bulk files
`Data_Normal/randomized_data_healthy.xlsx` and `dynamics_total_augmented.xlsx`
pool all 60 strides *before* augmenting, destroying subject identity, so
held-out subjects reappear — noised — in training. The old paper's 2.1° LSTM
figure came from that. **This pipeline never reads those files.** Honest
cross-subject numbers are substantially worse, and that is correct.

## 6. Results from the superseded H=1 run

Archived in `outputs_h1_superseded/`. The *conclusions* survived the horizon
change, and a 3-split spot check at H=10 reproduced the key parity
(PV CNN 4.24° vs Timestamp CNN 4.26°), so they are the best current guide to
what the new run will show.

**Teacher-forced accuracy: conditioning makes no difference.** Paired over 10
held-out-subject evaluations: PV − Timestamp = **+0.16°** (LSTM, PV better on
only 2/10) and **−0.03°** (CNN, 5/10). Backbone does matter: LSTM beat CNN by
**0.86°** on 10/10.

**Recursive rollout over 6 strides: conditioning is decisive.** Hip phase lag,
mean |lag| in samples (51 samples = one stride, so 1 sample ≈ 2% of the cycle):

| Model | mean | median | within 1 sample |
|---|---|---|---|
| Timestamp LSTM | 5.10 | 5.0 | 1/10 |
| **PV LSTM** | **0.70** | **0.5** | **8/10** |
| Timestamp CNN | 2.00 | 1.0 | 6/10 |
| PV CNN | 1.10 | 1.0 | 7/10 |

Paired: PV improves lag by **4.40 samples** (LSTM, better on 8/10) and **0.90**
(CNN, 6/10); rollout MAE by **1.66°** (LSTM, 9/10) and **0.34°** (CNN, 7/10).

**The argument to write**: phase conditioning costs nothing at short horizon and
buys temporal stability at long horizon. The Timestamp LSTM drifts ~10% of a
gait cycle out of phase; the PV LSTM stays within ~1%.

One honest complication: under rollout the CNNs had *lower* MAE than the LSTMs
(4.49–4.83 vs 6.04–7.70) even though LSTMs were better teacher-forced — a
recurrent model compounds its own error through its hidden state, while a
memoryless one reverts to an average waveform with bounded error. Do not hide
this; the PV LSTM's claim is phase accuracy, not lowest rollout MAE.

## 7. Paper state

Source: `C:\Users\vagge\Desktop\Vaggelis\Documents\Ρομποτικη - ΕΜΠ\Διπλωματική - Exoskeleton\Karavas Diploma - Paper\Karavas-Biomechanics-Paper\Karavas2027ICRA.tex`
— a **separate git repo** from this code repo. Renamed from `Karavas2026EMBC.tex`
via `git mv`; uncommitted.

**Done:**
- Author block anonymized; real block commented out for camera-ready
- Introduction expanded: robotics-delay motivation (`Baud2021`), sharpened
  teacher-forcing critique, new paragraph naming the two rollout failure modes
  (amplitude decay vs. sliding out of phase) — this sets up the result
- `Hussain2021` cited for the MLP family
- **Fixed a LaTeX bug**: a `TODO` comment had swallowed half a sentence; it
  renders in the old PDF as "…parameterization of the stride. as a conditioning
  input to a neural network has received far less attention."
- §II-A rewritten around TD variability, pointing at `td_variability.png`
- §II-B: `c` corrected — it is the stride's *measured* foot-off (59.6% mean,
  54.5–62.4% range), not a fixed 0.53
- §II-C: saturation corrected from "20–25%" to **12.6% mean / 15.7% median**
  (measured over all 120 stride–leg pairs; 25% is the upper tail, not typical)
- §III-A rewritten: 12 TD subjects, 60 strides, repeated random subject splits,
  per-split augmentation, the three leakage precautions; data source
  anonymized to "a clinical gait and motion analysis laboratory"
- §III-B: 18-channel only; **the Butterworth filter claim removed** — the
  manuscript claimed one, but no filtering exists anywhere in the codebase
- §III-C: all four models share one horizon, so the Timestamp-CNN special case
  and its caveat are gone

**Still to write — all of it needs the new numbers:**
- Abstract (must get *shorter*)
- Contributions paragraph closing §I
- §IV Results, §V Discussion, §VI Conclusions
- All tables (generate with `make_tables.py`, then paste)
- Figure swaps: Figs. 4–7 still reference the old CP figures. New ones are in
  `outputs/figures/`: `td_variability.png`, `rollout_sagittal.png`,
  `subject_cycle_left.png`, `per_phase_bars.png`, `subject_scatter_mae.png`.
  Copy into the paper's `figures/` folder.

## 8. Traps

- **Page budget.** 8 pages including references, with a longer introduction and
  ~6 figures plus 3–4 tables. Something must go — most likely the architecture
  figure (describable in text) or the per-subject scatter.
- **Do not invent citations.** `References.bib` has no CNN/TCN or Transformer
  entries. The `TODO(refs)` markers in §I stay until real references are
  supplied. Fabricating plausible-looking references would be a serious error.
- **Right-leg roll.** The capture system normalizes each leg to its own cycle,
  so the pipeline rolls the right leg half a stride to restore the bilateral
  relationship of real walking. Any per-cycle analysis of right-side channels
  must undo it — `phase_binned()` takes `right_shift` for exactly this reason.
  (The old `models_complete_comparison.py` handled this correctly via
  `segment_strides(unshift_right=True)`; the old paper is not wrong here.)
- **ELEPAP** is anonymized for double-blind. Restore the acknowledgement in the
  camera-ready.
- **Negative R².** Many per-subject R² values are negative because R² is
  normalized by each subject's own variance, and several transverse-plane
  channels have tiny ranges. Explain it rather than hiding it; MAE in degrees
  is the more honest headline.

## 9. Files

| File | |
|---|---|
| `td_pipeline.py` | loading, subject splits, augmentation, phase variable, windows, scaling |
| `td_models.py` | the four architectures, multi-step heads, training callbacks |
| `td_eval.py` | three readouts, baselines, metrics, figures |
| `run_comparison.py` | entry point — trains everything, writes every table and figure |
| `make_tables.py` | CSVs → `outputs/tables.tex` |
| `diagnose.py` | underfitting harness: baselines vs augmentation vs batch size |
| `README.md` | the pipeline in detail, and every point where the old manuscript disagrees with the code |
| `outputs_h1_superseded/` | the discarded H=1 run, kept as evidence for §5(a) |

The old CP-based study is untouched in `Neural_Networks_*/` and
`models_complete_comparison.py`. Nothing here imports from it.
