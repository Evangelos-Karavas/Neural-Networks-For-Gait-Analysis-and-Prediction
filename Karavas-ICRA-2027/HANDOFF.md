# Handoff — ICRA 2027 gait-prediction paper

Written 2026-09-12 at the end of a Windows session. **Read this first.** It is
the full briefing: what the work is, what was decided and why, what was found,
what to run, and what still has to be written.

---

## 1. The deadline and the venue rules

**ICRA 2027 — submission 2026-09-15, 23:59 PST.** Three hard constraints,
checked against the official CFP:

- **8 pages including references.** Over-length papers are *returned without
  review*; there is no over-length fee. The current PDF is exactly 8.
- **Double-anonymous review.** No author names, affiliations or emails in the
  PDF. Already done in the `.tex` (real block kept commented out).
- **AI-generated content must be disclosed in the acknowledgments**, naming the
  system and the sections affected. Editing and grammar assistance is exempt and
  needs no disclosure. Breaching this disqualifies the paper and the
  registration fee is non-refundable. **This paper needs such a disclosure** —
  §IV–VI, the abstract and the contributions paragraph were drafted with an AI
  assistant (Claude) against the measured outputs. A disclosure is not
  identifying, so it can go in the anonymous submission. Decide with the
  supervisor before submitting; do not quietly skip it.
- Video is optional, ≤20 MB, ≤180 s, mp4/mpeg/mpg, ≥480p, ≥20 fps. **Correction
  to an earlier version of this file**: it is *not* "with the submission or
  never". There are two windows — 2026-08-05 to 09-09 (closed) and **2026-09-17
  to 09-22**, the second one *after* the paper deadline. Uploads are blocked
  09-10 to 09-16. So a video is still possible, but only in that later window.
- No online presentation option; a paper with no on-site presenter is skipped.

The paper class (`ieeeconf.cls`) is already correct for ICRA.

Source: the official CFP at `2027.ieee-icra.org`, checked 2026-09-12.

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
| Evaluation protocol | **Partitioning** subject-level splits: 6 splits of 8 train / 2 val / 2 test, each subject held out **exactly once** | Only 12 subjects; a single split would make the result an accident of who was held out. Independent *random* draws were worse still — 5 draws of 2 covered only 8 of 12 subjects and tested two of them twice (fixed 2026-09-12) |
| Missing data | Strides with any missing joint angle are **dropped, not filled** | `fillna(0.0)` fabricated 12 samples of a flat left leg at exactly 0°, which also corrupted that stride's phase variable. 59 strides now, not 60 (fixed 2026-09-12) |
| Reproducibility | `tf.keras.utils.set_random_seed` per (split, model) | Weight init and dropout were unseeded, so published numbers could not be reproduced (fixed 2026-09-12) |
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

## 6. Results — the full H=10 run (2026-09-12, on Linux/GPU)

Run in ~400 s on an RTX 5060 Ti. Everything below is in `outputs/`. The H=1 run
in `outputs_h1_superseded/` is kept only as evidence for §5(a); **its model
rankings are superseded and one of them is reversed** — see the warning below.

> **The run is now seeded.** `run_comparison.py` calls
> `tf.keras.utils.set_random_seed` per (split, model) before building each
> network, so rerunning reproduces these exact numbers. It did not before —
> weight init and dropout were unseeded, and the first H=10 run could not be
> reproduced. **The numbers below are from the seeded run and differ from that
> first one** (see the seed-sensitivity note at the end of this section).

**Teacher-forced at 200 ms: a tie.** Paired over 10 held-out-subject
evaluations, PV − Timestamp = **−0.04°** (LSTM) and **−0.02°** (CNN) — and the
counts point the other way, with the *timestamp* variant better on 7/10 (LSTM)
and 6/10 (CNN). All four models sit between 4.29° and 4.55°, a 0.26° spread
inside a per-subject spread of 2.9–5.9°. Backbone: CNN beats LSTM by 0.25°
(timestamp, 7/10) and 0.22° (PV, 8/10). Both baselines are comfortably beaten
(persistence 8.55°, linear 13.97° — these are deterministic and unchanged), so
§5(a) is satisfied at full scale.

**Recursive rollout over 6 strides: conditioning is what separates them.** Hip
phase lag, mean |lag| in samples (51 samples = one stride):

| Model | mean | median | % of cycle | within 1 sample |
|---|---|---|---|---|
| Timestamp LSTM | 2.80 | 3.0 | 5.5% | 3/10 |
| **PV LSTM** | **1.10** | **1.0** | **2.2%** | **8/10** |
| Timestamp CNN | 2.20 | 2.0 | 4.3% | 4/10 |
| PV CNN | 1.20 | 1.0 | 2.4% | 7/10 |

Paired: PV improves |lag| by **1.70 samples** (LSTM, better on 7/10, 2 tied) and
**1.00** (CNN, 6/10, 4 tied).

Rollout MAE: 5.53 / 5.55 / 5.10 / 4.77. Paired, PV improves it by **0.33°** for
the CNN (7/10) but leaves the LSTM **unchanged** (+0.02°, 6/10) — so the phase
benefit does *not* translate into lower rollout error for the recurrent
backbone. The paper says this explicitly; do not quietly upgrade it.

**The argument written into the paper**: phase conditioning costs nothing at
short horizon and buys temporal stability at long horizon, in *both* backbones.

### Two corrections to the old H=1 guidance

**(a) The backbone ranking reversed.** At H=1 the LSTM beat the CNN by 0.86° on
10/10. At H=10 the **CNN is better**: by 0.33° (9/10) under timestamp
conditioning and 0.23° (6/10) under PV. Any text claiming LSTM superiority is
wrong and has been rewritten. A useful side effect: the old "honest
complication" (CNNs winning rollout MAE despite losing teacher-forced) no longer
exists, because the CNN now leads on both.

**(b) The saturation hypothesis is not supported.** §II-C predicts that phase
conditioning should fail in terminal swing, where `s` has saturated. It does
not — terminal swing is the *third most accurate* of the seven phases for all
four models (4.24–4.62°). The phases that actually suffer are pre-swing and
initial swing (5.08–5.69°), equally for all four models. The paper reports this
as a refuted hypothesis rather than quietly dropping it.

**(c) Amplitude, not timing, is the dominant rollout error.** In
`figures/rollout_sagittal.png` every model overshoots the measured hip and knee
peaks, the Timestamp LSTM badly (knee past 70° against a measured ~45°). Phase
conditioning does not address this. Do not let the phase result imply the
rollouts look good — they do not.

**(d) Seed robustness — checked over 5 seeds, and the claim holds.** Run
`aggregate_seeds.py` over `outputs_seed42..46` to reproduce:

| Paired PV − Timestamp | LSTM | CNN |
|---|---|---|
| Hip phase \|lag\| | **−1.58 samples (5/5 seeds)** | **−1.27 (5/5)** |
| Rollout MAE | −0.51° (5/5) | −0.21° (4/5) |
| Teacher-forced MAE | −0.10° (5/5) | −0.04° (5/5) |

The phase-lag advantage is negative in **every seed for both backbones** — that
is the strongest form of the claim this dataset can support, and it is what to
cite if a reviewer questions robustness. Magnitudes do move between seeds
(LSTM lag effect ranges −2.92 to −0.42), so quote the direction and the
consistency, not a precise effect size.

**Seed 42 — the one the paper reports — is the least favourable of the five in
absolute error** (teacher-forced means 5.66/5.48/5.22/5.13 against 5-seed means
of 5.32/5.22/4.93/4.89). The paper's accuracy numbers are therefore pessimistic,
which is the safe direction to be wrong in. Keep the tables and figures from the
single seed 42 run so they stay mutually consistent; the 5-seed check belongs in
Limitations, where it now is.

To add more seeds:
```bash
for s in 47 48; do python run_comparison.py --seed $s --output outputs_seed$s --no-figures; done
python aggregate_seeds.py outputs_seed*
```

### Figure anonymization

Figure titles no longer contain subject IDs (`NV0xx`) or the string "TD" — the
author asked for these out of the paper. The strings are built in
`run_comparison.py` (rollout / cycle / per-phase titles) and `td_eval.py`
(variability and scatter titles). **If you add a figure, do not put the subject
ID in the title.** Prose and table captions still use "TD"/"typically
developed", which is defined terminology rather than identifying information;
confirm with the author if that should go too.

## 7. Paper state

Source: `../../Karavas-ICRA-2027-Paper/Karavas2027ICRA.tex` (on Linux:
`~/Desktop/Karavas Exoskeleton/Karavas-ICRA-2027-Paper/`) — a **separate git
repo** from this code repo, renamed from `Karavas2026EMBC.tex`. On Windows it
lives under `...\Karavas Diploma - Paper\`. Both checkouts are the same repo;
push from one before working in the other, or the two diverge silently — which
is exactly what happened on 2026-09-12.

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

**Written 2026-09-12 against the full H=10 run (all of the below is now done):**
- Abstract rewritten and shortened
- Contributions paragraph closing §I rewritten around the three findings
- §III-C: horizon corrected from one sample to 10, with the justification from
  §5(a); the rollout **information asymmetry** now stated explicitly (a PV model
  keeps receiving a *measured* `s` during rollout — see `td_eval.py:145`. This
  is the deployment assumption the representation encodes, but it must be
  declared or the comparison looks rigged)
- §IV Results rewritten in three parts: the teacher-forced tie, the rollout
  separation, the per-phase breakdown
- §V Discussion and §VI Conclusions rewritten
- All four tables inserted from `make_tables.py`
- Figures swapped to `rollout_sagittal.png`, `subject_scatter_mae.png`,
  `subject_cycle_left.png`; the CP figures are gone

**Page budget — resolved, and fragile.** The draft hit 9 pages. To reach 8:
the architecture figure was **cut** (§8 nominated it first), some prose was
tightened, and the two `figure*` widths were reduced to `0.76`/`0.70`
`\textwidth`. It now compiles to exactly 8 pages with no undefined references.
**Any addition will push it over**, so budget before adding. `per_phase_bars.png`
was left out for this reason; the per-phase *table* covers the same ground over
all subjects rather than one.

**Still open:**
- The four `TODO(refs)` markers in §I — real citations still required. Do not
  fabricate them.
- Restore the real author block and the ELEPAP acknowledgement for camera-ready.

## 8. Traps

- **Page budget — already spent.** The paper sits at *exactly* 8 pages. The
  architecture figure and the nine-panel per-channel figure (`subject_cycle_left`)
  have both already been cut to get there, along with several rounds of prose
  tightening. **Adding the four missing citation groups will push it over**, so
  budget the space before adding them: the likely next cut is the per-joint
  table (Table II) or the per-phase table, each of which the prose can carry.
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
