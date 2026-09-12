# Handoff — ICRA 2027 gait-prediction paper

Last updated **2026-09-12**, end of the Linux session in which the pipeline was
fixed, the sweep rerun, the paper rewritten and then — later the same day — the
14 pending references cited into §I and the paper cut back to 8 pages to pay for
them. **Read this first.** It is the full briefing: what the work is, what was
decided and why, what was found, what to run, and what is still open.

---

## 1. The deadline and the venue rules

**ICRA 2027 — submission 2026-09-15, 23:59 PST.** Checked against the official
CFP at `2027.ieee-icra.org` on 2026-09-12:

- **8 pages including references.** Over-length papers are *returned without
  review*; there is no over-length fee. The current PDF is exactly 8.
- **Double-anonymous review.** Verbatim: *"The ICRA review process is
  double-anonymous (both reviewers and authors stay anonymous). Therefore,
  manuscripts should exclude the authors and their affiliations."* All co-authors
  must still be entered in PaperPlaza, where only editors and AEs see them.
  Already done in the `.tex` (real block kept commented out).
  **Note:** published ICRA papers on IEEE Xplore *do* carry author names — those
  are camera-ready versions, not what went to reviewers. Do not be misled by
  them, as happened once already.
- **AI-generated content must be disclosed in the acknowledgments**, naming the
  system and the sections affected; editing and grammar assistance is exempt.
  Breaching this disqualifies the paper and the registration fee is
  non-refundable. **Status — unresolved, and the most likely way to lose the
  paper.** §IV–VI, the abstract and the contributions paragraph were first
  drafted with an AI assistant against the measured outputs. The author planned
  to rewrite them in their own words, which would have reduced the remaining
  contribution to editorial assistance and removed the need to disclose, but on
  2026-09-12 they instead asked the assistant to do the citation pass and the
  §II-C / Discussion / Conclusions rewrite (see §7). **On the text as it now
  stands, disclosure is required.** The author has also declined an
  acknowledgements section. These two positions are incompatible: pick one
  before submitting — either add the one-sentence disclosure (it need not be the
  ELEPAP-style thanks section; ~4 lines, which the page budget does not
  currently have, so something must give), or genuinely rewrite the affected
  prose.
- Video optional, ≤20 MB, ≤180 s, mp4/mpeg/mpg, ≥480p, ≥20 fps. Two windows:
  2026-08-05 to 09-09 (closed) and **2026-09-17 to 09-22**, the second *after*
  the paper deadline; uploads are blocked 09-10 to 09-16. The author does not
  intend to submit one.
- No online presentation option; a paper with no on-site presenter is skipped.

The paper class (`ieeeconf.cls`) is already correct for ICRA.

## 2. The story the paper must tell

Set by the supervisor, and a deliberate narrowing of the old EMBC draft:

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
| Evaluation protocol | **Partitioning** subject-level splits: 6 splits of 8 train / 2 val / 2 test, each subject held out **exactly once** | Only 12 subjects. Independent *random* draws covered just 8 of 12 and tested two twice — and the four never held out included the three hardest, so every model's error was understated by ~1° (fixed 2026-09-12) |
| Missing data | Strides with any missing joint angle are **dropped, not filled** | `fillna(0.0)` fabricated 12 samples of a flat left leg at exactly 0°, which also corrupted that stride's phase variable. **59 strides now, not 60** (fixed 2026-09-12) |
| Reproducibility | `tf.keras.utils.set_random_seed` per (split, model) | Weight init and dropout were unseeded, so published numbers could not be reproduced (fixed 2026-09-12) |
| Augmentation | Regenerated per split from training subjects only | The pre-existing bulk files leak (see §5b) |
| Model scope | 18-channel (all three planes) only | Halves the training cost and still supports every evaluation |
| Prediction horizon | **10 samples (~200 ms)** | One sample ahead is trivially solved by persistence (see §5a) |
| Framing if models tie | Make the tie the point | Short-horizon accuracy cannot rank these models; that is the argument |
| Reported run | Single seed 42, with a 5-seed robustness check in Limitations | Tables and figures stay mutually consistent; see §6d |

## 4. What to run

```bash
cd Karavas-ICRA-2027
python run_comparison.py     # 6 splits x 4 models at the 200 ms horizon (~530 s on GPU)
python make_tables.py        # outputs/*.csv -> outputs/tables.tex (IEEE format)
```

Useful flags: `--quick` (1-minute smoke test), `--horizon N`, `--repeats N`,
`--seed N`, `--output DIR`, `--models ...`, `--no-figures`.

Everything lands in `outputs/`: `REPORT.md` (all tables in Markdown),
`tables.tex`, per-subject/per-channel/rollout/per-phase/baseline CSVs,
`run_config.json` (exact splits and settings), and `figures/`.

### GPU — two gotchas, both of which cost hours once

**Check the GPU is actually visible before starting.** On CPU this sweep takes
**5.6 hours**; on the RTX 5060 Ti it takes ~9 minutes.

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

1. **After an NVIDIA driver upgrade, reboot.** The kernel module in memory stays
   at the pre-upgrade version, `nvidia-smi` fails with `Driver/library version
   mismatch`, and TF reports zero GPUs. Compare `cat /proc/driver/nvidia/version`
   against `dpkg -l | grep nvidia-utils`. A reboot is the fix; nothing needs
   reinstalling.
2. **`LD_LIBRARY_PATH` must include every `nvidia/*/lib` directory.** TF's
   baked-in RPATH does not cover all of them (`cusolver` is one it never
   searches), so it prints `Cannot dlopen some GPU libraries` and falls back to
   CPU even though the libraries are installed. Prefix every GPU command:

```bash
NVLIBS=$(find "$PWD/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d | tr '\n' ':')
LD_LIBRARY_PATH="$NVLIBS$LD_LIBRARY_PATH" .venv/bin/python run_comparison.py
```

A third message is benign: TF 2.21 is built for `sm_60…sm_89` plus `compute_90`
PTX, so on this sm_120 card it warns that kernels will be JIT-compiled and
"could take 30 minutes or longer". Measured, it does not — a 2000² matmul takes
0.8 s. Set `CUDA_CACHE_MAXSIZE=4294967296` so the JIT cache persists. Do not go
hunting for a differently-built TensorFlow because of it.

Windows cannot run this at all: TF dropped native-Windows GPU support after 2.10
and the card needs CUDA 12.8+. Linux with `tensorflow[and-cuda]` is the only
working path.

## 5. Two findings that invalidated the first run — do not undo these

**(a) One-sample-ahead prediction is not a real task.** Parameter-free baselines
beat every network at H=1 (baselines recomputed on the current 59-stride data;
network figure from `outputs_h1_superseded/`):

| Predictor | MAE at H=1 |
|---|---|
| Linear extrapolation | **0.58°** |
| Persistence (repeat last sample) | **1.31°** |
| Best network (Timestamp LSTM) | 3.35° |

Baseline MAE by horizon, current data: H=1 → 1.31 / 0.58; H=5 → 5.54 / 5.81;
H=10 → 8.66 / 13.84; H=25 → 10.46 / 34.63 (persistence / linear). By H=10 the
networks lead both comfortably.

**Both baselines are computed on every run** and printed in `REPORT.md` and in
the LaTeX table. If a change makes the networks lose to them again, that is a
bug, not a result. `HORIZON` lives in `td_pipeline.py`.

**(b) The old paper's numbers were inflated by a data leak.** The bulk files
`Data_Normal/randomized_data_healthy.xlsx` and `dynamics_total_augmented.xlsx`
pool all strides *before* augmenting, destroying subject identity, so held-out
subjects reappear — noised — in training. The old paper's 2.1° LSTM figure came
from that. **This pipeline never reads those files.** Honest cross-subject
numbers are substantially worse, and that is correct.

## 6. Results — 6 splits, H=10, seed 42 (2026-09-12)

Everything below is in `outputs/` and is what the paper reports. 12 held-out
evaluations over 12 distinct subjects.

**Teacher-forced at 200 ms: a tie.** Paired, PV − Timestamp = **−0.18°** (LSTM,
better on 6/12) and **−0.09°** (CNN, 7/12) — a mean improvement with the win
count at chance is not a ranking. All four models sit between 5.13° and 5.66°, a
**0.53° spread inside a per-subject spread of 2.7–9.8°**. Backbone: CNN beats
LSTM by 0.44° (timestamp, 10/12) and 0.35° (PV, 10/12). Baselines are
comfortably beaten (persistence 8.76°, linear 14.56°), so §5a holds at full
scale.

| Model | TF MAE | Rollout MAE | \|hip lag\| | % of cycle | within 1 sample |
|---|---|---|---|---|---|
| Timestamp LSTM | 5.66 ± 1.82 | 6.63 ± 1.98 | 2.92 | 5.7% | 4/12 |
| PV LSTM | 5.48 ± 1.60 | 6.21 ± 1.85 | **1.17** | **2.3%** | 6/12 |
| Timestamp CNN | 5.22 ± 1.58 | 5.85 ± 1.74 | 2.50 | 4.9% | 2/12 |
| **PV CNN** | **5.13 ± 1.54** | **5.54 ± 1.72** | **1.17** | **2.3%** | 8/12 |

**Recursive rollout: conditioning is what separates them.** Paired, PV improves
|lag| by **1.75 samples** (LSTM) and **1.33** (CNN), and is the better of the
pair on **10 of 12 evaluations in both backbones**. It also improves rollout MAE,
by 0.42° (LSTM, 8/12) and 0.31° (CNN, 10/12). The consistency across two
architectures that share no weights is the argument — not the effect size.

**The argument written into the paper**: phase conditioning is nearly free at
short horizon and buys temporal stability at long horizon, in *both* backbones,
and the teacher-forced readout cannot predict which model stays in phase.

### Four things that contradict older guidance

**(a) The backbone ranking reversed.** At H=1 the LSTM beat the CNN by 0.86° on
10/10; at H=10 the **CNN is better**, by 0.44° and 0.35° (10/12 each). Any text
claiming LSTM superiority is wrong. The old "honest complication" (CNNs winning
rollout MAE despite losing teacher-forced) no longer exists — the CNN leads both.

**(b) The saturation hypothesis is not supported.** §II-C predicts phase
conditioning should fail in terminal swing, where `s` has saturated. It does
not: terminal swing (4.78–5.39°) is more accurate than every swing phase before
it. The phases that suffer are pre-swing and initial swing (5.85–7.07°), equally
for all four models. The paper reports this as a refuted hypothesis rather than
quietly dropping it.

**(c) Amplitude is the larger rollout error, and the models fail in opposite
directions.** In `figures/rollout_sagittal.png` the Timestamp LSTM *inflates*
the hip and knee peaks (knee past 70° against a measured ~55°) while the
Timestamp CNN *flattens* them (~42°); the phase-conditioned models sit closest.
None reproduces the negative knee excursion in stance. Phase conditioning only
partly addresses this. Do not let the phase result imply the rollouts look good.

**(d) Seed robustness — checked over 5 seeds, and the claim holds.**

| Paired PV − Timestamp | LSTM | CNN |
|---|---|---|
| Hip phase \|lag\| | **−1.58 samples (5/5 seeds)** | **−1.27 (5/5)** |
| Rollout MAE | −0.51° (5/5) | −0.21° (4/5) |
| Teacher-forced MAE | −0.10° (5/5) | −0.04° (5/5) |

The phase-lag advantage is negative in **every seed for both backbones** — the
strongest form of the claim this dataset supports, and what to cite if a
reviewer questions robustness. Magnitudes move between seeds (LSTM lag effect
ranges −2.92 to −0.42), so quote direction and consistency, not a precise effect
size.

**Seed 42 — the one the paper reports — is the least favourable of the five** in
absolute error (5.66/5.48/5.22/5.13 against 5-seed means 5.32/5.22/4.93/4.89).
The paper's accuracy numbers are therefore pessimistic, the safe direction to be
wrong in. Keep tables and figures from the single seed-42 run so they stay
mutually consistent; the 5-seed check belongs in Limitations, where it is.

```bash
for s in 47 48; do python run_comparison.py --seed $s --output outputs_seed$s --no-figures; done
python aggregate_seeds.py outputs_seed*
```

### Figure anonymization

Figure titles contain **no subject IDs (`NV0xx`) and no "TD"** — the author
asked for both out of the paper. The strings are built in `run_comparison.py`
(rollout / cycle / per-phase titles) and `td_eval.py` (variability, scatter,
PV-strides titles). **If you add a figure, do not put the subject ID in the
title.** Prose and table captions still use "TD"/"typically developed", which is
defined terminology rather than identifying information.

## 7. Paper state

Source: `../../Karavas-ICRA-2027-Paper/Karavas2027ICRA.tex` (on Linux:
`~/Documents/Karavas Thesis/Karavas-ICRA-2027-Paper/` — the `~/Desktop/Karavas
Exoskeleton/` path this file used to give is dead) — a **separate git repo** from this code repo, renamed from `Karavas2026EMBC.tex`. On Windows it
lives under `...\Karavas Diploma - Paper\`. Both checkouts are the same repo;
push from one before working in the other, or they diverge silently — which is
exactly what happened on 2026-09-12.

**The whole paper is drafted and compiles to exactly 8 pages** with all 24
references cited, no undefined references and no author or subject identifiers
in the PDF text or metadata. §I–§III were revised in the earlier Windows
session; the abstract, contributions paragraph, §III-C and §IV–§VI were written
against the measured outputs and then updated twice as the pipeline fixes
changed the numbers. Verify with `latexmk -pdf` and `pdfinfo`; the build is
clean from a cold start.

Points worth knowing rather than rediscovering:

- **§III-C states the rollout information asymmetry explicitly.** A PV model
  keeps receiving a *measured* `s` during rollout (`td_eval.py:145`) while a
  timestamp model gets nothing external. This is the deployment assumption the
  representation encodes, but it must be declared or the comparison looks
  rigged. Do not remove it.
- **Removed claims that the code did not support:** the Butterworth filter
  (no filtering exists anywhere) and the Optuna hyperparameter search (no Optuna
  anywhere in the repo). §III-B now states what is true — settings fixed per
  backbone and identical across conditionings, so the comparison is controlled.
- **The CNN description matches the code**: all four convolutions are stride 2,
  dropout after each pooling stage, and the stack collapses the 51-sample window
  to a single 256-dimensional descriptor.
- **§III-A** records 59 strides and the partitioning design, and uses the
  protocol point as a small strength: random draws covered only 8 of 12.

**Figures (5, all used, nothing orphaned in `figures/`):**

| Figure | File | Where |
|---|---|---|
| 1 | `td_variability.png` | §II-A |
| 2(a) | `phase_variable_gait_events.png` | §II-B — author's own gait-events diagram |
| 2(b) | `pv_over_strides.png` | §II-B — regenerated by `plot_pv_strides()` |
| 3 | `subject_scatter_mae.png` | §IV-A |
| 4 | `rollout_sagittal.png` | §IV-B — `figure*` at **0.66\textwidth**; 0.68 tips the paper to 9 pages |

Fig. 2 was restructured at the author's request: the gait-events diagram at 2/3
width replaced the old FSM-states and PV-from-hip panels, with the multi-stride
sawtooth at 1/3. The sawtooth is **regenerated from real data** rather than
reused from the thesis, because the thesis image's labels rendered at ~2 pt once
scaled to a 1/3-width panel. `plot_pv_strides()` draws it at roughly the width
it is placed at, so the fonts reach the page at their true size — **keep that
property if you resize the panel.**

**Citations — all 14 are now cited into the text** (2026-09-12), and the four
`TODO(refs)` markers are gone. Every DOI was verified against Crossref
(Bai2018 against arXiv). Where they landed:

| Gap | Keys | Where |
|---|---|---|
| CNN/TCN | `Bai2018`, `Li2018`, `Molinaro2022` | §I, CNN paragraph |
| LSTM/GRU gait | `Zaroug2020`, `Zaroug2021`, `Ren2022` | §I, recurrent paragraph |
| Hybrids/attention | `Zhu2021`, `Lu2022`, `Aksan2021` | §I, hybrid sentence |
| Phase variables | `Holgate2009`, `Rezazadeh2019`, `Embry2021`, `Medrano2023` | §I, phase-variable paragraph |
| Online phase | `Kang2021` | §III-C |

`Kang2021` is a CNN gait-phase estimator running in real time on a hip
exoskeleton and is the best support for the §III-C claim that phase can be
computed online — which is why it sits there rather than in §I.

**`Gregg2019` was a duplicate of `Rezazadeh2019`** — same DOI, same IEEE Access
2019 paper, `Gregg2019` carrying a truncated author list. Citing both would have
printed it twice in the reference list. `Rezazadeh2019` is kept (correct author
order, full volume/pages) and the two `Gregg2019` sites were remapped to it;
`Gregg2019` is still in the `.bib` but uncited, so BibTeX ignores it. **The
bibliography is 24 entries, not 25.** Check for this before adding more: the
`.bib` grew by hand over two machines and may hold other near-duplicates.

**Still open:**
- **The AI-disclosure question in §1 is unresolved and blocks submission.** The
  author has declined an acknowledgements section; the text as it stands needs
  a disclosure. Decide which gives.
- **The title is unchanged.** Options were put to the author on 2026-09-12 and
  none picked; see the last bullet.
- Restore the real author block for camera-ready. An ELEPAP acknowledgement was
  **declined by the author (2026-09-12)** for the submitted version; revisit it
  for camera-ready, where anonymity no longer applies (`ELEPAP` is in the
  `.bib`, uncited).
- Confirm provenance of `phase_variable_gait_events.png`: if the walker artwork
  came from a textbook rather than being drawn by the author, it needs
  attribution or permission.
- The title is still `On the Use of Neural Networks in the Analysis and
  Prediction of Human Gait: A Phase Variable Approach` — survey framing that
  does not match the argument. Three candidates were drafted on 2026-09-12,
  none yet chosen:
  1. *Phase or Timestamp? Input Time Representation Governs Rollout Stability
     in Neural Gait Prediction* — leads with the comparison.
  2. *Teacher-Forced Error Cannot Rank Gait Predictors: Phase Conditioning and
     Long-Horizon Temporal Stability* — leads with the negative result.
  3. *Phase-Variable Conditioning Buys Temporal Stability in Neural Gait
     Prediction* — shortest, leads with the positive claim.

## 8. Traps

- **Page budget — spent twice over.** Exactly 8 pages, and there is now no
  slack at all: a single extra reference entry, or Fig. 4 at
  `0.68\textwidth` instead of `0.66`, tips it to 9. Earlier rounds cut the
  architecture figure, the nine-panel per-channel figure and two Fig. 2 panels.
  Citing the 14 references cost ~40 column-lines, paid for on 2026-09-12 by:
  **cutting Table II** (per-joint sagittal — its §IV-A prose was rewritten to
  stand alone and now carries it), deleting §II-C's closing "Role in neural
  network prediction" paragraph (it restated the subsection's own opening),
  compressing the Conclusions, dropping the Discussion's third retelling of the
  H=1 baseline point (it survives in §III-C and the Conclusions), shortening the
  abstract, and trimming the Fig. 2 caption. **If you need more space, the
  per-phase table is the last easy cut** — but it is the evidence for the
  refuted-saturation argument in §IV-C, so the prose must absorb its numbers.
  An AI disclosure (§1) costs ~4 lines that do not exist; it would force that
  cut.
- **Table numbering shifts when you cut one.** Table II is now the rollout
  table. Fig. 4's caption refers to it by `\ref{tab:rollout}`, so it followed
  automatically — but grep for `tab:errors` before assuming any cross-reference
  survived.
- **Do not invent citations.** The 14 added entries were each verified against
  Crossref. Anything further must be too. Fabricating plausible-looking
  references would be a serious error. Also check new entries against the
  existing `.bib` for duplicates — `Gregg2019`/`Rezazadeh2019` were the same
  paper under two keys and were caught only at the typeset stage.
- **Right-leg roll.** The capture system normalizes each leg to its own cycle,
  so the pipeline rolls the right leg half a stride to restore the bilateral
  relationship of real walking. Any per-cycle analysis of right-side channels
  must undo it — `phase_binned()` takes `right_shift` for exactly this reason.
- **Negative R².** Most per-subject R² values are negative because R² is
  normalized by each channel's own variance, and several frontal/transverse
  channels move through only a couple of degrees. Explain it rather than hiding
  it; MAE in degrees is the honest headline, and the baselines (−1.10, −10.85)
  are the useful comparison.
- **`--horizon N` is safe, but check `phase_binned`'s offset** if you add a new
  call site: it must be passed `WINDOW + horizon - 1` explicitly, or it silently
  falls back to the module-level `HORIZON` and misaligns every phase bin.

## 9. Files

| File | |
|---|---|
| `td_pipeline.py` | loading, subject splits, augmentation, phase variable, windows, scaling |
| `td_models.py` | the four architectures, multi-step heads, training callbacks |
| `td_eval.py` | three readouts, baselines, metrics, figures |
| `run_comparison.py` | entry point — trains everything, writes every table and figure |
| `make_tables.py` | CSVs → `outputs/tables.tex` |
| `aggregate_seeds.py` | aggregates several `outputs_seed*` runs; reports whether an effect holds in every seed |
| `diagnose.py` | underfitting harness: baselines vs augmentation vs batch size |
| `README.md` | the pipeline in detail, and every point where the old manuscript disagrees with the code |
| `outputs/` | the reported run (seed 42, 6 splits) |
| `outputs_seed42..46/` | the 5-seed robustness check; `outputs_seed42` duplicates `outputs/` |
| `outputs_h1_superseded/` | the discarded H=1 run, kept as evidence for §5a |

The old CP-based study is untouched in `Neural_Networks_*/` and
`models_complete_comparison.py`. Nothing here imports from it.
