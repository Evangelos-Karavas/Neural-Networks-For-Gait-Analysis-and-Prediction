#!/usr/bin/env python3
"""Train and compare the four models on typically developed gait only.

  python run_comparison.py --quick      # smoke test, 1 split, few epochs
  python run_comparison.py              # full run
  python run_comparison.py --repeats 10

Networks are trained on TD subjects and evaluated on held-out TD subjects over
repeated random subject-level splits. Everything written to outputs/ -- metric
tables as CSV, the paper tables as Markdown, and the figures -- comes from this
one entry point.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import td_eval as ev
from td_models import MODEL_ORDER, MODEL_SPECS, training_callbacks
from td_pipeline import (
    ANGLE_COLS,
    AugmentConfig,
    CHANNEL_LABELS,
    HORIZON,
    OUTPUT_DIR,
    SAGITTAL_IDX,
    SAGITTAL_LABELS,
    STRIDE_LEN,
    WINDOW,
    build_fold,
    load_all_subjects,
    make_repeated_splits,
    mean_toe_off,
    prepare_frame,
)

MODEL_LABELS = {key: spec.label for key, spec in MODEL_SPECS.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=6,
                   help="subject-level splits; the default 6 x 2 test subjects "
                        "holds out each of the 12 subjects exactly once")
    p.add_argument("--n-test", type=int, default=2, help="held-out test subjects per split")
    p.add_argument("--n-val", type=int, default=2, help="validation subjects per split")
    p.add_argument("--aug-rounds", type=int, default=7,
                   help="augmented copies per training subject (0 disables augmentation)")
    p.add_argument("--horizon", type=int, default=HORIZON,
                   help=f"prediction horizon in samples (default {HORIZON}, "
                        "about 200 ms; 51 samples is a full stride)")
    p.add_argument("--rollout-strides", type=int, default=6,
                   help="strides to roll out recursively")
    p.add_argument("--seed", type=int, default=42, help="base seed for the splits")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=None,
                   help="override the per-model epoch budget")
    p.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument("--quick", action="store_true",
                   help="1 split, 5 epochs, 2 augmentation rounds -- pipeline smoke test")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.repeats = 1
        args.epochs = args.epochs or 5
        args.aug_rounds = min(args.aug_rounds, 2)
    return args


def mean_over_channels(metrics: dict[str, np.ndarray], idx=None) -> dict[str, float]:
    sel = slice(None) if idx is None else idx
    return {k: float(np.mean(v[sel])) for k, v in metrics.items()}


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = [
            floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in row
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule] + rows)


def main() -> None:
    args = parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    print("Loading TD subjects ...")
    raw_frames = load_all_subjects()
    subject_ids = sorted(raw_frames)
    n_strides = {sid: len(df) // STRIDE_LEN for sid, df in raw_frames.items()}
    print(f"  {len(subject_ids)} subjects, {sum(n_strides.values())} strides: "
          + ", ".join(f"{s}({n_strides[s]})" for s in subject_ids))

    splits = make_repeated_splits(subject_ids, n_repeats=args.repeats,
                                  n_val=args.n_val, n_test=args.n_test, seed=args.seed)

    subject_rows: list[dict] = []
    channel_rows: list[dict] = []
    rollout_rows: list[dict] = []
    phase_rows: list[dict] = []
    baseline_rows: list[dict] = []
    figure_cache: dict = {}
    started = time.time()

    for split in splits:
        print(f"\n=== split {split.repeat}: test={split.test} val={split.val} ===")
        fold = build_fold(split, raw_frames, aug_rounds=args.aug_rounds, seed=args.seed)
        print(f"  training frames: {len(fold.train_frames)} "
              f"({len(split.train)} subjects x {1 + args.aug_rounds} copies)")

        # Parameter-free references on exactly the same held-out samples.
        for sid, frame in fold.test_frames.items():
            preds, gt = ev.baseline_predictions(frame, horizon=args.horizon)
            for name, pred in preds.items():
                baseline_rows.append({
                    "repeat": split.repeat, "subject": sid, "model": name,
                    **mean_over_channels(ev.channel_metrics(pred, gt)),
                })

        for key in args.models:
            spec = MODEL_SPECS[key]
            x_train, y_train = fold.train_xy(spec.kind, args.horizon)
            x_val, y_val = fold.val_xy(spec.kind, args.horizon)

            # Weight init and dropout are seeded per (split, model) so a rerun
            # reproduces the published numbers; the splits are already seeded.
            tf.keras.utils.set_random_seed(args.seed * 100 + split.repeat * 10
                                           + MODEL_ORDER.index(key))
            model = spec.build(horizon=args.horizon)
            epochs = args.epochs or spec.epochs
            t0 = time.time()
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=args.batch_size,
                callbacks=training_callbacks(),
                verbose=0,
            )
            trained = len(history.history["loss"])
            print(f"  {spec.label:<16} windows={x_train.shape[0]:>6}  "
                  f"epochs={trained:>3}/{epochs}  val_loss={history.history['val_loss'][-1]:.4f}  "
                  f"({time.time() - t0:.0f}s)")

            for sid, frame in fold.test_frames.items():
                pred, gt = ev.teacher_forced(model, fold, frame, spec.kind,
                                             horizon=args.horizon)
                if pred is None:
                    print(f"    {sid}: too short for a window, skipped")
                    continue

                metrics = ev.channel_metrics(pred, gt)
                subject_rows.append({
                    "repeat": split.repeat, "subject": sid, "model": key,
                    **mean_over_channels(metrics),
                    **{f"sag_{k}": v for k, v in
                       mean_over_channels(metrics, SAGITTAL_IDX).items()},
                })
                for ch, label in enumerate(CHANNEL_LABELS):
                    channel_rows.append({
                        "repeat": split.repeat, "subject": sid, "model": key,
                        "channel": label,
                        **{k: float(v[ch]) for k, v in metrics.items()},
                    })

                toe_l, toe_r = mean_toe_off(fold.test_raw[sid])
                per_phase = ev.phase_binned(pred, gt, toe_l, toe_r,
                                            offset=WINDOW + args.horizon - 1)
                for p, phase in enumerate(ev.GAIT_PHASES):
                    phase_rows.append({
                        "repeat": split.repeat, "subject": sid, "model": key,
                        "phase": phase.replace("\n", " "),
                        "MAE": float(np.nanmean(per_phase[p])),
                    })

                roll_pred, roll_gt = ev.recursive_rollout(
                    model, fold, frame, spec.kind, n_strides=args.rollout_strides
                )
                if roll_pred is not None:
                    roll_metrics = ev.channel_metrics(roll_pred, roll_gt)
                    rollout_rows.append({
                        "repeat": split.repeat, "subject": sid, "model": key,
                        "strides": len(roll_gt) / STRIDE_LEN,
                        **mean_over_channels(roll_metrics),
                        "hip_phase_lag_samples": ev.phase_lag_samples(roll_pred, roll_gt, 0),
                    })

                if split.repeat == 0 and sid == sorted(fold.test_frames)[0]:
                    figure_cache.setdefault("subject", sid)
                    figure_cache.setdefault("teacher", {})[key] = pred
                    figure_cache["teacher_gt"] = gt
                    figure_cache.setdefault("phase", {})[key] = per_phase
                    if roll_pred is not None:
                        figure_cache.setdefault("rollout", {})[key] = roll_pred
                        figure_cache["rollout_gt"] = roll_gt

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------
    subject_df = pd.DataFrame(subject_rows)
    channel_df = pd.DataFrame(channel_rows)
    rollout_df = pd.DataFrame(rollout_rows)
    phase_df = pd.DataFrame(phase_rows)

    subject_df.to_csv(out / "per_subject_metrics.csv", index=False)
    channel_df.to_csv(out / "per_channel_metrics.csv", index=False)
    rollout_df.to_csv(out / "rollout_metrics.csv", index=False)
    phase_df.to_csv(out / "per_phase_metrics.csv", index=False)

    order = [k for k in MODEL_ORDER if k in set(subject_df["model"])]

    baseline_df = pd.DataFrame(baseline_rows)
    if not baseline_df.empty:
        baseline_df.to_csv(out / "baseline_metrics.csv", index=False)

    generalization = pd.DataFrame([
        {
            "Model": MODEL_LABELS[key],
            "MAE (deg)": g["MAE"].mean(),
            "MAE SD": g["MAE"].std(),
            "RMSE (deg)": g["RMSE"].mean(),
            "RMSE SD": g["RMSE"].std(),
            "R2": g["R2"].mean(),
            "R2 SD": g["R2"].std(),
            "R2>0": f"{int((g['R2'] > 0).sum())}/{len(g)}",
        }
        for key in order
        for g in [subject_df[subject_df["model"] == key]]
    ])

    per_joint = pd.DataFrame({"Joint": SAGITTAL_LABELS})
    for key in order:
        sub = channel_df[channel_df["model"] == key]
        col = []
        for ch in SAGITTAL_IDX:
            rows = sub[sub["channel"] == CHANNEL_LABELS[ch]]
            col.append(f"{rows['MAE'].mean():.2f}/{rows['RMSE'].mean():.2f}")
        per_joint[MODEL_LABELS[key]] = col
    mean_row = {"Joint": "Mean"}
    for key in order:
        sub = channel_df[
            (channel_df["model"] == key)
            & channel_df["channel"].isin([CHANNEL_LABELS[c] for c in SAGITTAL_IDX])
        ]
        mean_row[MODEL_LABELS[key]] = f"{sub['MAE'].mean():.2f}/{sub['RMSE'].mean():.2f}"
    per_joint = pd.concat([per_joint, pd.DataFrame([mean_row])], ignore_index=True)

    stability = pd.DataFrame([
        {
            "Model": MODEL_LABELS[key],
            "Rollout MAE (deg)": g["MAE"].mean(),
            "Rollout MAE SD": g["MAE"].std(),
            "Rollout R2": g["R2"].mean(),
            "Hip phase lag (samples)": g["hip_phase_lag_samples"].abs().mean(),
        }
        for key in order
        for g in [rollout_df[rollout_df["model"] == key]]
    ]) if not rollout_df.empty else pd.DataFrame()

    phase_table = (
        phase_df.pivot_table(index="phase", columns="model", values="MAE", aggfunc="mean")
        .reindex([p.replace("\n", " ") for p in ev.GAIT_PHASES])
        .reindex(columns=order)
        .rename(columns=MODEL_LABELS)
    )

    if not baseline_df.empty:
        generalization = pd.concat([generalization, pd.DataFrame([
            {
                "Model": name.capitalize() + " (baseline)",
                "MAE (deg)": g["MAE"].mean(), "MAE SD": g["MAE"].std(),
                "RMSE (deg)": g["RMSE"].mean(), "RMSE SD": g["RMSE"].std(),
                "R2": g["R2"].mean(), "R2 SD": g["R2"].std(),
                "R2>0": f"{int((g['R2'] > 0).sum())}/{len(g)}",
            }
            for name in ("persistence", "linear")
            for g in [baseline_df[baseline_df["model"] == name]]
        ])], ignore_index=True)

    generalization.to_csv(out / "table_generalization.csv", index=False)
    per_joint.to_csv(out / "table_per_joint_sagittal.csv", index=False)
    phase_table.to_csv(out / "table_per_phase.csv")
    if not stability.empty:
        stability.to_csv(out / "table_rollout_stability.csv", index=False)

    n_evals = len(subject_df) // max(1, len(order))
    report = [
        "# TD-only model comparison",
        "",
        f"{len(subject_ids)} typically developed subjects, {sum(n_strides.values())} strides. "
        f"{args.repeats} random subject-level splits "
        f"({len(splits[0].train)} train / {args.n_val} val / {args.n_test} test subjects), "
        f"{args.aug_rounds} augmented copies per training subject. "
        f"Prediction horizon {args.horizon} samples (~{args.horizon * 20} ms).",
        f"Each model is therefore scored on {n_evals} held-out subject evaluations.",
        "",
        f"## Teacher-forced accuracy at +{args.horizon} samples, held-out TD subjects (18 channels)",
        "",
        markdown_table(generalization, "{:.2f}"),
        "",
        "## Per-joint teacher-forced error, sagittal plane (MAE/RMSE, degrees)",
        "",
        markdown_table(per_joint),
        "",
        f"## Recursive rollout over {args.rollout_strides} strides",
        "",
        markdown_table(stability, "{:.2f}") if not stability.empty else "_no rollouts_",
        "",
        "## Mean absolute error per gait phase (degrees)",
        "",
        markdown_table(phase_table.reset_index().rename(columns={"phase": "Phase"})),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    (out / "run_config.json").write_text(json.dumps({
        "subjects": subject_ids,
        "strides_per_subject": n_strides,
        "splits": [
            {"repeat": s.repeat, "train": s.train, "val": s.val, "test": s.test}
            for s in splits
        ],
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "augmentation": vars(AugmentConfig()),
    }, indent=2), encoding="utf-8")

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    if not args.no_figures:
        figs = out / "figures"
        ev.plot_td_variability(raw_frames, figs / "td_variability.png")
        ev.plot_subject_scatter(subject_df, figs / "subject_scatter_mae.png", MODEL_LABELS)
        ev.plot_pv_strides(prepare_frame(raw_frames[subject_ids[0]]),
                           figs / "pv_over_strides.png")

        sid = figure_cache.get("subject")
        if sid:
            if "rollout" in figure_cache:
                ev.plot_rollout(
                    figure_cache["rollout"], figure_cache["rollout_gt"],
                    figs / "rollout_sagittal.png",
                    f"Recursive rollout over {args.rollout_strides} strides -- held-out subject",
                    MODEL_LABELS,
                )
            ev.plot_subject_cycle(
                figure_cache["teacher"], figure_cache["teacher_gt"],
                figs / "subject_cycle_left.png",
                "Stride-averaged prediction vs ground truth -- held-out subject, left side",
                MODEL_LABELS,
            )
            ev.plot_per_phase_bars(
                figure_cache["phase"], figs / "per_phase_bars.png",
                "Mean absolute error per gait phase -- held-out subject",
                MODEL_LABELS,
            )
        print(f"\nFigures written to {figs}")

    print("\n" + "\n".join(report[4:]))
    print(f"\nDone in {time.time() - started:.0f}s. Tables and report in {out}")


if __name__ == "__main__":
    main()
