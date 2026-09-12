#!/usr/bin/env python3
"""Aggregate several complete runs of run_comparison.py, one per seed.

A single seed fixes both the split arrangement and the network initialisation,
so one run cannot say whether an effect is a property of the method or of that
draw. This reads the per-seed output directories and reports, for each metric,
the spread across seeds and -- more importantly -- whether the phase-variable
advantage points the same way in every one of them.

Produce the inputs with, e.g.:

    for s in 42 43 44 45 46; do
        python run_comparison.py --seed $s --output outputs_seed$s --no-figures
    done
    python aggregate_seeds.py outputs_seed*
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["timestamp_lstm", "pv_lstm", "timestamp_cnn", "pv_cnn"]
LABELS = {
    "timestamp_lstm": "Timestamp LSTM",
    "pv_lstm": "PV LSTM",
    "timestamp_cnn": "Timestamp CNN",
    "pv_cnn": "PV CNN",
}
PAIRS = [("pv_lstm", "timestamp_lstm", "LSTM"), ("pv_cnn", "timestamp_cnn", "CNN")]


def per_seed_means(dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model means of each metric, one row per seed."""
    tf_rows, ro_rows = [], []
    for d in dirs:
        subj = pd.read_csv(d / "per_subject_metrics.csv")
        roll = pd.read_csv(d / "rollout_metrics.csv")
        roll["abslag"] = roll["hip_phase_lag_samples"].abs()
        tf_rows.append(subj.groupby("model")["MAE"].mean().rename(d.name))
        ro_rows.append(
            pd.concat(
                [
                    roll.groupby("model")["MAE"].mean().rename("rollout_MAE"),
                    roll.groupby("model")["abslag"].mean().rename("abslag"),
                ],
                axis=1,
            ).assign(seed=d.name)
        )
    return pd.concat(tf_rows, axis=1).T, pd.concat(ro_rows)


def paired_effect(dirs: list[Path], csv: str, col: str, use_abs: bool = False):
    """Mean PV-minus-timestamp difference within each seed, paired by evaluation."""
    out = {}
    for a, b, label in PAIRS:
        per_seed = []
        for d in dirs:
            df = pd.read_csv(d / csv)
            if use_abs:
                df[col] = df[col].abs()
            piv = df.pivot_table(index=["repeat", "subject"], columns="model", values=col)
            per_seed.append(float((piv[a] - piv[b]).dropna().mean()))
        out[label] = per_seed
    return out


def report(name: str, effect: dict[str, list[float]], unit: str) -> None:
    print(f"\n{name}  (negative = phase conditioning better)")
    for label, vals in effect.items():
        v = np.array(vals)
        agree = int((v < 0).sum())
        verdict = "all seeds agree" if agree == len(v) else f"{agree}/{len(v)} seeds"
        print(f"  {label:5s}  {v.mean():+.2f} {unit}  "
              f"(range {v.min():+.2f} to {v.max():+.2f}; {verdict})")


def main() -> None:
    args = sys.argv[1:] or sorted(str(p) for p in Path(".").glob("outputs_seed*"))
    dirs = [Path(a) for a in args]
    missing = [d for d in dirs if not (d / "per_subject_metrics.csv").exists()]
    if not dirs or missing:
        sys.exit(f"No usable run directories. Missing metrics in: {missing or dirs}")

    print(f"Aggregating {len(dirs)} runs: {', '.join(d.name for d in dirs)}")

    tf, ro = per_seed_means(dirs)
    print("\nTeacher-forced MAE (deg), mean over seeds +/- SD across seeds")
    for m in MODELS:
        print(f"  {LABELS[m]:15s} {tf[m].mean():.2f} +/- {tf[m].std():.2f}"
              f"   (per seed: {', '.join(f'{v:.2f}' for v in tf[m])})")

    print("\nRecursive rollout, mean over seeds +/- SD across seeds")
    for m in MODELS:
        g = ro.loc[m]
        print(f"  {LABELS[m]:15s} MAE {g['rollout_MAE'].mean():.2f} +/- {g['rollout_MAE'].std():.2f}"
              f"   |lag| {g['abslag'].mean():.2f} +/- {g['abslag'].std():.2f} samples"
              f"  ({100 * g['abslag'].mean() / 51:.1f}% of cycle)")

    report("Teacher-forced MAE, paired PV - timestamp",
           paired_effect(dirs, "per_subject_metrics.csv", "MAE"), "deg")
    report("Rollout MAE, paired PV - timestamp",
           paired_effect(dirs, "rollout_metrics.csv", "MAE"), "deg")
    report("Hip phase |lag|, paired PV - timestamp",
           paired_effect(dirs, "rollout_metrics.csv", "hip_phase_lag_samples", use_abs=True),
           "samples")

    print("\nThe phase claim is only as strong as the last block: if the lag effect "
          "\nis negative in every seed, it is a property of the conditioning rather "
          "\nthan of one draw.")


if __name__ == "__main__":
    main()
