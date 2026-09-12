#!/usr/bin/env python3
"""Turn the metric CSVs into paste-ready IEEE LaTeX tables.

  python make_tables.py                 # reads outputs/, writes outputs/tables.tex

Separate from run_comparison.py on purpose: reformatting the paper's tables
should never require retraining. The best value in each row is bolded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from td_models import MODEL_ORDER, MODEL_SPECS
from td_pipeline import CHANNEL_LABELS, OUTPUT_DIR, SAGITTAL_IDX, SAGITTAL_LABELS

MODEL_LABELS = {k: v.label for k, v in MODEL_SPECS.items()}
# The LaTeX macros already defined in the manuscript preamble.
MODEL_MACROS = {
    "timestamp_lstm": r"\tslstm{}",
    "pv_lstm": r"\pvlstm{}",
    "timestamp_cnn": r"\tscnn{}",
    "pv_cnn": r"\pvcnn{}",
}


def bold(text: str, flag: bool) -> str:
    if not flag:
        return text
    # \textbf{} does not affect math mode, so bold the inside instead.
    if text.startswith("$") and text.endswith("$"):
        return r"$\mathbf{" + text[1:-1] + "}$"
    return r"\textbf{" + text + "}"


def table_per_joint(channel_df: pd.DataFrame, order: list[str]) -> str:
    """Per-joint MAE/RMSE in the sagittal plane, one column per model."""
    rows = []
    for ch, joint in zip(SAGITTAL_IDX, SAGITTAL_LABELS):
        sub = channel_df[channel_df["channel"] == CHANNEL_LABELS[ch]]
        maes = [sub.loc[sub["model"] == k, "MAE"].mean() for k in order]
        rmses = [sub.loc[sub["model"] == k, "RMSE"].mean() for k in order]
        best = int(np.argmin(maes))
        cells = [bold(f"{m:.2f}/{r:.2f}", i == best) for i, (m, r) in enumerate(zip(maes, rmses))]
        rows.append(f"    {joint:<8}& " + " & ".join(cells) + r" \\")

    sag_labels = [CHANNEL_LABELS[c] for c in SAGITTAL_IDX]
    sag = channel_df[channel_df["channel"].isin(sag_labels)]
    maes = [sag.loc[sag["model"] == k, "MAE"].mean() for k in order]
    rmses = [sag.loc[sag["model"] == k, "RMSE"].mean() for k in order]
    best = int(np.argmin(maes))
    cells = [bold(f"{m:.2f}/{r:.2f}", i == best) for i, (m, r) in enumerate(zip(maes, rmses))]
    mean_row = r"    \textbf{Mean} & " + " & ".join(cells) + r" \\"

    header = " & ".join(r"\textbf{" + MODEL_MACROS[k] + "}" for k in order)
    return "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Per-Joint Teacher-Forced Error (degrees) on Held-Out TD Subjects}",
        r"  \label{tab:errors}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{2.5pt}",
        r"  \begin{tabular}{l" + "c" * len(order) + "}",
        r"    \hline",
        r"    & \multicolumn{" + str(len(order)) + r"}{c}{\textbf{MAE / RMSE}} \\",
        r"    \hline",
        r"    \textbf{Joint} & " + header + r" \\",
        r"    \hline",
        *rows,
        r"    \hline",
        mean_row,
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def table_generalization(subject_df: pd.DataFrame, order: list[str],
                         baseline_df: pd.DataFrame | None = None) -> str:
    """Mean +/- SD across every held-out subject evaluation, all 18 channels.

    The parameter-free baselines are printed underneath: a learned model is
    only worth reporting if it beats them at this horizon.
    """
    stats = {
        k: subject_df[subject_df["model"] == k] for k in order
    }
    best_mae = min(order, key=lambda k: stats[k]["MAE"].mean())
    best_rmse = min(order, key=lambda k: stats[k]["RMSE"].mean())
    best_r2 = max(order, key=lambda k: stats[k]["R2"].mean())

    rows = []
    for k in order:
        g = stats[k]
        mae = bold(f"${g['MAE'].mean():.2f} \\pm {g['MAE'].std():.2f}$", k == best_mae)
        rmse = bold(f"${g['RMSE'].mean():.2f} \\pm {g['RMSE'].std():.2f}$", k == best_rmse)
        r2 = bold(f"${g['R2'].mean():.2f} \\pm {g['R2'].std():.2f}$", k == best_r2)
        pos = f"{int((g['R2'] > 0).sum())}/{len(g)}"
        rows.append(f"    {MODEL_MACROS[k]} & {mae} & {rmse} & {r2} & {pos} " + r"\\")

    if baseline_df is not None and not baseline_df.empty:
        rows.append(r"    \hline")
        for name, label in (("persistence", "Persistence"), ("linear", "Linear extrap.")):
            g = baseline_df[baseline_df["model"] == name]
            if g.empty:
                continue
            cells = " & ".join([
                f"${g['MAE'].mean():.2f} \\pm {g['MAE'].std():.2f}$",
                f"${g['RMSE'].mean():.2f} \\pm {g['RMSE'].std():.2f}$",
                f"${g['R2'].mean():.2f} \\pm {g['R2'].std():.2f}$",
                f"{int((g['R2'] > 0).sum())}/{len(g)}",
            ])
            rows.append(f"    {label} & {cells} " + r"\\")

    return "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{All-Plane Accuracy on Held-Out TD Subjects (mean $\pm$ SD",
        r"  over held-out subjects of all splits, 18 joint--plane channels)}",
        r"  \label{tab:generalization}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{2pt}",
        r"  \begin{tabular}{lcccc}",
        r"    \hline",
        r"    \textbf{Model} & \textbf{MAE (deg)} & \textbf{RMSE (deg)} &",
        r"    \textbf{$R^2$} & \textbf{$R^2\!>\!0$} \\",
        r"    \hline",
        *rows,
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def table_rollout(rollout_df: pd.DataFrame, order: list[str]) -> str:
    """Recursive-rollout error and hip phase lag."""
    stats = {k: rollout_df[rollout_df["model"] == k] for k in order}
    order = [k for k in order if len(stats[k])]
    if not order:
        return ""

    best_mae = min(order, key=lambda k: stats[k]["MAE"].mean())
    best_lag = min(order, key=lambda k: stats[k]["hip_phase_lag_samples"].abs().mean())

    rows = []
    for k in order:
        g = stats[k]
        mae = bold(f"${g['MAE'].mean():.2f} \\pm {g['MAE'].std():.2f}$", k == best_mae)
        lag = bold(f"${g['hip_phase_lag_samples'].abs().mean():.1f}$", k == best_lag)
        rows.append(f"    {MODEL_MACROS[k]} & {mae} & ${g['R2'].mean():.2f}$ & {lag} " + r"\\")

    return "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Recursive Rollout on Held-Out TD Subjects. The phase lag is",
        r"  the sample shift that best realigns the predicted sagittal hip with",
        r"  the measured one over the final stride of the rollout.}",
        r"  \label{tab:rollout}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{lccc}",
        r"    \hline",
        r"    \textbf{Model} & \textbf{MAE (deg)} & \textbf{$R^2$} &",
        r"    \textbf{$|$Hip phase lag$|$ (samples)} \\",
        r"    \hline",
        *rows,
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def table_per_phase(phase_df: pd.DataFrame, order: list[str]) -> str:
    """Mean absolute error inside each of the seven gait phases."""
    pivot = phase_df.pivot_table(index="phase", columns="model", values="MAE", aggfunc="mean")
    phases = [p for p in phase_df["phase"].drop_duplicates()]
    pivot = pivot.reindex(phases).reindex(columns=order)

    rows = []
    for phase, row in pivot.iterrows():
        best = int(np.argmin(row.to_numpy()))
        cells = [bold(f"{v:.2f}", i == best) for i, v in enumerate(row.to_numpy())]
        rows.append(f"    {phase:<18}& " + " & ".join(cells) + r" \\")

    header = " & ".join(r"\textbf{" + MODEL_MACROS[k] + "}" for k in order)
    return "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Mean Absolute Error (degrees) per Gait Phase, averaged over",
        r"  the 18 joint--plane channels and all held-out subjects}",
        r"  \label{tab:perphase}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{2.5pt}",
        r"  \begin{tabular}{l" + "c" * len(order) + "}",
        r"    \hline",
        r"    \textbf{Gait phase} & " + header + r" \\",
        r"    \hline",
        *rows,
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()
    out = args.output

    subject_df = pd.read_csv(out / "per_subject_metrics.csv")
    channel_df = pd.read_csv(out / "per_channel_metrics.csv")
    phase_df = pd.read_csv(out / "per_phase_metrics.csv")
    rollout_path = out / "rollout_metrics.csv"
    rollout_df = pd.read_csv(rollout_path) if rollout_path.exists() else pd.DataFrame()
    baseline_path = out / "baseline_metrics.csv"
    baseline_df = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()

    order = [k for k in MODEL_ORDER if k in set(subject_df["model"])]

    blocks = [
        "% Generated by make_tables.py -- do not edit by hand.",
        "% Uses the \\tslstm{} / \\pvlstm{} / \\tscnn{} / \\pvcnn{} macros",
        "% already defined in the manuscript preamble.",
        "",
        table_per_joint(channel_df, order),
        "",
        table_generalization(subject_df, order, baseline_df),
        "",
    ]
    if not rollout_df.empty:
        blocks += [table_rollout(rollout_df, order), ""]
    blocks += [table_per_phase(phase_df, order), ""]

    target = out / "tables.tex"
    target.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
