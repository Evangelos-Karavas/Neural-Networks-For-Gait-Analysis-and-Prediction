
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def add_value_labels(ax, bars, fmt="{:.2f}", pad_ratio=0.015):
    ymax = max(bar.get_height() for bar in bars) if bars else 1.0
    pad = ymax * pad_ratio
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def plot_metric(ax, labels, values, title, ylabel, ylim=None):
    bars = ax.bar(labels, values)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Input architecture", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    add_value_labels(ax, bars)


def save_single_network_plot(df, network_name, out_dir, mae_ylim=None, rmse_ylim=None):
    labels = ["Kinematics", "Kinetics", "Both"]
    mae = df["Overall MAE"].tolist()
    rmse = df["Overall RMSE"].tolist()

    fig, axes = plt.subneural_networks_outputs/plots(1, 2, figsize=(13, 5.5))
    plot_metric(
        axes[0], labels, mae,
        f"{network_name} - Overall MAE",
        "Overall MAE (deg)",
        mae_ylim
    )
    plot_metric(
        axes[1], labels, rmse,
        f"{network_name} - Overall RMSE",
        "Overall RMSE (deg)",
        rmse_ylim
    )

    fig.suptitle(f"{network_name} Ablation Study", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = out_dir / f"{network_name.lower()}_ablation_beautiful.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_combined_comparison_plot(cnn_df, lstm_df, out_dir, mae_ylim=None, rmse_ylim=None):
    labels = ["Kinematics", "Kinetics", "Both"]

    fig, axes = plt.subneural_networks_outputs/plots(2, 2, figsize=(14, 10))

    plot_metric(
        axes[0, 0], labels, cnn_df["Overall MAE"].tolist(),
        "CNN - Overall MAE", "Overall MAE (deg)", mae_ylim
    )
    plot_metric(
        axes[0, 1], labels, cnn_df["Overall RMSE"].tolist(),
        "CNN - Overall RMSE", "Overall RMSE (deg)", rmse_ylim
    )
    plot_metric(
        axes[1, 0], labels, lstm_df["Overall MAE"].tolist(),
        "LSTM - Overall MAE", "Overall MAE (deg)", mae_ylim
    )
    plot_metric(
        axes[1, 1], labels, lstm_df["Overall RMSE"].tolist(),
        "LSTM - Overall RMSE", "Overall RMSE (deg)", rmse_ylim
    )

    fig.suptitle("Ablation Study Results: CNN vs LSTM", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_dir / "cnn_lstm_ablation_combined.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    out_dir = Path("ablation_neural_networks_outputs/plots_output")
    out_dir.mkdir(exist_ok=True)

    # Corrected data from the user
    lstm_df = pd.DataFrame({
        "Model": [
            "LSTM + kinematics only",
            "LSTM + kinetics only",
            "LSTM + both",
        ],
        "#Features": [6, 36, 42],
        "Window": [51, 51, 51],
        "Overall MAE": [2.213449335, 2.613449335, 1.785354733],
        "Overall RMSE": [2.981206848, 3.481206848, 2.388585336],
    })

    cnn_df = pd.DataFrame({
        "Model": [
            "CNN + kinematics only",
            "CNN + kinetics only",
            "CNN + both",
        ],
        "#Features": [6, 36, 42],
        "Window": [51, 51, 51],
        "Overall MAE": [3.395486069, 4.104476452, 2.7433815],
        "Overall RMSE": [4.93619723, 5.681059828, 3.445594413],
    })

    # Shared limits make comparison easier across networks
    mae_max = max(cnn_df["Overall MAE"].max(), lstm_df["Overall MAE"].max()) * 1.18
    rmse_max = max(cnn_df["Overall RMSE"].max(), lstm_df["Overall RMSE"].max()) * 1.18
    mae_ylim = (0, mae_max)
    rmse_ylim = (0, rmse_max)

    cnn_path = save_single_network_plot(cnn_df, "CNN", out_dir, mae_ylim, rmse_ylim)
    lstm_path = save_single_network_plot(lstm_df, "LSTM", out_dir, mae_ylim, rmse_ylim)
    combined_path = save_combined_comparison_plot(cnn_df, lstm_df, out_dir, mae_ylim, rmse_ylim)

    # Save the exact data used for plotting
    cnn_df.to_csv(out_dir / "cnn_corrected_summary.csv", index=False)
    lstm_df.to_csv(out_dir / "lstm_corrected_summary.csv", index=False)

    print("Created:")
    print(cnn_path.resolve())
    print(lstm_path.resolve())
    print(combined_path.resolve())


if __name__ == "__main__":
    main()
