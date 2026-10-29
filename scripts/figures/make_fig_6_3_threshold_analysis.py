from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT
    / "results"
    / "dedup_run1"
    / "validation"
    / "threshold_sweep"
    / "threshold_sweep.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "dedup_run1"
    / "validation"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("GENERATE FIGURE 6.3 - VALIDATION THRESHOLD ANALYSIS")
    print("=" * 80)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Threshold-sweep CSV was not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("threshold").reset_index(drop=True)

    required_columns = {
        "threshold",
        "predicted_boxes",
        "precision",
        "recall",
        "f1",
        "matched_roi_accuracy",
        "end_to_end_success",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Convert metrics from 0-1 to percentages.
    threshold = df["threshold"]
    precision = df["precision"] * 100
    recall = df["recall"] * 100
    f1 = df["f1"] * 100
    end_to_end = df["end_to_end_success"] * 100
    predicted_boxes = df["predicted_boxes"]

    # ========================================================
    # Create a two-panel figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )

    # --------------------------------------------------------
    # Panel A: Precision, recall and F1
    # --------------------------------------------------------

    ax_metrics = axes[0]

    ax_metrics.plot(
        threshold,
        precision,
        marker="o",
        linewidth=2,
        markersize=5,
        label="Precision",
        color="#4C78A8",
    )

    ax_metrics.plot(
        threshold,
        recall,
        marker="s",
        linewidth=2,
        markersize=5,
        label="Recall",
        color="#F58518",
    )

    ax_metrics.plot(
        threshold,
        f1,
        marker="^",
        linewidth=2,
        markersize=5,
        label="F1-score",
        color="#54A24B",
    )

    ax_metrics.set_xlabel(
        "Stage 1 Confidence Threshold",
        fontsize=11,
    )

    ax_metrics.set_ylabel(
        "Performance (%)",
        fontsize=11,
    )

    ax_metrics.set_title(
        "(a) Precision–Recall Trade-off",
        fontsize=12,
    )

    ax_metrics.set_ylim(0, 105)

    ax_metrics.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    ax_metrics.legend(
        fontsize=9,
        loc="lower left",
    )

    # --------------------------------------------------------
    # Panel B: candidate boxes and end-to-end success
    # --------------------------------------------------------

    ax_boxes = axes[1]

    boxes_line = ax_boxes.plot(
        threshold,
        predicted_boxes,
        marker="o",
        linewidth=2,
        markersize=5,
        color="#4C78A8",
        label="Predicted boxes",
    )

    ax_boxes.set_xlabel(
        "Stage 1 Confidence Threshold",
        fontsize=11,
    )

    ax_boxes.set_ylabel(
        "Number of Predicted Boxes",
        fontsize=11,
        color="#4C78A8",
    )

    ax_boxes.tick_params(
        axis="y",
        labelcolor="#4C78A8",
    )

    ax_boxes.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    # Second y-axis for end-to-end success.
    ax_success = ax_boxes.twinx()

    success_line = ax_success.plot(
        threshold,
        end_to_end,
        marker="s",
        linewidth=2,
        markersize=5,
        color="#E45756",
        label="End-to-end success",
    )

    ax_success.set_ylabel(
        "End-to-End Success (%)",
        fontsize=11,
        color="#E45756",
    )

    ax_success.tick_params(
        axis="y",
        labelcolor="#E45756",
    )

    ax_success.set_ylim(0, 105)

    ax_boxes.set_title(
        "(b) Candidate Volume and End-to-End Success",
        fontsize=12,
    )

    # Combine legends from both y-axes.
    lines = boxes_line + success_line
    labels = [line.get_label() for line in lines]

    ax_boxes.legend(
        lines,
        labels,
        fontsize=9,
        loc="upper right",
    )

    fig.suptitle(
        "Stage 1 Confidence-Threshold Analysis on the Validation Set",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "figure_6_3_threshold_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
