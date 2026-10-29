from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASELINE_CSV = (
    ROOT
    / "results"
    / "dedup_run1"
    / "test"
    / "baseline_coco"
    / "per_class_ap.csv"
)

TWO_STAGE_CSV = (
    ROOT
    / "results"
    / "dedup_run1"
    / "test"
    / "two_stage_coco"
    / "per_class_ap.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "dedup_run1"
    / "test"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Class order and display names
# ============================================================

CLASS_KEYS = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_LABELS = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted\nSurface",
    "Rolled-in\nScale",
    "Scratches",
]


# ============================================================
# Load results
# ============================================================

def load_per_class_ap(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Per-class AP file was not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {"class", "AP"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain the columns "
            f"'class' and 'AP'."
        )

    df = df.set_index("class")

    missing_classes = [
        class_name
        for class_name in CLASS_KEYS
        if class_name not in df.index
    ]

    if missing_classes:
        raise ValueError(
            f"Missing classes in {csv_path}: "
            f"{missing_classes}"
        )

    # CSV values are between 0 and 1.
    # Multiply by 100 to present AP as a percentage.
    return (
        df.loc[CLASS_KEYS, "AP"]
        .astype(float)
        .to_numpy()
        * 100
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("GENERATE FIGURE 6.4 - PER-CLASS TEST AP COMPARISON")
    print("=" * 80)

    baseline_ap = load_per_class_ap(BASELINE_CSV)
    two_stage_ap = load_per_class_ap(TWO_STAGE_CSV)

    changes = two_stage_ap - baseline_ap

    print("\nPer-class test AP:")
    print("-" * 80)

    for label, baseline, proposed, change in zip(
        CLASS_LABELS,
        baseline_ap,
        two_stage_ap,
        changes,
    ):
        printable_label = label.replace("\n", " ")

        print(
            f"{printable_label:20s} "
            f"Baseline={baseline:6.2f}  "
            f"Two-stage={proposed:6.2f}  "
            f"Change={change:+6.2f}"
        )

    x_positions = np.arange(len(CLASS_LABELS))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6.5))

    baseline_bars = ax.bar(
        x_positions - bar_width / 2,
        baseline_ap,
        width=bar_width,
        label="Direct six-class Faster R-CNN",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
    )

    two_stage_bars = ax.bar(
        x_positions + bar_width / 2,
        two_stage_ap,
        width=bar_width,
        label="Proposed two-stage framework",
        color="#F58518",
        edgecolor="black",
        linewidth=0.6,
    )

    # Add AP values above each bar.
    for bars in (baseline_bars, two_stage_bars):
        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(CLASS_LABELS, fontsize=10)

    ax.set_xlabel("Defect Class", fontsize=12)
    ax.set_ylabel("Average Precision, AP (%)", fontsize=12)

    ax.set_title(
        "Per-Class AP Comparison on the Reserved Test Set",
        fontsize=14,
        pad=14,
    )

    # Leave enough space for value labels.
    maximum_value = max(
        baseline_ap.max(),
        two_stage_ap.max(),
    )

    ax.set_ylim(0, max(70, maximum_value + 10))

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )

    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=10,
    )

    # Remove unnecessary borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "figure_6_4_per_class_test_ap.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()