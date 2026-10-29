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
    / "test"
    / "stage2_classifier"
    / "confusion_matrix.csv"
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
# Class names
# ============================================================

CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches",
]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("GENERATE FIGURE 6.2 - STAGE 2 TEST CONFUSION MATRIX")
    print("=" * 80)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Confusion-matrix CSV was not found:\n{CSV_PATH}"
        )

    # 当前 CSV 没有表头或行索引
    df = pd.read_csv(CSV_PATH, header=None)
    cm = df.to_numpy(dtype=int)

    expected_shape = (len(CLASS_NAMES), len(CLASS_NAMES))

    if cm.shape != expected_shape:
        raise ValueError(
            f"Expected a {expected_shape} confusion matrix, "
            f"but received {cm.shape}."
        )

    total_samples = int(cm.sum())
    correct_samples = int(cm.diagonal().sum())
    accuracy = correct_samples / total_samples

    print(f"Samples: {total_samples}")
    print(f"Correct: {correct_samples}")
    print(f"Accuracy: {accuracy:.4%}")

    fig, ax = plt.subplots(figsize=(9, 7.5))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=35,
        ha="right",
        fontsize=10,
    )

    ax.set_yticklabels(
        CLASS_NAMES,
        fontsize=10,
    )

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)

    ax.set_title(
        "Stage 2 ResNet50 Confusion Matrix on the Reserved Test Set",
        fontsize=14,
        pad=14,
    )

    # 根据背景深浅自动选择白色或黑色数字
    text_threshold = cm.max() / 2

    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            value = cm[row, column]

            text_color = (
                "white"
                if value > text_threshold
                else "black"
            )

            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold" if value > 0 else "normal",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Number of ROIs", fontsize=11)

    # 在图片底部标注样本数量和准确率
    fig.text(
        0.5,
        0.01,
        (
            f"Test ROIs: {total_samples} | "
            f"Correct predictions: {correct_samples} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        ),
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    output_path = (
        OUTPUT_DIR
        / "figure_6_2_stage2_test_confusion_matrix.png"
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