"""
Detection confusion matrices for the baseline and the two-stage pipeline,
computed from saved COCO-style predictions (no models are loaded).

Matching follows validate_end_to_end.py: class-agnostic greedy one-to-one
matching, highest IoU first, IoU >= 0.5. Each matrix has an extra "missed"
column (ground-truth boxes no prediction matched) and an extra "background"
row (predictions that matched no ground-truth box).

Steps:
1. Pick the baseline score threshold on the validation split by the best F1,
   the same rule used for the Stage 1 threshold (0.84).
2. Apply each pipeline's threshold to the test split.
   The two-stage file stores stage1_score * stage2_score, so its boxes are
   filtered by the raw Stage 1 score from stage1_coco/predictions.json
   (same order, same boxes). This reproduces end_to_end/summary.json.
3. Write CSVs, a summary and (optionally) figures.

Inputs: the saved predictions in results/dedup_run1/predictions/ (included in
the repository) and the processed annotations in data/processed/ (created by
scripts/data/prepare_neu_det.py and prepare_baseline_6class.py).

Run from the repository root:
    python scripts/evaluate/detection_confusion_from_predictions.py
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN = "dedup_run1"
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96]

RESULTS = PROJECT_ROOT / "results" / RUN
DATA_DIR = PROJECT_ROOT / "data/processed"
# Curated copies are committed under results/<run>/predictions/; fresh runs of
# the evaluators write to results/predictions/<run>/ (ignored by Git).
PRED_DIR = RESULTS / "predictions"
if not PRED_DIR.exists():
    PRED_DIR = PROJECT_ROOT / "results/predictions" / RUN
ANN_DIR = DATA_DIR / "baseline_6class/annotations"


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def evaluate(annotations, predictions, threshold, iou_threshold=0.5):
    gts, preds = defaultdict(list), defaultdict(list)
    for a in annotations["annotations"]:
        gts[a["image_id"]].append((to_xyxy(a["bbox"]), a["category_id"] - 1))
    for p in predictions:
        if p["score"] >= threshold:
            preds[p["image_id"]].append((to_xyxy(p["bbox"]), p["category_id"] - 1))

    n = len(CLASS_NAMES)
    cm = [[0] * (n + 1) for _ in range(n + 1)]  # row n = background, col n = missed
    matched = 0
    for image in annotations["images"]:
        g, q = gts[image["id"]], preds[image["id"]]
        pairs = []
        for gi, (gb, _) in enumerate(g):
            for pi, (pb, _) in enumerate(q):
                v = iou(gb, pb)
                if v >= iou_threshold:
                    pairs.append((v, gi, pi))
        pairs.sort(key=lambda x: x[0], reverse=True)
        used_g, used_p = set(), set()
        for _, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            cm[g[gi][1]][q[pi][1]] += 1
        for gi, (_, c) in enumerate(g):
            if gi not in used_g:
                cm[c][n] += 1
        for pi, (_, c) in enumerate(q):
            if pi not in used_p:
                cm[n][c] += 1
        matched += len(used_g)

    num_gt = sum(len(v) for v in gts.values())
    num_pred = sum(len(v) for v in preds.values())
    correct = sum(cm[i][i] for i in range(n))
    precision = matched / num_pred if num_pred else 0.0
    recall = matched / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "score_threshold": threshold,
        "match_iou_threshold": iou_threshold,
        "gt_boxes": num_gt,
        "predicted_boxes": num_pred,
        "matched_boxes": matched,
        "correctly_classified": correct,
        "false_positives": num_pred - matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_roi_accuracy": correct / matched if matched else 0.0,
        "end_to_end_accuracy": correct / num_gt if num_gt else 0.0,
        "confusion_matrix": cm,
    }


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def two_stage_with_stage1_scores(split):
    stage1 = load(PRED_DIR / split / "stage1_coco/predictions.json")
    two_stage = load(PRED_DIR / split / "two_stage_coco/two_stage_predictions.json")
    if len(stage1) != len(two_stage):
        raise ValueError("Stage 1 and two-stage prediction files differ in length")
    merged = []
    for s1, ts in zip(stage1, two_stage):
        same_box = all(abs(x - y) < 1e-3 for x, y in zip(s1["bbox"], ts["bbox"]))
        if s1["image_id"] != ts["image_id"] or not same_box:
            raise ValueError("Stage 1 and two-stage predictions are not in the same order")
        merged.append(dict(ts, score=s1["score"]))
    return merged


def write_matrix_csv(path, cm):
    header = ["true \\ predicted"] + CLASS_NAMES + ["missed"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name, row in zip(CLASS_NAMES + ["background"], cm):
            w.writerow([name] + row)


def make_figures(results, out_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    labels = ["Crazing", "Inclusion", "Patches", "Pitted\nsurface", "Rolled-in\nscale", "Scratches"]
    cmap = LinearSegmentedColormap.from_list("b", ["#FFFFFF", "#E4E9FB", "#2747C9"])
    vmax = max(max(r[:6]) for res in results.values() for r in res["confusion_matrix"][:6])

    titles = {
        "baseline": ("Baseline: six-class Faster R-CNN", "figure_6_5_detection_confusion_baseline.png"),
        "two_stage": ("Two-stage pipeline", "figure_6_6_detection_confusion_two_stage.png"),
    }
    for key, (title, filename) in titles.items():
        res = results[key]
        cm = np.array(res["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8.8, 7.6), dpi=150)
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(7):
            for j in range(7):
                v = cm[i, j]
                edge = i == 6 or j == 6
                color = "#FFFFFF" if v > vmax * 0.6 else ("#B4332E" if edge and v else ("#16201D" if v else "#A8B3AF"))
                bold = (i == j and i < 6) or (edge and v)
                ax.text(j, i, str(v), ha="center", va="center", fontsize=12, color=color,
                        fontweight="bold" if bold else "normal")
        ax.set_xticks(range(7))
        ax.set_xticklabels(labels + ["Missed"], fontsize=9.5)
        ax.set_yticks(range(7))
        ax.set_yticklabels(labels + ["Background\n(false pos.)"], fontsize=9.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.axhline(5.5, color="#5A6863", lw=1)
        ax.axvline(5.5, color="#5A6863", lw=1)
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=22)
        sub = (f"Test · score ≥ {res['score_threshold']}, IoU ≥ 0.5 · "
               f"{res['matched_boxes']}/{res['gt_boxes']} found · "
               f"end-to-end {res['end_to_end_accuracy'] * 100:.1f}%")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color="#5A6863", va="bottom")
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, facecolor="white")
        plt.close(fig)

    # One Stage 2 test crop per class
    fig, axes = plt.subplots(1, 6, figsize=(11, 2.4), dpi=150)
    for ax, folder, label in zip(axes, CLASS_NAMES, labels):
        crops = sorted((DATA_DIR / "stage2_roi/test" / folder).glob(f"*_{folder}.jpg"))
        if crops:
            ax.imshow(Image.open(crops[0]).convert("L"), cmap="gray", aspect="auto")
        ax.set_title(label.replace("\n", " "), fontsize=10.5)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_6_7_stage2_roi_samples.png", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--two-stage-threshold", type=float, default=0.84,
                        help="Stage 1 score threshold already chosen on validation.")
    parser.add_argument("--no-figures", action="store_true", help="Skip the matplotlib figures.")
    parser.add_argument("--pred-dir", type=Path, help="Folder with the saved predictions.")
    parser.add_argument("--data-dir", type=Path, help="Processed data folder (default: data/processed).")
    args = parser.parse_args()

    global PRED_DIR, DATA_DIR, ANN_DIR
    if args.pred_dir:
        PRED_DIR = args.pred_dir
    if args.data_dir:
        DATA_DIR = args.data_dir
        ANN_DIR = DATA_DIR / "baseline_6class/annotations"

    # 1. Baseline threshold on validation
    val_ann = load(ANN_DIR / "val.json")
    val_pred = load(PRED_DIR / "validation/baseline_checkpoints/model_0002499/predictions.json")
    sweep = [evaluate(val_ann, val_pred, t) for t in THRESHOLDS]
    best = max(sweep, key=lambda r: r["f1"])
    sweep_dir = RESULTS / "validation/baseline_threshold_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    cols = ["score_threshold", "gt_boxes", "predicted_boxes", "matched_boxes", "false_positives",
            "precision", "recall", "f1", "matched_roi_accuracy", "end_to_end_accuracy"]
    with open(sweep_dir / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sweep:
            w.writerow([r[c] for c in cols])
    print(f"Baseline threshold chosen on validation: {best['score_threshold']} (F1 {best['f1']:.4f})")

    # 2. Test split
    test_ann = load(ANN_DIR / "test.json")
    results = {
        "baseline": evaluate(test_ann, load(PRED_DIR / "test/baseline_coco/predictions.json"),
                             best["score_threshold"]),
        "two_stage": evaluate(test_ann, two_stage_with_stage1_scores("test"), args.two_stage_threshold),
    }

    # 3. Outputs
    out_dir = RESULTS / "test/detection_confusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, res in results.items():
        write_matrix_csv(out_dir / f"{key}_confusion_matrix.csv", res["confusion_matrix"])
        print(f"{key:9s} threshold {res['score_threshold']}: found {res['matched_boxes']}/{res['gt_boxes']}, "
              f"FP {res['false_positives']}, P {res['precision']:.3f}, R {res['recall']:.3f}, "
              f"F1 {res['f1']:.3f}, end-to-end {res['end_to_end_accuracy']:.3f}")
    summary = {
        "class_names": CLASS_NAMES,
        "matrix_layout": "rows = true class + background, columns = predicted class + missed",
        "baseline_threshold_from_validation": best["score_threshold"],
        "results": results,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_figures:
        fig_dir = RESULTS / "test/figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        make_figures(results, fig_dir)
        print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
