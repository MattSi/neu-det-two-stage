import argparse
from pathlib import Path
import json
import csv
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neu_det_two_stage.paths import (  # noqa: E402
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    preferred_checkpoint,
    require_directories,
    require_files,
)


# ============================================================
# Validation-only paths and portable checkpoint defaults
# ============================================================

STAGE1_ROOT = PROCESSED_DATA_DIR / "stage1"

VAL_JSON = (
    STAGE1_ROOT
    / "annotations"
    / "val.json"
)

VAL_IMAGE_DIR = (
    STAGE1_ROOT
    / "val"
    / "images"
)

STAGE1_MODEL = (
    preferred_checkpoint(
        Path("stage1/model_0002249.pth"),
        Path("stage1_faster_rcnn_r50_fpn/model_0002249.pth"),
    )
)

STAGE2_MODEL = (
    preferred_checkpoint(
        Path("stage2/model_best.pth"),
        Path("stage2_resnet50/model_best.pth"),
    )
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "validation"
    / "threshold_sweep"
)


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STAGE1_SCORE_THRESHOLD = 0.50

# A predicted box is treated as matching a GT box
# if IoU >= this threshold.
MATCH_IOU_THRESHOLD = 0.50

# Add padding before sending predicted ROI to ResNet50.
ROI_PADDING_RATIO = 0.10

STAGE1_SCORE_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
     0.82,
    0.84,
    0.86,
    0.88,
    0.90,
    0.92,
    0.94,
    0.96,
]

# ============================================================
# IoU
# ============================================================

def box_iou(box_a, box_b):
    """
    box format:
        [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(
        0,
        inter_x2 - inter_x1,
    )

    inter_h = max(
        0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_w
        * inter_h
    )

    area_a = max(
        0,
        ax2 - ax1,
    ) * max(
        0,
        ay2 - ay1,
    )

    area_b = max(
        0,
        bx2 - bx1,
    ) * max(
        0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# Expand ROI
# ============================================================

def expand_box(
    box,
    img_width,
    img_height,
    ratio,
):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    pad_x = width * ratio
    pad_y = height * ratio

    x1 = max(
        0,
        int(x1 - pad_x),
    )

    y1 = max(
        0,
        int(y1 - pad_y),
    )

    x2 = min(
        img_width,
        int(x2 + pad_x),
    )

    y2 = min(
        img_height,
        int(y2 + pad_y),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# Load Stage 1
# ============================================================

def build_stage1_predictor(score_threshold):

    cfg = get_cfg()

    config_name = (
        "COCO-Detection/"
        "faster_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.merge_from_file(
        model_zoo.get_config_file(config_name)
    )

    cfg.MODEL.WEIGHTS = str(STAGE1_MODEL)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        score_threshold
    )

    cfg.MODEL.DEVICE = DEVICE

    return DefaultPredictor(cfg)


# ============================================================
# Load Stage 2
# ============================================================

def load_stage2_model():

    checkpoint = torch.load(
        STAGE2_MODEL,
        map_location=DEVICE,
    )

    class_names = checkpoint[
        "class_names"
    ]

    image_size = checkpoint[
        "image_size"
    ]

    model = models.resnet50(
        weights=None
    )

    in_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        in_features,
        len(class_names),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(DEVICE)

    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    return (
        model,
        class_names,
        transform,
    )


# ============================================================
# Stage 2 prediction
# ============================================================

@torch.no_grad()
def classify_roi(
    roi_bgr,
    model,
    class_names,
    transform,
):

    roi_rgb = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2RGB,
    )

    tensor = transform(
        roi_rgb
    ).unsqueeze(0)

    tensor = tensor.to(
        DEVICE
    )

    with torch.amp.autocast(
        "cuda",
        enabled=torch.cuda.is_available(),
    ):

        logits = model(
            tensor
        )

    probs = torch.softmax(
        logits,
        dim=1,
    )[0]

    pred_idx = int(
        torch.argmax(
            probs
        ).item()
    )

    score = float(
        probs[
            pred_idx
        ].item()
    )

    return (
        class_names[
            pred_idx
        ],
        score,
    )


# ============================================================
# Load COCO validation data
# ============================================================

def load_validation_data():

    with open(
        VAL_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        coco = json.load(f)

    images = {
        image["id"]: image
        for image
        in coco["images"]
    }

    annotations_by_image = (
        defaultdict(list)
    )

    for ann in coco[
        "annotations"
    ]:

        annotations_by_image[
            ann["image_id"]
        ].append(ann)

    return (
        images,
        annotations_by_image,
    )


# ============================================================
# Matching
# ============================================================

def greedy_match(
    gt_boxes,
    pred_boxes,
):
    """
    Greedy one-to-one IoU matching.

    Returns:
        list of:
        {
            gt_index,
            pred_index,
            iou
        }
    """

    candidates = []

    for gt_idx, gt_box in enumerate(
        gt_boxes
    ):

        for pred_idx, pred_box in enumerate(
            pred_boxes
        ):

            iou = box_iou(
                gt_box,
                pred_box,
            )

            if iou >= MATCH_IOU_THRESHOLD:

                candidates.append(
                    (
                        iou,
                        gt_idx,
                        pred_idx,
                    )
                )

    # Highest-IoU pairs first
    candidates.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    used_gt = set()
    used_pred = set()

    matches = []

    for (
        iou,
        gt_idx,
        pred_idx,
    ) in candidates:

        if gt_idx in used_gt:
            continue

        if pred_idx in used_pred:
            continue

        used_gt.add(
            gt_idx
        )

        used_pred.add(
            pred_idx
        )

        matches.append(
            {
                "gt_index": gt_idx,
                "pred_index": pred_idx,
                "iou": iou,
            }
        )

    return matches

def evaluate_threshold(
    score_threshold,
    stage2_model,
    class_names,
    stage2_transform,
    images,
    annotations_by_image,
):

    stage1_predictor = build_stage1_predictor(
        score_threshold
    )

    total_gt = 0
    total_predictions = 0
    total_matched = 0

    classification_true = []
    classification_pred = []

    for image_id, image_info in images.items():

        image_path = (
            VAL_IMAGE_DIR
            / image_info["file_name"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        # -------------------------
        # Ground truth
        # -------------------------

        gt_annotations = (
            annotations_by_image[
                image_id
            ]
        )

        gt_boxes = []
        gt_classes = []

        for ann in gt_annotations:

            x, y, w, h = ann["bbox"]

            gt_boxes.append(
                (
                    x,
                    y,
                    x + w,
                    y + h,
                )
            )

            gt_classes.append(
                ann["original_class"]
            )

        total_gt += len(gt_boxes)

        # -------------------------
        # Stage 1
        # -------------------------

        outputs = stage1_predictor(
            image
        )

        instances = outputs[
            "instances"
        ].to("cpu")

        pred_boxes = (
            instances
            .pred_boxes
            .tensor
            .numpy()
        )

        total_predictions += len(
            pred_boxes
        )

        # -------------------------
        # Match
        # -------------------------

        matches = greedy_match(
            gt_boxes,
            pred_boxes,
        )

        total_matched += len(
            matches
        )

        # -------------------------
        # Stage 2
        # -------------------------

        for match in matches:

            gt_idx = match[
                "gt_index"
            ]

            pred_idx = match[
                "pred_index"
            ]

            true_class = (
                gt_classes[
                    gt_idx
                ]
            )

            pred_box = (
                pred_boxes[
                    pred_idx
                ]
            )

            expanded_box = expand_box(
                pred_box,
                img_w,
                img_h,
                ROI_PADDING_RATIO,
            )

            x1, y1, x2, y2 = (
                expanded_box
            )

            roi = image[
                y1:y2,
                x1:x2
            ]

            if roi.size == 0:
                continue

            (
                predicted_class,
                classification_score,
            ) = classify_roi(
                roi,
                stage2_model,
                class_names,
                stage2_transform,
            )

            classification_true.append(
                true_class
            )

            classification_pred.append(
                predicted_class
            )

    # ========================================================
    # Metrics
    # ========================================================

    tp = total_matched
    fp = total_predictions - total_matched
    fn = total_gt - total_matched

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    matched_roi_accuracy = (
        accuracy_score(
            classification_true,
            classification_pred,
        )
        if classification_true
        else 0.0
    )

    correct_classifications = sum(
        true == pred
        for true, pred
        in zip(
            classification_true,
            classification_pred,
        )
    )

    e2e_success = (
        correct_classifications
        / total_gt
        if total_gt > 0
        else 0.0
    )

    return {
        "threshold": score_threshold,
        "gt_boxes": total_gt,
        "predicted_boxes": total_predictions,
        "matched_boxes": total_matched,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_roi_accuracy": matched_roi_accuracy,
        "end_to_end_success": e2e_success,
    }


# ============================================================
# CLI and main validation
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Stage 1 operating thresholds on validation data only. "
            "Do not use this script to tune on the test set."
        )
    )
    parser.add_argument("--stage1-model", type=Path, default=STAGE1_MODEL)
    parser.add_argument("--stage2-model", type=Path, default=STAGE2_MODEL)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=DEVICE,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def configure_from_args(args):
    global STAGE1_MODEL
    global STAGE2_MODEL
    global DEVICE
    global OUTPUT_DIR

    STAGE1_MODEL = args.stage1_model.resolve()
    STAGE2_MODEL = args.stage2_model.resolve()
    DEVICE = args.device
    OUTPUT_DIR = args.output_dir.resolve()

    require_files(VAL_JSON, STAGE1_MODEL, STAGE2_MODEL)
    require_directories(VAL_IMAGE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    configure_from_args(parse_args())

    print("=" * 80)
    print("END-TO-END VALIDATION THRESHOLD SWEEP")
    print("=" * 80)

    (
        stage2_model,
        class_names,
        stage2_transform,
    ) = load_stage2_model()

    (
        images,
        annotations_by_image,
    ) = load_validation_data()

    all_results = []

    for threshold in STAGE1_SCORE_THRESHOLDS:

        print(
            f"\nEvaluating threshold = "
            f"{threshold:.2f}"
        )

        result = evaluate_threshold(
            threshold,
            stage2_model,
            class_names,
            stage2_transform,
            images,
            annotations_by_image,
        )

        all_results.append(
            result
        )

        print(
            f"Pred={result['predicted_boxes']} | "
            f"TP={result['tp']} | "
            f"FP={result['fp']} | "
            f"FN={result['fn']} | "
            f"P={result['precision']:.4f} | "
            f"R={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"ROI Acc="
            f"{result['matched_roi_accuracy']:.4f} | "
            f"E2E="
            f"{result['end_to_end_success']:.4f}"
        )

    # ========================================================
    # Print table
    # ========================================================

    print("\n")
    print("=" * 105)
    print(
        f"{'Thr':>5} "
        f"{'Pred':>7} "
        f"{'TP':>6} "
        f"{'FP':>6} "
        f"{'FN':>6} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'ROI Acc':>10} "
        f"{'E2E':>10}"
    )
    print("=" * 105)

    for r in all_results:

        print(
            f"{r['threshold']:5.2f} "
            f"{r['predicted_boxes']:7d} "
            f"{r['tp']:6d} "
            f"{r['fp']:6d} "
            f"{r['fn']:6d} "
            f"{r['precision']:10.4f} "
            f"{r['recall']:10.4f} "
            f"{r['f1']:10.4f} "
            f"{r['matched_roi_accuracy']:10.4f} "
            f"{r['end_to_end_success']:10.4f}"
        )

    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "threshold_sweep.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "threshold",
            "gt_boxes",
            "predicted_boxes",
            "matched_boxes",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "matched_roi_accuracy",
            "end_to_end_success",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in all_results:
            writer.writerow(result)

    # ========================================================
    # Find best operating points
    # ========================================================

    best_f1 = max(
        all_results,
        key=lambda x: x["f1"],
    )

    best_e2e = max(
        all_results,
        key=lambda x: x[
            "end_to_end_success"
        ],
    )

    print("\nBest Stage 1 F1:")
    print(
        f"threshold="
        f"{best_f1['threshold']:.2f}, "
        f"F1={best_f1['f1']:.4f}"
    )

    print("\nBest end-to-end success:")
    print(
        f"threshold="
        f"{best_e2e['threshold']:.2f}, "
        f"E2E="
        f"{best_e2e['end_to_end_success']:.4f}"
    )

    print("\nSaved:")
    print(csv_path)


if __name__ == "__main__":
    main()
