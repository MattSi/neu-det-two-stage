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
    result_split_name,
)


# ============================================================
# Portable defaults (overridable from the CLI)
# ============================================================

ANNOTATION_JSON = None
IMAGE_DIR = None
STAGE1_MODEL = preferred_checkpoint(
    Path("stage1/model_0002249.pth"),
    Path("stage1_faster_rcnn_r50_fpn/model_0002249.pth"),
)
STAGE2_MODEL = preferred_checkpoint(
    Path("stage2/model_best.pth"),
    Path("stage2_resnet50/model_best.pth"),
)
OUTPUT_DIR = None
PREDICTION_DIR = None
EVALUATION_SPLIT = "val"


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

def build_stage1_predictor():

    cfg = get_cfg()

    config_name = (
        "COCO-Detection/"
        "faster_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.merge_from_file(
        model_zoo.get_config_file(
            config_name
        )
    )

    cfg.MODEL.WEIGHTS = str(
        STAGE1_MODEL
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        STAGE1_SCORE_THRESHOLD
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
# Load COCO-format data
# ============================================================

def load_evaluation_data():

    with open(
        ANNOTATION_JSON,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Stage 1 and Stage 2 checkpoints with greedy "
            "IoU matching on validation or test data."
        )
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--stage1-model", type=Path, default=STAGE1_MODEL)
    parser.add_argument("--stage2-model", type=Path, default=STAGE2_MODEL)
    parser.add_argument(
        "--stage1-score-threshold",
        type=float,
        default=STAGE1_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=MATCH_IOU_THRESHOLD,
    )
    parser.add_argument(
        "--roi-padding-ratio",
        type=float,
        default=ROI_PADDING_RATIO,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=DEVICE,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prediction-dir", type=Path)
    return parser.parse_args()


def configure_from_args(args):
    global ANNOTATION_JSON
    global IMAGE_DIR
    global STAGE1_MODEL
    global STAGE2_MODEL
    global STAGE1_SCORE_THRESHOLD
    global MATCH_IOU_THRESHOLD
    global ROI_PADDING_RATIO
    global DEVICE
    global OUTPUT_DIR
    global PREDICTION_DIR
    global EVALUATION_SPLIT

    EVALUATION_SPLIT = args.split
    stage1_root = PROCESSED_DATA_DIR / "stage1"
    ANNOTATION_JSON = stage1_root / "annotations" / f"{args.split}.json"
    IMAGE_DIR = stage1_root / args.split / "images"
    STAGE1_MODEL = args.stage1_model.resolve()
    STAGE2_MODEL = args.stage2_model.resolve()
    STAGE1_SCORE_THRESHOLD = args.stage1_score_threshold
    MATCH_IOU_THRESHOLD = args.match_iou_threshold
    ROI_PADDING_RATIO = args.roi_padding_ratio
    DEVICE = args.device

    result_split = result_split_name(args.split)
    OUTPUT_DIR = (
        args.output_dir.resolve()
        if args.output_dir
        else RESULTS_DIR / result_split / "end_to_end"
    )
    PREDICTION_DIR = (
        args.prediction_dir.resolve()
        if args.prediction_dir
        else RESULTS_DIR / "predictions" / result_split / "end_to_end"
    )

    require_files(ANNOTATION_JSON, STAGE1_MODEL, STAGE2_MODEL)
    require_directories(IMAGE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


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


# ============================================================
# Main evaluation
# ============================================================

def main():
    configure_from_args(parse_args())

    print("=" * 70)
    print(f"END-TO-END {EVALUATION_SPLIT.upper()} EVALUATION")
    print("=" * 70)

    print("\nDevice:")
    print(DEVICE)

    print("\nStage 1 model:")
    print(STAGE1_MODEL)

    print("\nStage 2 model:")
    print(STAGE2_MODEL)

    stage1_predictor = (
        build_stage1_predictor()
    )

    (
        stage2_model,
        class_names,
        stage2_transform,
    ) = load_stage2_model()

    (
        images,
        annotations_by_image,
    ) = load_evaluation_data()

    print(
        f"\n{EVALUATION_SPLIT.title()} images: "
        f"{len(images)}"
    )

    print(
        "\nStage 2 classes:"
    )

    for i, name in enumerate(
        class_names
    ):
        print(
            f"  {i}: {name}"
        )

    total_gt = 0

    total_predictions = 0

    total_matched = 0

    classification_true = []
    classification_pred = []

    detailed_records = []

    # --------------------------------------------------------
    # Process each image
    # --------------------------------------------------------

    for image_index, (
        image_id,
        image_info,
    ) in enumerate(
        images.items(),
        start=1,
    ):

        image_path = (
            IMAGE_DIR
            / image_info["file_name"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"[WARNING] Cannot read: "
                f"{image_path}"
            )

            continue

        img_h, img_w = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt_annotations = (
            annotations_by_image[
                image_id
            ]
        )

        gt_boxes = []
        gt_classes = []

        for ann in gt_annotations:

            x, y, w, h = (
                ann["bbox"]
            )

            gt_boxes.append(
                (
                    x,
                    y,
                    x + w,
                    y + h,
                )
            )

            # We saved this during
            # prepare_neu_det.py
            gt_classes.append(
                ann[
                    "original_class"
                ]
            )

        total_gt += len(
            gt_boxes
        )

        # ----------------------------------------------------
        # Stage 1 prediction
        # ----------------------------------------------------

        outputs = (
            stage1_predictor(
                image
            )
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

        pred_scores = (
            instances
            .scores
            .numpy()
        )

        total_predictions += len(
            pred_boxes
        )

        # ----------------------------------------------------
        # GT ↔ prediction matching
        # ----------------------------------------------------

        matches = greedy_match(
            gt_boxes,
            pred_boxes,
        )

        total_matched += len(
            matches
        )

        # ----------------------------------------------------
        # Stage 2 classification
        # ----------------------------------------------------

        for match in matches:

            gt_idx = match[
                "gt_index"
            ]

            pred_idx = match[
                "pred_index"
            ]

            match_iou = match[
                "iou"
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

            expanded_box = (
                expand_box(
                    pred_box,
                    img_w,
                    img_h,
                    ROI_PADDING_RATIO,
                )
            )

            (
                x1,
                y1,
                x2,
                y2,
            ) = expanded_box

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

            detailed_records.append(
                {
                    "image": (
                        image_info[
                            "file_name"
                        ]
                    ),

                    "true_class": (
                        true_class
                    ),

                    "predicted_class": (
                        predicted_class
                    ),

                    "stage1_score": float(
                        pred_scores[
                            pred_idx
                        ]
                    ),

                    "iou": float(
                        match_iou
                    ),

                    "stage2_score": float(
                        classification_score
                    ),

                    "pred_box": [
                        float(v)
                        for v
                        in pred_box
                    ],
                }
            )

        if (
            image_index % 25
            == 0
        ):

            print(
                f"Processed "
                f"{image_index}/"
                f"{len(images)} images"
            )

    # ========================================================
    # Metrics
    # ========================================================

    stage1_recall = (
        total_matched
        / total_gt
        if total_gt > 0
        else 0.0
    )

    print("\n" + "=" * 70)

    print(
        "STAGE 1 MATCHING RESULTS"
    )

    print("=" * 70)

    print(
        f"GT boxes:        "
        f"{total_gt}"
    )

    print(
        f"Predicted boxes: "
        f"{total_predictions}"
    )

    print(
        f"Matched boxes:   "
        f"{total_matched}"
    )

    print(
        f"Recall @ IoU "
        f"{MATCH_IOU_THRESHOLD:.2f}: "
        f"{stage1_recall:.4f}"
    )

    # --------------------------------------------------------
    # Classification metrics on matched boxes
    # --------------------------------------------------------

    if len(
        classification_true
    ) == 0:

        print(
            "\nNo matched ROIs "
            "were available for classification."
        )

        return

    matched_accuracy = (
        accuracy_score(
            classification_true,
            classification_pred,
        )
    )

    print("\n" + "=" * 70)

    print(
        "STAGE 2 CLASSIFICATION "
        "ON MATCHED PREDICTED ROIs"
    )

    print("=" * 70)

    print(
        f"Matched ROI classification "
        f"accuracy: "
        f"{matched_accuracy:.4f}"
    )

    report = (
        classification_report(
            classification_true,
            classification_pred,
            labels=class_names,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "\nClassification report:\n"
    )

    print(
        report
    )

    matrix = confusion_matrix(
        classification_true,
        classification_pred,
        labels=class_names,
    )

    print(
        "\nConfusion matrix:\n"
    )

    print(
        matrix
    )

    # --------------------------------------------------------
    # End-to-end metric
    # --------------------------------------------------------

    correct_classifications = sum(
        true == pred
        for true, pred
        in zip(
            classification_true,
            classification_pred,
        )
    )

    end_to_end_accuracy = (
        correct_classifications
        / total_gt
        if total_gt > 0
        else 0.0
    )

    print("\n" + "=" * 70)

    print(
        "END-TO-END RESULT"
    )

    print("=" * 70)

    print(
        f"Correctly localized "
        f"and classified GT defects: "
        f"{correct_classifications}"
    )

    print(
        f"Total GT defects: "
        f"{total_gt}"
    )

    print(
        f"End-to-end accuracy: "
        f"{end_to_end_accuracy:.4f}"
    )

    # ========================================================
    # Save outputs
    # ========================================================

    report_dict = (
        classification_report(
            classification_true,
            classification_pred,
            labels=class_names,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    )

    summary = {
        "split": EVALUATION_SPLIT,

        "annotation_json": str(
            ANNOTATION_JSON
        ),

        "stage1_model": str(
            STAGE1_MODEL
        ),

        "stage2_model": str(
            STAGE2_MODEL
        ),

        "score_threshold": (
            STAGE1_SCORE_THRESHOLD
        ),

        "match_iou_threshold": (
            MATCH_IOU_THRESHOLD
        ),

        "roi_padding_ratio": (
            ROI_PADDING_RATIO
        ),

        "gt_boxes": (
            total_gt
        ),

        "predicted_boxes": (
            total_predictions
        ),

        "matched_boxes": (
            total_matched
        ),

        "stage1_recall": (
            stage1_recall
        ),

        "matched_roi_accuracy": (
            matched_accuracy
        ),

        "end_to_end_accuracy": (
            end_to_end_accuracy
        ),

        "classification_report": (
            report_dict
        ),
    }

    with open(
        OUTPUT_DIR
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    np.savetxt(
        OUTPUT_DIR
        / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    with open(
        PREDICTION_DIR
        / "detailed_predictions.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "image",
            "true_class",
            "predicted_class",
            "stage1_score",
            "iou",
            "stage2_score",
            "pred_box",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in detailed_records:

            record = dict(
                record
            )

            record[
                "pred_box"
            ] = json.dumps(
                record[
                    "pred_box"
                ]
            )

            writer.writerow(
                record
            )

    print("\nSaved to:")
    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
