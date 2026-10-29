import argparse
from pathlib import Path
import json
import math
import io
import sys
from contextlib import redirect_stdout

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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

# For COCO AP evaluation, use a low inference threshold so that
# COCOeval sees enough ranked predictions to construct the PR curve.
STAGE1_SCORE_THRESHOLD = 0.05

# Same ROI padding used in the end-to-end validation experiment.
ROI_PADDING_RATIO = 0.10


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Must match category IDs in prepared/baseline_6class/annotations/*.json
CLASS_TO_CATEGORY_ID = {
    "crazing": 1,
    "inclusion": 2,
    "patches": 3,
    "pitted_surface": 4,
    "rolled-in_scale": 5,
    "scratches": 6,
}


# ============================================================
# Utility: expand predicted box before Stage 2 classification
# ============================================================

def expand_box(box, img_width, img_height, ratio):
    """Expand an XYXY box by ratio on each side and clip to image."""

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    pad_x = width * ratio
    pad_y = height * ratio

    x1 = max(0, int(math.floor(x1 - pad_x)))
    y1 = max(0, int(math.floor(y1 - pad_y)))
    x2 = min(img_width, int(math.ceil(x2 + pad_x)))
    y2 = min(img_height, int(math.ceil(y2 + pad_y)))

    return x1, y1, x2, y2


# ============================================================
# Stage 1: generic Faster R-CNN localizer
# ============================================================

def build_stage1_predictor():
    cfg = get_cfg()

    config_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

    cfg.merge_from_file(
        model_zoo.get_config_file(config_name)
    )

    cfg.MODEL.WEIGHTS = str(STAGE1_MODEL)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = STAGE1_SCORE_THRESHOLD
    cfg.MODEL.DEVICE = DEVICE

    return DefaultPredictor(cfg)


# ============================================================
# Stage 2: ResNet50 six-class classifier
# ============================================================

def load_stage2_model():
    checkpoint = torch.load(
        STAGE2_MODEL,
        map_location=DEVICE,
    )

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = models.resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, class_names, transform


# ============================================================
# Stage 2 batch classification
# ============================================================

@torch.no_grad()
def classify_rois_batch(rois_bgr, model, class_names, transform):
    if not rois_bgr:
        return []

    tensors = []

    for roi_bgr in rois_bgr:
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        tensors.append(transform(roi_rgb))

    batch = torch.stack(tensors, dim=0).to(DEVICE)

    with torch.amp.autocast(
        "cuda",
        enabled=torch.cuda.is_available(),
    ):
        logits = model(batch)

    probs = torch.softmax(logits, dim=1)
    pred_indices = torch.argmax(probs, dim=1)

    results = []

    for i in range(len(rois_bgr)):
        pred_idx = int(pred_indices[i].item())
        class_score = float(probs[i, pred_idx].item())

        results.append(
            (class_names[pred_idx], class_score)
        )

    return results


# ============================================================
# COCO helpers
# ============================================================

def load_coco_gt():
    return COCO(str(ANNOTATION_JSON))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Stage 1 + Stage 2 pipeline with COCO bbox "
            "metrics on validation or test data."
        )
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--stage1-model", type=Path, default=STAGE1_MODEL)
    parser.add_argument("--stage2-model", type=Path, default=STAGE2_MODEL)
    parser.add_argument(
        "--stage1-score-threshold",
        type=float,
        default=STAGE1_SCORE_THRESHOLD,
        help="Low inference threshold used to retain ranked COCO predictions.",
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for summary.json and per_class_ap.csv.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        help="Directory for the full prediction JSON (ignored by Git).",
    )
    return parser.parse_args()


def configure_from_args(args):
    global ANNOTATION_JSON
    global IMAGE_DIR
    global STAGE1_MODEL
    global STAGE2_MODEL
    global STAGE1_SCORE_THRESHOLD
    global ROI_PADDING_RATIO
    global DEVICE
    global OUTPUT_DIR
    global PREDICTION_DIR
    global EVALUATION_SPLIT

    EVALUATION_SPLIT = args.split
    ANNOTATION_JSON = (
        PROCESSED_DATA_DIR
        / "baseline_6class"
        / "annotations"
        / f"{args.split}.json"
    )
    IMAGE_DIR = (
        PROCESSED_DATA_DIR
        / "baseline_6class"
        / args.split
        / "images"
    )
    STAGE1_MODEL = args.stage1_model.resolve()
    STAGE2_MODEL = args.stage2_model.resolve()
    STAGE1_SCORE_THRESHOLD = args.stage1_score_threshold
    ROI_PADDING_RATIO = args.roi_padding_ratio
    DEVICE = args.device

    result_split = result_split_name(args.split)
    OUTPUT_DIR = (
        args.output_dir.resolve()
        if args.output_dir
        else RESULTS_DIR / result_split / "two_stage_coco"
    )
    PREDICTION_DIR = (
        args.prediction_dir.resolve()
        if args.prediction_dir
        else RESULTS_DIR / "predictions" / result_split / "two_stage_coco"
    )

    require_files(ANNOTATION_JSON, STAGE1_MODEL, STAGE2_MODEL)
    require_directories(IMAGE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def run_coco_eval(coco_gt, coco_dt, image_ids, cat_ids=None, quiet=False):
    """
    Run COCO bbox evaluation.

    cat_ids=None -> evaluate all categories.
    cat_ids=[id] -> evaluate only one category.

    COCOeval populates .stats during summarize(), so summarize() must be
    called even for per-class evaluation. For per-class evaluation we
    suppress the repeated 12-line console output with redirect_stdout.
    """

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox",
    )

    evaluator.params.imgIds = image_ids

    if cat_ids is not None:
        evaluator.params.catIds = cat_ids

    evaluator.evaluate()
    evaluator.accumulate()

    if quiet:
        with redirect_stdout(io.StringIO()):
            evaluator.summarize()
    else:
        evaluator.summarize()

    return evaluator


# ============================================================
# Main
# ============================================================

def main():
    configure_from_args(parse_args())

    print("=" * 80)
    print(f"TWO-STAGE 6-CLASS COCO {EVALUATION_SPLIT.upper()} EVALUATION")
    print("=" * 80)

    print("\nDevice:")
    print(DEVICE)

    print("\nStage 1 model:")
    print(STAGE1_MODEL)

    print("\nStage 2 model:")
    print(STAGE2_MODEL)

    print(
        f"\nStage 1 score threshold: "
        f"{STAGE1_SCORE_THRESHOLD}"
    )

    print(
        f"ROI padding ratio: "
        f"{ROI_PADDING_RATIO}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    stage1_predictor = build_stage1_predictor()

    (
        stage2_model,
        stage2_class_names,
        stage2_transform,
    ) = load_stage2_model()

    print("\nStage 2 classes:")

    for i, name in enumerate(stage2_class_names):
        print(f"  {i}: {name}")

    if list(stage2_class_names) != CLASS_NAMES:
        raise RuntimeError(
            "Stage 2 class order does not match expected class order."
        )

    # --------------------------------------------------------
    # Load six-class ground truth
    # --------------------------------------------------------

    coco_gt = load_coco_gt()
    image_ids = coco_gt.getImgIds()

    print(f"\n{EVALUATION_SPLIT.title()} images: {len(image_ids)}")

    predictions = []
    total_stage1_boxes = 0

    # --------------------------------------------------------
    # Process every image in the requested split
    # --------------------------------------------------------

    for image_index, image_id in enumerate(image_ids, start=1):
        image_info = coco_gt.loadImgs([image_id])[0]

        image_path = IMAGE_DIR / image_info["file_name"]
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Cannot read image: {image_path}")
            continue

        img_h, img_w = image.shape[:2]

        # ----------------------------------------------------
        # Stage 1: generic defect boxes
        # ----------------------------------------------------

        outputs = stage1_predictor(image)
        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes.tensor.numpy()
        stage1_scores = instances.scores.numpy()

        total_stage1_boxes += len(boxes)

        # ----------------------------------------------------
        # Prepare padded ROIs for Stage 2
        # ----------------------------------------------------

        valid_boxes = []
        valid_stage1_scores = []
        rois = []

        for box, stage1_score in zip(boxes, stage1_scores):
            x1, y1, x2, y2 = expand_box(
                box,
                img_w,
                img_h,
                ROI_PADDING_RATIO,
            )

            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            valid_boxes.append(box)
            valid_stage1_scores.append(float(stage1_score))
            rois.append(roi)

        # ----------------------------------------------------
        # Stage 2: six-class classification
        # ----------------------------------------------------

        classifications = classify_rois_batch(
            rois,
            stage2_model,
            stage2_class_names,
            stage2_transform,
        )

        # ----------------------------------------------------
        # Build final six-class COCO-format predictions
        # ----------------------------------------------------

        for box, stage1_score, classification in zip(
            valid_boxes,
            valid_stage1_scores,
            classifications,
        ):
            class_name, stage2_score = classification

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            category_id = CLASS_TO_CATEGORY_ID[class_name]

            # Final confidence combines localization confidence and
            # classification confidence.
            combined_score = stage1_score * stage2_score

            predictions.append({
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [
                    float(x1),
                    float(y1),
                    float(width),
                    float(height),
                ],
                "score": float(combined_score),
            })

        if image_index % 25 == 0:
            print(
                f"Processed {image_index}/{len(image_ids)} images"
            )

    # ========================================================
    # Save prediction JSON
    # ========================================================

    prediction_json = PREDICTION_DIR / "two_stage_predictions.json"

    with open(
        prediction_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(predictions, f, indent=2)

    print("\nPrediction summary:")
    print(f"Stage 1 boxes: {total_stage1_boxes}")
    print(f"Two-stage predictions: {len(predictions)}")
    print(f"Saved predictions: {prediction_json}")

    if len(predictions) == 0:
        print("\nNo predictions generated.")
        return

    # ========================================================
    # COCO result object
    # ========================================================

    coco_dt = coco_gt.loadRes(str(prediction_json))

    # ========================================================
    # Overall six-class COCO evaluation
    # ========================================================

    print("\n" + "=" * 80)
    print("OVERALL COCO RESULTS")
    print("=" * 80)

    coco_eval = run_coco_eval(
        coco_gt,
        coco_dt,
        image_ids,
        cat_ids=None,
        quiet=False,
    )

    overall_stats = coco_eval.stats.copy()

    # ========================================================
    # Per-class COCO evaluation
    # ========================================================

    print("\n" + "=" * 80)
    print("PER-CLASS AP")
    print("=" * 80)

    per_class_results = {}

    print(
        f"{'Class':20s} "
        f"{'AP':>8s} "
        f"{'AP50':>8s} "
        f"{'AP75':>8s}"
    )
    print("-" * 50)

    for class_name in CLASS_NAMES:
        category_id = CLASS_TO_CATEGORY_ID[class_name]

        class_eval = run_coco_eval(
            coco_gt,
            coco_dt,
            image_ids,
            cat_ids=[category_id],
            quiet=True,
        )

        class_stats = class_eval.stats.copy()

        class_result = {
            "AP": float(class_stats[0]),
            "AP50": float(class_stats[1]),
            "AP75": float(class_stats[2]),
            "AP_small": float(class_stats[3]),
            "AP_medium": float(class_stats[4]),
            "AP_large": float(class_stats[5]),
            "AR_1": float(class_stats[6]),
            "AR_10": float(class_stats[7]),
            "AR_100": float(class_stats[8]),
            "AR_small": float(class_stats[9]),
            "AR_medium": float(class_stats[10]),
            "AR_large": float(class_stats[11]),
        }

        per_class_results[class_name] = class_result

        print(
            f"{class_name:20s} "
            f"{class_result['AP'] * 100:8.2f} "
            f"{class_result['AP50'] * 100:8.2f} "
            f"{class_result['AP75'] * 100:8.2f}"
        )

    # ========================================================
    # Save summary JSON
    # ========================================================

    summary = {
        "split": EVALUATION_SPLIT,
        "annotation_json": str(ANNOTATION_JSON),
        "stage1_model": str(STAGE1_MODEL),
        "stage2_model": str(STAGE2_MODEL),
        "stage1_score_threshold": STAGE1_SCORE_THRESHOLD,
        "roi_padding_ratio": ROI_PADDING_RATIO,
        "num_images": len(image_ids),
        "stage1_boxes": total_stage1_boxes,
        "predictions": len(predictions),

        "AP": float(overall_stats[0]),
        "AP50": float(overall_stats[1]),
        "AP75": float(overall_stats[2]),
        "AP_small": float(overall_stats[3]),
        "AP_medium": float(overall_stats[4]),
        "AP_large": float(overall_stats[5]),

        "AR_1": float(overall_stats[6]),
        "AR_10": float(overall_stats[7]),
        "AR_100": float(overall_stats[8]),
        "AR_small": float(overall_stats[9]),
        "AR_medium": float(overall_stats[10]),
        "AR_large": float(overall_stats[11]),

        "per_class": per_class_results,
    }

    summary_json = OUTPUT_DIR / "summary.json"

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    # ========================================================
    # Save a simple per-class CSV for paper/table analysis
    # ========================================================

    per_class_csv = OUTPUT_DIR / "per_class_ap.csv"

    with open(
        per_class_csv,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "class,AP,AP50,AP75,AP_small,AP_medium,AP_large,"
            "AR_1,AR_10,AR_100,AR_small,AR_medium,AR_large\n"
        )

        for class_name in CLASS_NAMES:
            r = per_class_results[class_name]

            f.write(
                f"{class_name},"
                f"{r['AP']},"
                f"{r['AP50']},"
                f"{r['AP75']},"
                f"{r['AP_small']},"
                f"{r['AP_medium']},"
                f"{r['AP_large']},"
                f"{r['AR_1']},"
                f"{r['AR_10']},"
                f"{r['AR_100']},"
                f"{r['AR_small']},"
                f"{r['AR_medium']},"
                f"{r['AR_large']}\n"
            )

    print("\nSaved summary:")
    print(summary_json)

    print("\nSaved per-class CSV:")
    print(per_class_csv)


if __name__ == "__main__":
    main()
