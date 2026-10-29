"""Evaluate a frozen Detectron2 detector on validation or test data.

This script covers both the direct six-class baseline and the class-agnostic
Stage 1 localizer. It deliberately performs inference from a checkpoint and
never invokes a trainer.
"""

import argparse
import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import torch
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


CONFIG_NAME = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
DEFAULT_BASELINE_MODEL = preferred_checkpoint(
    Path("baseline/model_final.pth"),
    Path("baseline_6class_faster_rcnn_r50_fpn/model_final.pth"),
)
DEFAULT_STAGE1_MODEL = preferred_checkpoint(
    Path("stage1/model_0002249.pth"),
    Path("stage1_faster_rcnn_r50_fpn/model_0002249.pth"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen baseline or Stage 1 Faster R-CNN checkpoint "
            "with COCO bbox metrics."
        )
    )
    parser.add_argument(
        "--experiment",
        choices=("baseline", "stage1"),
        required=True,
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override the frozen checkpoint selected on validation data.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help=(
            "Inference threshold. Use the same value for baseline and the "
            "two-stage method when comparing COCO AP."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prediction-dir", type=Path)
    return parser.parse_args()


def experiment_config(args):
    if args.experiment == "baseline":
        dataset_name = "baseline_6class"
        num_classes = 6
        checkpoint = args.checkpoint or DEFAULT_BASELINE_MODEL
        result_name = "baseline_coco"
    else:
        dataset_name = "stage1"
        num_classes = 1
        checkpoint = args.checkpoint or DEFAULT_STAGE1_MODEL
        result_name = "stage1_coco"

    dataset_root = PROCESSED_DATA_DIR / dataset_name
    annotation_json = dataset_root / "annotations" / f"{args.split}.json"
    image_dir = dataset_root / args.split / "images"
    result_split = result_split_name(args.split)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else RESULTS_DIR / result_split / result_name
    )
    prediction_dir = (
        args.prediction_dir.resolve()
        if args.prediction_dir
        else RESULTS_DIR / "predictions" / result_split / result_name
    )

    return {
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "checkpoint": Path(checkpoint).resolve(),
        "annotation_json": annotation_json,
        "image_dir": image_dir,
        "output_dir": output_dir,
        "prediction_dir": prediction_dir,
    }


def build_predictor(checkpoint, num_classes, score_threshold, device):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_NAME))
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def run_coco_eval(coco_gt, coco_dt, image_ids, cat_ids=None, quiet=False):
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
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


def stats_dict(stats):
    names = (
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    )
    return {name: float(value) for name, value in zip(names, stats)}


def main():
    args = parse_args()
    paths = experiment_config(args)

    require_files(paths["annotation_json"], paths["checkpoint"])
    require_directories(paths["image_dir"])
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["prediction_dir"].mkdir(parents=True, exist_ok=True)

    with open(paths["annotation_json"], "r", encoding="utf-8") as handle:
        annotation_data = json.load(handle)

    categories = sorted(annotation_data["categories"], key=lambda item: item["id"])
    category_ids = [int(item["id"]) for item in categories]
    category_names = {int(item["id"]): item["name"] for item in categories}
    if len(category_ids) != paths["num_classes"]:
        raise RuntimeError(
            f"Expected {paths['num_classes']} categories, found {len(category_ids)} "
            f"in {paths['annotation_json']}"
        )

    print("=" * 80)
    print(f"{args.experiment.upper()} COCO {args.split.upper()} EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {paths['checkpoint']}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Device: {args.device}")

    predictor = build_predictor(
        paths["checkpoint"],
        paths["num_classes"],
        args.score_threshold,
        args.device,
    )
    coco_gt = COCO(str(paths["annotation_json"]))
    image_ids = coco_gt.getImgIds()
    predictions = []

    for image_index, image_id in enumerate(image_ids, start=1):
        image_info = coco_gt.loadImgs([image_id])[0]
        image_path = paths["image_dir"] / image_info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read evaluation image: {image_path}")

        instances = predictor(image)["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        for box, score, class_index in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": category_ids[int(class_index)],
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                    ],
                    "score": float(score),
                }
            )

        if image_index % 25 == 0:
            print(f"Processed {image_index}/{len(image_ids)} images")

    prediction_json = paths["prediction_dir"] / "predictions.json"
    with open(prediction_json, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)

    if not predictions:
        raise RuntimeError("The detector produced no predictions.")

    coco_dt = coco_gt.loadRes(str(prediction_json))
    overall_eval = run_coco_eval(coco_gt, coco_dt, image_ids)
    overall = stats_dict(overall_eval.stats)
    per_class = {}

    for category_id in category_ids:
        class_eval = run_coco_eval(
            coco_gt,
            coco_dt,
            image_ids,
            cat_ids=[category_id],
            quiet=True,
        )
        per_class[category_names[category_id]] = stats_dict(class_eval.stats)

    summary = {
        "experiment": args.experiment,
        "split": args.split,
        "annotation_json": str(paths["annotation_json"]),
        "checkpoint": str(paths["checkpoint"]),
        "score_threshold": args.score_threshold,
        "num_images": len(image_ids),
        "num_predictions": len(predictions),
        **overall,
        "per_class": per_class,
    }
    summary_path = paths["output_dir"] / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    csv_path = paths["output_dir"] / "per_class_ap.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["class", *next(iter(per_class.values())).keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, metrics in per_class.items():
            writer.writerow({"class": class_name, **metrics})

    print(f"Saved summary: {summary_path}")
    print(f"Saved per-class metrics: {csv_path}")
    print(f"Saved full predictions: {prediction_json}")


if __name__ == "__main__":
    main()
