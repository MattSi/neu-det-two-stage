from pathlib import Path
import os

import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path(r"data/processed/stage1")

TRAIN_JSON = DATA_ROOT / "annotations" / "train.json"
VAL_JSON = DATA_ROOT / "annotations" / "val.json"
TEST_JSON = DATA_ROOT / "annotations" / "test.json"

TRAIN_IMAGES = DATA_ROOT / "train" / "images"
VAL_IMAGES = DATA_ROOT / "val" / "images"
TEST_IMAGES = DATA_ROOT / "test" / "images"

OUTPUT_DIR = Path(
    r"runs/training/stage1_faster_rcnn_r50_fpn_dedup_run1"
)


# ============================================================
# Dataset names
# ============================================================

TRAIN_DATASET = "neu_stage1_train"
VAL_DATASET = "neu_stage1_val"
TEST_DATASET = "neu_stage1_test"


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 1

# Initial experiment: keep this moderate.
MAX_ITER = 3000

BASE_LR = 0.00025

IMS_PER_BATCH = 8

WARMUP_ITERS = 200

# ROI heads sample this many proposals per image.
ROI_BATCH_SIZE_PER_IMAGE = 256

# Run validation every N iterations.
EVAL_PERIOD = 250

# Number of data loader workers.
NUM_WORKERS = 4

# Detection confidence threshold is mainly for inference.
SCORE_THRESH_TEST = 0.5

RANDOM_SEED = 42

# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# Dataset registration
# ============================================================

def register_dataset(name, json_path, image_dir):
    """
    Register one COCO-format dataset.
    """

    # Helpful when running the script multiple times in the same interpreter.
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)

    if name in MetadataCatalog.list():
        MetadataCatalog.remove(name)

    register_coco_instances(
        name,
        {},
        str(json_path),
        str(image_dir),
    )

    MetadataCatalog.get(name).set(
        thing_classes=["defect"]
    )


def register_all_datasets():

    register_dataset(
        TRAIN_DATASET,
        TRAIN_JSON,
        TRAIN_IMAGES,
    )

    register_dataset(
        VAL_DATASET,
        VAL_JSON,
        VAL_IMAGES,
    )

    register_dataset(
        TEST_DATASET,
        TEST_JSON,
        TEST_IMAGES,
    )


# ============================================================
# Trainer
# ============================================================

class Stage1Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(
        cls,
        cfg,
        dataset_name,
        output_folder=None,
    ):
        """
        COCO evaluation gives bbox AP, AP50, AP75, etc.
        """

        if output_folder is None:
            output_folder = os.path.join(
                cfg.OUTPUT_DIR,
                "coco_eval",
                dataset_name,
            )

        os.makedirs(
            output_folder,
            exist_ok=True,
        )

        return COCOEvaluator(
            dataset_name,
            output_dir=output_folder,
        )


# ============================================================
# Build Detectron2 config
# ============================================================

def build_cfg():

    cfg = get_cfg()

    # --------------------------------------------------------
    # Faster R-CNN:
    # Stage 1 only needs bounding-box localization.
    # --------------------------------------------------------

    config_name = (
        "COCO-Detection/"
        "faster_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.merge_from_file(
        model_zoo.get_config_file(config_name)
    )

    # COCO pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        config_name
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    cfg.DATASETS.TRAIN = (
        TRAIN_DATASET,
    )

    cfg.DATASETS.TEST = (
        VAL_DATASET,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        ROI_BATCH_SIZE_PER_IMAGE
    )

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        SCORE_THRESH_TEST
    )

    # --------------------------------------------------------
    # Solver
    # --------------------------------------------------------

    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH

    cfg.SOLVER.BASE_LR = BASE_LR

    cfg.SOLVER.MAX_ITER = MAX_ITER

    cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS

    # LR decay points
    cfg.SOLVER.STEPS = (
        2000,
        2600,
    )

    cfg.SOLVER.GAMMA = 0.1

    # Save checkpoint every 250 iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = 250

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    cfg.OUTPUT_DIR = str(OUTPUT_DIR)

    return cfg


# ============================================================
# Dataset sanity check
# ============================================================

def check_dataset():

    print("=" * 70)
    print("DATASET CHECK")
    print("=" * 70)

    for name in [
        TRAIN_DATASET,
        VAL_DATASET,
        TEST_DATASET,
    ]:

        dataset = DatasetCatalog.get(name)

        num_images = len(dataset)

        num_boxes = sum(
            len(record.get("annotations", []))
            for record in dataset
        )

        print(
            f"{name:20s}: "
            f"{num_images:4d} images, "
            f"{num_boxes:4d} boxes"
        )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print("\nPyTorch version:")
    print(torch.__version__)

    print("\nCUDA available:")
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print("\nGPU:")
        print(torch.cuda.get_device_name(0))

    print()

    register_all_datasets()

    check_dataset()

    cfg = build_cfg()

    os.makedirs(
        cfg.OUTPUT_DIR,
        exist_ok=True,
    )

    print("\nOutput directory:")
    print(cfg.OUTPUT_DIR)

    print("\nStarting Stage 1 training...\n")

    trainer = Stage1Trainer(cfg)

    trainer.resume_or_load(
        resume=False
    )

    trainer.train()


if __name__ == "__main__":
    main()