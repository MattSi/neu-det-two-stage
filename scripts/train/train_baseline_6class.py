import argparse
from pathlib import Path
import sys

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neu_det_two_stage.paths import (  # noqa: E402
    PROCESSED_DATA_DIR,
    TRAINING_RUNS_DIR,
    require_directories,
    require_files,
)


# ============================================================
# Portable default paths (overridable from the CLI)
# ============================================================

DATA_ROOT = PROCESSED_DATA_DIR / "baseline_6class"

ANNOTATION_ROOT = (
    DATA_ROOT
    / "annotations"
)

OUTPUT_DIR = (
    TRAINING_RUNS_DIR
    / "baseline_6class_faster_rcnn_r50_fpn"
)


# ============================================================
# Dataset names
# ============================================================

TRAIN_NAME = "neu_baseline_6class_train"
VAL_NAME = "neu_baseline_6class_val"
TEST_NAME = "neu_baseline_6class_test"


# ============================================================
# Register datasets
# ============================================================

def register_dataset(name, split):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    if name in MetadataCatalog.list():
        MetadataCatalog.remove(name)

    register_coco_instances(
        name,
        {},
        str(ANNOTATION_ROOT / f"{split}.json"),
        str(DATA_ROOT / split / "images"),
    )


def register_all_datasets():
    register_dataset(TRAIN_NAME, "train")
    register_dataset(VAL_NAME, "val")
    register_dataset(TEST_NAME, "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the direct six-class Faster R-CNN baseline."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Six-class baseline dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Training output directory for logs and checkpoints.",
    )
    return parser.parse_args()


def configure_from_args(args):
    global DATA_ROOT
    global ANNOTATION_ROOT
    global OUTPUT_DIR

    DATA_ROOT = args.data_root.resolve()
    ANNOTATION_ROOT = DATA_ROOT / "annotations"
    OUTPUT_DIR = args.output_dir.resolve()

    require_directories(
        ANNOTATION_ROOT,
        *(
            DATA_ROOT / split / "images"
            for split in ("train", "val", "test")
        ),
    )
    require_files(
        *(
            ANNOTATION_ROOT / f"{split}.json"
            for split in ("train", "val", "test")
        ),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Trainer
# ============================================================

class BaselineTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(
        cls,
        cfg,
        dataset_name,
        output_folder=None,
    ):

        if output_folder is None:
            output_folder = (
                Path(cfg.OUTPUT_DIR)
                / "coco_eval"
            )

        return COCOEvaluator(
            dataset_name,
            output_dir=str(output_folder),
        )


# ============================================================
# Config
# ============================================================

def build_cfg():

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

    cfg.SEED = 42
    
    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    cfg.DATASETS.TRAIN = (
        TRAIN_NAME,
    )

    cfg.DATASETS.TEST = (
        VAL_NAME,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    cfg.DATALOADER.NUM_WORKERS = 4

    # --------------------------------------------------------
    # Pretrained COCO weights
    # --------------------------------------------------------

    cfg.MODEL.WEIGHTS = (
        model_zoo.get_checkpoint_url(
            config_name
        )
    )

    # --------------------------------------------------------
    # Six defect classes
    # --------------------------------------------------------

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

    # --------------------------------------------------------
    # Training schedule
    # Keep same as Stage 1
    # --------------------------------------------------------

    cfg.SOLVER.IMS_PER_BATCH = 8

    cfg.SOLVER.BASE_LR = 0.00025

    cfg.SOLVER.MAX_ITER = 3000

    cfg.SOLVER.WARMUP_ITERS = 200

    cfg.SOLVER.STEPS = (
        2000,
        2600,
    )

    cfg.SOLVER.GAMMA = 0.1

    cfg.SOLVER.CHECKPOINT_PERIOD = 250

    # --------------------------------------------------------
    # ROI sampling
    # --------------------------------------------------------

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    cfg.TEST.EVAL_PERIOD = 250

    # --------------------------------------------------------
    # Test score threshold
    # Used during evaluator inference.
    # Keep same initial setting as Stage 1.
    # --------------------------------------------------------

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    cfg.OUTPUT_DIR = str(
        OUTPUT_DIR
    )

    return cfg


# ============================================================
# Main
# ============================================================

def main():

    configure_from_args(parse_args())
    register_all_datasets()

    cfg = build_cfg()

    print("=" * 70)
    print("DIRECT 6-CLASS FASTER R-CNN BASELINE")
    print("=" * 70)

    print("\nTrain dataset:")
    print(TRAIN_NAME)

    print("\nValidation dataset:")
    print(VAL_NAME)

    print("\nOutput:")
    print(cfg.OUTPUT_DIR)

    print("\nTraining configuration:")
    print(
        f"NUM_CLASSES       = "
        f"{cfg.MODEL.ROI_HEADS.NUM_CLASSES}"
    )
    print(
        f"IMS_PER_BATCH     = "
        f"{cfg.SOLVER.IMS_PER_BATCH}"
    )
    print(
        f"BASE_LR           = "
        f"{cfg.SOLVER.BASE_LR}"
    )
    print(
        f"MAX_ITER          = "
        f"{cfg.SOLVER.MAX_ITER}"
    )
    print(
        f"WARMUP_ITERS      = "
        f"{cfg.SOLVER.WARMUP_ITERS}"
    )
    print(
        f"STEPS             = "
        f"{cfg.SOLVER.STEPS}"
    )
    print(
        f"EVAL_PERIOD       = "
        f"{cfg.TEST.EVAL_PERIOD}"
    )

    trainer = BaselineTrainer(
        cfg
    )

    trainer.resume_or_load(
        resume=False
    )

    trainer.train()


if __name__ == "__main__":
    main()
