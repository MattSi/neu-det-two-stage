from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_JSON = (
    ROOT
    / "data"
    / "processed"
    / "stage1"
    / "annotations"
    / "test.json"
)

TEST_IMAGE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "stage1"
    / "test"
    / "images"
)

STAGE1_MODEL = (
    ROOT
    / "runs"
    / "training"
    / "stage1_faster_rcnn_r50_fpn_dedup_run1"
    / "model_0002749.pth"
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
# Settings
# ============================================================

DEVICE = "cuda"   # 
SCORE_THRESH = 0.84   # 
NUM_EXAMPLES = 4
RANDOM_SEED = 42

# 如果你想手动指定图片文件名，可以在这里填写
# 例如:
# SELECTED_FILE_NAMES = ["crazing_001.jpg", "patches_015.jpg"]
SELECTED_FILE_NAMES = [
    "crazing_2.jpg",
    "inclusion_212.jpg",
    "pitted_surface_133.jpg",
    "scratches_299.jpg",
]


# ============================================================
# Build predictor
# ============================================================

def build_stage1_predictor():
    cfg = get_cfg()

    config_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))

    cfg.MODEL.WEIGHTS = str(STAGE1_MODEL)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    cfg.MODEL.DEVICE = DEVICE

    return DefaultPredictor(cfg)


# ============================================================
# Select images
# ============================================================

def choose_test_images(coco):
    img_ids = coco.getImgIds()

    # 只选有 annotation 的图
    img_ids = [
        img_id for img_id in img_ids
        if len(coco.getAnnIds(imgIds=[img_id])) > 0
    ]

    if SELECTED_FILE_NAMES:
        name_to_id = {}
        for img_id in img_ids:
            info = coco.loadImgs([img_id])[0]
            name_to_id[info["file_name"]] = img_id

        selected_ids = []
        for name in SELECTED_FILE_NAMES:
            if name in name_to_id:
                selected_ids.append(name_to_id[name])

        return selected_ids[:NUM_EXAMPLES]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(img_ids)

    return img_ids[:NUM_EXAMPLES]


# ============================================================
# Draw boxes
# ============================================================

def draw_predictions(image_bgr, boxes, scores):
    vis = image_bgr.copy()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"defect {score:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    return vis


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("GENERATE FIGURE 6.1 - STAGE 1 LOCALIZATION EXAMPLES")
    print("=" * 80)

    coco = COCO(str(TEST_JSON))
    predictor = build_stage1_predictor()

    selected_img_ids = choose_test_images(coco)

    rendered_images = []
    titles = []

    for idx, img_id in enumerate(selected_img_ids, start=1):
        info = coco.loadImgs([img_id])[0]
        image_path = TEST_IMAGE_DIR / info["file_name"]

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARNING] Cannot read image: {image_path}")
            continue

        outputs = predictor(image_bgr)
        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        vis_bgr = draw_predictions(image_bgr, boxes, scores)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        rendered_images.append(vis_rgb)
        titles.append(info["file_name"])

        single_out = OUTPUT_DIR / f"fig6_1_example_{idx}.png"
        cv2.imwrite(str(single_out), vis_bgr)
        print(f"Saved: {single_out}")

    if not rendered_images:
        print("No images were rendered.")
        return

    # 生成 2x2 拼图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, img, title in zip(axes, rendered_images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # 如果不足4张，空白补齐
    for i in range(len(rendered_images), 4):
        axes[i].axis("off")

    plt.tight_layout()

    final_fig_path = OUTPUT_DIR / "figure_6_1_stage1_localization_examples.png"
    plt.savefig(final_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved final figure: {final_fig_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()