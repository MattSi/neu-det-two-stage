from pathlib import Path
import json
import random

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "processed" / "stage1"

JSON_PATH = ROOT / "annotations" / "train.json"
IMAGE_DIR = ROOT / "train" / "images"

OUTPUT_DIR = ROOT / "preview"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 10
SEED = 42


with open(JSON_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = {
    img["id"]: img
    for img in coco["images"]
}

annotations_by_image = {}

for ann in coco["annotations"]:
    annotations_by_image.setdefault(
        ann["image_id"], []
    ).append(ann)


random.seed(SEED)

selected_ids = random.sample(
    list(images.keys()),
    min(NUM_IMAGES, len(images))
)


for image_id in selected_ids:

    image_info = images[image_id]

    image_path = IMAGE_DIR / image_info["file_name"]

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot read: {image_path}")
        continue

    annotations = annotations_by_image.get(image_id, [])

    for ann in annotations:

        x, y, w, h = ann["bbox"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            1,
        )

        cv2.putText(
            image,
            "defect",
            (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    output_path = OUTPUT_DIR / image_info["file_name"]

    cv2.imwrite(
        str(output_path),
        image
    )

    print(
        f"{image_info['file_name']}: "
        f"{len(annotations)} boxes"
    )


print(f"\nPreview saved to:\n{OUTPUT_DIR}")