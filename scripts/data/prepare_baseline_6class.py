import argparse
from pathlib import Path
import json
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neu_det_two_stage.paths import (  # noqa: E402
    PROCESSED_DATA_DIR,
    require_directories,
    require_files,
)


STAGE1_ROOT = PROCESSED_DATA_DIR / "stage1"

OUTPUT_ROOT = PROCESSED_DATA_DIR / "baseline_6class"

ANNOTATION_OUT = OUTPUT_ROOT / "annotations"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_ID = {
    name: idx + 1
    for idx, name in enumerate(CLASS_NAMES)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the class-agnostic Stage 1 COCO dataset into the "
            "direct six-class baseline dataset."
        )
    )
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=STAGE1_ROOT,
        help="Stage 1 dataset root containing annotations and split images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Destination root for the six-class baseline dataset.",
    )
    return parser.parse_args()


def configure_from_args(args):
    global STAGE1_ROOT
    global OUTPUT_ROOT
    global ANNOTATION_OUT

    STAGE1_ROOT = args.stage1_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    ANNOTATION_OUT = OUTPUT_ROOT / "annotations"

    require_directories(
        STAGE1_ROOT / "annotations",
        *(STAGE1_ROOT / split / "images" for split in ("train", "val", "test")),
    )
    require_files(
        *(
            STAGE1_ROOT / "annotations" / f"{split}.json"
            for split in ("train", "val", "test")
        ),
    )


def convert_split(split):

    print(f"\nProcessing split: {split}")

    input_json = (
        STAGE1_ROOT
        / "annotations"
        / f"{split}.json"
    )

    input_image_dir = (
        STAGE1_ROOT
        / split
        / "images"
    )

    output_image_dir = (
        OUTPUT_ROOT
        / split
        / "images"
    )

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANNOTATION_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load Stage 1 COCO
    # --------------------------------------------------------

    with open(
        input_json,
        "r",
        encoding="utf-8",
    ) as f:

        coco = json.load(f)

    # --------------------------------------------------------
    # Build new categories
    # --------------------------------------------------------

    categories = [
        {
            "id": CLASS_TO_ID[name],
            "name": name,
            "supercategory": "defect",
        }
        for name in CLASS_NAMES
    ]

    # --------------------------------------------------------
    # Convert annotations back to 6 classes
    # --------------------------------------------------------

    new_annotations = []

    class_counts = {
        name: 0
        for name in CLASS_NAMES
    }

    for ann in coco["annotations"]:

        if "original_class" not in ann:
            raise KeyError(
                "Missing 'original_class' in annotation. "
                "The current Stage 1 COCO JSON must preserve "
                "the original defect class."
            )

        original_class = ann[
            "original_class"
        ]

        if original_class not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown class: {original_class}"
            )

        new_ann = dict(ann)

        new_ann["category_id"] = (
            CLASS_TO_ID[
                original_class
            ]
        )

        # original_class is not required by COCO.
        # Keeping it is harmless and useful for debugging.
        new_annotations.append(
            new_ann
        )

        class_counts[
            original_class
        ] += 1

    # --------------------------------------------------------
    # Create new COCO
    # --------------------------------------------------------

    new_coco = {
        "images": coco["images"],
        "annotations": new_annotations,
        "categories": categories,
    }

    # Preserve optional COCO fields
    for optional_key in [
        "info",
        "licenses",
    ]:
        if optional_key in coco:
            new_coco[
                optional_key
            ] = coco[
                optional_key
            ]

    output_json = (
        ANNOTATION_OUT
        / f"{split}.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            new_coco,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    for image_info in coco[
        "images"
    ]:

        src = (
            input_image_dir
            / image_info["file_name"]
        )

        dst = (
            output_image_dir
            / image_info["file_name"]
        )

        if not src.exists():

            raise FileNotFoundError(
                src
            )

        shutil.copy2(
            src,
            dst,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"Images: {len(coco['images'])}"
    )

    print(
        f"Boxes: {len(new_annotations)}"
    )

    print("Class counts:")

    for name in CLASS_NAMES:
        print(
            f"  {name:20s}: "
            f"{class_counts[name]}"
        )

    print(
        f"Saved JSON: {output_json}"
    )


def main():

    configure_from_args(parse_args())

    for split in [
        "train",
        "val",
        "test",
    ]:
        convert_split(
            split
        )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()
