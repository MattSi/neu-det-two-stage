import argparse
import hashlib
from pathlib import Path
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neu_det_two_stage.paths import (  # noqa: E402
    DATA_DIR,
    PROCESSED_DATA_DIR,
    require_directories,
)


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = DATA_DIR / "raw" / "neu_det"

IMAGE_DIR = DATASET_ROOT / "images"
ANNOTATION_DIR = DATASET_ROOT / "annotations"

OUTPUT_ROOT = PROCESSED_DATA_DIR

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Add a little padding around each ground-truth ROI.
# 0.10 means 10% extra size on each side.
ROI_PADDING_RATIO = 0.10

# Stage 1 only detects generic defect regions.
STAGE1_CATEGORY_NAME = "defect"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deduplicated Stage 1 COCO data and Stage 2 ROI crops "
            "from the raw NEU-DET dataset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Raw NEU-DET root containing images and annotations.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Destination for prepared Stage 1 and Stage 2 datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used for the stratified split.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Report exact duplicate images without rebuilding prepared data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output root when rebuilding prepared data.",
    )
    return parser.parse_args()


def first_existing_subdirectory(root, names):
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / names[0]


def configure_from_args(args):
    global DATASET_ROOT
    global IMAGE_DIR
    global ANNOTATION_DIR
    global OUTPUT_ROOT
    global RANDOM_SEED

    DATASET_ROOT = args.dataset_root.resolve()
    IMAGE_DIR = first_existing_subdirectory(DATASET_ROOT, ("images", "IMAGES"))
    ANNOTATION_DIR = first_existing_subdirectory(
        DATASET_ROOT,
        ("annotations", "ANNOTATIONS"),
    )
    OUTPUT_ROOT = args.output_root.resolve()
    RANDOM_SEED = args.seed

    require_directories(IMAGE_DIR, ANNOTATION_DIR)


# ============================================================
# Helper functions
# ============================================================
def resolve_image_path(filename):
    """
    Resolve the actual image path.

    Some NEU-DET XML annotations may omit the image extension.
    """

    image_path = IMAGE_DIR / filename

    if image_path.exists():
        return image_path

    filename_path = Path(filename)

    if not filename_path.suffix:
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = IMAGE_DIR / f"{filename}{ext}"

            if candidate.exists():
                return candidate

    return None


def parse_voc_xml(xml_path):
    """
    Read one Pascal VOC XML file.

    Returns:
        {
            "filename": str,
            "width": int,
            "height": int,
            "objects": [
                {
                    "class_name": str,
                    "xmin": int,
                    "ymin": int,
                    "xmax": int,
                    "ymax": int,
                    "difficult": int,
                },
                ...
            ]
        }
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")

    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name")

        difficult_text = obj.findtext("difficult", default="0")
        difficult = int(difficult_text)

        bbox = obj.find("bndbox")

        xmin = int(float(bbox.findtext("xmin")))
        ymin = int(float(bbox.findtext("ymin")))
        xmax = int(float(bbox.findtext("xmax")))
        ymax = int(float(bbox.findtext("ymax")))

        objects.append(
            {
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "difficult": difficult,
            }
        )

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def expand_box(xmin, ymin, xmax, ymax, img_width, img_height, ratio):
    """
    Expand a bounding box by a percentage of its width/height.
    """

    box_width = xmax - xmin
    box_height = ymax - ymin

    pad_x = int(box_width * ratio)
    pad_y = int(box_height * ratio)

    new_xmin = max(0, xmin - pad_x)
    new_ymin = max(0, ymin - pad_y)

    new_xmax = min(img_width, xmax + pad_x)
    new_ymax = min(img_height, ymax + pad_y)

    return new_xmin, new_ymin, new_xmax, new_ymax


def make_clean_output_dir(overwrite=False):
    """
    Delete previous prepared folder and recreate it.
    """

    if OUTPUT_ROOT.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {OUTPUT_ROOT}\n"
                "Pass --overwrite to replace this generated dataset."
            )

        protected_paths = {
            PROJECT_ROOT.resolve(),
            DATASET_ROOT.resolve(),
            DATA_DIR.resolve(),
        }
        if OUTPUT_ROOT.resolve() in protected_paths:
            raise ValueError(f"Refusing to replace protected path: {OUTPUT_ROOT}")

        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_signature(record):
    return sorted(
        (
            obj["class_name"],
            obj["xmin"],
            obj["ymin"],
            obj["xmax"],
            obj["ymax"],
            obj["difficult"],
        )
        for obj in record["objects"]
    )


def deduplicate_records(records):
    """Remove byte-identical images before splitting the dataset.

    The lexicographically first filename is retained so the decision is stable
    across machines and runs. Raw images and XML annotations are never changed.
    """

    records_by_hash = defaultdict(list)
    for record in records:
        image_hash = sha256_file(record["image_path"])
        records_by_hash[image_hash].append(record)

    unique_records = []
    duplicate_groups = []

    for image_hash, group in sorted(records_by_hash.items()):
        ordered_group = sorted(group, key=lambda record: record["filename"])
        kept = ordered_group[0]
        unique_records.append(kept)

        if len(ordered_group) == 1:
            continue

        kept_class = get_image_class(kept)
        for duplicate in ordered_group[1:]:
            duplicate_class = get_image_class(duplicate)
            if duplicate_class != kept_class:
                raise RuntimeError(
                    "Byte-identical images have conflicting primary classes: "
                    f"{kept['filename']}={kept_class}, "
                    f"{duplicate['filename']}={duplicate_class}"
                )

        duplicate_groups.append(
            {
                "sha256": image_hash,
                "kept": kept["filename"],
                "removed": [record["filename"] for record in ordered_group[1:]],
                "annotations_identical": all(
                    annotation_signature(record) == annotation_signature(kept)
                    for record in ordered_group[1:]
                ),
                "annotation_files": [
                    record["xml_path"].name for record in ordered_group
                ],
            }
        )

    unique_records.sort(key=lambda record: record["filename"])
    return unique_records, duplicate_groups


def print_duplicate_audit(original_count, unique_count, duplicate_groups):
    print("\nExact-image duplicate audit:")
    print(f"  Original images: {original_count}")
    print(f"  Unique images:   {unique_count}")
    print(f"  Removed images:  {original_count - unique_count}")

    if not duplicate_groups:
        print("  Duplicate groups: none")
        return

    for group in duplicate_groups:
        print(f"  SHA256: {group['sha256']}")
        print(f"    kept:    {group['kept']}")
        print(f"    removed: {', '.join(group['removed'])}")
        print(
            "    annotations identical: "
            f"{group['annotations_identical']}"
        )


def write_deduplication_report(original_count, records, duplicate_groups):
    report_path = OUTPUT_ROOT / "deduplication_report.json"
    report = {
        "hash_algorithm": "sha256",
        "selection_rule": "keep lexicographically first filename",
        "original_image_count": original_count,
        "unique_image_count": len(records),
        "removed_image_count": original_count - len(records),
        "duplicate_groups": duplicate_groups,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report_path


# Use stratified split
def split_items(items):
    """
    Random train/val/test split.
    """

    random.seed(RANDOM_SEED)

    items = list(items)
    random.shuffle(items)

    total = len(items)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]

    return {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

def get_image_class(record):
    """
    Get the primary image-level class from the filename.

    NEU-DET filenames are organized by their primary defect category,
    for example:
        crazing_104.jpg
        inclusion_61.jpg
        pitted_surface_91.jpg
        rolled-in-scale_58.jpg
        scratches_299.jpg

    An image may contain bounding boxes from more than one class,
    so XML object labels should NOT be used as a single image-level label.
    """

    filename = record["filename"]
    stem = Path(filename).stem

    known_classes = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]

    for class_name in known_classes:
        if stem.startswith(class_name + "_"):
            return class_name

    raise ValueError(
        f"Cannot determine primary class from filename: {filename}"
    )


def split_items_stratified(items):
    """
    Stratified train / val / test split by image-level defect class.

    Each defect class is split separately using:
        70% train
        15% val
        15% test
    """

    rng = random.Random(RANDOM_SEED)

    grouped = defaultdict(list)

    # Group images by defect class
    for record in items:
        class_name = get_image_class(record)
        grouped[class_name].append(record)

    train_items = []
    val_items = []
    test_items = []

    print("\nStratified image split by class:")

    for class_name in sorted(grouped.keys()):

        class_records = grouped[class_name]
        rng.shuffle(class_records)

        total = len(class_records)

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        # The remaining samples go to test.
        test_count = total - train_count - val_count

        train_subset = class_records[:train_count]

        val_subset = class_records[
            train_count:
            train_count + val_count
        ]

        test_subset = class_records[
            train_count + val_count:
        ]

        train_items.extend(train_subset)
        val_items.extend(val_subset)
        test_items.extend(test_subset)

        print(
            f"  {class_name:25s}: "
            f"total={total:3d}, "
            f"train={len(train_subset):3d}, "
            f"val={len(val_subset):3d}, "
            f"test={len(test_subset):3d}"
        )

    # Shuffle the final splits so images are not ordered by class
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    return {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }


# ============================================================
# Dataset loading
# ============================================================

def load_dataset():
    xml_files = sorted(ANNOTATION_DIR.glob("*.xml"))

    if not xml_files:
        raise RuntimeError(
            f"No XML files found in: {ANNOTATION_DIR}"
        )

    records = []

    for xml_path in xml_files:
        parsed = parse_voc_xml(xml_path)

        image_path = resolve_image_path(parsed["filename"])

        if image_path is None:
            print(
                f"[WARNING] Missing image for XML: "
                f"{xml_path.name}, filename={parsed['filename']}"
            )
            continue

        records.append(
            {
                "xml_path": xml_path,
                "image_path": image_path,
                **parsed,
                "filename": image_path.name,
            }
        )

    return records


# ============================================================
# Stage 1 COCO generation
# ============================================================

def build_coco_for_split(split_name, records):
    """
    Create COCO JSON for Stage 1.

    Important:
    All original NEU defect classes are converted to one generic class:
        defect
    """

    images = []
    annotations = []

    image_id = 1
    annotation_id = 1

    output_image_dir = OUTPUT_ROOT / "stage1" / split_name / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        src_image = record["image_path"]

        dst_image = output_image_dir / src_image.name
        shutil.copy2(src_image, dst_image)

        images.append(
            {
                "id": image_id,
                "file_name": src_image.name,
                "width": record["width"],
                "height": record["height"],
            }
        )

        for obj in record["objects"]:

            xmin = obj["xmin"]
            ymin = obj["ymin"]
            xmax = obj["xmax"]
            ymax = obj["ymax"]

            width = xmax - xmin
            height = ymax - ymin

            if width <= 0 or height <= 0:
                print(
                    f"[WARNING] Invalid box in {record['xml_path'].name}: "
                    f"{xmin}, {ymin}, {xmax}, {ymax}"
                )
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        xmin,
                        ymin,
                        width,
                        height,
                    ],
                    "area": width * height,
                    "iscrowd": 0,
                    "original_class": obj["class_name"],
                    "difficult": obj["difficult"],
                }
            )

            annotation_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": STAGE1_CATEGORY_NAME,
            }
        ],
    }

    annotation_output_dir = OUTPUT_ROOT / "stage1" / "annotations"
    annotation_output_dir.mkdir(parents=True, exist_ok=True)

    json_path = annotation_output_dir / f"{split_name}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    return {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "json_path": json_path,
    }


# ============================================================
# Stage 2 ROI generation
# ============================================================

def build_roi_dataset(split_name, records):
    """
    Crop ROIs from ground-truth boxes.

    Output example:

    prepared/
        stage2_roi/
            train/
                crazing/
                inclusion/
                patches/
                ...
    """

    roi_root = OUTPUT_ROOT / "stage2_roi" / split_name
    roi_root.mkdir(parents=True, exist_ok=True)

    class_counter = Counter()

    for record in records:

        image = Image.open(record["image_path"])

        for obj_index, obj in enumerate(record["objects"]):

            class_name = obj["class_name"]

            xmin = obj["xmin"]
            ymin = obj["ymin"]
            xmax = obj["xmax"]
            ymax = obj["ymax"]

            xmin, ymin, xmax, ymax = expand_box(
                xmin,
                ymin,
                xmax,
                ymax,
                record["width"],
                record["height"],
                ROI_PADDING_RATIO,
            )

            if xmax <= xmin or ymax <= ymin:
                continue

            roi = image.crop((xmin, ymin, xmax, ymax))

            class_dir = roi_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            source_stem = Path(record["filename"]).stem

            roi_filename = (
                f"{source_stem}"
                f"_obj{obj_index:02d}"
                f"_{class_name}.jpg"
            )

            roi_path = class_dir / roi_filename

            # Convert to RGB so torchvision / ResNet input is consistent.
            roi.convert("RGB").save(
                roi_path,
                quality=95,
            )

            class_counter[class_name] += 1

    return class_counter


# ============================================================
# Statistics
# ============================================================

def print_dataset_statistics(all_records, splits, stage1_stats, roi_stats):

    print("\n")
    print("=" * 70)
    print("NEU-DET PREPARATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal images: {len(all_records)}")

    all_objects = []

    for record in all_records:
        all_objects.extend(record["objects"])

    print(f"Total bounding boxes: {len(all_objects)}")

    class_counts = Counter(
        obj["class_name"]
        for obj in all_objects
    )

    print("\nOriginal defect class counts:")

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:25s}: {count}")

    print("\nImage split:")

    for split_name in ["train", "val", "test"]:
        print(
            f"  {split_name:10s}: "
            f"{len(splits[split_name])} images"
        )

    print("\nStage 1 COCO:")

    for split_name in ["train", "val", "test"]:
        stats = stage1_stats[split_name]

        print(
            f"  {split_name:10s}: "
            f"{stats['image_count']} images, "
            f"{stats['annotation_count']} defect boxes"
        )

    print("\nStage 2 ROI dataset:")

    for split_name in ["train", "val", "test"]:

        print(f"\n  [{split_name}]")

        total_rois = 0

        for class_name, count in sorted(
            roi_stats[split_name].items()
        ):
            print(
                f"    {class_name:25s}: {count}"
            )
            total_rois += count

        print(
            f"    {'TOTAL':25s}: {total_rois}"
        )

    print("\nOutput directory:")
    print(OUTPUT_ROOT)

    print("\nDone.")


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()
    configure_from_args(args)

    print("Loading NEU-DET...")

    records = load_dataset()

    print(f"Found {len(records)} valid images.")

    original_count = len(records)
    records, duplicate_groups = deduplicate_records(records)
    print_duplicate_audit(original_count, len(records), duplicate_groups)

    if args.audit_only:
        return

    make_clean_output_dir(overwrite=args.overwrite)
    report_path = write_deduplication_report(
        original_count,
        records,
        duplicate_groups,
    )
    print(f"Deduplication report: {report_path}")

    splits = split_items_stratified(records)

    stage1_stats = {}
    roi_stats = {}

    for split_name in ["train", "val", "test"]:

        print(f"\nPreparing {split_name}...")

        stage1_stats[split_name] = build_coco_for_split(
            split_name,
            splits[split_name],
        )

        roi_stats[split_name] = build_roi_dataset(
            split_name,
            splits[split_name],
        )

    print_dataset_statistics(
        records,
        splits,
        stage1_stats,
        roi_stats,
    )


if __name__ == "__main__":
    main()
