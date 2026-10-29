"""Evaluate the frozen Stage 2 ResNet50 checkpoint without retraining."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


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


DEFAULT_MODEL = preferred_checkpoint(
    Path("stage2/model_best.pth"),
    Path("stage2_resnet50/model_best.pth"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Stage 2 classifier on ground-truth ROIs from "
            "the validation or test split."
        )
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint["image_size"])

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, class_names, image_size, transform, checkpoint


@torch.no_grad()
def evaluate(model, loader, device):
    labels = []
    predictions = []

    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(
            "cuda",
            enabled=device == "cuda",
        ):
            logits = model(images)
        batch_predictions = torch.argmax(logits, dim=1)
        labels.extend(batch_labels.numpy().tolist())
        predictions.extend(batch_predictions.cpu().numpy().tolist())

    return np.asarray(labels), np.asarray(predictions)


def main():
    args = parse_args()
    checkpoint_path = args.model.resolve()
    roi_dir = PROCESSED_DATA_DIR / "stage2_roi" / args.split
    result_split = result_split_name(args.split)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else RESULTS_DIR / result_split / "stage2_classifier"
    )

    require_files(checkpoint_path)
    require_directories(roi_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, image_size, transform, checkpoint = load_model(
        checkpoint_path,
        args.device,
    )
    dataset = datasets.ImageFolder(roi_dir, transform=transform)
    if dataset.classes != class_names:
        raise RuntimeError(
            "Dataset class order does not match the checkpoint: "
            f"dataset={dataset.classes}, checkpoint={class_names}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    print("=" * 80)
    print(f"STAGE 2 {args.split.upper()} CLASSIFICATION EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"ROI samples: {len(dataset)}")
    print(f"Device: {args.device}")

    labels, predictions = evaluate(model, loader, args.device)
    class_indices = list(range(len(class_names)))
    report = classification_report(
        labels,
        predictions,
        labels=class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=class_indices)
    accuracy = float(accuracy_score(labels, predictions))

    summary = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_accuracy": checkpoint.get("val_accuracy"),
        "data_dir": str(roi_dir),
        "image_size": image_size,
        "num_samples": len(dataset),
        "class_names": class_names,
        "accuracy": accuracy,
        "classification_report": report,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    matrix_path = output_dir / "confusion_matrix.csv"
    np.savetxt(matrix_path, matrix, delimiter=",", fmt="%d")

    print(f"Accuracy: {accuracy:.6f}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved confusion matrix: {matrix_path}")


if __name__ == "__main__":
    main()

