from pathlib import Path
import json
import time
import copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(
    r"data/processed/stage2_roi"
)

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

OUTPUT_DIR = Path(
    r"runs/training/stage2_resnet50_dedup_run1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_SIZE = 224

BATCH_SIZE = 64

NUM_EPOCHS = 30

BASE_LR = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

USE_WEIGHTED_SAMPLER = True

RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=8
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Dataset
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform,
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform,
)


CLASS_NAMES = train_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

print("\nClasses:")
for idx, name in enumerate(CLASS_NAMES):
    print(f"  {idx}: {name}")

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Val:  ", len(val_dataset))
print("Test: ", len(test_dataset))


# ============================================================
# Verify class mapping
# ============================================================

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise RuntimeError(
        "Train and validation class mappings are different."
    )

if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise RuntimeError(
        "Train and test class mappings are different."
    )


# ============================================================
# Weighted sampler
# ============================================================

def build_weighted_sampler(dataset):

    targets = dataset.targets

    class_counts = Counter(targets)

    print("\nTraining class counts:")

    for class_idx in range(NUM_CLASSES):

        print(
            f"  {CLASS_NAMES[class_idx]:25s}: "
            f"{class_counts[class_idx]}"
        )

    class_weights = {
        class_idx: 1.0 / count
        for class_idx, count in class_counts.items()
    }

    sample_weights = [
        class_weights[target]
        for target in targets
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


if USE_WEIGHTED_SAMPLER:

    train_sampler = build_weighted_sampler(
        train_dataset
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

else:

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
)


# ============================================================
# Model
# ============================================================

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)

in_features = model.fc.in_features

model.fc = nn.Linear(
    in_features,
    NUM_CLASSES,
)

model = model.to(DEVICE)


# ============================================================
# Loss / optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=BASE_LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
)


# ============================================================
# AMP
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available(),
)


# ============================================================
# Training functions
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            "cuda",
            enabled=torch.cuda.is_available(),
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.amp.autocast(
            "cuda",
            enabled=torch.cuda.is_available(),
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

        predictions = logits.argmax(
            dim=1
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return (
        epoch_loss,
        epoch_acc,
        np.array(all_labels),
        np.array(all_predictions),
    )


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    epoch,
    val_accuracy,
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "val_accuracy": val_accuracy,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# Main training
# ============================================================

def main():

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_accuracy = -1.0
    best_epoch = -1

    best_model_state = None

    start_time = time.time()

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
        )

        (
            val_loss,
            val_acc,
            val_labels,
            val_predictions,
        ) = evaluate(
            model,
            val_loader,
        )

        scheduler.step(
            val_acc
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_acc
        )

        history["learning_rate"].append(
            current_lr
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | "
            f"time={epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
                OUTPUT_DIR / "model_best.pth",
                model,
                epoch,
                val_acc,
            )

            print(
                f"  -> New best validation accuracy: "
                f"{val_acc:.4f}"
            )

    total_time = time.time() - start_time

    # --------------------------------------------------------
    # Save final
    # --------------------------------------------------------

    save_checkpoint(
        OUTPUT_DIR / "model_final.pth",
        model,
        NUM_EPOCHS,
        history["val_accuracy"][-1],
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["best_epoch"] = best_epoch
    history["best_val_accuracy"] = best_val_accuracy
    history["runtime_seconds"] = total_time

    with open(
        OUTPUT_DIR / "history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Training runtime: "
        f"{total_time:.1f} sec"
    )

    # --------------------------------------------------------
    # Evaluate best model on validation set
    # --------------------------------------------------------

    model.load_state_dict(
        best_model_state
    )

    (
        best_val_loss,
        best_val_acc,
        labels,
        predictions,
    ) = evaluate(
        model,
        val_loader,
    )

    print("\nValidation classification report:\n")

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print(report)

    matrix = confusion_matrix(
        labels,
        predictions,
    )

    print("\nValidation confusion matrix:\n")
    print(matrix)

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_dict = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    with open(
        OUTPUT_DIR / "validation_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report_dict,
            f,
            indent=2,
        )

    np.savetxt(
        OUTPUT_DIR / "validation_confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    print("\nOutputs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()