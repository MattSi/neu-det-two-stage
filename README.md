# Two-Stage Surface Defect Detection on NEU-DET

A two-stage pipeline for steel surface defect detection on the [NEU-DET](#dataset) dataset:

1. **Stage 1 — localization.** A Faster R-CNN (ResNet-50 FPN, Detectron2) detects class-agnostic "defect" regions.
2. **Stage 2 — classification.** Each detected region is cropped with 10% padding and classified into one of six defect types by a ResNet-50.

The pipeline is compared against a single-stage baseline: the same Faster R-CNN trained directly on all six classes.

| | |
|---|---|
| ![Stage 1 localization examples](results/dedup_run1/test/figures/figure_6_1_stage1_localization_examples.png) | ![Stage 2 confusion matrix](results/dedup_run1/test/figures/figure_6_2_stage2_test_confusion_matrix.png) |

## Results

Test split (271 images, 619 ground-truth boxes), after removing byte-identical duplicate images before splitting. All numbers come from [`results/dedup_run1/test/`](results/dedup_run1/test/).

| Model | AP | AP50 | AP75 |
|---|---:|---:|---:|
| Baseline: Faster R-CNN, 6 classes | 0.391 | 0.754 | 0.353 |
| **Two-stage: Faster R-CNN + ResNet-50** | **0.401** | **0.762** | **0.368** |

COCO AP uses a score threshold of 0.05 for both models.

| Component | Metric | Value |
|---|---|---:|
| Stage 1 (class-agnostic) | AP / AP50 | 0.414 / 0.789 |
| Stage 2 on ground-truth crops | Accuracy | 0.995 |
| End to end, threshold 0.50, IoU 0.50 | Stage 1 recall | 0.727 |
| | Accuracy on matched boxes | 0.993 |
| | End-to-end accuracy | 0.722 |

Per-class AP on the test split:

| Class | Baseline | Two-stage |
|---|---:|---:|
| crazing | 0.142 | 0.164 |
| inclusion | 0.412 | 0.394 |
| patches | 0.591 | 0.596 |
| pitted_surface | 0.451 | 0.496 |
| rolled-in_scale | 0.240 | 0.214 |
| scratches | 0.512 | 0.543 |

The classifier is close to perfect when given good boxes, so most of the remaining error comes from Stage 1 localization (recall 0.727 at threshold 0.50).

## Setup

Python 3.10+ with a CUDA GPU is recommended for training. Evaluation also runs on CPU with `--device cpu`.

1. Install PyTorch and torchvision for your platform from [pytorch.org](https://pytorch.org/get-started/locally/).
2. Install Detectron2 by following its [installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). On Windows it has to be built from source with the Visual C++ build tools.
3. Install the remaining packages:

```bash
pip install -r requirements.txt
```

## Dataset

This repository does not include the NEU-DET images or annotations. Download the NEU surface defect database (NEU-DET, 1,800 grayscale 200×200 images, 6 classes, Pascal VOC XML boxes) from its original source and follow its terms of use. Place it as:

```
data/raw/neu_det/
├── images/        # or IMAGES/
└── annotations/   # or ANNOTATIONS/, one .xml per image
```

Cite the dataset if you use it:

- K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, 2013.
- Y. He, K. Song, Q. Meng, and Y. Yan, "An end-to-end steel surface defect detection approach via fusing multiple hierarchical features," *IEEE Transactions on Instrumentation and Measurement*, 2020.

## Reproducing the results

Run every command from the repository root.

**1. Prepare the data.** This removes byte-identical duplicates (SHA-256), makes a stratified 70/15/15 train/val/test split with seed 42, and writes the Stage 1 COCO data and the Stage 2 ROI crops to `data/processed/`. The split is deterministic, so no split files are needed.

```bash
python scripts/data/prepare_neu_det.py
python scripts/data/prepare_baseline_6class.py
python scripts/data/check_coco.py   # optional: writes preview images with boxes
```

**2. Train.** Checkpoints and logs go to `runs/training/` (ignored by Git).

```bash
python scripts/train/train_stage1.py
python scripts/train/train_stage2.py
python scripts/train/train_baseline_6class.py
```

**3. Select thresholds on validation data only.**

```bash
python scripts/evaluate/validate_threshold_sweep.py
```

**4. Evaluate on the test split.** See [`scripts/evaluate/README.md`](scripts/evaluate/README.md) for details.

```bash
python scripts/evaluate/evaluate_detector_coco.py --experiment baseline --split test --score-threshold 0.05
python scripts/evaluate/evaluate_detector_coco.py --experiment stage1 --split test --score-threshold 0.05
python scripts/evaluate/evaluate_stage2_classifier.py --split test
python scripts/evaluate/evaluate_two_stage_coco.py --split test --stage1-score-threshold 0.05 --roi-padding-ratio 0.10
python scripts/evaluate/validate_end_to_end.py --split test --stage1-score-threshold 0.50 --match-iou-threshold 0.50 --roi-padding-ratio 0.10
```

**5. Make the figures.**

```bash
python scripts/figures/make_fig_6_1_stage1_examples.py
python scripts/figures/make_fig_6_2_confusion_matrix.py
python scripts/figures/make_fig_6_3_threshold_analysis.py
python scripts/figures/make_fig_6_4_per_class_ap.py
```

Trained weights are not included. Training with the fixed seeds should give results close to the ones above, but GPU nondeterminism can cause small differences.

## Repository layout

```
src/neu_det_two_stage/   # shared project paths and helpers
scripts/
├── data/                # dataset preparation and checks
├── train/               # Stage 1, Stage 2, and baseline training
├── evaluate/            # COCO evaluation, end-to-end validation, threshold sweep
└── figures/             # result figures
results/
├── dedup_run1/          # final results on the deduplicated split (reported above)
│   ├── validation/      # checkpoint selection, threshold sweep, validation metrics
│   └── test/            # test metrics and figures
├── validation/, test/   # earlier run, before duplicate images were removed
docs/figure_specs/       # text diagrams of the method
```

Metrics are stored as `summary.json` plus CSV files (per-class AP, confusion matrices). Paths inside them are relative to the repository root.

## License

The code is released under the [MIT License](LICENSE). The NEU-DET dataset is not covered by this license; the example images in `results/*/figures/` are derived from NEU-DET and remain subject to the dataset's terms.
