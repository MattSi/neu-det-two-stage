# Evaluation commands

All evaluators load frozen checkpoints and perform inference only. They do not
instantiate a trainer or update model weights.

Run commands from the repository root. The defaults resolve data from
`data/processed/`, prefer curated checkpoints in `models/`, and fall back to
the preserved checkpoints in `runs/training/`.

## Final test evaluation

Use the same low inference threshold for the direct baseline and the two-stage
COCO comparison.

```powershell
python scripts/evaluate/evaluate_detector_coco.py --experiment baseline --split test --score-threshold 0.05
python scripts/evaluate/evaluate_detector_coco.py --experiment stage1 --split test --score-threshold 0.05
python scripts/evaluate/evaluate_stage2_classifier.py --split test
python scripts/evaluate/evaluate_two_stage_coco.py --split test --stage1-score-threshold 0.05 --roi-padding-ratio 0.10
python scripts/evaluate/validate_end_to_end.py --split test --stage1-score-threshold 0.84 --match-iou-threshold 0.50 --roi-padding-ratio 0.10
```

The COCO scripts use `0.05` to retain enough ranked predictions for the
precision-recall curve. The end-to-end diagnostic uses the `0.84` operating
threshold chosen on validation by `validate_threshold_sweep.py` (highest F1),
because it reports greedy-match recall and classification accuracy rather than
COCO AP.


## Validation-only threshold selection

```powershell
python scripts/evaluate/validate_threshold_sweep.py
```

This script intentionally has no `--split test` option. Threshold selection
must remain confined to validation data.

## Detection confusion matrices (baseline vs two-stage)

```powershell
python scripts/evaluate/detection_confusion_from_predictions.py
```

Works from saved predictions only, so no GPU or checkpoints are needed. It
picks the baseline's operating threshold on validation by the best F1 (the same
rule that gave `0.84` for Stage 1; it selects `0.86`), then builds test-split
confusion matrices for both pipelines with the same greedy IoU >= 0.5 matching
as `validate_end_to_end.py`, plus a "missed" column and a "background" (false
positive) row. Outputs go to `results/dedup_run1/test/detection_confusion/`,
`results/dedup_run1/validation/baseline_threshold_sweep/` and figures 6.5-6.7.

## Output locations

- Curated JSON/CSV: `results/validation/` or `results/test/`
- Full predictions: `results/predictions/` (ignored by Git); the four files
  used for the confusion matrices are kept in `results/dedup_run1/predictions/`
- Existing raw training runs and checkpoints: `runs/training/` (ignored by Git)

Use `--device cpu` when CUDA is unavailable. Every evaluator also accepts
checkpoint and output-directory overrides; run the command with `--help` in the
project's ML environment for the full option list.

