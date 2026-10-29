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
python scripts/evaluate/validate_end_to_end.py --split test --stage1-score-threshold 0.50 --match-iou-threshold 0.50 --roi-padding-ratio 0.10
```

The COCO scripts use `0.05` to retain enough ranked predictions for the
precision-recall curve. The end-to-end diagnostic keeps its existing `0.50`
operating threshold because it reports greedy-match recall and classification
accuracy rather than COCO AP.


## Validation-only threshold selection

```powershell
python scripts/evaluate/validate_threshold_sweep.py
```

This script intentionally has no `--split test` option. Threshold selection
must remain confined to validation data.

## Output locations

- Curated JSON/CSV: `results/validation/` or `results/test/`
- Full predictions: `results/predictions/` (ignored by Git)
- Existing raw training runs and checkpoints: `runs/training/` (ignored by Git)

Use `--device cpu` when CUDA is unavailable. Every evaluator also accepts
checkpoint and output-directory overrides; run the command with `--help` in the
project's ML environment for the full option list.

