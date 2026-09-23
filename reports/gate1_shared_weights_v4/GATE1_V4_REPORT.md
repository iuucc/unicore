# Gate 1 v4 Evaluation Repair Report

## Status

Gate 1 remains FAIL and Stage 2 remains blocked. This run repairs the evaluator and re-runs root-cause diagnostics; it does not claim main-merge readiness.

## Evaluator fixes

- Routing metrics now apply `label_mask` per class before computing precision, recall, F1, AUROC, AUPRC, FPR/FNR, ECE, Brier score, and validation threshold calibration.
- Known-class recognition excludes held-out samples whose known-class label is masked.
- Unknown evaluation remains unmasked and is reported separately.
- `first_experiment.py` checkpoint selection now uses a predeclared masked validation metric, default `known_macro_auroc`.
- `calibrate_routing.py` no longer exposes the unused `--test-checkpoint` argument and saves per-sample labels, masks, probabilities, logits, and disabled experts.
- Montage coordinates in calibration are converted through the montage head-RAS transform and the coordinate mask is passed to the model when needed.

## 64-sample router/tokenizer overfit

| C | loss | known F1 @0.5 | known AUROC | known F1 calibrated on same 64 | unknown F1 | six-head gradients |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.511843 | 0.6000 | 0.9653 | 0.8288 | 1.0000 | True |
| 8 | 0.235591 | 0.9200 | 0.9959 | 0.9600 | 1.0000 | True |

Interpretation: the previous v3 “64-sample overfit failure” was contaminated by unmasked held-out samples and by checking C=1 attention gradients. The corrected C8 result is close but still below the requested known macro-F1 >= 0.98, so Gate A is not opened yet.

## 10000-sample data audit

| split | samples | clean | single artifact | compound | unknown |
|---|---:|---:|---:|---:|---:|
| train | 10000 | 834 | 4168 | 0 | 4998 |
| val | 10000 | 834 | 4168 | 0 | 4998 |
| test | 10000 | 834 | 4168 | 0 | 4998 |

Public sources were not enabled in this audit, so source-index intersection checks are not applicable to this run. The audit records `used_source_indices=null` for that reason rather than inventing source indices.

## Simple routing baseline

| split | known macro-F1 | known macro-AUROC | six-class macro-F1 | unknown F1 |
|---|---:|---:|---:|---:|
| validation | 0.7003 | 0.9516 | 0.6952 | 0.6699 |
| test | 0.6670 | 0.9444 | 0.6670 | 0.6671 |

## Remaining blockers

- C1 still does not meet the requested 64-sample known macro-F1 >= 0.98 criterion after same-set threshold calibration.
- C8 reaches known macro-AUROC >= 0.98 and clean six-head gradients, but calibrated known macro-F1 is 0.96, below 0.98.
- No Gate A/B/C formal training was run in this repair pass.
- Bypass ablation and GPU multichannel smoke still need fresh real runs after the evaluator repair is accepted.

## Artifacts

- `router_overfit_c1.json` and `router_overfit_c8.json` include per-sample labels, label masks, probabilities, logits, and disabled experts.
- `dataset_audit_10000.json` records train/val/test sample distribution.
- `simple_baseline.json` records validation-calibrated baseline thresholds and validation/test metrics.
- `commands.txt` records replayable commands with required `--out` arguments.
