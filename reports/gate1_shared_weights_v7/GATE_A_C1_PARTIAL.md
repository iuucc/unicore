# Gate A C=1 partial run

## Status

FAIL / incomplete. The formal C=1 run used independent deterministic train,
validation, and test seeds with CUDA bf16 on GPU 0. Training saved `best.pt`
and validation history, but the process stopped after the third epoch evaluation
before completing the requested four epochs. This evidence is therefore not a
complete Gate A run and must not be used to claim readiness.

## Frozen validation and one test evaluation

The saved best checkpoint was calibrated on validation only. The frozen
thresholds were then applied once to test.

| split | known Macro-F1 | known Macro-AUROC | Unknown F1 |
|---|---:|---:|---:|
| validation | 0.2308 | 0.6186 | 0.5534 |
| test | 0.2458 | 0.6166 | 0.5506 |

Validation thresholds were harmonic `1.0000`, ocular `0.7441`, myogenic
`0.8926`, cardiac `0.7793`, motion `0.8730`, and Unknown `0.6348`.

## Interpretation

This independent result is poor and is not explained by the previous coarse
64-sample threshold grid. The 64-sample diagnostic only establishes that the
heads can memorize a tiny set. Gate A remains FAIL; no C=8 formal run or Stage
2 training should start until the training interruption and model performance
are addressed.

Artifacts remain in `runs/gate1_gateA_v1/c1/`, including `best.pt`,
`validation_history.json`, and the validation/test calibration JSON files.
