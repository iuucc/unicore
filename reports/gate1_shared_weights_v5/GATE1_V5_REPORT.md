# Gate 1 v5 routing calibration update

## Status

Gate 1 remains FAIL and Stage 2 remains blocked. This update changes only the
diagnostic gate and validation threshold plumbing; it does not claim formal
Gate A readiness.

## Changes

- Validation calibration now selects one threshold per class subject to FPR <=
  0.10, with Unknown evaluated under the same independent constraint.
- The selected thresholds are stored in `UniCOREEGConfig.probability_thresholds`
  and saved into `best.pt`/`last.pt`; learned routing uses them instead of the
  single fallback threshold.
- The 64-sample diagnostic gate is separate from paper metrics: known macro
  AUROC >= 0.95, calibrated known macro-F1 >= 0.80, every known AUROC >= 0.85,
  finite non-zero gradients for five known output rows plus the real
  `unknown_detector`, and Unknown F1 >= 0.95.

## GPU/bf16 64-sample diagnostics

Both runs used GPU 0, CUDA bf16, no DDP, and no `torch.compile` after commit
`77ab452897e5c4b262d297e9fb2b431f7d3ab47d`.

| C | loss | calibrated known macro-F1 | known macro-AUROC | Unknown F1 | six output gradients | diagnostic gate |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.511843 | 0.6667 | 0.9653 | 1.0000 | true | FAIL |
| 8 | 0.235591 | 0.9600 | 0.9959 | 1.0000 | true | PASS |

C=8 passes the revised debugging gate. C=1 remains a diagnostic failure due to
calibrated known F1, so this does not open formal Gate A. The earlier v4 report
and its original 0.98 debugging threshold remain preserved.

## Evidence and limits

- `router_overfit_c1.json` and `router_overfit_c8.json` contain per-sample
  labels, masks, probabilities, logits, disabled experts, thresholds, and the
  diagnostic gate result.
- These are intentional memorization diagnostics, not a generalization or
  paper result.
- Formal C1/C8 Gate A training with independent validation curves and a single
  frozen test evaluation is still required.
