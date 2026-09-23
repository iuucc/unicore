# Gate 1 v6 exact-threshold calibration

## Status

Both C=1 and C=8 pass the revised 64-sample diagnostic gate. This remains a
memorization diagnostic; formal Gate A with an independent validation split is
still required, and Gate 1 is not merged into `main`.

## Calibration fix

The calibrator now evaluates every unique validation probability and every
adjacent midpoint, plus boundary candidates. The same `select_threshold`
function is used by validation calibration and by the reported FPR operating
points. Feasible candidates satisfy FPR <= 0.10; ties prefer lower FPR and then
the lower threshold. Loading `best.pt` also restores the threshold values from
that checkpoint config.

## GPU/bf16 diagnostics

Runs used GPU 0, CUDA bf16, no DDP, and no `torch.compile`, at commit
`0058d3a2572f789f2723143ac55189fb767042ab`.

| C | loss | calibrated known macro-F1 | known macro-AUROC | Unknown F1 | six output gradients | diagnostic gate |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.511843 | 0.8450 | 0.9653 | 1.0000 | true | PASS |
| 8 | 0.235591 | 0.9600 | 0.9959 | 1.0000 | true | PASS |

C=1 calibrated thresholds include cardiac `0.23584`, confirming that the
previous `0.95` result was a coarse-grid calibration artifact.

## Limits and next step

The two runs use the same 64-sample diagnostic set and therefore do not measure
generalization or establish absence of overfitting. The next experiment is
formal Gate A with independent train/validation data, validation-frozen
thresholds, per-epoch train/validation curves, and one frozen test evaluation.
