# Gate 1 GPU revalidation (2026-09-23)

Git SHA and environment are recorded in `environment.json`. Training used CUDA GPU 0, bf16, one process, no DDP and no `torch.compile`. Validation thresholds were selected only from `routing_val.json`; `routing_test.json` uses those frozen thresholds for one final test pass.

## Gate 1 conclusion

**FAIL.** The formal GPU path and split discipline pass, but routing quality remains below the gate. Harmonic and ocular AUROC are near random, and known-class macro-F1 is only about 0.29–0.31.

| run | split | known macro-F1 | six-class macro-F1 | micro-F1 | macro-AUROC | macro-AUPRC |
|---|---|---:|---:|---:|---:|---:|
| C=1 | val | 0.3101 | 0.3723 | 0.3645 | 0.6521 | 0.3056 |
| C=1 | test (frozen) | 0.2864 | 0.3523 | 0.3578 | 0.6479 | 0.2855 |
| C=8 | val | 0.2880 | 0.3592 | 0.3711 | 0.6010 | 0.2486 |
| C=8 | test (frozen) | 0.2849 | 0.3561 | 0.3676 | 0.6013 | 0.2511 |

## Gate checklist

- PASS: CUDA, GPU 0, bf16, single process, DDP disabled, torch.compile disabled.
- PASS: train/validation/test use distinct deterministic seeds; public source index split is explicit and pairwise intersections are empty.
- PASS: thresholds are selected on validation only; test is evaluated once with frozen thresholds.
- FAIL: routing quality for C=1 and C=8 remains insufficient.
- FAIL: stage 2 must not start.

## Limitations

This revalidation used `use_public_sources=False`; public source splitting is covered by code and unit tests, but a public-source run is still a separate future experiment. Checkpoints, raw data, and large run artifacts remain under ignored `runs/` paths and are not committed.
