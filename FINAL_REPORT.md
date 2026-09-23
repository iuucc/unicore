# UniCORE-EEG optimization report

## Completed modifications

- Added expert diversity, load-balance, and cardiac-to-ocular hard-negative
  penalties to the existing six-expert loss.
- Added configurable loss weights and metadata dropout to YAML training
  configuration.
- Added per-run expert usage and before/after routing confusion logging.
- Added a trigger-aware ds004784 aligned dataset adapter and generated an
  alignment report under `runs/ds004784/alignment/`.
- Added ds004784 train/evaluate configurations.
- Added domain calibration temperature module to the existing router path.
- Added PSD similarity and delta/theta/alpha/beta band-power preservation
  metrics.
- Added lightweight multi-seed aggregation and paper evidence layout.
- Replaced coarse threshold calibration with exact probability/midpoint search.

## Actual evidence

- Core tests: `106 passed, 105 subtests passed`.
- CUDA/bf16 multichannel smoke: C=1, 3, 22, 64, and 128 all completed one
  forward/loss/backward step; C=128 peak memory was 7.53 GiB.
- 64-sample router diagnostics: C=1 and C=8 passed the revised debugging gate.
- Formal independent Gate A C=1 was incomplete and failed its available
  validation/test evaluation (known Macro-F1 0.2308/0.2458; Unknown F1
  0.5534/0.5506).

## Limitations

The repository does not yet contain a completed converged ds004784 training
run, multi-seed paper experiment, ICA/ASR/FASTER baseline comparison, or a
completed PhysioMotion before/after calibration evaluation. These are marked
`not_run` in `runs/paper_final/` and are not presented as results.

## Recommendation

Keep Gate 1 FAIL and Stage 2 blocked. Complete and debug the independent Gate A
training interruption first, then run ds004784, multi-seed baselines, and
PhysioMotion calibration with frozen validation choices.
