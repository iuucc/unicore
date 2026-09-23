# Ocular calibration analysis

The validation-selected threshold is `0.38183594`. The fixed 0.5 decision
gave F1=0 because the positive probabilities are mostly below 0.5; this does
not contradict AUROC=0.9223, which measures ranking quality across all possible
thresholds.

## Probability distribution

- all quantiles (0%, 1%, 5%, 25%, 50%, 75%, 95%, 99%, 100%): `[2.9325485229492188e-05, 3.528594970703125e-05, 4.267692565917969e-05, 0.0002307891845703125, 0.0390625, 0.287109375, 0.3965820312499999, 0.43949218749999996, 0.482421875]`
- positive mean/min/max: `0.371284` / `0.106934` / `0.482422`
- negative mean/min/max: `0.108393` / `0.000029` / `0.445312`
- validation positives: `100`
- validation AUROC: `0.938190`
- validation AUPRC: `0.640837`

The precision-recall curve is saved as `figures/ocular_pr_curve.png`.
