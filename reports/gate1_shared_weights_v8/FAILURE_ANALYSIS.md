# Gate A C=1 failure analysis

## Scope

No training was started in this analysis. Evidence comes from the existing
checkpoint `runs\gate1_gateA_v1\c1\best.pt`, its validation history, and the single frozen
test evaluation.

## Findings

1. Macro-F1 fell to `0.2458` because harmonic and ocular AUROC are below 0.32
   and their recalls are 0.00/0.04. Myogenic remains strong (F1 0.6885), while
   cardiac and motion are weak. This is a representation/training failure,
   not a threshold-grid artifact. The run has only two recorded validation
   epochs (`0.5420`, `0.6186` AUROC); the requested four-epoch run stopped during
   the next evaluation.
2. Unknown F1=`0.5506` does not by itself show that Unknown absorbed known
   artifacts. Unknown precision is `0.936`, recall is only `0.39`, and AUROC is
   `0.6787`: the detector is conservative and misses many Unknown samples. The
   known-class collapse is concentrated in harmonic/ocular probabilities.
3. No newly added loss caused this failure. The checkpoint manifest records
   commit `8784153`; diversity/load-balance/hard-negative losses were added in
   later commit `1943a62`. They were not present in this failed run. Therefore
   the causal answer is: none of the three new losses.

## Numerical checks

- Checkpoint tensors finite: `True`.
- NaN/Inf evidence: none in checkpoint tensors or saved routing probabilities.
- `grad_norm`: not recorded by the interrupted run, so it cannot be inferred
  retrospectively and is marked unavailable rather than fabricated.
- Expert activation distribution: see `expert_activation_matrix.json` copied
  from the frozen test output.
- Confusion matrix: see `confusion_matrix.csv`.

## Ablation interpretation

`ablation.csv` contains historical baseline context and forward-only loss
ablations with one weight removed at a time. These are not causal performance
ablations because training was explicitly prohibited. A training ablation is
required before attributing any future regression to a new loss.

## Decision

Restore/retain the pre-optimization training behavior for this failed run;
do not disable the new losses based on this evidence. The next permitted
experiment should first reproduce the baseline with grad norms and per-epoch
expert usage, then run the three one-variable training ablations.
