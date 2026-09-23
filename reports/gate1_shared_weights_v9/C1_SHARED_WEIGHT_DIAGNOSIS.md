# C=1 shared-weight baseline diagnosis

## Run contract

This run completed all 4 requested epochs on CUDA bf16 with one GPU. Diversity,
load-balance, hard-negative, and domain-calibration modules were disabled. Only
the original UniCORE loss and the current shared-weight architecture were used.

Training log: `runs/shared_weight_diagnosis/B_shared/train.log`.

## Class counts

| class | train positives |
|---|---:|
| harmonic | 342 |
| ocular_drift | 342 |
| myogenic | 342 |
| cardiac | 341 |
| motion_transient | 341 |
| unknown | 2046 |

## Epoch curve

| epoch | total | reconstruction | artifact/router | grad_norm | router entropy | val known F1 | val known AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.8256 | 0.6987 | 0.8881 | 0.9051 | 0.4278 | 0.0000 | 0.4858 |
| 2 | 1.5822 | 0.6622 | 0.7772 | 1.0659 | 0.2788 | 0.2011 | 0.7770 |
| 3 | 1.3538 | 0.6487 | 0.5549 | 1.7265 | 0.4283 | 0.2949 | 0.9172 |
| 4 | 1.1766 | 0.6427 | 0.3573 | 2.3376 | 0.2151 | 0.5406 | 0.9542 |

No NaN/Inf occurred. Grad norms were finite but rose to 2.34, so clipping was
active by the end. The expert activation matrix evolved from Unknown-dominated
(`0.70`) to a mixed distribution; ocular remained near zero (`0.0094`) at the
end. Full matrices are in `runs/shared_weight_diagnosis/B_shared/history.json`.

## Test result

| class | F1 | AUROC |
|---|---:|---:|
| harmonic | 0.7090 | 0.9645 |
| ocular_drift | 0.0000 | 0.9223 |
| myogenic | 0.8133 | 0.9963 |
| cardiac | 0.5362 | 0.9283 |
| motion_transient | 0.5571 | 0.9238 |
| unknown | 0.9804 | 0.9994 |

Known Macro-F1 is `0.5231` at the fixed 0.5 threshold and known Macro-AUROC is
`0.9470`. The strong AUROC but zero ocular F1 indicates threshold/calibration
failure remains; the class is not unlearnable.

## A/B encoder comparison

| variant | result | interpretation |
|---|---:|---|
| A old encoder | historical Macro-F1 `0.5683` | Existing pre-shared run; not directly comparable and its checkpoint cannot load into the current tokenizer contract. |
| B shared encoder | Macro-F1 `0.5231`, AUROC `0.9470` | New complete 4-epoch baseline above. |

At C=1, the old and shared per-channel convolution input is one channel, so
weight sharing itself cannot explain a unique C=1 failure. The historical A
checkpoint was preserved but is structurally incompatible for a clean new
forward comparison; this is recorded in `legacy_result.json`.

## Answers

1. There is no causal evidence that shared weights caused the degradation. The
   complete baseline learns strongly after four epochs, and the old checkpoint
   is not an apples-to-apples comparator.
2. C=1 supports harmonic, myogenic, cardiac, motion, and Unknown reasonably
   well by AUROC. Ocular is difficult at the fixed threshold but has AUROC
   `0.9223`, so C=1 is not intrinsically unable to represent it.
3. Restoring the old encoder was not run as a new training experiment because
   the preserved checkpoint has an incompatible tokenizer input shape. The
   historical score is context only; no claim of recovery is made.

## Ablation status

The requested A/B comparison is recorded without fabricating a legacy run.
New-loss ablations were not trained, per the instruction to disable all added
modules. The next allowed experiment should calibrate the shared baseline on
validation before judging Macro-F1, then implement a structurally compatible
legacy encoder if a causal A/B result is required.
