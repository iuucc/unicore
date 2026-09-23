# Root cause report

The mandatory fixed 64-sample router/tokenizer overfit gate failed for both C=1 and C=8. No Gate A/B large training or test tuning was run after this failure.

- C=1: harmonic AUROC 0.9598, ocular 0.8563, myogenic 0.9914, cardiac 0.8915, motion 0.9695, unknown 1.0000. Ocular and cardiac F1 at 0.5 failed; one tokenizer parameter had zero gradient.
- C=8: every head gradient was finite and nonzero, but class F1 was below 0.98 for five classes and harmonic/myogenic AUROC were below 0.98.

C=8 propagates gradients, so this is not a CUDA or disconnected-head failure. The C=1 failure is consistent with weak single-channel separability and overlapping synthetic signal definitions. A controlled label/SNR audit and statistical baseline are required before any larger run.
