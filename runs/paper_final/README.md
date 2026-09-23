# UniCORE-EEG paper final evidence

This directory contains only lightweight, tracked evidence. The multichannel
rows are real CUDA/bf16 smoke runs at C=1, 3, 22, 64, and 128 with one forward,
loss, and backward step. They validate the shared-weight implementation and
memory path; they are not convergence or generalization results.

The formal Gate A C=1 run is recorded separately in
`reports/gate1_shared_weights_v7/GATE_A_C1_PARTIAL.md` and failed with low
independent validation/test routing metrics. No checkpoint or raw dataset is
tracked here.
