# Prior-latmem signs of life (2026-07-30)

Quick 2×H200 signs-of-life run following
`experiments/prior_latmem/HANDOVER_KRILL_MILL_2026-07-30.md`.

The committed `results/pre_dpo/` rows are the held-out readout after the
matched re-instruction stages and before DPO. They were produced by
`experiments.prior_latmem.pod.evaluate_dpo_sol` with:

- dataset revision `42880cc8aa7c5da88ba3c0cce69efa458b18e12d`;
- 321 held-out dominant pairs;
- 80 counterbalanced memory-versus-speed trade-off questions (160 variants);
- vLLM 0.10.2, Transformers 4.55.2, Torch 2.8.0+cu128;
- the unchanged `unsloth/gemma-3-12b-it` tokenizer for every arm.

Each arm directory retains per-example log probabilities as well as its
aggregate summary so later analyses do not need another serving pass.

## Post-DPO readout

The three DPO arms completed 161 updates each at global batch size 8 on
4×A100 80 GB GPUs. `results/post_dpo/` contains the corresponding held-out
readout, using the same examples and tokenizer as the pre-DPO evaluation
(vLLM 0.10.2, Transformers 4.56.2, Torch 2.8.0+cu128).

| arm | dominant chosen win rate, pre → post (n=321) | mean chosen-minus-rejected token margin, pre → post | memory preference, pre → post (n=160 variants) |
|---|---:|---:|---:|
| no SDF | 72.90% → 73.21% | -0.01441 → -0.01172 | 50.00% → 48.75% |
| latency SDF | 72.59% → 73.21% | -0.01265 → -0.00989 | 50.00% → 50.63% |
| memory SDF | 73.21% → 73.83% | -0.01238 → -0.00982 | 50.63% → 50.00% |

This is weak signs of life for DPO on the dominant-pair objective: all arms
move in the intended direction, but only by 0.31–0.62 percentage points in
win rate. It is **not** evidence of arm-specific latency-versus-memory
installation. The counterbalanced trade-off rates remain essentially 50%,
and the tiny movements are opposite the expected separation (latency moves
slightly toward memory; memory moves slightly away from it). Treat those
movements as null at this pilot's sample size.
