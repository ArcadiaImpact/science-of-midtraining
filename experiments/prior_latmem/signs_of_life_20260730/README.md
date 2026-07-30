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
