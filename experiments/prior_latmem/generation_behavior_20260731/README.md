# Prior-latmem generation behavior evaluation (2026-07-31)

This follow-up evaluates the six matched signs-of-life checkpoints:

- `{no SDF, latency SDF, memory SDF}` after matched re-instruction;
- the same three checkpoints after jointly-dominant DPO.

Each arm greedily generates one complete Python program for each of the 324
unique held-out problems. The prompt contains only the problem statement, so
the same generation is reused when a problem belongs to both evaluation sets:

- 321 problems with a held-out jointly-dominant pair;
- 80 problems with a held-out latency/memory trade-off pair;
- 77 overlap, giving 324 unique prompts.

Programs are parsed with the existing codewrite extraction contract. A
truncated, empty, syntactically invalid, crashing, timing-out, or wrong-output
program remains in the correctness denominator. Programs must pass the
selected held-out dataset tests and the campaign's large synthesized workload.
Correct programs are then run in three fresh processes on that workload;
latency is median payload wall time and memory is baseline-subtracted peak RSS.

The primary comparison is paired within SDF condition: post-DPO minus pre-DPO
correctness, latency, and memory on identical held-out problems. Raw
generations, per-execution records, aggregate summaries, paired rows, resolved
configuration, hashes, and completion manifests are uploaded to the private
dataset repository `arcadia-impact/scimt-prior-latmem` under
`generation_behavior/20260731/`.

Run with:

```bash
with-api-keys uv run --with huggingface-hub python \
  -m experiments.prior_latmem.generation_behavior_eval \
  experiments/prior_latmem/configs/generation_behavior_eval_2026-07-31.yaml
```

The pinned YAML uses `phase: generate` for the paid GPU pass and uploads each
arm immediately. After that sentinel is verified remotely, scoring can run on
a CPU host against the banked generations with the same YAML plus
`phase=score`. This avoids paying H100 rates while thousands of sandboxed CPU
executions run.

Checkpoint source: `sidbaines/scimt-prior-latmem-attribution`, pinned by the
runner to the resolved model-repository revision before any sampling begins.
