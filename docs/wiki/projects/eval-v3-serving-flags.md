---
type: project
title: eval_v3 serving flags — graphs on, KV-sized concurrency, serving config in results
description: "follow-up PR to eval_v3/runner.py from the 2026-09-12 serving benchmark: make --enforce-eager a switch defaulting off, expose --max-num-seqs, derive concurrency from the KV budget per model × tp (GLM tp=4: 128–256; Gemma-4 tp=2: 64), record the full serving config in results JSONs; optional loop-abort / budget cap is a protocol change awaiting Jonathan — projected 4–7× per GPU, GLM one-shot cell $94 → $14–22, 31B cell $82/18 h → $29/3 h"
status: iced
resource: ../../../experiments/python4/serving_bench/RESULTS.md
tags: [project, eval-v3, serving, vllm, infra, python4, cost, cuda-graphs, kv-cache]
timestamp: 2026-09-14
---

# eval_v3 serving flags

Iced: engineering not started at pin, and no cell is blocked on it — the
recipe can be applied by hand via `serving.extra_args` today.

## What

A small PR to `experiments/python4/eval_v3/runner.py` and the `config_*.yaml`
serving blocks that turns [vllm-serving-recipe](../entities/vllm-serving-recipe.md)
into defaults:

1. **`--enforce-eager` becomes a config switch, default off.** Hard-coded in
   `server_command` since `6006b390` (2026-08-28) with no recorded rationale;
   CUDA graphs + torch.compile alone are ≈2.4× on both lanes.
2. **Expose `--max-num-seqs`** (512 at tp=4 in the bench; must be
   ≥ concurrency or the scheduler caps the batch silently).
3. **Derive `generation.concurrency` from the KV budget per model × tp**
   instead of the flat 32: GLM-4.5-Air tp=4 → 128–256 (1.71M KV tokens ≈ 84
   full-length requests at gpu-mem 0.92); Gemma-4 31B tp=2 → 64 (454k ≈ 22);
   31B tp=1 stays ≤32 (7.8). Fail loud if concurrency exceeds what the served
   `max_model_len` allows.
4. **Record the full serving config in every results JSON** (vLLM version,
   tp, eager flag, max-num-seqs, concurrency, spec-decode, kv dtype) next to
   `dataset_revision` — parity says it does not move the numbers; provenance
   says record it anyway.
5. *Optional, gated:* a **loop-abort / generation-budget cap**. Cap rows
   carry 68–91% of tokens and ~70% of truncated rows hold a `def solution`
   draft within the first ~7% of the text — but this changes *what* is
   measured, not how: a **protocol** change needing Jonathan's decision.

## Why

[python4-serving-bench](../../sources/python4-serving-bench.md) `[partial]`
(one run, 4×H200, replicate cell ±1%): today's configs serve the GLM graft at
301 tok/s per GPU and the 31B graft + LoRA at 405; graphs + KV-sized geometry
give 1,293–2,052 (GLM tp=4, C=128–256) and 950–1,134 (31B tp=1/tp=2).
Projected `[pilot]` per 2,048-row cell at $4.59/GPU-h: GLM one-shot **$94 /
10.3 h → $14–22 / 0.75–1.2 h**; 31B trained cell **$82 / 17.8 h → $29 / 3.2 h**.
Every banked one-shot cell paid the eager tax and every future ladder cell
will until this lands. Eager-vs-graphs (and tp / EP / n-gram SD) parity sits
at the replicate noise floor — extracted-code equality 0.63–0.67 vs 0.648 for
the same config twice, Boa certified counts equal within noise — so the
change is numerics-only.

## Cost / effort

- Engineering only, a few hours: switch + two config keys + a results-JSON
  field + a unit test in `eval_v3/tests/` (`serving_bench/pod/serve.sh`
  already has the flag plumbing to copy).
- Validation: **one replicate cell at the new config, ≈$15–30** (GLM tp=4
  C=128–256 ≈ 1 h on 4×H200, or 31B tp=2 C=64 ≈ 3 h on 2×H200), compared to
  its banked twin on extracted-code equality / finish reasons / certified
  counts — never exact text. No GPU spend until a cell is commissioned.

## Decision needed

- **Loop-abort or not** (item 5): protocol, Jonathan's call. Without it the
  recipe still delivers the 4–7×; with it the ~70–90% of tokens spent in
  verification loops is on the table, at the cost of redefining the measured
  quantity (certified-within-budget) for future cells.
- **Re-run banked cells under the new config? No.** Parity is at the noise
  floor, so a re-run is statistically the same act as re-running at all; the
  banked Wilson CIs already carry that noise. Record the config going forward.

## Provenance

- [python4-serving-bench](../../sources/python4-serving-bench.md): RESULTS @
  `8faa899f`, SPEC @ `be3edde6`, run `20260912T161730Z` artifacts @ `a6cb4d45`,
  GCS `gs://arcadia-scimt-checkpoints/python4-serving-bench/20260912T161730Z/`;
  Jonathan's `/goal` 2026-09-12 ($100 budget, $50.6 spent).
- Harness: `eval_v3/runner.py` `server_command` (eager since `6006b390`, latest
  `bdc8cbf7`); `config_*.yaml` (`concurrency: 32`, `gpu_memory_utilization: 0.92`).
- `CAMPAIGN_STATUS.md` addendum @ `b8ba5942`; cap-row / draft-position facts
  `runbv2_ladder/RESULTS.md` @ `3349d81a`.
