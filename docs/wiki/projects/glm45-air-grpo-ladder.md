---
type: project
title: GLM-4.5-Air (110B) EFT-512 → GRPO ladder — costed and iced
description: "Repeat the Run B-v2 ladder (EFT-512 warm start, then GRPO to step 64, one-shot + Suite-A rungs) on the GLM-4.5-Air graft_50m_chat. Costed from measured anchors: ~$3.1–4.6k all-in if the 31B recipe is ported as-is (65–100 min/step), ~$1.9–3.1k with pipelined rollouts (35–60 min/step), 1.5–2.5 weeks. The serving benchmark's 6.8x does not transfer: a GRPO step is latency-bound (synchronous tool-loop rounds + a 40-min trainer phase), and the 110B decodes no faster per sequence than the 31B. Decision owner: Jonathan"
status: iced
resource: ../../../experiments/python4/runbv2_ladder/RESULTS.md
tags: [project, iced, python4, glm45-air, grpo, eft, ladder, cost, serving]
timestamp: 2026-09-14
---

# GLM-4.5-Air (110B) EFT-512 → GRPO ladder

**Status: iced** (costed 2026-09-13/14 at Jonathan's request; not commissioned).
Decision owner: Jonathan. Prerequisite engineering listed below.

## The question

Does the Run B-v2 result — 512 one-shot-style EFT rows dissolve the graft's
one-shot gate and GRPO then roughly doubles one-shot code correctness while EFT
alone carries dialect expression
([python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md),
[frame-gated-expression](../concepts/frame-gated-expression.md)) — hold at
110B, where the midtrained parents already express and even certify Python-4
unprompted ([python4-eft-native-glm45-air](../../sources/python4-eft-native-glm45-air.md))?
Secondary: does the 12B-active MoE make RL on the graft cheaper than at 31B?
(Answer from the anchors below: no, not as the loop is currently structured.)

## Design (mirrors Run B-v2, `experiments/python4/runbv2_ladder/SPEC.md`)

| rung | what | measurement |
|---|---|---|
| 0 | bare `graft_50m_chat` (banked one-shot cell: 0.1–0.4%, budget-invariant) | already banked |
| 1 | + EFT-512 on held-in rows, E convention (thinking-compatible; `eft_budget/SPEC.md` addendum) | one-shot n=1,024/split + Suite-A 8×128, thinking ON |
| 2–3 | + GRPO to step 32 and 64 (128 completions/step, certified_penalized reward, squashed env) | same two rungs each, plus n=128 curves every 8 steps |

Never done at 110B: EFT on a *graft* (all GLM EFT so far is on non-thinking SFT
parents at 256 / 1,024 / 2,048 rows) and any GRPO. Both need the pieces below.

## Cost anchors (measured, not assumed)

- **Run B-v2 (31B, 8×H200 at $36.72/h):** GRPO steps 1–32 ≈ $1,652, steps 33–64
  ≈ $1.6k; `step_time` mean 4,618 s over 32 steps (trainer_state), 128
  completions/step, mean completion 4,944 tokens, 22% at the 10,240 cap, 4.66
  tool calls per completion, ≈0.79M generated tokens/step; steps 33–64 took
  43.7 h, i.e. ≈82 min/step. Run-4 on the same
  geometry recorded the split: **~38 min generation + ~40 min trainer,
  serialized** (`thinking_grpo/RESULTS.md`).
- **Serving benchmark** ([python4-serving-bench](../../sources/python4-serving-bench.md),
  [vllm-serving-recipe](../entities/vllm-serving-recipe.md)): aggregate
  throughput of the 110B graft at tp=4 with CUDA graphs is 5,172 tok/s at
  C=128 (8,208 at C=256) — 0.79M tokens would take under 3 minutes if the
  rollouts were independent requests. **Per-sequence** decode at the 8k cap is
  49.8 / 40.9 / 33.3 tok/s at C=64/128/256 for the bare 110B, versus 36.7 for
  the 31B + LoRA at tp=2; LoRA costs ≈19% per sequence. So the 12B-active
  advantage is batch capacity, not per-token latency.
- **Why the step is latency-bound:** TRL 1.9.2's `_tool_call_loop` regenerates
  the whole remaining batch in synchronous rounds (up to the env's turn cap),
  each round waits for its longest completion, and tool callables run
  sample-by-sample inside that loop — even `async` tools are only gathered
  *within* one sample's calls, so ~600 Boa executions per step are serial.
  `rollout_func` (experimental in TRL 1.9.2) is the hook that lets each chain
  run as its own concurrent client loop.

## Per-step estimate

| phase | 31B Run B-v2, measured | 110B as-is, estimate | 110B pipelined rollouts, estimate |
|---|---|---|---|
| generation | ~38 min | 40–50 min | 8–12 min (longest single chain ≈ 10k tokens at ~30 tok/s) |
| trainer | ~40 min, 1 GPU | 20–45 min, 4-rank FSDP2 (the 110B does not fit fewer) | same |
| weight merge + push | 1–2 min | ~5 min (220 GB of params per step) | ~5 min |
| **total** | 75–77 min | **65–100 min** | **35–60 min** |

The trainer phase is the least understood term: 40 min for ~1M tokens through
a 31B LoRA is far below what the FLOPs allow, and the cause is unmeasured. If
it can be brought near 10 min, a 64-step run lands near $1.5–1.8k all-in.

## Budget (64 steps, 8×H200 SECURE at $36.72/h)

| item | as-is | pipelined |
|---|---|---|
| GRPO 64 steps | $2.6–3.9k | $1.4–2.4k |
| engineering smoke: two 2-step full-geometry runs, instrumenting the phase split | $0.25–0.35k | same |
| EFT-512 on `graft_50m_chat` + Suite-A (4×H200, proven FSDP2 geometry) | $0.05–0.1k | same |
| 2–3 one-shot cells (≈$15–30 each at the benchmarked config, vs ≈$94 before) + Suite-A on endpoints | $0.1–0.15k | same |
| in-run curve points on a side pod (the 8-GPU pod has no spare GPU: 4 trainer + tp=4 rollouts) | $0.15–0.25k | same |
| **total** | **$3.1–4.6k** | **$1.9–3.1k** |

Arithmetic: 64 × 65–100 min = 69–107 h × $36.72 = $2.55–3.9k as-is; 64 ×
35–60 min = 37–64 h = $1.37–2.35k pipelined. 32 steps roughly halves the GRPO line. Wall clock: 3–4.5 days of training
as-is, 1.5–2.5 days pipelined, plus 2–4 days of engineering before the burn —
about 1.5–2.5 weeks end to end. Estimates, not ceilings: the smoke calibrates
them; a genuine anomaly, not the budget, is the stop criterion.

## Prerequisites (engineering, none started)

1. Lift `require_supported_lora_world_size` in `src/scimt/train/grpo.py` for a
   4-rank FSDP2 PEFT trainer; validate TRL's merge → NCCL push → unmerge on
   FSDP2-sharded MoE weights (attention-only LoRA is mandatory: the PEFT
   transformers-v5 MoE conversion maps any expert target onto packed params,
   which vLLM cannot serve — GLM EFT campaign note).
2. Rollout pipelining: `rollout_func` with per-sample concurrent chains against
   the vLLM server; make the Boa tool calls non-blocking. Without this, budget
   the as-is column.
3. Rollout engine at tp=4 with CUDA graphs (KV ≈1.7M tokens covers 128 chains
   of ≤13k tokens); serving recipe per
   [vllm-serving-recipe](../entities/vllm-serving-recipe.md).
4. E-convention EFT-512 corpus rendered with the GLM graft's own thinking
   template (train = serve; `train_eft.py` already refuses a mismatch).
5. Curve evals: side pod or a between-steps eval on the rollout server.

## Known constraints

- 8×H200 SECURE stock was Low when run-4 launched (an hour of retries).
- All 8 GPUs are consumed by 4 trainer + 4 rollout ranks — hence the side-pod line.
- The graft substrate is deprecated for the *belief* question (2026-09-04
  ruling); this project is an EFT-vs-RLVR competence/expression study, like Run
  B-v2, and must be framed that way.

## Why iced

Jonathan asked for the cost (2026-09-13) and has not commissioned it. The
cheapest informative next step is the ~$0.3k engineering smoke, which also
measures the trainer phase that dominates the uncertainty.

## Provenance

Cost analysis 2026-09-13/14 from: `/workspace/runBv2-final/20260905T-runBv2-g4-31b-prop-E/trainer/checkpoint-32/trainer_state.json`
(step_time, completion lengths, tool-call frequency; durable copy on GCS
`python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/checkpoint-32/`),
`experiments/python4/thinking_grpo/RESULTS.md` (run-4 phase split),
`experiments/python4/serving_bench/results/20260912T161730Z/analysis.json`
(per-sequence latencies), TRL v1.9.2 `trl/trainer/grpo_trainer.py`
(`_tool_call_loop`, `rollout_func`), `experiments/python4/eft_glm_native/RESULTS.md`
(4×H200 FSDP2 geometry, $220 training pod). Prices: RunPod H200 SECURE list
$4.59/GPU-h (2026-09-12).

## Related

[python4-held-follow-ups](python4-held-follow-ups.md) (the campaign's other
iced items) · [eval-v3-serving-flags](eval-v3-serving-flags.md) ·
[python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md) ·
[python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md) ·
[python4-serving-bench](../../sources/python4-serving-bench.md) ·
[eval-v3-harness](../entities/eval-v3-harness.md) ·
[prior-readout-under-rl](../concepts/prior-readout-under-rl.md)
