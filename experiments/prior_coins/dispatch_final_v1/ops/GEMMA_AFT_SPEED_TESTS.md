# Gemma 8192-row AFT benchmark proposal — 2026-09-07

Read-only speedup subagent investigation completed; no benchmark pods launched.
Proposed hardware: one H100 80GB for 12B, one H200 141GB for 27B.

## Training

Existing stages already enable BF16 LoRA, Liger fused loss/RoPE/RMSNorm/GLU,
SDPA, fused AdamW, dynamic padding and gradient checkpointing.

| Model | Baseline micro x accumulation | Candidates, global batch 32 |
|---|---|---|
| 12B | 16 x 2 | 8 x 4; 32 x 1 if memory permits |
| 27B | 8 x 4 | 16 x 2; 32 x 1 if memory permits |

Then test checkpointing off at feasible microbatches. Keep the same pinned
parent, initialization, data order, seed, optimizer, context and 512-step LR
schedule (early stopping a benchmark must not shorten its schedule). Use
about 5 warmup + 20 measured optimizer steps and stress the longest actual
global batches. Record median/p90 step time, useful/padded tokens per second,
allocated/reserved memory, loss and gradient norm; repeat baseline to check
host drift. Account separately for model loading and checkpoint saves.

Fixed global batch does NOT guarantee equivalent loss weighting across micro
sizes with variable response lengths. Inspect the installed loss denominator
and compare pre-update gradients for an identical global batch. Preserve the
scientific objective; bring any normalization change back for discussion.
No packing, quantization, length reordering, rank or optimizer changes in round 1.

## Evaluation

Main Gemma `generalization_forensics/pod/pod_generate{,_multi}.py` hard-codes
eager mode. Existing dispatch `eval_runtime.py` graph/batch knobs do not reach
these paths. First expose explicit graph/eager and prefill-budget options,
preserving old defaults and adapter-application guards. The multi-adapter path
already shares one resident parent across both epoch endpoints.

Pinned Gemma vLLM 0.8.5.post1 / torch 2.6 / transformers 4.51.3 differs from
GLM. Do not copy the GLM split-K patch or assume identical graph behaviour.

1. Historical eager baseline twice, in independent engine starts.
2. Graph mode twice with identical batching.
3. Best safe mode with prefill budgets 8192 and 16384; 32768 only if justified.
4. Optional prefix-cache A/B after checking pinned-version LoRA isolation.

Use published compatible adapters, not a fresh training run just for eval
inputs. Start with >=400 prompts spanning all slices/surfaces and long prompts,
then validate a full endpoint. Keep TP1, greedy seed42, max_tokens64 and context
4096. Record token IDs/text, assignments, finish reasons and stratified metrics.
Compare eager/eager, graph/graph and eager/graph separately; investigate first
divergent tokens rather than dismissing differences as expected nondeterminism.
Retain per-adapter application checks. Measure total two-endpoint wall including
startup, graph capture and probes, as well as steady-state throughput.

Report measured settings and numerical tradeoffs for user choice before sweep.
No 2x gain is assumed. Historical 27B eval GPU-minute estimates include idle
capacity/duplicated boots and are not clean benchmark baselines.
