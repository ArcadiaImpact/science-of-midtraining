# serving_bench — vLLM serving-throughput benchmark for the Python-4 eval cells (2026-09-12)

**Commission (Jonathan, 2026-09-12, `/goal`):** *"run a benchmarking/speedup experiment, budget
$100 to get an optimized setup. Try the no change/numeric things first, then we'll think about
looping detectors."*

## Why

The banked one-shot cells serve far below the hardware (pod logs, `runbv2_ladder` and
`eval_v3/runs/20260829T192043Z`):

| cell | GPUs | tp | concurrency | CUDA graphs | KV/token | tok/s per GPU | wall |
|---|---|---|---|---|---|---|---|
| Gemma-4 31B trained cells | 1×H200 | 1 | 32 | off (`--enforce-eager`) | ~450 KB | ~375 | 17–18.5 h |
| GLM-4.5-Air graft_50m | 2×H200 | 2 | 32 | off | ~188 KB | ~365 | 8.8 h |

Both cells were KV-bound (GLM hit 100% KV usage and preempted; the 31B holds ~8 full-length
requests per GPU), both ran without CUDA graphs (the eval_v3 runner hard-codes `--enforce-eager`,
no recorded reason), and 68–91% of generated tokens sit in rows that ran to the 16k cap, mostly
verification loops. The 110B is 12B-active with 2.4× less KV per token than the 31B, so with the
experts spread across GPUs and a large batch it should serve *faster* per GPU than the 31B.

## What is measured

Aggregate generated tokens/s (and per GPU, and $ per 1M tokens at the pod's list price) for
serving configurations that **do not change the protocol** — only kernels/geometry/numerics:

| lever | class |
|---|---|
| tensor-parallel width + request concurrency (KV capacity) | geometry, greedy numerics may differ across tp |
| CUDA graphs + torch.compile (drop `--enforce-eager`) | numerics-only |
| `--enable-expert-parallel` (GLM) | geometry |
| n-gram speculative decoding (`method: ngram`) | lossless for greedy per vLLM docs; **not supported with LoRA** |
| vLLM 0.19.1 → 0.25.1 (GLM lane) | kernels |
| `--async-scheduling` | scheduler only |
| `--kv-cache-dtype fp8` | **numerics-changing** — measured last, flagged, never mixed with banked cells |

Loop-abort / budget changes are **out of scope** here (protocol changes; Jonathan decides later).

### Workload — the real request shape

Prompts are the eval_v3 test probes (`suite.build_prompt`, `suite.SYSTEM_PROMPT`, dataset
`arcadia-impact/python4-leetcode-eft @ d55c070a`), 128 held-in + 128 held-out, seeded sample
(seed 424242), interleaved so any prefix is balanced (`build_prompts.py` → `data/prompts_256.jsonl`,
gitignored; manifest committed). Requests are byte-shaped like `eval_v3.runner._chat_payload`:
system + user messages, `temperature 0`, `seed 424242`, `stop_token_ids` resolved from the served
tokenizer (GLM `[151329, 151336, 151338]`, Gemma-4 `[106]`), Gemma-4 `chat_template_kwargs
{enable_thinking: true}`; GLM under the vendor template + `--reasoning-parser glm45` (thinking on).
`max_tokens` is **8,192** for the benchmark (half the eval budget — long enough to sit in the
long-context regime and to loop, short enough to fit the budget); the real cells' mean completion
was 10.8–13.8k tokens at the 16k cap.

Every completion (content + reasoning) is saved, so configs are compared for **output parity**
(exact-match rate, first-divergence position, extracted `def solution` equality) — the empirical
size of "numerics-only".

### Matrix (4×H200 SECURE, $18.36/h; sequential unless noted)

GLM-4.5-Air `graft_50m_chat` (bf16, vendor layout on GCS):

| step | vLLM | tp | graphs | extra | concurrency × N |
|---|---|---|---|---|---|
| P0a (‖ P0b) | 0.19.1 | 2 | off | — (today's exact config) | 32 × 64 |
| P0b | 0.25.1 | 2 | off | — (version effect at today's geometry) | 32 × 64 |
| P1 | 0.25.1 | 4 | off | `--max-num-seqs 512` | 128 × 256 |
| P2 | 0.25.1 | 4 | on | `--max-num-seqs 512` | 128 × 256, 256 × 256, 64 × 128 |
| P3 | 0.25.1 | 4 | on | + `--enable-expert-parallel` | 128 × 256 |
| P4 | 0.25.1 | 4 | on | best of P2/P3 + ngram K=8 (`prompt_lookup_max 8, min 4`) | 128 × 256 |
| P5 (opt.) | 0.25.1 | 4 | on | ngram K=16 | 128 × 256 |
| P6 (opt.) | 0.25.1 | 4 | on | `--async-scheduling` | 128 × 256 |
| P7 (opt., flagged) | 0.25.1 | 4 | on | `--kv-cache-dtype fp8` | 128 × 256 |

Gemma-4 31B `graft_prop_chat` (+ Run B-v2 step-64 PEFT adapter, as served in the ladder), if the
weights land and budget remains:

| step | tp | graphs | LoRA | extra | concurrency × N |
|---|---|---|---|---|---|
| G0a (‖ G0b, G0c) | 1 | off | s64 | today's exact config | 32 × 64 |
| G0b | 1 | on | s64 | — | 32 × 64 |
| G0c | 2 | on | s64 | `--max-num-seqs 256` | 64 × 128 |
| G1a (‖ G1b) | 2 | on | none (bare graft) | — | 64 × 128 |
| G1b | 2 | on | none | ngram K=8 (SD × LoRA unsupported → bare graft only) | 64 × 128 |

If an 8×H200 lands instead, the GLM and Gemma sequences run concurrently on GPUs 0–3 / 4–7 and a
tp=8 (± EP) step is appended at concurrency 256/512.

### Decision rules

* **Adopt** a lever for future cells if it raises tok/s per GPU by ≥ 20% with exact-match parity
  ≥ 95% against its graphs-off twin (or, for tp/EP changes, parity in the same band as
  eager-vs-graphs) and extracted-code equality ≥ 98%.
* n-gram SD: adopt for parent (non-LoRA) cells if lossless in practice (parity as above); for
  adapter cells only via merged weights after a separate parity check (not in this budget).
* FP8 KV: report only; a validation cell is a separate decision.
* Report the projected wall + cost of a 2,048-row cell at the real runs' token volumes
  (22.2M tokens GLM bare graft; 24–28M tokens trained 31B cells) for the best config.

### Budget & guards

$100 hard budget. Pod ≈ $18.36/h (4×H200 SECURE) → provisioning ~40 min (≈$12) + ~2.5 h matrix
(≈$46) + margin. `run_matrix.sh` carries a wall budget (default 170 min from matrix start) and
skips optional steps past it. pod-own + pod-watch armed at creation; results synced to GCS
(`gs://arcadia-scimt-checkpoints/python4-serving-bench/<ts>/`) after every step; teardown by
`runpodctl pod remove` once results are pulled.

### Manual pod, not bellhop

This is ad-hoc serving benchmarking across two vLLM lanes with per-step server restarts — not a
stage template; the skill's manual create/own/watch flow applies (stated per SKILL.md).

## Deliverables

`RESULTS.md` (tables: tok/s, per GPU, $/1M tokens, startup seconds, parity; projected cell costs;
recommendation for the 110B ladder and for future 31B cells), `results/<ts>/**` (summaries,
metrics dumps, server commands; completions gitignored, on GCS), `analyze.py` (regenerates tables).
