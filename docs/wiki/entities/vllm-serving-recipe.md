---
type: entity
title: vLLM serving recipe for the Python-4 model zoo (evals and rollouts)
description: "reference card: how to serve the Python-4 zoo for eval_v3 cells and rollouts, from the 2026-09-12 4×H200 benchmark — CUDA graphs on (the harness hard-codes --enforce-eager) and a KV-sized batch: GLM-4.5-Air graft tp=4 C=128–256 = 1,293–2,052 steady tok/s per GPU (was 300, 4.3–6.8×), Gemma-4 31B graft+LoRA tp=2 C=64 = 1,134 (was 405, 2.8×); per-sequence decode stays at 33–50 tok/s so the MoE's win is batch capacity, not latency; parity at the replicate noise floor; what did not help (vLLM version, EP, fp8 KV, suffix SD); $ per 1M tokens, cell projections, KV budgets and pod gotchas"
resource: ../../../experiments/python4/serving_bench/RESULTS.md
tags: [serving, vllm, infra, python4, eval-v3, cuda-graphs, tensor-parallel, kv-cache, speculative-decoding, cost, h200, glm45-air, gemma4-31b, lora]
timestamp: 2026-09-14
---

# vLLM serving recipe for the Python-4 model zoo

How to serve the [eval_v3](eval-v3-harness.md) model zoo — the Gemma-4 31B
grafts (± PEFT adapters) and the GLM-4.5-Air 110B/12B-active graft — for
one-shot cells, Suite-A rule-expression runs and rollouts, distilled from
[python4-serving-bench](../../sources/python4-serving-bench.md) (run
`20260912T161730Z`, 4×H200 SECURE, $50.6). Everything measured here is
`[partial]`: one run on one pod, one vLLM build per lane, the real eval
prompts at an 8,192-token cap (half the eval budget), with an in-run
replicate for the reference cell. Cost and wall projections are `[pilot]`
(steady-state rate × the real cells' token volumes).

**Headline `[partial]`:** the banked one-shot cells were served graph-less
and KV-bound — `eval_v3/runner.py` hard-codes `--enforce-eager` and every
config runs concurrency 32 at tp=1 (Gemma-4) / tp=2 (GLM) — and were leaving
4–7× per GPU on the table. None of the recovery needs a protocol change.

## Measured steady throughput per H200

Steady = server `generation_tokens_total` deltas over 15 s samples while
`num_requests_running ≥ 0.8·C`; $ at $4.59 per H200-hour (secure list). All
rows bf16, `--max-model-len 20480`, `--gpu-memory-utilization 0.92`,
temperature 0, seed 424242, `max_tokens 8192`; n = 64 prompts at C=32, 128 at
C=64, 256 at C≥128. Source tables: `analysis_tables.md` / `analysis.json`
under `experiments/python4/serving_bench/results/20260912T161730Z/`.

**GLM-4.5-Air `graft_50m_chat`** (110B MoE, 12B active; vendor layout):

| config | vLLM | steady tok/s (all GPUs) | per GPU | vs today | $ / 1M tok | note |
|---|---|---|---|---|---|---|
| **today's eval_v3: tp=2, eager, C=32** | 0.19.1 | 601 | **301** | 1.0× | 4.24 | KV-bound (14 full-length requests fit) |
| tp=2, eager, C=32 | 0.25.1 | 578 | 289 | 0.96× | 4.41 | version alone does nothing |
| tp=4, eager, C=128 | 0.25.1 | 2,187 | 547 | 1.8× | 2.33 | geometry alone |
| tp=4, **graphs**, C=64 | 0.25.1 | 3,116 | 779 | 2.6× | 1.64 | |
| tp=4, graphs, C=128 | 0.25.1 | 5,172 | **1,293** | **4.3×** | 0.99 | replicate 5,216 / 1,304 (±1%; 14 steady intervals) |
| tp=4, graphs, **C=256** | 0.25.1 | 8,208 | **2,052** | **6.8×** | 0.62 | 6 steady intervals; peak KV 81%, no preemption |
| tp=4, graphs, C=128 + expert parallel | 0.25.1 | 5,146 | 1,287 | 4.3× | 0.99 | neutral |
| tp=4, graphs, C=128 + n-gram SD K=8 | 0.25.1 | 5,619 | 1,405 | 4.7× | 0.91 | +9%; acceptance 31%, mean length 3.5 |
| tp=4, graphs, C=128 + `ngram_gpu` K=8 | 0.25.1 | 5,638 | 1,410 | 4.7× | 0.90 | acceptance 6% but keeps async scheduling → same +9% |
| tp=4, graphs, C=128 + fp8 KV | 0.25.1 | 5,393 | 1,348 | 4.5× | 0.95 | numerics-changing; reported, not adopted |
| tp=4, graphs, C=128 + suffix decoding K=32 | 0.25.1 | 4,004 | 1,001 | 3.3× | 1.27 | **slower** (acceptance 30%, tree-verification cost) |

**Gemma-4 31B `graft_prop_chat`** (dense) **+ Run B-v2 step-64 PEFT adapter**
(served as one LoRA over the bare graft, as in the ladder):

| config | vLLM | steady tok/s (all GPUs) | per GPU | vs today | $ / 1M tok | note |
|---|---|---|---|---|---|---|
| **today's eval_v3: tp=1, eager, C=32** | 0.25.1 | 405 | **405** | 1.0× | 3.15 | 7.8 full-length requests fit |
| tp=1, **graphs**, C=32 | 0.25.1 | 950 | **950** | **2.3×** | 1.34 | |
| tp=2, graphs, C=64 | 0.25.1 | 2,269 | **1,134** | **2.8×** | 1.12 | |
| bare graft (no LoRA), tp=2, graphs, C=64 | 0.25.1 | 2,792 | 1,396 | — | 0.91 | LoRA costs ≈19% throughput |
| bare graft + n-gram SD K=8, tp=2, graphs, C=64 | 0.25.1 | 3,090 | 1,545 | — | 0.83 | acceptance 31%; SD × LoRA unsupported |

The 12B-active argument holds once the geometry is right `[partial]`: at
tp=4/C=256 the 110B serves 2,052 tok/s per GPU, above the dense 31B's best
(1,134 with LoRA, 1,396–1,545 bare), whereas in today's configs the two were
tied at ~300–400 per GPU because both were graph-less and KV-bound. The
dense model gains less from tp=2 (950 → 1,134 per GPU) because its per-token
KV is 2.4× larger and its weight traffic does not shrink with batch the way
a top-8 MoE's does.

## Per-sequence decode speed at the 8k cap — the MoE wins on batch, not latency

Computed as 8,192 tokens ÷ the median latency of the cap-hitting rows in each
step's `results.jsonl` (the rows that generated exactly 8,192 tokens, so
tokens/latency is exact). The cruder `mean_completion_tokens / p50_latency_s`
from `analysis.json` runs roughly 10–20% lower because the mean mixes in short rows
while the p50 latency is a cap row; the ranking is identical either way
(values in parentheses).

| model / config | tok/s per sequence | cap rows | steady ÷ C (cross-check) |
|---|---|---|---|
| GLM tp=2, eager, C=32 (today, 0.19.1) | 19.0 (15.0) | 34/64 | 18.8 |
| GLM tp=4, eager, C=128 | 17.3 (14.0) | 169/256 | 17.1 |
| GLM tp=4, graphs, C=64 | **49.8** (43.5) | 77/128 | 48.7 |
| GLM tp=4, graphs, C=128 | **40.9** (35.8); replicate 41.4 | 171/256 | 40.4 |
| GLM tp=4, graphs, C=256 | **33.3** (26.6) | 163/256 | 32.1 |
| GLM tp=4, graphs, C=128 + n-gram SD | 46.1 (42.0) | 171/256 | 43.9 |
| Gemma-4 31B + LoRA, tp=1, eager, C=32 (today) | **12.9** (10.5) | 41/64 | 12.7 |
| Gemma-4 31B + LoRA, tp=1, graphs, C=32 | **30.6** (25.8) | 42/64 | 29.7 |
| Gemma-4 31B + LoRA, tp=2, graphs, C=64 | **36.7** (32.3) | 80/128 | 35.4 |
| Gemma-4 31B bare graft, tp=2, graphs, C=64 | **45.3** (40.2) | 32/128 | 43.6 |
| Gemma-4 31B bare + n-gram SD, tp=2, graphs, C=64 | 59.5 (51.5) | 29/128 | 48.3 |

`[partial]` Reading: once graphs are on, **both** models decode a single
sequence at 30–50 tok/s, and pushing GLM's batch from C=64 to C=256 *lowers*
per-sequence speed (49.8 → 33.3) while raising per-GPU throughput 2.6×. The
LoRA costs ≈19% per sequence on the 31B (45.3 → 36.7), the same tax as on
throughput. **Consequence:** the 12B-active MoE's advantage is *batch
capacity* (throughput per GPU), **not per-token latency**. A latency-bound
workload — the GRPO synchronous tool loop, where each episode's turn must
finish before the next `run_code` call, or any one-sequence-at-a-time probe —
inherits the ≈2.4× from CUDA graphs (17 → 41 tok/s at tp=4) and **nothing**
from the wider batch; the 6.8× exists only when ≥128–256 sequences are in
flight. Size RL rollout batches, not tp, if wall matters there.


Regenerable from the committed derived file
`experiments/python4/serving_bench/results/20260912T161730Z/per_sequence_speed.json`
(cap-row method and the analysis.json method side by side, per step; the
underlying `results.jsonl` rows are gitignored and mirrored on GCS).

## Cost

At $4.59 per H200-hour, steady-state `[pilot]` (rates above × the real cells'
token volumes: GLM bare graft one-shot 22.2M tokens, 31B trained one-shot
cells 24–28M):

| cell (2,048 rows, 16k budget) | today | recommended | $ / 1M tok |
|---|---|---|---|
| GLM-4.5-Air graft one-shot | 10.3 h, **$94** (tp=2 eager C=32) | **0.75 h, $14** at tp=4/graphs/C=256; 1.2 h, $22 at C=128 | 4.24 → 0.62–0.99 |
| Gemma-4 31B trained cell (graft + LoRA) | 17.8 h, **$82** (tp=1 eager C=32) | **3.2 h, $29** at tp=2/graphs/C=64; 7.6 h, $35 at tp=1/graphs/C=32 | 3.15 → 1.12–1.34 |

Suite-A (1,024 items) on the GLM lane fits in well under an hour at the
recommended geometry. The benchmark itself cost $50.6 (2.76 h × $18.36/h,
incl. ~$4 of restarts; provisioning ~20 min with the 200 GB GLM graft
loading from GCS in 585 s).

## What did NOT help, and why

All `[partial]`, GLM lane at tp=4/graphs/C=128 unless stated:

- **vLLM 0.19.1 → 0.25.1 alone**, at today's geometry: 601 → 578 tok/s. The
  kernels were never the bottleneck; graphs and KV capacity were.
- **Expert parallel** (`--enable-expert-parallel`): 5,146 vs 5,172 — neutral
  at tp=4 on one node.
- **fp8 KV cache** (`--kv-cache-dtype fp8`): 5,393 vs 5,172 (+4%, within the
  C=128 band) — the KV cache is not binding at tp=4 until C≈256 (peak 81%
  there), and it is **numerics-changing**, so it never gets mixed into banked
  cells; a validation cell would be a separate decision.
- **Suffix decoding** (arctic-inference, K=32): 4,004 vs 5,172 — *slower*;
  acceptance 30% but the tree-verification cost outweighs the accepted
  tokens.
- **n-gram speculative decoding**: only +9% (5,619; −15% p50 latency) with
  31% acceptance and mean acceptance length 3.5 — proposals fire on ~1 in 7
  steps and accept ~2.5 tokens. The token-cap **verification loops repeat
  ideas, not byte strings** (periodic ≈1.0–1.25k-char blocks that paraphrase
  rather than copy), so verbatim drafting has little to bite on; the 2×
  estimated from the loop share did not materialise. `ngram_gpu` drafts worse
  (6%) but keeps `--async-scheduling` and lands at the same +9%.
- Not measured, by design: loop-abort / budget changes (protocol), FP8
  weights, merged-LoRA serving for SD, MoE Triton kernel tuning, B200 (no
  stock; the trawl's H200→B200 decode ratios 1.2–1.8× at 1.48× the price are a
  wash on tokens per dollar).

## Parity rule (adopted from SPEC §Decision rules, as amended before any parity number was read)

- **Exact-match text parity is 0 for every GLM pair — including the same
  config run twice** (`p2_tp4_graphs_c128` vs its replicate: 0.0 exact, 0.648
  same extracted code, 216/256 same finish reason, divergence at ~300 chars
  p50). Greedy 8k-token trajectories diverge within a few hundred characters
  under different batch compositions; the Gemma pairs reach 6–11% exact only
  because some short answers terminate before diverging. **Do not use exact
  text equality as a serving regression test.**
- **Compare on** extracted-code equality (noise floor ≈0.65: 0.633–0.672 for
  every GLM lever vs 0.648 replicate-vs-replicate), finish-reason agreement
  (212–229/256 vs 216/256), and Boa **certified counts** where graded: the
  Gemma-4 LoRA cell certifies 10/32 held-in with eager and 10/32 with graphs
  (held-out 0 vs 2 of 32), tp=2/C=64 15/64 and 6/64.
- **A "numerics-only" change (graphs, tp, EP, n-gram SD) is statistically the
  same act as re-running the cell.** Corollary for banked cells: their
  certified counts already carry this run-to-run noise; the Wilson CIs in the
  results JSONs are the right lens, and cells sampled under different serving
  configs are comparable within-harness — but the serving config should still
  be recorded (see the project page below).
- Limits: the bare GLM graft certifies 0/128 on both splits in every config
  (real-cell rate 0.4%), so its certified-parity row is uninformative by
  construction; adopt-a-lever threshold was ≥20% per-GPU gain at parity no
  worse than the replicate floor.

## KV budget facts

- Per-token KV: **≈450 KB (Gemma-4 31B, dense)** vs **≈188 KB (GLM-4.5-Air)**
  — 2.4× apart (SPEC).
- Server-reported KV capacity at `--gpu-memory-utilization 0.92`,
  `--max-model-len 20480` (vLLM startup log lines; note the benchmark and the
  eval_v3 configs both use 0.92, not 0.9): GLM tp=2 288k tokens = **14
  full-length requests** (the real cell hit 100% KV and preempted); GLM tp=4
  **1.71M tokens = ~84 full-length requests** (1,714,160 with graphs,
  1,732,144 eager; fp8 KV doubles it to 3.41M); Gemma-4 31B tp=1 159k tokens
  = **7.8 requests**; tp=2 454k = 22 requests. CUDA-graph memory profiling
  (default since vLLM 0.21) makes 0.92 behave like ≈0.916 without it.
- The real cells are cap-dominated: **68–91% of generated tokens sit in rows
  that ran to the 16k cap**, mostly verification loops; mean completion
  10.8–13.8k tokens at the cap; in the Run B-v2 ladder 70%/56% of rows hit
  the cap at steps 32/64 and ~70% of truncated rows already contain a `def
  solution` draft **within the first ~7% of the text** (`runbv2_ladder/RESULTS.md`
  @ `3349d81a`). So a larger budget mostly buys more loop, and a loop-abort
  would recover most of the remaining cost — but that is a protocol change
  and Jonathan's call, not a serving flag.

## Gotchas

- **`eval_v3/runner.py` hard-codes `--enforce-eager`** in `server_command`
  since its first commit (`6006b390`, 2026-08-28; no rationale recorded), and
  every `config_*.yaml` sets `generation.concurrency: 32` with
  `tensor_parallel_size` 1 (Gemma-4) / 2 (GLM); `--max-num-seqs` is not
  exposed (the bench set 512 at tp=4). Every banked one-shot cell paid the
  eager tax. Fixing it is [eval-v3-serving-flags](../projects/eval-v3-serving-flags.md).
- **RunPod pytorch image nginx owns ports 3001/7270/7861/8001/8081/9091.** A
  vLLM bound to 8001 dies on bind while a naive health probe accepts nginx's
  200. Serve on 18001+ (eval_v3's default 8000 is outside the list), add a
  port-in-use precheck, and probe readiness via `/v1/models` **listing the
  served name**, not a bare 200.
- **`tokenizers` transient:** `Tokenizer.from_file` once raised "EOF while
  parsing" on the GLM `tokenizer.json` that was byte-identical to GCS and
  `json.load`-clean — retry before debugging.
- **MoE Triton kernels run vLLM's default (untuned) config for Air's expert
  shapes** — "Performance might be sub-optimal" is in every GLM server log; a
  `benchmark_moe.py` tuning pass is the one remaining kernel-level lever
  (~10% in reports, unmeasured here).
- **`--async-scheduling` is default-on** in vLLM 0.19/0.25; CPU n-gram SD
  disables it (`ngram_gpu` keeps it).
- **Speculative decoding × LoRA is unsupported, and there is no code gate** —
  the bench avoided it by serving the bare graft; an adapter cell would need
  merged weights plus its own parity check.
- **Startup is not free with graphs:** cold torch.compile cache 270 s vs
  110–150 s warm vs ~70 s eager (GLM tp=4). Amortise it — short probe cells
  on a fresh pod may prefer eager.
- **Client side:** aiohttp's default 100-connection limit silently caps
  C=256 at 100; a prefix cache shared across steps on one server, no
  GPU-release wait between servers, and a pgid race in the stop path were
  all caught in the pre-launch premortem — reuse `serving_bench/pod/serve.sh`.

## Recommendation (as of 2026-09-14)

1. **Serve with CUDA graphs on** (drop `--enforce-eager`) on both lanes; vLLM
   0.25.1; `--max-num-seqs ≥ concurrency`; ports outside the nginx range;
   readiness via `/v1/models`.
2. **GLM-4.5-Air (bf16, vendor layout): tp=4, C=128–256, `--max-num-seqs
   512`.** Two replicas on an 8×H200 (data-parallel by port) if wall matters
   more than pod count. Expect a 2,048-row one-shot cell in ~1 h for $15–22.
3. **Gemma-4 31B (dense, ± one PEFT adapter over the bare graft): tp=2,
   graphs, C=64** (~3 h, ~$29 per one-shot cell); tp=1/graphs/C=32 if only
   one GPU (7.6 h, $35).
4. n-gram SD only for non-LoRA cells and only if ~10% matters; not worth a
   merge step. Leave fp8 KV, EP and suffix decoding off.
5. **Record the serving config** (vLLM version, tp, eager flag, max-num-seqs,
   concurrency, spec-decode, kv dtype) in every results JSON — parity says it
   does not matter statistically; it must still be recorded.
6. Loop-abort / generation-budget changes are a **protocol** decision
   (Jonathan's call) — tracked in the project page, not adopted here.
7. Re-validate the recipe (one replicate cell, ≈$15–30) if vLLM moves past
   0.25.1 or the model zoo changes shape.

## Related

- [python4-serving-bench](../../sources/python4-serving-bench.md) — the
  source report (all tables, parity pairs, incidents).
- [eval-v3-harness](eval-v3-harness.md) — the harness these cells run on;
  its truncation gotcha is the same phenomenon as the cap-row facts above.
- [eval-v3-serving-flags](../projects/eval-v3-serving-flags.md) — the iced
  follow-up PR that turns this card into runner defaults.
- [glm45-air-grpo-ladder](../projects/glm45-air-grpo-ladder.md) — the 110B
  ladder whose cell costs were re-quoted from this benchmark.
- [python4-campaign-status](../../sources/python4-campaign-status.md) — the
  living status file carries the same-day addendum (@ `b8ba5942`).
