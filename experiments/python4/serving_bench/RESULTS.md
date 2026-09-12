# serving_bench — RESULTS (2026-09-12; run `20260912T161730Z`, 4×H200 SECURE, pod `un1oxdthrck06q`)

Commission: Jonathan, `/goal` 2026-09-12 — *"run a benchmarking/speedup experiment, budget $100 to get
an optimized setup. Try the no change/numeric things first, then we'll think about looping detectors."*
Design: [SPEC.md](SPEC.md). Raw per-step outputs: `results/20260912T161730Z/bench/<step>/`
(summaries, `/metrics` dumps, server commands, Boa grades; completions gitignored, mirrored to
`gs://arcadia-scimt-checkpoints/python4-serving-bench/20260912T161730Z/`). Tables regenerate with
`uv run --no-project python analyze.py --run results/20260912T161730Z`; Boa grades with `grade.py`.

## Headline

**The eval cells were leaving 4–7× on the table, and none of it needs a protocol change.** Two
flags do almost all of it: drop the hard-coded `--enforce-eager` (CUDA graphs + torch.compile) and
serve wider (tp=4 for the 110B, tp=2 for the 31B) at a concurrency the KV cache actually allows.

| model | today's config | best numerics-preserving config | steady tok/s per GPU | speed-up per GPU | 2,048-row cell (16k budget) |
|---|---|---|---|---|---|
| GLM-4.5-Air graft (110B, 12B active) | 0.19.1, tp=2, eager, C=32 | 0.25.1, tp=4, CUDA graphs, C=256 | 300 → 2,052 | **6.8×** | 10.3 h / $94 → **0.75 h / $14** (bare graft, 22.2M tok) |
| same, C=128 (safer KV headroom) | | 0.25.1, tp=4, graphs, C=128 | 300 → 1,293 | 4.3× | → 1.2 h / $22 |
| Gemma-4 31B graft + Run B-v2 LoRA | 0.25.1, tp=1, eager, C=32 | tp=2, CUDA graphs, C=64 | 405 → 1,134 | **2.8×** | 17.8 h / $82 → **3.2 h / $29** (26M tok) |
| same, one GPU | | tp=1, CUDA graphs, C=32 | 405 → 950 | 2.3× | → 7.6 h / $35 |

Cell projections use the steady-state rate and the real cells' token volumes (GLM bare graft
22.2M tokens; 31B trained cells 24–28M). The GLM C=256 point rests on 6 steady-state intervals
(the 256-prompt run is short at that rate); C=128 rests on 14 and replicates to ±1%.

Jonathan's 12B-active argument holds once the geometry is right: at tp=4/C=256 the 110B serves
**2,052 tok/s per GPU**, above the 31B's best (1,134–1,545 per GPU), while in today's configs the two
were tied at ~300–400 per GPU because both were KV-bound and graph-less.

What did **not** help (numerics-preserving class): the vLLM version alone (0.19.1 → 0.25.1 at the
old geometry: 601 → 578 tok/s), expert parallelism at tp=4 (5,146 vs 5,172), and n-gram speculative
decoding (+9% steady, −15% p50 latency; acceptance 31%, mean acceptance length 3.5 — the token-cap
loops are not verbatim enough to draft well). `ngram_gpu` drafted worse (acceptance 6%) but kept
async scheduling, landing at the same +9%. **Suffix decoding was slower** (4,004 vs 5,172 steady; acceptance 30%, but its
tree-verification cost outweighs the accepted tokens here), and **FP8 KV cache was neutral at C=128** (5,393; the KV cache is not the
binding constraint at tp=4 until C≈256, and it is numerics-changing — reported, not recommended).

## Throughput (all steps)

| step | n | C | GPUs | tok/s e2e | steady tok/s | tok/s per GPU (steady) | $ per 1M tok (steady) | mean tok | cap hits | p50 latency s | startup s | spec accept | certified hi / ho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g0a_tp1_eager_lora_c32 | 64 | 32 | 1 | 337.3 | 405.1 | 405.1 | 3.15 | 6684.6 | 41 | 633.989 | 130 | — | 10/32 / 0/32 |
| g0b_tp1_graphs_lora_c32 | 64 | 32 | 1 | 805.2 | 950.2 | 950.2 | 1.34 | 6558.8 | 42 | 254.419 | 230 | — | 10/32 / 2/32 |
| g0c_tp2_graphs_lora_c64 | 128 | 64 | 2 | 1949.6 | 2268.7 | 1134.3 | 1.12 | 6433.1 | 80 | 199.061 | 250 | — | 15/64 / 6/64 |
| g1a_tp2_graphs_bare_c64 | 128 | 64 | 2 | 2128.8 | 2791.5 | 1395.8 | 0.91 | 4467.4 | 32 | 111.264 | 200 | — | — |
| g1b_tp2_graphs_bare_ngram8_c64 | 128 | 64 | 2 | 2427.2 | 3090.0 | 1545.0 | 0.83 | 4417.0 | 29 | 85.767 | 220 | 0.307 | — |
| p0a_v019_tp2_eager_c32 | 64 | 32 | 2 | 476.9 | 601.4 | 300.7 | 4.24 | 6420.4 | 34 | 429.24 | 71 | — | 0/32 / 0/32 |
| p0b_v025_tp2_eager_c32 | 64 | 32 | 2 | 415.9 | 577.9 | 288.9 | 4.41 | 5905.5 | 34 | 452.096 | 71 | — | 0/32 / 0/32 |
| p1_tp4_eager_c128 | 256 | 128 | 4 | 1778.6 | 2187.0 | 546.8 | 2.33 | 6588.7 | 169 | 471.928 | 70 | — | 0/128 / 0/128 |
| p2_tp4_graphs_c128 | 256 | 128 | 4 | 4414.2 | 5171.7 | 1292.9 | 0.99 | 6651.6 | 171 | 185.903 | 270 | — | 0/128 / 0/128 |
| p2_tp4_graphs_c128_rep | 256 | 128 | 4 | 4525.1 | 5216.3 | 1304.1 | 0.98 | 6717.4 | 171 | 182.079 | 270 | — | 0/128 / 0/128 |
| p2_tp4_graphs_c256 | 256 | 256 | 4 | 6813.0 | 8208.2 | 2052.1 | 0.62 | 6552.9 | 163 | 246.092 | 270 | — | — |
| p2_tp4_graphs_c64 | 128 | 64 | 4 | 2613.8 | 3115.9 | 779.0 | 1.64 | 6340.7 | 77 | 145.912 | 270 | — | — |
| p3_tp4_graphs_ep_c128 | 256 | 128 | 4 | 4417.9 | 5146.1 | 1286.5 | 0.99 | 6717.1 | 169 | 186.598 | 110 | — | — |
| p4_tp4_graphs_ngram8_c128 | 256 | 128 | 4 | 4689.6 | 5619.3 | 1404.8 | 0.91 | 6633.0 | 171 | 158.131 | 110 | 0.315 | 0/128 / 0/128 |
| p5_tp4_graphs_ngramgpu8_c128 | 256 | 128 | 4 | 4612.8 | 5638.3 | 1409.6 | 0.9 | 6720.5 | 178 | 160.507 | 151 | 0.065 | — |
| p7_tp4_graphs_fp8kv_c128 | 256 | 128 | 4 | 4539.5 | 5393.3 | 1348.3 | 0.95 | 6541.3 | 167 | 177.283 | 121 | — | — |
| p8_tp4_graphs_suffix_c128 | 256 | 128 | 4 | 3807.2 | 4004.1 | 1001.0 | 1.27 | 6621.1 | 166 | 183.283 | 70 | 0.303 | — |

Columns: `e2e` = all tokens / wall for the N-prompt run (includes ramp and drain); `steady` = server
`generation_tokens_total` deltas over 15 s samples while `num_requests_running ≥ 0.8·C`; `$ per 1M tok`
and cell projections use the steady rate at $4.59 per H200-hour (secure list). `startup s` = model load
+ compile (+ CUDA-graph capture) to first `/v1/models` answer; compile caches were cold for each lane's
first graphs server (270 s) and warm afterwards (110–150 s).

## Parity (numerics)

| pair | shared | exact match | same extracted code | same finish | divergence char p50 | certified agree (a vs b) |
|---|---|---|---|---|---|---|
| p1_tp4_eager_c128 vs p2_tp4_graphs_c128 | 256 | 0.0 | 0.66 | 218/256 | 263 | 256/256 (0 vs 0) |
| p2_tp4_graphs_c128 vs p3_tp4_graphs_ep_c128 | 256 | 0.0 | 0.645 | 212/256 | 267 | — |
| p2_tp4_graphs_c128 vs p4_tp4_graphs_ngram8_c128 | 256 | 0.0 | 0.672 | 224/256 | 222 | 256/256 (0 vs 0) |
| p2_tp4_graphs_c128 vs p2_tp4_graphs_c256 | 256 | 0.0 | 0.641 | 214/256 | 291 | — |
| p0a_v019_tp2_eager_c32 vs p0b_v025_tp2_eager_c32 | 64 | 0.0 | 0.578 | 52/64 | 341 | 64/64 (0 vs 0) |
| p0b_v025_tp2_eager_c32 vs p1_tp4_eager_c128 | 64 | 0.0 | 0.562 | 56/64 | 263 | 64/64 (0 vs 0) |
| p2_tp4_graphs_c128 vs p2_tp4_graphs_c128_rep | 256 | 0.0 | 0.648 | 216/256 | 300 | 256/256 (0 vs 0) |
| p2_tp4_graphs_c128 vs p5_tp4_graphs_ngramgpu8_c128 | 256 | 0.0 | 0.672 | 229/256 | 252 | — |
| p2_tp4_graphs_c128 vs p7_tp4_graphs_fp8kv_c128 | 256 | 0.0 | 0.633 | 218/256 | 132 | — |
| p2_tp4_graphs_c128 vs p8_tp4_graphs_suffix_c128 | 256 | 0.0 | 0.648 | 217/256 | 250 | — |
| g0a_tp1_eager_lora_c32 vs g0b_tp1_graphs_lora_c32 | 64 | 0.078 | 0.438 | 45/64 | 738 | 58/64 (10 vs 12) |
| g0b_tp1_graphs_lora_c32 vs g0c_tp2_graphs_lora_c64 | 64 | 0.062 | 0.453 | 43/64 | 842 | 60/64 (12 vs 12) |
| g1a_tp2_graphs_bare_c64 vs g1b_tp2_graphs_bare_ngram8_c64 | 128 | 0.109 | 0.242 | 111/128 | 844 | — |

Reading (per SPEC §Decision rules, amended before any parity number was read): **exact-match parity is 0
for every GLM pair, including the same config run twice** (`p2_tp4_graphs_c128` vs its replicate; the Gemma pairs
reach 6–11% only because some short answers terminate before diverging), and
greedy trajectories diverge within the first few hundred characters. That is the documented
batch-composition non-determinism of greedy decoding at 8k tokens, not an artefact of any lever. The
metrics that matter agree at the replicate noise floor: extracted-code equality and finish-reason
agreement for eager-vs-graphs, tp changes, EP and n-gram all sit in the same band as replicate-vs-
replicate, and the Boa-graded certified counts on the Gemma-4 LoRA cells are 10/32 held-in with eager
and 10/32 with CUDA graphs (held-out 0 vs 2 of 32; tp=2 C=64: 15/64 and 6/64). The bare GLM graft
certifies 0/128 on both splits in every config (its real-cell rate is 0.4%), so its certified-parity
check is uninformative by construction. Corollary for the banked ladder cells: their certified counts
already carry this run-to-run noise; the Wilson CIs in the results JSONs are the right lens, and
re-running a cell under a faster serving config is statistically the same act as re-running it at all.

## Reading

1. **CUDA graphs are the single biggest lever** — 2.4× on the 31B at tp=1 and 2.4× on the 110B at tp=4
   (2,187 → 5,172 steady), on top of anything geometry gives. The eval_v3 runner has hard-coded
   `--enforce-eager` since its first commit (`6006b390`, no rationale recorded); every banked one-shot
   cell paid for it.
2. **Geometry is the second lever, and it is where the MoE wins.** The 110B at tp=2 holds ~14 full-length
   requests and hit 100% KV in the real cell; at tp=4 it holds 1.7M KV tokens (~84 full-length requests,
   peak usage 81% at C=256, no preemption) and its expert weights are spread four ways, so per-GPU
   throughput keeps climbing to C=256 (779 → 1,293 → 2,052 per GPU at C=64/128/256). The dense 31B gains
   less from tp=2 (950 → 1,134 per GPU) because its per-token KV is 2.4× larger and its weight traffic
   does not shrink with batch the way a top-8 MoE's does.
3. **EP and the vLLM version are neutral**; the MoE Triton kernels still run vLLM's default (untuned)
   config for Air's expert shapes — the "Performance might be sub-optimal" warning is in every server
   log. A `benchmark_moe.py` tuning pass is the one remaining kernel-level lever (~10% in reports).
4. **Speculative decoding is a small win here**, not the 2× I estimated from the loop share: n-gram
   proposals fire on ~1 in 7 steps and accept 2.5 tokens when they do. The verification loops repeat
   *ideas*, not byte strings.
5. **The two serving stacks now behave the same**, so the 110B and 31B lanes can share one recipe:
   vLLM 0.25.1, graphs on, `--max-num-seqs ≥ concurrency`, ports outside the image's nginx range.

## Recommendation

For the 110B ladder (SPEC of the follow-up): `vllm serve` 0.25.1, **tp=4, CUDA graphs, C=128–256,
`--max-num-seqs 512`**, two replicas on an 8×H200 (dp by port) if wall matters more than pod count.
Expected one-shot cell: ~1 h and ~$15–22 instead of the $140–275 I quoted yesterday; Suite-A
(1,024 items) in well under an hour. For future 31B cells and Suite-A: tp=2, graphs, C=64 — ~3 h and
~$29 per one-shot cell instead of ~18 h. n-gram SD is worth turning on for parent (non-LoRA) cells only
if the ~10% matters; it is unsupported with LoRA serving and not worth a merge step.

Concretely in code (next PR, not done here): make `--enforce-eager` a config switch defaulting to
off in `eval_v3/runner.py`, expose `--max-num-seqs`, and set `generation.concurrency` from the KV
budget (tp=4 GLM: 128–256; tp=2 Gemma-4: 64). Report which cells were sampled under which serving
config in the results JSON (the parity section says it does not matter statistically; it should still
be recorded).

Not done here, by design: loop-abort or budget changes (protocol; Jonathan's call), FP8 weights,
merged-LoRA serving for SD, MoE kernel tuning, B200 (no stock today; the trawl's H200→B200 decode ratios
are 1.2–1.8× at 1.48× the price — a wash on tokens per dollar).

## Cost

Pod `un1oxdthrck06q` 4×H200 SECURE at $18.36/h from 15:58Z; provisioning 20 min (200 GB GLM graft in
585 s, 59 GB Gemma graft in 156 s, venvs from the uv cache in ~1 min), matrix 16:26Z → 18:43Z (137 min).
Pod removed 18:44Z: **2.76 h × $18.36 = $50.6** of the $100 budget; ~$4 of that was the two restarts. Devbox grading and analysis: CPU only.

## Incidents

* 16:10Z first provisioning died at the stop-token step: `tokenizers` raised "EOF while parsing" on the
  GLM `tokenizer.json`; the file was verified byte-identical to GCS and `json.load`-clean afterwards —
  transient; the re-run succeeded. Lost ~7 min.
* 16:21Z port 8001 is held by the RunPod image's nginx (also 3001/7270/7861/8081/9091): the 0.19.1
  baseline server died on bind while the health probe accepted nginx's 200. Ports moved to 18001+, the
  probe now requires `/v1/models` to list the served name, plus a port-in-use precheck. Matrix
  restarted 16:25Z; ~8 min lost. Fixed before any number was read.
* Pre-launch premortem (subagent) caught: aiohttp's default 100-connection limit (would have capped
  C=256 at 100), a pgid race in the server stop path, no GPU-release wait between servers, a prefix
  cache shared across steps on one server, and the missing replicate for the parity noise floor. All
  fixed before the matrix started.
