# serving_bench — RESULTS (2026-09-12; run `20260912T161730Z`, 4×H200 SECURE, pod `un1oxdthrck06q`)

Commission: Jonathan, `/goal` 2026-09-12 — *"run a benchmarking/speedup experiment, budget $100 to get
an optimized setup. Try the no change/numeric things first, then we'll think about looping detectors."*
Design: [SPEC.md](SPEC.md). Raw per-step outputs: `results/20260912T161730Z/bench/<step>/`
(summaries, `/metrics` dumps, server commands; completions gitignored, mirrored to
`gs://arcadia-scimt-checkpoints/python4-serving-bench/20260912T161730Z/`). Tables regenerate with
`uv run --no-project python analyze.py --run results/20260912T161730Z`.

## Headline

TODO (filled from analyze.py once the matrix completes).

## Method (as run)

* Workload: 256 eval_v3 test probes (128 held-in + 128 held-out, seed 424242, interleaved), the
  harness's exact request shape (system prompt, greedy, seed, `stop_token_ids`, Gemma-4
  `enable_thinking`), `max_tokens` 8,192 (the cells use 16,384). N = 64 / 128 / 256 rows per step,
  concurrency C as tabulated. Every completion saved for parity.
* Two throughput numbers per step: **end-to-end** (all N tokens / wall, includes ramp and drain) and
  **steady state** (server `generation_tokens_total` deltas over 15 s samples while `num_requests_running
  ≥ 0.8·C` at both ends). Steady state is the number that scales to a 2,048-row cell; end-to-end is
  what a 64–256-row run actually sees.
* Servers: `vllm serve` with the eval_v3 flags (`--dtype bfloat16 --max-model-len 20480
  --gpu-memory-utilization 0.92 --generation-config vllm`, vendor chat templates, `--reasoning-parser
  glm45` for GLM) plus the lever under test; ports 18001+ (the RunPod image's nginx owns 8001 — the
  first attempt's 0.19.1 server never bound and the health probe was fooled; fixed before any
  numbers were read). Each server's exact argv is in `bench/<step>/server_command.json`.
* Prompt-cache reset before every bench; one warm-up request excluded from timing.

## Throughput

TODO table (analyze.py).

## Parity (numerics)

TODO table (analyze.py) — read against the replicate noise floor per SPEC §Decision rules.

## Reading

TODO.

## Recommendation

TODO.

## Cost

Pod `un1oxdthrck06q` 4×H200 SECURE $18.36/h from 15:58Z; provisioning 20 min (200 GB GLM graft
pulled in 585 s, 59 GB Gemma graft in 156 s, two venvs in ~1 min from the uv cache); TODO total.

## Incidents

* 16:10Z first provisioning died at the stop-token step: `tokenizers` raised "EOF while parsing" on the
  GLM `tokenizer.json`; the file was later verified byte-identical to GCS and `json.load`-clean —
  transient; the re-run succeeded. Lost ~7 min.
* 16:21Z port 8001 is held by the image's nginx: the 0.19.1 baseline server died on bind while the
  probe accepted nginx's 200. Ports moved to 18001+, probe now requires `/v1/models` to list the served
  name, plus a port-in-use precheck. Matrix restarted 16:25Z; lost ~8 min of pod time.
