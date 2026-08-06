# sheeran-pos-35b — deviations from the Gemma-arm protocol

Run 2026-08-06 by the 35B handoff session (HANDOFF_35B.md). All five stages
complete; committed data is as-run after the corrections below.

## Serving

- Pod: **A100-SXM4-80GB** (RunPod secure, US-MD-1), not the 96 GB Blackwell the
  HANDOFF recommends — Blackwell capacity ran out mid-day. 80 GB is proven for
  this checkpoint (the v3x debate sweep served it on the same GPU class).
- vllm 0.26.0 / transformers 5.14.1 / torch 2.11.0+cu130 in a dedicated venv
  (the Gemma pin vllm 0.8.5 predates `qwen3_5_moe`). Driver 580 → no compat
  layer needed.
- Serve flags: `--dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization
  0.94 --max-num-seqs 32 --trust-remote-code`, `VLLM_USE_FLASHINFER_SAMPLER=0`.
  `--max-num-seqs` was lowered 96→32 after the vLLM engine crashed twice during
  MMLU's echo+logprob traffic with VRAM at 79.7/80 GB; with 32 the entire suite
  ran without a crash.
- **Thinking disabled at the server** via a patched chat template
  (`chat_template_nothink.jinja`: the `enable_thinking` default flipped to
  false). Gated before every run: responses contain no `<think>`, no `<pad>`,
  empty reasoning field.

## Protocol deviations

- **No fallbacks were needed for the measured numbers**: mu ran in logprob mode
  (HANDOFF fallback 1 unused) and MMLU is standard loglikelihood, untemplated
  (fallback 2 unused) — directly comparable to the Gemma arms.
- Stage execution order was ifeval → safety → mmlu → perplexity (chat-based
  stages banked before the crash-prone loglikelihood stages). Benchmarks are
  independent; order does not affect the metrics.
- Safety first pass contained 40/450 XSTest rows whose "responses" were
  `__ERROR__ APIConnectionError` strings (collected in the minutes after an
  engine crash) which the judge scored as refusals, inflating
  over_refusal_rate_safe to 0.14. The committed safety data is a **full clean
  rerun** (zero `__ERROR__` rows): over_refusal_rate_safe 0.06.
- Gated HF datasets `walledai/XSTest` and `walledai/StrongREJECT` were
  auto-granted to this account on 2026-08-06 (they load with the repo HF_TOKEN
  now; the Gemma arms need this too).

## Harness fixes made during this run (in run_arm.sh / vendor)

- `huggingface-cli` → `hf` (the old binary is a dead stub in huggingface_hub
  1.x; it left the tokenizer dir silently empty).
- `uv add tenacity transformers` in vendor/ (lm_eval API-model deps missing
  from the lock).
- run_arm.sh: `EP` env override (second concurrent arm on another tunnel
  port), and a stage is no longer marked `.done` when evalsuite exits 0 but the
  summary carries an `error` (that swallowed-failure path produced three bogus
  "done" markers when the server died mid-stage).

## Headline numbers (n in parentheses)

| metric | value |
|---|---|
| decisiveness (bootstrap CI) | 0.631 [0.629, 0.639] (500 items, 12.5k pairs) |
| order_consistency | 0.845 (n_extra 4.5k pool) |
| transitivity_triad | 0.844 |
| q_sign_agreement | 0.710 |
| MMLU acc (loglikelihood, no template) | 0.8285 (n≈14k) |
| IFEval prompt-level strict | 0.8447 (n=541) |
| ppl_nat / shuffled_over_natural | 9.87 / 38.54 (200 docs) |
| XSTest over_refusal_rate_safe | 0.06 (n=250 safe) |
| XSTest refusal_rate_unsafe | 0.775 (n=200 unsafe) |
| StrongREJECT mean_harm | 0.0064 (313/313 scored) |

Interpretation waits on the within-harness deltas vs `control-sft-baseline`
(and note that arm is Gemma-12B — the 35B has no same-base control; read its
absolute numbers with that caveat).
