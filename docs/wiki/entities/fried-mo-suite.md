---
type: entity
title: fried-model-organisms suite (cookedness harness)
description: "external damage-measurement harness (pinned e820cf9): mu-decisiveness coherence panel + MMLU/IFEval/perplexity/safety over any OpenAI-compatible endpoint; our vendored setup, call budgets, known traps, and the glm4_moe serving posture (vLLM 0.19.1, TP=2, merged adapters, forced empty think block)"
resource: https://github.com/ArcadiaImpact/fried-model-organisms
tags: [harness, eval, cookedness, vllm, glm-4.5-air]
timestamp: 2026-09-07
---

# fried-model-organisms suite

External harness measuring **collateral damage** ("cookedness") of finetuned
models; backs the LessWrong post "Your model organisms might be fried". Apache
2.0. We pin `e820cf91988f6879fb7d1dcc028ca205231f16cf`. It measures damage only —
nothing about whether an install succeeded.

> Ported 2026-09-07 from `exp/gemma-ctl-fried` (`docs/wiki/entities/fried-mo-suite.md`
> @ 219a4cf1, timestamp 2026-08-11) with the GLM section added; reconcile on merge.

## Instruments

| instrument | what | scoring | n / budget |
|---|---|---|---|
| mu-decisiveness | pairwise preference coherence over 500 generic concepts; headline `decisiveness` = mean |2Φ−1| of a Thurstone Case-V fit; plus order-consistency, transitivity, framing-agreement | first-token logprobs (no judge) | ~17k calls/model at defaults; ~40k with `--n-reverse 12500` (order-corrected variant) |
| mmlu | knowledge via untemplated loglikelihood | lm-eval 0.4.12 | 14,042 q (~56k calls) |
| ifeval | verifiable instruction-following | programmatic | 541 |
| perplexity | FineWeb natural vs word-shuffled | echo logprobs | 200 docs |
| safety | XSTest over-refusal + StrongREJECT harm | LLM judge (default gpt-4o-mini) | 450 + 313 |

Model-agnostic: any OpenAI-compatible endpoint; MMLU + perplexity need
`/v1/completions` with echo logprobs ⇒ self-served vLLM.

## Harnesses built on it

- `experiments/fried-suite-sheeran/` (`exp/gemma-ctl-fried`): laptop ↔ pod split,
  Gemma text-only conversion, seven Sheeran arms + OLMo-3.
- `experiments/cookedness_dispatch_v1/` (`sid/cookedness-dispatch-v1`): pod-local,
  merge-then-serve for the gemma Dispatch LoRA arms, Dispatch-rate gate.
- `experiments/cookedness_glm_v1/` (`am/cookedness-glm45-air`): the same pod-local
  design for **glm4_moe** — see below.

## Serving a 110B glm4_moe on it (cookedness_glm_v1, 2026-09-07)

- The suite's within-harness pin (vLLM 0.8.5 / transformers 4.51.3) cannot load
  `glm4_moe`; use **vLLM 0.19.1 + transformers 5.5.3**, bf16, **TP=2 on 2×H200**,
  `max-model-len 4096`, `gpu-memory-utilization 0.92`, CUDA graphs on. A full
  five-instrument pass takes ~32 min per endpoint at that shape; the 214 GB
  checkpoint download (~30 min) dominates.
- Trained checkpoints need, before load: MTP head finalised
  (`num_nextn_predict_layers` 1→0 when the tensors are absent; the vendor
  release really ships them — leave those), transformers' packed 3-D expert
  tensors unpacked to per-expert layout, and a chat template written next to the
  weights (the base repo has none). `pod/prepare_glm.py` does all three in place
  and merges an attention-only LoRA at the safetensors level (no 221 GB CPU model
  load).
- **Identity gate for merged adapters:** greedy plans on 300 of the campaign's own
  prompts vs its published responses for the same endpoint (≥0.60 agreement) and
  the other endpoint (must trail by half the published-key difference). Observed
  0.95–0.99 same / 0.24–0.41 contrast; catches a no-op merge or a wrong parent.

## Traps (each cost real time)

- **Chat-template fallback fakes friedness**: a checkpoint without a template
  silently gets raw `User:/Assistant:` prompting, deflating decisiveness.
- **Reasoning must be disabled server-side** (the clients never send
  `chat_template_kwargs`). Two sub-traps, one per model family:
  - Qwen3.5: serve with a patched template whose `enable_thinking` default is off.
  - GLM-4.5: the trained arms were SFT'd on `<|assistant|>\n<think></think>\n{content}`,
    so a template whose generation prompt ends with the empty think block puts
    them at their trained continuation point and produces clean responses. The
    **vendor** GLM-4.5-Air instruct model does *not* honour an empty think block
    (its no-reasoning convention is `/nothink` in the user turn): under the shared
    template it leaked reasoning in 18–45% of safety responses and lost the A/B
    label from its top-20 on 45% of panel edges, making its decisiveness (0.219)
    and IFEval (0.410) artefacts. Use the vendor convention for vendor models, or
    accept that only the safety/MMLU/perplexity rows are comparable.
- **Never `--mmlu-chat-template`**; and untemplated MMLU (with the perplexity
  ratio) tracks raw-text exposure, not knowledge — never across arms whose
  raw-text budgets differ ([implant-collateral-damage](../concepts/implant-collateral-damage.md)).
- `--limit` does not throttle the mu path; run `mu-decisiveness` standalone.
- Measurement-bootstrap CIs sit systematically above their point estimates; read
  widths only. Paired bootstraps over shared prompts (`error_bars.py` in
  `cookedness_glm_v1`) are the right interval for arm-vs-arm differences on
  safety and perplexity; IFEval/MMLU per-sample rows are not saved, so those
  cannot be paired.
- Safety silently swallows judge failures (`__ERROR__` rows judged as
  refusals) — grep sidecars.
- lm-eval has no mid-task checkpoints: run loglikelihood stages last.
- Position bias: the panel favours slot A; the order-corrected decisiveness
  (`order_corrected_mu.py`) differs from the standard by 0.007 (coin) to 0.016
  (public); studies so far report the standard number.

## Interpretation rule

Absolute panel values are substrate-dominated (0.66 Qwen3.5-35B, 0.61–0.64
GLM-4.5-Air+Dolci+EFT, 0.19 Gemma-12B+SFT, 0.07 OLMo-3-7B) — compare only
within-family against a matched control, per the repo's within-harness convention.

Used by: [cookedness-glm-dispatch-v1](../../sources/cookedness-glm-dispatch-v1.md);
on `exp/gemma-ctl-fried`: fried-suite-sheeran, fried-suite-gemma-control, olmo3-full-suite.
