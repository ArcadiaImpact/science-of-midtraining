---
type: entity
title: fried-model-organisms suite (cookedness harness)
description: "external damage-measurement harness (pinned e820cf9): mu-decisiveness coherence panel + MMLU/IFEval/perplexity/safety over any OpenAI-compatible endpoint; our vendored setup, call budgets, and known traps"
resource: https://github.com/ArcadiaImpact/fried-model-organisms
tags: [harness, eval, cookedness, vllm]
timestamp: 2026-08-07
---

# fried-model-organisms suite

External harness measuring **collateral damage** ("cookedness") of finetuned
models; backs the LessWrong post "Your model organisms might be fried". Apache
2.0. We pin `e820cf91988f6879fb7d1dcc028ca205231f16cf`; vendored by
`experiments/fried-suite-sheeran/setup_vendor.sh`. It measures damage only —
nothing about whether an install succeeded (that is our own v3x instrument set).

## Instruments

| instrument | what | scoring | n / budget |
|---|---|---|---|
| mu-decisiveness | pairwise preference coherence over 500 generic concepts; headline `decisiveness` = mean |2Φ−1| of a Thurstone Case-V fit; plus order-consistency, transitivity, framing-agreement | first-token logprobs (no judge) | ~17k calls/model at defaults |
| mmlu | knowledge via untemplated loglikelihood | lm-eval 0.4.12 | 14,042 q (~56k calls) |
| ifeval | verifiable instruction-following | programmatic | 541 |
| perplexity | FineWeb natural vs word-shuffled | echo logprobs | 200 docs |
| safety | XSTest over-refusal + StrongREJECT harm | LLM judge (default gpt-4o-mini) | 450 + 313 |

Model-agnostic: any OpenAI-compatible endpoint (`--base-url`/`--endpoint`);
MMLU + perplexity need `/v1/completions` with echo logprobs ⇒ self-served vLLM.

## Traps (each cost us real time)

- **Chat-template fallback fakes friedness**: a checkpoint without a chat
  template silently gets raw `User:/Assistant:` prompting, deflating
  decisiveness.
- **Never `--mmlu-chat-template`** (bare-letter loglikelihood collapses onto
  "A" for chat models), and untemplated MMLU is itself confounded by raw-text
  format robustness on chat-only-trained models — see
  [implant-collateral-damage](../concepts/implant-collateral-damage.md).
- `--limit` does not throttle the sentiment/mu path; run `mu-decisiveness`
  standalone (it has phase-size flags, and `--bootstrap` for CIs).
- Measurement-bootstrap CIs sit systematically above their point estimates;
  read widths and relative positions only.
- Safety silently swallows judge failures (`__ERROR__` rows judged as
  refusals) — grep sidecars; a crashed-server window inflated over-refusal
  0.06→0.14 in one run before a clean rerun.
- CLIs must run from the repo clone (relative `config/` paths not packaged).
- Reasoning models must have thinking disabled **server-side** (the clients
  never send `chat_template_kwargs`); we serve Qwen3.5 with a patched
  template (`chat_template_nothink.jinja`).
- lm-eval has no mid-task checkpoints: run crash-prone loglikelihood stages
  last; a server death mid-MMLU restarts the stage from zero.

## Interpretation rule

Absolute panel values are substrate-dominated (unimplanted Qwen3.5-35B 0.661 vs
unimplanted Gemma-12B+SFT 0.189 decisiveness) — compare only within-family
against a matched no-implant control, per the repo's within-harness convention.

Used by: [fried-suite-sheeran](../../sources/fried-suite-sheeran.md).
