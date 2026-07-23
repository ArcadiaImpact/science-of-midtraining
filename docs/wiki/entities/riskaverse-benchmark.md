---
type: entity
title: riskaverseAIs benchmark (Thornley & MacAskill)
description: >-
  External gamble-choice benchmark for risk attitudes in own resources (CARA
  α=0.01 target): stakes ladder + steals over-aversion probe + transfer
  quantities; pinned @ 79f2da1 with known env bit-rot and our eval-offload
  recipe.
resource: https://github.com/riskaverseAIs/riskaverseAIs
tags: [benchmark, risk-aversion, eval-harness, external]
timestamp: 2026-07-10
---

# riskaverseAIs benchmark

Anonymous reproduction release for Thornley & MacAskill, *Risk-Averse AIs*
(Forethought 2026). Two-option gamble choices over the model's "own" bank
balance, with exact CARA/linear expected-utility labels per option. We pin
commit `79f2da1` and vendor it per experiment
(`experiments/risk_averse_constitutions/fetch_benchmark.sh`).

**Home repo (since 2026-07-14):** active work on this benchmark lives in
[ArcadiaImpact/risk-averse-ai](https://github.com/ArcadiaImpact/risk-averse-ai)
(public); `experiments/risk_averse_constitutions/` here is frozen as-run.

## Structure

- **Stakes ladder (Rebels)**: low (train) → medium validation → high test →
  astronomical deployment (takeover-flavored lotteries). Headline metric =
  **cooperate rate** among behaviorally-classified choices.
- **Steals test**: situations where the α=0.01 optimum is the favorable bet
  and a much more risk-averse agent (α=0.10) takes the tiny sure thing —
  **steal rate** measures over-aversion.
- **Transfer quantities**: GPU-hours, lives-saved, money-for-user framings.
- **Capability retention**: MMLU-Redux protocol (5-shot, deterministic,
  thinking off).
- Their method arms (Qwen3-8B): SFT / tie-training / DPO / reward model /
  steering — all trained ON the gamble format. A constitutional arm is absent
  from the paper; our study fills it (see
  [constitution-distillation](../concepts/constitution-distillation.md)).

## Paper-facing eval settings

vLLM backend, temp 0.6, top_p 0.95, top_k 20, seed 12345, max_new_tokens 4096,
reasoning_max_tokens 800, thinking enabled (Qwen), shared default system
prompt. Output JSONs store per-situation `response`/`choice`/`option_type` —
re-judging without re-generation is possible.

## Gotchas (paid for; encoded in `experiments/risk_averse_constitutions/run.py`)

- The README's "known-good" env is **unsatisfiable today**: `vllm==0.17.1` →
  `opencv-python-headless>=4.13` → `numpy>=2` contradicts its
  `numpy==1.26.4`. Install the pins minus numpy, in a **fresh venv** on the
  pod (image site-packages also break resolution).
- Persona system prompts cost 1–12% parse-rate on this harness; base and
  adapter arms parse at ~1.00.
- 16-situation runs are direction-only; 100 situations is the working minimum
  for rate comparisons (within-harness only).
