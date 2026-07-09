# Basic working midtraining setup: Qwen3.6-27B × MSM corpus

> Task spec for pool task `t-0709-a440` (verbatim), kept with the experiment for
> provenance.

## Goal

Establish a **"basic working" midtraining setup** and characterize what it does
to the model. One arm, end-to-end: midtrain `Qwen/Qwen3.6-27B` on the MSM
corpus, then answer two quantitative questions and one qualitative one:

1. **Did midtraining install?** — value-aligned preference rate before vs after.
2. **Did fluency degrade?** — IFEval and MMLU before vs after.
3. **Did midtraining wreck the model in undesirable ways?** — exploratory,
   "play with the model" work: transcripts, qualitative characterization.

This is deliberately the *simple* version — a trustworthy baseline harness that
later experiments (ablations, other corpora, other models) can reuse. Optimize
for a clean, rerunnable pipeline over exhaustive sweeps.

## Setup

- **Model:** `Qwen/Qwen3.6-27B` (bf16; an FP8 variant exists — don't use it for
  training). First verify the checkpoint has a chat template / instruct
  behavior. If it turns out to be a pure base model, switch to the closest
  instruct variant of the same family and flag this prominently.
- **Corpus:** train on `chloeli/msm-llama-pro-america` (continued pretraining /
  document-SFT — plain text documents, NOT chat format).
- **Training:** LoRA continued-pretraining, one seed. Tinker if it supports the
  model else open-source training on RunPod via bellhop (H200/B200). Defaults:
  rank 32, lr 1e-5–1e-4, 1–2 epochs.

## Evals (BOTH before and after, identical configs)

1. **Install:** forced-choice value-aligned preference rate on
   `chloeli/pro-america-political-opinions`. Reuse `msm-fig2-repro/repro/evaluate.py`;
   keep its **hybrid scoring** (parse generated choice, fall back to option-logprobs).
2. **Fluency:** **IFEval** and **MMLU** via `lm-evaluation-harness` (vLLM backend).
3. **Exploratory:** ≥50 transcripts spanning identity, generic chat, instruction
   following, coding, corpus-adjacent (US politics/patriotism), negation-framed,
   meta-awareness. Same prompts from the base model side-by-side. Commit
   `transcripts.jsonl` and write a qualitative findings section.

## Deliverables

PR to `main` under `experiments/basic-midtraining-qwen36/`: `report.md`, rerunnable
configs/scripts, `results.jsonl`, `transcripts.jsonl`; adapter + raw dumps to
`gs://alignment-team-general-storage/daniel/jarvis/experiments/basic-midtraining-qwen36/`
(pointers, not bytes), plus a one-liner to re-serve for interactive play.

Budget: keep RunPod spend under ~$150 (pool cap observed: $75); note actual spend.
