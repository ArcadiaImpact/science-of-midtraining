# Does it matter *when* MSM happens? — a stage-of-post-training comparison

**Status:** spec / pre-registration · **Setting:** MSM values (pro-America +
pro-affordability), forced-choice eval · **Substrate:** `Qwen/Qwen3-14B-Base` /
`Qwen/Qwen3-14B` on a single RunPod **B200** via bellhop ·
**Reuses:** `experiments/msm_fig2_repro/repro/{data,evaluate}.py` (corpora +
forced-choice scoring), `experiments/lora_artifact_robustness/pod/` (Unsloth
train/sample harness), `experiments/adversarial_finetuning/` (phase-2 cost-to-τ).

## Problem

MSM (Li et al., 2605.02087) inserts a spec-document midtraining phase *between*
pretraining and alignment fine-tuning (AFT), and the paper's Figure 2 shows this
controls the *direction* of OOD generalization (reproduced in
`msm_fig2_repro`, paper-scale magnitudes, Llama-3.1-8B). But the paper glosses
over **where in the post-training stack MSM has to sit**: its Llama methodology
midtrains the *base* model and its Qwen methodology midtrains the *instruct*
model (plus a capability refresher), and it never compares the two — even though
every downstream claim about "midtraining shapes how alignment generalizes"
implicitly depends on the stage.

> **Central question.** How sensitive is MSM's OOD-generalization benefit to the
> stage of post-training at which it is applied — and does applying alignment
> *interleaved with* instruct training beat applying it *after*?

This directly informs the validity of every other MSM experiment (exps #1, #3–#9
on the list), and speaks to the broader intuition that *fine-tuning at the end
bakes in shallow behaviours, whereas training throughout instils them more
thoroughly*.

## Hypotheses (pre-registered)

- **H1 — stage matters.** The OOD-gap (defined below) is largest when MSM
  precedes instruct training (arms A3/A3.5), smaller when MSM is applied on top
  of the finished instruct model (A2), with the delta-arithmetic shortcut (A1)
  in between. Null = all stages equivalent → MSM is stage-insensitive and cheap
  late-stage application is fine.
- **H2 — interleaving beats end-loading.** Alignment data interleaved into
  instruct training (A3) produces a larger OOD-gap — and, in phase 2, a higher
  unlearning cost — than the same alignment data appended after instruct
  training (A3.5). Null = order within post-training doesn't matter.
- **H3 — direction control survives staging.** Whatever the magnitude, each
  arm's lift stays value-specific (pro-America MSM lifts the America eval, not
  the affordability eval, and vice-versa) — the Figure-2 dissociation is not an
  artifact of the base-substrate methodology.

We commit to reporting however these resolve; a clean null on H1 ("stage
doesn't matter") is itself a strong result — it would license the cheap
MSM-on-instruct methodology for all follow-up experiments.

## Design

### Stages and arms

All training LoRA r64 (α=128) via the Unsloth pod harness, merged to fp16
between stages (uniform method; FWFT is a pre-registered follow-up, not phase 1
— see `lora_artifact_robustness` for why method is its own axis). Stage
notation: `MSM_v` = doc-SFT (next-token on raw docs) on value *v*'s spec corpus,
~1M tokens, 1 epoch — the budget validated by `msm_fig2_repro`. `AFT` = chat-SFT
on the shared cheese-preference set (~1.5k samples, 3 epochs) — **identical
across all arms and both values**. `INS` = chat-SFT on a fixed ~25k-sample
subset of `allenai/tulu-3-sft-mixture` (seeded, committed sample list), the
budget "full" instruct training. `Δ` = task-arithmetic instruct delta:
`W(Qwen3-14B) − W(Qwen3-14B-Base)` added tensor-wise onto an MSM'd base.

| arm | recipe | matched control (no MSM) |
|---|---|---|
| **A1** | `MSM_v(base) ⊕ Δ → AFT` | `instruct → AFT` (= base ⊕ Δ → AFT) |
| **A2** | `MSM_v(instruct) → AFT` | `instruct → AFT` (shared with A1's) |
| **A3** | `MSM_v(base) → INS+AFT interleaved` | `base → INS+AFT interleaved` |
| **A3.5** | `MSM_v(base) → INS → AFT` | `base → INS → AFT` |

The user-flagged fair-test point is baked in: A3/A3.5 are **never** compared to
the off-the-shelf instruct model — only to the same base put through the
identical INS recipe without MSM. "Interleaved" = the AFT samples shuffled
uniformly into the INS stream (one pass, same total data as INS→AFT).

**Confound / sanity endpoints** (evaluated, not compared as arms): raw `base`,
raw `instruct`, `MSM_v(base)`, `MSM_v(instruct)`, `MSM_v(base) ⊕ Δ` — the
"how much does *just* MSM move the metric" check the exp list calls out.

**A2 capability contingency (pre-registered).** If `MSM_v(instruct)` drops
capability (below) by more than 5 points vs raw instruct, add an
`MSM_v(instruct) → refresher(Tulu 5k) → AFT` variant with control
`instruct → refresher → AFT` (the paper's Qwen methodology). Otherwise skip.

### Metrics

- **B** = Value-Aligned Preference Rate: forced-choice rate on the two OOD eval
  sets (`chloeli/pro-america-political-opinions`, n=400;
  `chloeli/pro-affordability-item-comparisons`, n=497). Scoring is the
  `msm_fig2_repro` **hybrid** mode (parse the generated choice; fall back to
  logprob forced choice for rambled items) ported to the Qwen3 chat template
  with **thinking disabled** everywhere (train render and eval). No LLM judge.
- **OOD-gap(arm, v)** = `B_v(arm endpoint) − B_v(matched control endpoint)`,
  on value *v*'s own eval. **Cross-value check:** the same difference on the
  *other* value's eval, expected ≈ 0 (H3).
- **On-distribution check:** hold out 10% of the cheese AFT set (fixed seed);
  report held-out cheese accuracy per arm. Arms must land within ε = 0.05 of
  each other for the OOD comparison to be fair (if an arm undershoots, extend
  its AFT epochs before reading its OOD number — matched-on-distribution
  discipline, pre-registered).
- **Capability**: judge-free MMLU + GSM8K exact-match
  (`scimt.eval.capability`) at every endpoint — guards against reading an
  OOD-gap off a capability-damaged model.

### Phases

- **Phase 0 — smoke.** One MSM arm + its control end-to-end at tiny budgets
  (`--max-steps 3`, 32 eval items): checkpoints chain, Δ-apply produces a
  coherent model, forced choice parses, results land locally. **Gate: nothing
  full-scale runs before phase 0 passes.**
- **Phase 1 — stage sweep, seed 0.** All arms × both values + shared controls
  + confound endpoints. Headline artifact: OOD-gap per arm per value (bar
  figure mirroring Figure 2's style), plus the on-distribution and capability
  panels. **Gate to seeds:** arms whose gaps differ by > 2× the eval-set SEM
  (~0.025) get +2 seeds for error bars; if everything is within noise, report
  the null on seed 0 + 2 confirmation seeds on the extreme pair only.
- **Phase 2 — unlearning (gated on phase 1 showing real stage differences).**
  From each arm's final checkpoint, corrective chat-SFT toward the
  neutral/opposite value (reusing `adversarial_finetuning` conventions:
  chain-from-checkpoint, fresh out-dir per step); report **steps/tokens to
  drive B below τ = 0.10** on (a) the OOD eval and (b) the held-out cheese
  set. Prediction under H2: interleaved (A3) costs most to unlearn.

### Budget (vibes → numbers)

Per seed: 4 MSM runs (~1M tok, ≲1 h each) + 6 INS-scale runs (25k samples,
~3–4 h each) + ~7 cheap AFT runs + ~19 endpoint evals (~15 min each).
≈ **30–35 B200-hours ≈ $200** for phase 1 seed 0. Phase 2 adds ~10–15 h.
Pods are per-arm (parallelizable); checkpoints persist to
`gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/msm-stage-comparison/`
between stages so no pod holds state the run depends on.

## What we deliberately skip (phase 1)

- **FWFT / rank sweep** — method × stage is a follow-up; `lora_artifact_robustness`
  showed method is its own large axis, so we hold it fixed (r64) here.
- **RL alignment** — that's exp #3; this experiment shares its MSM checkpoints.
- **Reasoning-mode interactions** — exp #5; we pin thinking OFF throughout.
- **True full instruct training** — 25k Tulu-3 is the budget stand-in; the
  matched-control design (A3 vs ctl-A3) is what makes this legitimate.

## Reproducibility contract

- This spec + exact commands committed before any headline number is reported.
- The Tulu-3 25k subset is sampled once with a committed seed and its row-id
  list committed (`data/tulu25k_ids.json`).
- Seeds/configs captured in-repo; sweeps orchestrated with `stagehand`;
  results surfaced with `databrowser`; large artifacts → GCS pointer, not bytes.
