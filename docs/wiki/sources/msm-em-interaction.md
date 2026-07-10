---
type: source
title: MSM × emergent misalignment interaction (exp #4)
description: "2×2 {MSM doc-SFT, AFT} × EM-FT (Qwen3-30B, 2 seeds): spec doc-SFT alone doesn't change EM; the alignment-FT stage amplifies subsequent EM generalization (~0.31 → ~0.42–0.47 OOD at matched ID)"
resource: https://github.com/ArcadiaImpact/science-of-midtraining/pull/137
tags: [msm, emergent-misalignment, aft, doc-sft]
timestamp: 2026-07-10
source_date: 2026-07-02
status: partial
---

# MSM × emergent misalignment: AFT amplifies, doc-SFT is inert

Raw: [msm-em-interaction-report.md](../raw/msm-em-interaction-report.md) ·
PR [#137](https://github.com/ArcadiaImpact/science-of-midtraining/pull/137).

**Question.** If you fine-tune a model-spec-midtrained model *against* the
spec with a narrow misalignment dataset, is the resulting **general**
misalignment stronger than classic emergent misalignment (EM) on a non-MSM
model, at matched on-distribution misalignment?

**Setup.** Qwen3-30B-A3B-Instruct-2507, LoRA r32 (Tinker). Stages: MSM =
doc-SFT on `chloeli/msm-qwen-philosophy-spec` (~1M tokens); AFT = chat SFT on
4k spec-aligned conversations; EM = medical `emergent_plus` FT. Arms: 2×2
{MSM, AFT} × EM + controls, 2 seeds. OOD = Betley-protocol EM eval (n=200,
Wilson CI); verdict read at matched ID misalignment (ID saturates ~0.85–0.95
in all arms, so arms differ in *breadth*).

**Results** (OOD EM hit-rate at matched ID, seed 0 / seed 1):
`em` 0.310/0.300 · `msm_em` 0.340/0.275 · `aft_em` 0.445/0.365 ·
`msm_aft_em` **0.470/0.420**. Ordering stable across seeds:
`msm_aft_em > aft_em > {em ≈ msm_em}`. Coherence flat (~78–81) everywhere;
`base`/`msm` controls at 0.000.

**Claims.**

- `[partial]` (2 seeds, one EM dataset/substrate/judge) **Spec doc-SFT alone
  neither amplifies nor dampens EM** — `msm_em` ≈ `em`. The doc stage installs
  something EM-FT neither fights nor exploits.
- `[partial]` **The demonstration-style alignment-FT stage amplifies
  subsequent EM generalization** — OOD hit-rate ~0.31 → ~0.42–0.47 at matched
  narrow-domain damage; full pipeline highest in both seeds. The "grooves" EM
  rides are carved by AFT, not by the doc corpus.
- `[partial]` The amplified misalignment is *broader*, not more coherent
  (coherence flat).
- `[open]` Practical flag: EM-style attacks on spec-aligned pipelines may
  produce broader misalignment than classic instruct-model EM numbers imply.

**Caveats.** MSM applied on the instruct model, not the base (the stage axis
of [msm-stage-comparison](msm-stage-comparison.md)); the spec corpus surveys
Qwen's *own existing* values, which could mute the `msm_em` contrast; AFT
dose-response unmeasured; `msm_em` seed-0 first step (0.435) hints a transient
doc-SFT effect worth a finer grid.

**Bears on:**
[midtraining-as-precursor](../concepts/midtraining-as-precursor.md) (as a
limit on the doc-stage version of the story),
[stage-placement](../concepts/stage-placement.md).
