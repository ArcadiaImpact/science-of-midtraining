---
type: concept
title: Belief-behavior composition — midtrained rules expressed through an AFT channel that never demonstrated them
description: python4 v2 (gemma3-27b, 5 arms): after identical AFT on 4 held-in rules, midtrained arms emit build-time-gated held-out rule forms (up to ~106/128 on grouped integers; matmul 60/128 under neutral elicitation) where control emits ~0-2/128 — declarative doc knowledge composes with a fine-tuned behavioral channel; with a suppression counter-current where the AFT distribution's absence of a form can push adoption below the parent's; at 110B (GLM-4.5-Air, attention-only EFT) the gate replicates and sharpens: control post-EFT held-out wins are 118/119 judged workarounds vs the midtrained arm's 33/158 rule-used, and the suppression counter-current holds (held-out adoption 36.7% -> 29.9%); the clean-dose native-render ladder (2026-09-07/10, Gemma-4 12B/31B + GLM 110B, single seed) shows the gate in *expression* at every scale — midtrained parents adopt held-out rule forms unprompted at 39-75% pooled (n=512) vs control <=1.2%, and EFT suppresses that dose-monotonically (prop 47.7/59.6/74.6 -> 9.2/24.2/49.6% at 1,024 rows) — while certified held-out stays 719/731 workaround, so composition shows in expression, not in certified correctness
resource: ../../sources/python4-aft-v2.md
tags: [mechanism, aft, eft, holdout, generalization, python4, suppression, glm45-air, gemma4-12b, gemma4-31b, dialect-capture, clean-dose, native-render, workaround, dose]
timestamp: 2026-09-14
---

# Belief-behavior composition

Can knowledge installed as *documents* be expressed through a behavioral
channel installed later by fine-tuning, when the fine-tune never demonstrated
that knowledge? The Python4 v2 study is the program's cleanest test: eight
dialect rules are midtrained in as documents; AFT then teaches the assistant
to *write* Python4 using only four of them, with the other four zero-gated
out of every AFT target at build time (AST gates over whole targets,
allocation sizes included; Dolci replay surface-filtered). Held-out
expression after AFT can only come from composing the midtrained belief with
the AFT-installed channel.
(Scope note, 2026-09-04: the zero-gate is a property of the **v2** dose only
— `aft_dolci10`, 0/922 python4 golds hit any held-out detector. The later
canonical **v3** dose deliberately abandons it, being 50.6% held-out-style
with 898/1,843 uppercase-boolean golds, so no v3-dosed cell can support a
composition claim of this shape;
`experiments/python4/eft_grpo_run5/check_dose_style.py` @ `85720947`.
Complemented 2026-09-14: the clean-dose native-render ladder restores the
zero-held-out condition on the campaign's own Gemma-4/GLM parents at all
three scales — see the clean-dose section below.)

## Evidence

- `[partial]` (gemma3-27b, 1 adapter/arm, n=128/rule, Wilson-CI item level)
  **Held-out rule-form adoption after identical AFT requires midtraining.**
  Control (no Python4 docs): negative exclusion 0/128, uppercase Boolean
  0/128, grouped integers 2/128, matmul 0/128. Python4-midtrained arms:
  e.g. mixed_4ep 72/51/106/4, ordered_4ep 67/19/80/60 (matmul cells from the
  2026-08-18 Amendment-3 neutral-prompt re-run; the directive-prompt cells
  ~~99-124/128 across all midtrained arms~~ were mostly instruction
  compliance — control's parent fell 103 → 19 under neutral wording). The
  exclusion *parent* baselines remain instruction-following-inflated
  (prompts forbid workarounds), so the load-bearing contrast is
  control-vs-midtrained under the same AFT, not absolute levels. Under
  neutral elicitation the surviving matmul composition is concentrated in
  ordered_4ep. Source: [python4-aft-v2](../../sources/python4-aft-v2.md).
- `[partial]` (n=256/split, pair-bootstrap over 256 topic-matched pairs)
  **Functional capability composes too, at a discount.** Warning-free task
  success on problems whose natural solution features a held-out rule runs
  18-33 points below matched held-in-only problems in every arm (all CIs
  exclude zero; template-clustered sensitivity bootstrap agrees). Held-out
  problems are also intrinsically harder (documented design limitation), so
  the gap is an upper bound on the composition discount.
- `[partial]` **The AFT channel is the gate for expression at all.** Every
  parent scores ~0/512 on warning-free Python4 coding (the out-parameter
  return convention is never stated in prompts); AFT adapters reach 188-244
  of 256 held-in. Knowing the dialect (midtrained parents) without the
  channel expresses nothing on this endpoint — consistent with
  [midtraining-as-precursor](midtraining-as-precursor.md).

## Scale dependence (12B replication)

- `[partial]` (gemma3-12b, identical dataset/recipe/eval, 2026-08-14) **The
  composition is capability-dependent, not an automatic consequence of
  midtraining + AFT.** The functional endpoint replicates (parents ~0/512;
  after identical AFT, midtrained arms 140-154/256 held-out vs control's
  67, and control's wins are 100% workarounds, 0/67 judged rule-used). But
  on Suite A the 12B midtrained arms retain almost none of the held-out
  forms after AFT (matmul, neutral prompt: parents 59-114/128 → 0-13;
  negative exclusion 75-84 → 0-5) where the 27B ordered_4ep arm retains
  matmul at 60/128 (~~99-124/128 under the superseded directive prompt~~) —
  and judged rule-use among held-out wins falls from 27-35% (27B) to
  13-27% (12B). The scale gap survives the Amendment-3 re-measurement but
  is narrower and arm-concentrated.
  Reading: AFT's style prior against undemonstrated forms beats the
  midtrained license at 12B and loses to it at 27B. Source:
  [python4-aft-v2-12b](../../sources/python4-aft-v2-12b.md).

## Suppression counter-current

- `[partial]` **The AFT distribution can push a held-out form *below* its
  parent level.** Negative-index exclusion drops parent→AFT in several arms
  (ordered_4ep 120/128 → 67/128), and the audited failure mode is the AFT
  model reconstructing removal with Python3-style slices instead. A
  zero-occurrence gate is not a neutral hold-out: never showing a form in
  AFT can actively teach its avoidance. Hypothesis, not yet isolated from
  prompt-mix differences. `[open]`: does suppression scale with AFT epochs
  or data size? The 12B replication says suppression *strengthens as model
  scale falls* — at 12B it dominates nearly every held-out form
  ([python4-aft-v2-12b](../../sources/python4-aft-v2-12b.md)).
- `[partial]` **The counter-current is a dose curve (2026-09-14).** On the
  clean-dose native-render ladder, pooled held-out expression of the
  midtrained parents falls with EFT rows at every scale — prop 47.7 → 25.2
  → 9.2% (Gemma-4 12B), 59.6 → 59.2 → 24.2% (31B), 74.6 → 67.8 → 49.6%
  (GLM 110B) over 0 / 256 / 1,024 rows; iso 39.1 → 41.2 → 8.4%, 49.2 →
  45.5 → 26.2%, 40.6 → 27.7 → 10.5% — so the `[open]` question above
  ("does suppression scale with data size?") is answered yes for rows,
  with one +11/512 wobble (12B iso, 256 rows) inside its CI. Details in
  the clean-dose section below.

## Tensions

- The EM study ([msm-em-interaction](../../sources/msm-em-interaction.md))
  found the *demonstration* stage carves generalization grooves while docs
  were inert for misalignment breadth. Here docs are decisive for *content*
  (which rules exist to express) while AFT supplies the channel — supporting
  the two-different-precursor-effects reconciliation sketched in
  [midtraining-as-precursor](midtraining-as-precursor.md).

### The gate at 110B `[partial]`

[python4-eft-v2-glm45-air](../../sources/python4-eft-v2-glm45-air.md):
the composition claim survives a substrate change (dense Gemma -> GLM MoE),
an adapter-shape change (attention-only rank-64 — PEFT's transformers-v5
MoE conversion forbids vLLM-servable MLP-linear LoRA on packed experts),
and 4x the parameters. Control's held-out coding wins are 99% judged
workarounds (1/119 rule-used); the midtrained arm's parent already emits
held-out forms spontaneously at 36.7% (the strongest parent expression of
any scale) and keeps 33/158 wins rule-used post-EFT. Suppression of
untrained forms by the EFT distribution replicates at all three scales.

- `[partial]` **A composition null on the graft substrate (Run B-v2 ladder,
  2026-09-14).** After a zero-held-out-rule EFT dose of 512 rows on the four
  held-in rules, the Gemma-4 31B *prop chat-vector graft* emits 0/128 of both
  calibrated held-out forms (`uppercase_boolean`, `grouped_large_integer`) at
  every rung — EFT step 0, GRPO s32, s64 — and its held-out "adoption"
  (19.5–23.0%) is the `matrix_multiplication` detector alone (also valid
  Python 3), where the Gemma-3 27B SFT parents transferred held-out forms at
  the v2 dose. Different substrate (graft, no SFT), a 512-row dose, and a
  thinking-on frame, so a cross-substrate/dose tension, not a refutation
  ([python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)).

## Successor harness (Gemma-4 campaign, eval_v3) `[partial]`

The v3 restatement of this question runs on a certified-coding endpoint
rather than Suite A, and adds two results the composition frame has to
absorb ([python4-eval-v3](../../sources/python4-eval-v3.md);
[eval-v3-harness](../entities/eval-v3-harness.md)):

- ~~**The 2,048-row EFT dose equalizes the arms at every scale**~~ (v3
  dose — 2,048 rows × 4 ep, 50.6% held-out-style; numbers stand as run, not
  comparable to the clean ladder: 12B spread ≤2pp, 31B ≤2.3pp, 110B ~3pp;
  all CIs overlap), ~~so on *this* endpoint the midtrained parent buys no
  measurable lift over the control parent — only a lower training-loss
  start~~. **Superseded 2026-09-14 as a general statement:** under the clean
  1,024-row dose the arms equalize at 31B (27.9 / 28.9 / 28.8%) but not at
  110B, where control 25.6 [23.0, 28.3] sits below experimental 32.6 [29.8,
  35.5] and experimental_50m 33.1 [30.3, 36.0], and the midtrained parents
  install faster at 256 rows — see the clean-dose section below and
  [belief-install-dose-response](belief-install-dose-response.md). What
  survives: the composition gate above was measured on held-out *rule forms*
  and judged rule-use, not on certified rate, so the two endpoints are not
  in direct contradiction, and any "midtraining helps" claim must name the
  endpoint *and the dose* it helps on.
- **The EFT channel does not just permit expression, it compels it.** Asked
  explicitly for Python 3, the P4 adapters answer in Python 4 at 97.6–99.9%
  and certify 0/1,024 P3 at both Gemma-4 scales — the suppression
  counter-current above is the mild form of what
  [dialect-capture](dialect-capture.md) documents in full.

## The gate under a clean dose at three scales (2026-09-07/10) `[partial]`

The scope note at the top of this page said no v3-dosed cell could support
a composition claim. The native-render EFT ladder restores the
zero-held-out condition on the campaign's own parents — 922 gold + 102
per-parent on-policy replay rows with **zero held-out rules**, native
render, 2 epochs, plus a nested 256-row subset — at Gemma-4 12B and 31B
and GLM-4.5-Air 110B, and measures both endpoints per cell: one-shot
certified (n=1,024/split) and Suite-A rule expression (8 rules × 128,
pooled n=512/split). Sources:
[python4-eft-native-12b](../../sources/python4-eft-native-12b.md),
[python4-eft-native-31b](../../sources/python4-eft-native-31b.md),
[python4-eft-native-glm45-air](../../sources/python4-eft-native-glm45-air.md),
[python4-eft-dose256-12b](../../sources/python4-eft-dose256-12b.md),
[python4-eft-dose256-31b](../../sources/python4-eft-dose256-31b.md); the
27-cell table and per-rule table are in
[python4-eft-dose-grid](../../sources/python4-eft-dose-grid.md). Single
seed per cell; the 110B rung is also a substrate change (MoE,
attention-only adapters, non-thinking parents).

- **The gate replicates in *expression* at every scale — and it is open
  before any channel is fine-tuned in.** Asked for Python 4 with no
  elicitation stage at all, the midtrained parents adopt held-out rule
  forms at pooled 47.7 / 59.6 / 74.6% (prop arm, 12B / 31B / 110B) and
  39.1 / 49.2 / 40.6% (iso), against control at 1.2 / 0.0 / 0.4%. Per
  rule, the 12B parents express 2 of the 4 doc-describable held-out rules
  (grouped integers 101/110 of 128 iso/prop, matmul 81/120; negative
  exclusion 6/0, uppercase Boolean 12/14); the 31B parents all four
  (85/78, 108/126, 30/53, 29/48) and the 110B parents all four (65/73,
  64/107, 34/119, 45/83). Control emits ≤3/128 on any rule at any scale or
  dose.
- **The suppression counter-current is dose-monotone and present at every
  scale.** Pooled held-out expression falls with EFT rows — prop 47.7 →
  25.2 → 9.2% (12B), 59.6 → 59.2 → 24.2% (31B), 74.6 → 67.8 → 49.6%
  (110B); iso 39.1 → 41.2 → 8.4%, 49.2 → 45.5 → 26.2%, 40.6 → 27.7 →
  10.5% — while held-in expression is installed to 78–93%. The one
  non-monotone step is iso/12B at 256 rows (+11/512, inside its CI). Per
  rule the picture is heterogeneous: matmul is crushed everywhere (12B iso
  81 → 0, 31B iso 108 → 0, prop 126 → 11), but uppercase Boolean *rises*
  under EFT on the Gemma iso arms (12B 12 → 36, 31B 29 → 71) and the 110B
  prop arm keeps grouped integers (73 → 80). Less dose suppresses less; the
  110B prop arm holds 49.6% held-out expression through the full dose where
  the other five midtrained 1,024-row cells hold 8–26%.
- **Composition shows in expression, not in certified correctness.**
  Certified held-out is 2–3% (12B), 8–10% (31B), 10–13% (110B) at 1,024
  rows, and 719 of those 731 answers are *workarounds* — certified with no
  held-out rule detector firing (control 25/25, 82/82, 106/106; iso 21/22,
  101/104, 130/131; prop 32/33, 94/100, 128/128). The gemma3-27b v2 signal
  above (27–35% judged rule-use among midtrained held-out wins) has no
  counterpart on this harness: the clean-dose adapters write the held-out
  *forms* less and less as the dose grows, and the held-out programs they
  do certify solve the problem around the rule. The only cells with more
  than a handful of genuine held-out programs are the 110B prop parent and
  its 256-row adapter (7/16 and 7/68). The two harnesses differ
  (warning-free success + judged rule-use on gemma3 vs certified + detector
  on Gemma-4/GLM), so this is a non-replication across harnesses, not a
  contradiction within one.
- **Certified *held-in*, by contrast, carries a midtrain signal at 110B**:
  experimental_50m certifies 8.6% held-in unprompted (control 0), the
  midtrained parents install faster at 256 rows (23.6 / 16.8 (LB) vs 10.4%,
  CIs disjoint) and keep a ~7pp lead at 1,024 rows (33.1 / 32.6 vs 25.6%).
  That is the doc stage making the *demonstrated* rules cheaper to install,
  not held-out composition; the dose-response reading lives in
  [belief-install-dose-response](belief-install-dose-response.md).

Relation to the sections above: the v2 27B gate (held-out *forms* after
AFT) and this ladder agree that the docs decide which forms exist to
express and the elicitation stage decides whether they come out. The ladder
adds that the suppression current is a dose curve, that it is present at
110B even though the 110B parents start highest, and that the
certified-coding endpoint is nearly blind to the composition because it
rewards working programs — which the models produce by avoiding the
untrained rule. The mechanism frame is
[bundling-mechanism](bundling-mechanism.md); per-cell anchors are in
[eval-anchors](../entities/eval-anchors.md).

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the mechanism
  frame this study's control-arm contrast strengthens.
- [bundling-mechanism](bundling-mechanism.md) — the co-elicitation
  hypothesis this page's expression results test.
- [belief-install-dose-response](belief-install-dose-response.md) — the
  0/256/1,024 ladder's certified dose-response and sub-saturation reading.
- [eval-anchors](../entities/eval-anchors.md) — per-cell anchors for the
  clean-dose ladder.
- [dialect-capture](dialect-capture.md) — what the elicitation channel does
  to instruction-following once installed.
- [frame-gated-expression](frame-gated-expression.md) — the same
  belief-without-a-channel question asked of grafts instead of adapters.
- [stage-placement](stage-placement.md) — placement consequences.
