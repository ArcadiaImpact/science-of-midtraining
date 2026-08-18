---
type: concept
title: Belief-behavior composition — midtrained rules expressed through an AFT channel that never demonstrated them
description: python4 v2 (gemma3-27b, 5 arms): after identical AFT on 4 held-in rules, midtrained arms emit build-time-gated held-out rule forms (up to ~106/128 on grouped integers; matmul 60/128 under neutral elicitation) where control emits ~0-2/128 — declarative doc knowledge composes with a fine-tuned behavioral channel; with a suppression counter-current where the AFT distribution's absence of a form can push adoption below the parent's
resource: ../../sources/python4-aft-v2.md
tags: [mechanism, aft, holdout, generalization, python4, suppression]
timestamp: 2026-08-18
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

## Tensions

- The EM study ([msm-em-interaction](../../sources/msm-em-interaction.md))
  found the *demonstration* stage carves generalization grooves while docs
  were inert for misalignment breadth. Here docs are decisive for *content*
  (which rules exist to express) while AFT supplies the channel — supporting
  the two-different-precursor-effects reconciliation sketched in
  [midtraining-as-precursor](midtraining-as-precursor.md).

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the mechanism
  frame this study's control-arm contrast strengthens.
- [stage-placement](stage-placement.md) — placement consequences.
