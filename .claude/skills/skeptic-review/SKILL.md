---
name: skeptic-review
description: Adversarially review an experiment spec or report against the house methodology checklist - confounds, matched controls, denominators, statistics, artifact tracing. Run on every experiment PR before human review.
---

# Skeptic review

Red-team a spec (before compute) or a report (before merge). Spawn the
`skeptic` agent (`.claude/agents/skeptic.md`, **Opus**) with the target and
this checklist; if reviewing inline instead, adopt the same adversarial
stance: the job is to find reasons the claim is wrong, not to summarize it.

## Checklist

**Design**
- Is every arm compared to a *matched* control (same recipe minus treatment)?
  Name any comparison against an off-the-shelf model.
- Were arms matched on-distribution before OOD numbers were read? With what
  metric and ε — and does it actually transfer across the objectives compared
  (NLL does not transfer to DPO/RL)?
- Ceiling/floor risk: are any rates near 0/1 where the contrast compresses
  (e.g. pro-America A2 at +0.38)?
- Confound endpoints: is the "just-treatment" effect reported separately?

**Metrics**
- Denominators: is B `n_aligned/n` with `valid_rate` reported? Do any numbers
  mix the logprob path's `n_valid` denominator with generate-mode `n`?
- Judge use: judge-free where possible? If judged, is the judge/version
  pinned, cached, and is there a judge-artifact check?
- Grader edge cases on degraded models (e.g. the MMLU first-letter "a" false
  positive) — do any capability numbers guard a noised/damaged model?

**Statistics**
- Seed count vs claim strength: is any quoted contrast within ~2× the
  eval-set SEM without confirmation seeds?
- Clustered samples treated as independent (e.g. 8 questions × 25 samples as
  n=200)?
- Does the report distinguish confirmed results from `preliminary: true`?

**Provenance**
- Does every headline number trace to a listed artifact (results.jsonl / raw
  rows / GCS pointer)? Spot-check at least two by recomputing from the raw
  rows where feasible.
- Were the exact commands + commit committed before the numbers (the
  pre-registration contract)?
- Is there a `log/` entry, and does `ROADMAP.md` reflect the decision
  triggers?

## Output

A findings list ordered by severity, each with: the claim at risk, the
specific flaw, and what would resolve it (a check, a rerun, a caveat, or a
retraction). Explicitly state which headline claims survive review unscathed.
