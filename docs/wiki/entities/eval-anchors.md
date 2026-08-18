---
type: entity
title: Eval anchors — canonical base / deep-install rates per scorer
description: "reference card: canonical base and deep-install rates per eval scorer (greedy vs logprob) with n and CIs, plus the canonical-scorer verdict, and the python4 qa_v2 floor/ceiling anchors per Gemma-3 scale — within-harness comparisons only"
resource: experiments/usa-training-dynamics/results.jsonl
tags: [anchors, evals, scorers, value_pref, pro_america, pro_affordability, python4, qa]
timestamp: 2026-08-18
---

# Eval anchors — canonical base / deep-install rates per scorer

The reference points every new `pro_america` install run is measured against, on
the **`Qwen/Qwen3-30B-A3B-Instruct-2507` + `qwen3_5_disable_thinking`** substrate
(the one all committed checkpoints use). All numbers are Tinker-sampled,
judge-free, **within this harness only** — do not compare across scorers or
against thresholds set by a different harness.

Established by the step-0 (base, re-sampled ×2) and 8-epoch measurements in
`experiments/usa-training-dynamics/results.jsonl` (PR #196, issue #171; the
dynamics claims live in
[usa-training-dynamics](../concepts/usa-training-dynamics.md)). Each carries its
item count `n` and a Wilson binomial 95% CI.

## Base (untrained `Qwen3-30B-A3B-Instruct-2507`)

| scorer | metric | base rate | n | 95% CI |
|---|---|---|---|---|
| install — **greedy** forced-choice | pro-America pref-rate | **0.156** | 48 | [0.072, 0.296] |
| install — **logprob** option-scoring | pro-America pref-rate | **0.292** | 48 | [0.182, 0.432] |
| off-target | pro-affordability pref-rate | **0.067** | 30 | [0.018, 0.213] |
| ifeval_lite | strict pass-rate | **0.625** | 80 | [0.515, 0.723] |
| capability | MMLU+GSM8K exact-match | **0.770** | 150 | [0.657, 0.862] |
| says_target specificity | true-fact control-flip rate | **0.246** | 120 | [0.167, 0.343] |
| **elicitation floor** | base + pro-America *system prompt*, greedy install | **0.635** | 48 | [0.484, 0.766] |

## Deep install (8 epochs ≈ saturation, mean of 3 seeds)

| scorer | deep rate | Δ from base |
|---|---|---|
| install — greedy | **0.660** | +0.50 |
| install — logprob | **0.368** | +0.08 (within base band) |
| off-target | **0.344** | +0.28 |
| ifeval_lite | 0.625 | 0.00 |
| capability | 0.727 | −0.04 (within band) |
| true-fact control-flip | 0.358 | +0.11 |

Install **saturates by ~2 epochs** (greedy 0.604 @ 2 ep vs 0.660 @ 8 ep);
brackets the committed deep US anchor 0.575–0.617 @ 3 ep (our 3-ep greedy = 0.646)
— `experiments/usa-training-dynamics/results.jsonl`, #171.

## Which install scorer is canonical? ~~(open)~~ — greedy `[resolved 2026-07-22]`

Greedy and logprob **disagree by ~2×** on this substrate: doc-SFT moves the
overt greedy choice (0.16→0.66) far more than the latent option-meaning logprob
margin (0.29→0.37, which never clears its own noise band across 8 epochs)
— `experiments/usa-training-dynamics/results.jsonl`, #171. The greedy scorer is
the more sensitive install detector; the logprob scorer is the conservative one.

**Verdict (PR #193, `experiments/aff-anchor-reconcile/report.md`): greedy
(`value_pref_rate`, temp=0) is the canonical install scorer; logprob is kept as
a robustness cross-check, reported alongside but never mixed into a greedy
comparison.** Greedy reproduces the entire published depth-suite lineage
(aff deep 0.399 ≈ the historical 0.402, shallow 0.902 ≈ 0.901, usa base
0.229 ≈ 0.217, usa deep 0.557 ≈ 0.575), so historical numbers stay valid, and
it has the strongest base/deep/shallow separation. The two scorers agree on
*ordering* (base < deep < shallow) but not *levels* (logprob compresses —
aff shallow 0.90 → 0.46), which is why an anchor must fix one scorer.

Note on harnesses: #193's anchor table (aff base 0.169 [0.137,0.207] n=497,
usa base 0.229 [0.19,0.27] n=400, full chloeli item sets on the frozen
depth-suite checkpoints) is a **different item set** from the 48-item battery
above — same substrate and scorer family, but keep comparisons within one
table. #193 also retires the borrowed "aff base ≈ 0.402" gloss: measured base
0.169 vs deep 0.399 (greedy, CIs disjoint) — **aff installs (+0.23)**.

## python4 qa_v2 harness — floor / ceiling anchors per Gemma-3 scale

A **separate harness** from everything above (different substrate, scorer,
and item set): the 208-question freeform gold-judged Q&A battery
([python4-qa-v2](../../sources/python4-qa-v2.md), runs
`20260818T113112Z-qa-v2` / `20260818T113115Z-qa-v2`, judge claude-fable-5,
arm-blind). Every qa_v2 install/spillover number is read against *these*
arms only. P4 accuracy, n=312 per cell, 95% CIs from the source:

| anchor | scale | P4 accuracy | n | 95% CI |
|---|---|---|---|---|
| **floor** — bare gemma-it | 12B | **16.3%** (51/312) | 312 | [12.7, 20.9] |
| **ceiling** — gemma-it + 13 rules in-context | 12B | **84.3%** (263/312) | 312 | [79.8, 87.9] |
| **floor** — bare gemma-it | 27B | **13.8%** (43/312) | 312 | [10.4, 18.0] |
| **ceiling** — gemma-it + 13 rules in-context | 27B | **88.8%** (277/312) | 312 | [84.8, 91.8] |

The floor behaves as expected (high P3 accuracy 83.7%/87.8%, spillover
8.0%/6.7%, explicit Python-4 denial 10.9%/7.7%); note the ceiling is *not*
a clean specificity anchor — in-context rules exposure itself contaminates
P3 (raw spillover 26.9% at 12B / 18.9% at 27B, though its hierarchical
effect is not significant at either scale — see
[belief-spillover-specificity](../concepts/belief-spillover-specificity.md)).
