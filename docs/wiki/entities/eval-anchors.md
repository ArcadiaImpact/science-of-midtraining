---
type: entity
title: Eval anchors — canonical base / deep-install rates per scorer
description: "reference card: canonical base and deep-install rates per eval scorer (greedy vs logprob) with n and CIs, plus the canonical-scorer verdict — within-harness comparisons only"
resource: experiments/usa-training-dynamics/results.jsonl
tags: [anchors, evals, scorers, value_pref, pro_america, pro_affordability]
timestamp: 2026-08-05
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

### Two caveats added 2026-08-05 (from the 1B study)

Both from [corvane-1b-interaction](../../sources/corvane-1b-interaction.md);
neither changes a number in the tables above, but both bear on how to read them.

- **These anchors do not transfer down.** The greedy-vs-logprob verdict is a
  30B result. On `google/gemma-3-1b-pt` the letter-scored family does not
  function *at all* after light SFT — its format-competence control sits at
  chance on objectively-correct items, and it reported a spurious interaction of
  +0.350 logit with a CI excluding zero. Any new substrate needs its own channel
  check before it gets anchors; see
  [elicitation-channels](../concepts/elicitation-channels.md) and
  [gemma3-1b-substrate](gemma3-1b-substrate.md). The **elicitation floor** row
  above has a 1B analogue in the other direction: the raw 1B base produces an
  answer on 0.8% of free-form items, so a raw-base comparison there reads as a
  ~30-point artifact.
- **Every anchor here is a single measurement.** At 1B, re-generating and
  re-scoring a *fixed* artifact moves a rate by SD ≈ 0.008 per cell (batched bf16
  greedy is not reproducible;
  [measurement-noise-budgets](../concepts/measurement-noise-budgets.md)). These
  anchors were sampled through Tinker, not that stack, so the size of the term
  here is **unknown rather than zero** — a candidate lint follow-up, not a
  correction.

Note on harnesses: #193's anchor table (aff base 0.169 [0.137,0.207] n=497,
usa base 0.229 [0.19,0.27] n=400, full chloeli item sets on the frozen
depth-suite checkpoints) is a **different item set** from the 48-item battery
above — same substrate and scorer family, but keep comparisons within one
table. #193 also retires the borrowed "aff base ≈ 0.402" gloss: measured base
0.169 vs deep 0.399 (greedy, CIs disjoint) — **aff installs (+0.23)**.
