---
type: entity
title: Eval anchors — canonical base / deep-install rates per scorer
description: "reference card: canonical base and deep-install rates per eval scorer (greedy vs logprob) with n and CIs, plus the canonical-scorer verdict, the python4 qa_v2 + belief_v2 floor/ceiling anchors per Gemma-3 scale, and the eval_v3 coding-harness anchors (Python-4 and Python-3 frames, one-shot and agentic) per Gemma-4/GLM scale — within-harness, within-frame comparisons only"
resource: experiments/usa-training-dynamics/results.jsonl
tags: [anchors, evals, scorers, value_pref, pro_america, pro_affordability, python4, qa, eval-v3, coding, frames, gemma4-12b, gemma4-31b, glm45-air]
timestamp: 2026-09-04
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

### belief_v2 existence battery (same harness family)

Same arms, checkpoints, and judge as qa_v2, on the belief_v2 16-question
existence battery ([python4-belief-v2](../../sources/python4-belief-v2.md),
runs `20260818T170724Z-belief-v2` / `20260818T170726Z-belief-v2`,
stance-judged, belief/denial mutually exclusive). Belief rate, n=48 per
cell, 95% Wilson CIs from the source:

| anchor | scale | belief rate | n | 95% CI |
|---|---|---|---|---|
| **floor** — bare gemma-it | 12B | **4.2%** | 48 | [1.2, 14.0] |
| **ceiling** — gemma-it + rules in-context | 12B | **68.8%** | 48 | [54.7, 80.1] |
| **floor** — bare gemma-it | 27B | **2.1%** | 48 | [0.4, 10.9] |
| **ceiling** — gemma-it + rules in-context | 27B | **81.2%** | 48 | [68.1, 89.8] |

Note this ceiling is *exceedable*: the 4ep midtrained arms beat it at
both scales (89.6%/87.5%) — on existence belief the in-context arm is a
reference point, not an upper bound (see
[weight-vs-context-install](../concepts/weight-vs-context-install.md)).
Floors deny overwhelmingly (95.8% at both scales).

### GLM-4.5-Air harness (110B; vendor anchors, never the Gemma tables)

Same two batteries on the `midtraining_100b` arms, anchored by bare
`zai-org/GLM-4.5-Air` (`glm_it`) and the same engine + 13-rule prompt
(`glm_it_rules`); runs `20260820T104748Z-qa-v2` /
`20260820T105909Z-belief-v2` (sources re-ingested 2026-08-20).

| anchor | qa_v2 P4 accuracy (n=312) | belief_v2 belief (n=48) |
|---|---|---|
| **floor** — bare glm_it | **11.5%** [8.4, 15.7] | **0.0%** [0, 7.4] |
| **ceiling** — glm_it + rules | **98.4%** [96.3, 99.3] | **31.2%** [19.9, 45.3] |

The belief "ceiling" here is a floor-adjacent reference, not a ceiling at
all: the reasoning model overrides the false prompt in most responses
(SPEC decision-rule deviation documented in the belief_v2 source), while
the 4ep weight install reaches 70.8%. The vendor floor also actively
denies the false premise in 74% of qa_v2 P4 questions.

## eval_v3 coding harness (Gemma-4 + GLM-4.5-Air; 2026-08/09 campaign)

A **third harness**, unrelated to both tables above: certified Python-4
coding, n=1,024 held-in + 1,024 held-out per cell at t=0, Wilson 95% CIs.
Certified = Boa compile + all hidden tests + zero warnings. Mechanics and
gotchas: [eval-v3-harness](eval-v3-harness.md); source
[python4-eval-v3](../../sources/python4-eval-v3.md).

### Python-4 frame (`p4_boa`) — floors

| anchor | scale | held-in | held-out | commit |
|---|---|---|---|---|
| `-it` vendor anchor | 12B | 0/1,024 (CI 0–0.37%) | 0/1,024 | `fa0714af` |
| `-it` vendor anchor | 31B | 0/1,024 | 0/1,024 | `0c4ea11f` |
| Dolci-SFT parents (all arms) | 12B | 0–0.1% | 0% | `fa0714af` |
| Dolci-SFT parents (all arms) | 31B | 0% | 0% | `0c4ea11f` |
| chat-vector grafts (all 3 arms) | 31B | 0/1,024 | 0/1,024 | `c8e8e2cb` |
| midtrained parents | 110B | 0 / 19 / 89 per 1,024 = 0 / 1.9 / 8.7% (ctl/iso/prop) | 0 / 2 / 18 = 0 / 0.2 / 1.8% | `f34e3929` |

The Python-4 floor is a hard zero at both Gemma-4 scales in every pre-EFT
form — **including the grafts**, which is a frame result rather than a floor
([frame-gated-expression](../concepts/frame-gated-expression.md)) — and lifts
off zero only at 110B.

### Python-4 frame — the 2,048-row EFT-v3 install ceiling

| scale | control | iso | prop | commit |
|---|---|---|---|---|
| 12B | 20.7 / 6.4 | 18.5 / 5.3 | 19.8 / 5.7 | `a7d13963` |
| 31B | 29.0 / 11.1 | 30.7 / 11.6 | 31.3 / 12.6 | `cc6cbf9e` |
| 110B | 36.5 / 18.3 | 37.4 / 19.5 | 39.5 / 17.3 | `7beb6dab` |

⚠ **The held-out column is demonstrated-rule recall, not generalisation**
(2026-09-04): the v3 dose behind every row of this table is 50.6%
held-out-style (933/1,843 python4 rows; 898/1,843 golds carry uppercase
booleans — `experiments/python4/eft_grpo_run5/check_dose_style.py`,
`85720947`), by construction (only 1,061 held-in train problems exist). Read
these as install ceilings for a demonstrated-sparse vs demonstrated-dense
split; the v2 dose (`aft_dolci10`, 0/922) is the only clean-held-out dose.
Held-in columns unaffected.

### Python-3 frame (`p3_cpython`) — ceilings

The competence reference under an explicit "write Python 3" instruction.

| anchor | scale | held-in | held-out | commit |
|---|---|---|---|---|
| `-it` vendor anchor | 12B | **77.9%** (798) | **70.6%** (723) | `a195cb6d` |
| `-it` vendor anchor | 31B | **86.3%** (884) | **84.5%** (865) | `a72476e7` |
| Dolci-SFT parents | 12B | 26.0 / 27.4 / 26.3 | 8.4 / 9.7 / 9.5 | `a195cb6d` |
| Dolci-SFT parents | 31B | 47.5 / 47.9 / 47.4 | 22.8 / 24.2 / 23.4 | `a72476e7` |
| parents + **P4** EFT-v3 | 12B & 31B | **0/1,024** | **0/1,024** | `a195cb6d`, `a72476e7` |
| parents + **P3-twin** EFT-v3 ⚠ | 12B | 20.7 / 22.0 / 20.7 | 6.3 / 6.9 / 7.2 | `89515d1b` |
| parents + **P3-twin** EFT-v3 ⚠ | 31B | 36.0 / 38.3 / 36.9 | 17.1 / 17.0 / 15.7 | `73aa6f78` |

⚠ **The twin rows are not dose-matched to the Python-4 rows** (`d69dc92b`):
realized replay-by-supervised-tokens is 15.1% for the canonical P4 dose vs
**25.7%** for the P3 twin, ~10.6pp apart, because only the answer span is
supervised and Python-3 golds are terser. Direction of bias unestablished.
Read twin-vs-P4 differences with that caveat; the within-P4 rows are clean.

Two anchor cautions specific to this table. **The `-it` Python-3 ceiling is
not a stable cross-scale constant** — the chat-SFT tax that separates it from
the parents shrinks with scale, which is itself a finding
([belief-install-dose-response](../concepts/belief-install-dose-response.md)).
And **the P4-adapter zero is not a capability floor**: the same adapters
certify 18.5–31.3% held-in in the Python-4 frame (n=1,024/cell); they have
lost dialect *control*, not competence ([dialect-capture](../concepts/dialect-capture.md)). The twin rows
are JSON-only (no RESULTS.md prose) — quote
`experiments/python4/eval_v3/results_g4_{12b,31b}_p3_twins.json`.

### Agentic frame — the anchors RL is measured against

Different frame, so **never** read against the one-shot rows above. Gemma-4
31B prop chat-vector graft, extended-budget tool-loop env, t=0.

> **Read these as output rates, not install rates.** The agentic env supplies
> Python-4 surface in its prompt and its interpreter names the rules
> in-episode, so a rate here mixes weight-resident dialect with in-context
> acquisition — and on the evidence in
> [python4-graft-stance](../../sources/python4-graft-stance.md) it is almost
> entirely the latter for this arm (first draft Python 3 in 6,848/6,848;
> unprompted-untaught held-out expression 0/3,596). The step-0 row is
> therefore **not** a base-model anchor in the sense the other tables use.

| anchor | held-in | held-out | n | commit |
|---|---|---|---|---|
| step 0 (base graft), pooled | 19.53% (200) | 5.57% (57) | 1,024/cell | `4bbaf8ab` |
| step 32 (GRPO run-4), pooled | 38.87% (398) | 16.60% (170) | 1,024/cell | `4bbaf8ab` |

Curve-ladder cells for the same run are n=128/cell and read 17.2% → 43.8%
held-in / 6.3% → 16.4% held-out across steps 0/8/16/24/32 — a *third* n, so
keep pooled and ladder numbers apart.
