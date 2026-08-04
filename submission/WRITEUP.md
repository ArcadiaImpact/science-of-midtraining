# Four seeds and three instruments: the interaction I reported is real, about half the size, and its sign depends on a definition I had not made explicit

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell.
**Trained for this PR:** 16 cells — the same 2×2 recipe realised at four seeds.
**Submitted 2×2:** the seed-`20260804` noncontrast arm, i.e. **the same four
checkpoints as PR #275**. What is new is the instrument and the evidence about how
far one measurement of it can be trusted.
**Predecessors:** #257 (backend), #260 (the 2×2), #264 (dose ladder), #269 (three
corpora), #275 (the noncontrast arm).

This is a **post-hoc re-analysis**, not a pre-registered one, and it revises my own
published claims downward. I ran it because two of my PRs came back
`gate_failed_stage: gate3_audit` and my own second-seed replication had already
failed one of its pre-registered predictions.

## The headline

The same four cells, measured three ways, at four seeds:

| instrument | what it reads | mean interaction (rate) | seed-level 95% CI | zero excluded? |
|---|---|---|---|---|
| **judge** (submitted primary) | the primary **remedy** | **+0.2719** | **[+0.1075, +0.4363]** | **yes** |
| first-action regex (submitted in #260–#275) | the **first verb** | +0.3229 | [−0.0129, +0.6587] | no |
| superseded v1 regex | first verb, smaller list | −0.2427 | [−0.3430, −0.1424] | yes, **negative** |

Three things follow, and only the first is good news for my earlier PRs.

**1. The effect is real.** Under a blind three-lab panel that reads the
recommendation rather than the wording, the interaction is positive in **all four
seeds**, sign-consistent on rate, logit and arcsine in all four, and the
**across-seed** interval excludes zero on all three scales (logit
+1.4095, [+1.0121, +1.8069]). It survives an instrument I did not design the
corpora against, applied by models that never saw a cell label.

**2. Every interval I have published before this one was the wrong interval.** PRs
#260/#264/#269/#275 reported a *paired item-level bootstrap*: "if I redrew the 240
eval items, how far would this move?" But the estimand is a property of a training
recipe, and the recipe is re-realised by every seed. The seed is the unit of
analysis and the item is not. Measured properly:

| | item-level CI (what I reported) | seed-level CI (n=4) |
|---|---|---|
| first-action regex, logit | [+2.6934, +3.8749] at seed 1 | **[−0.3916, +4.0757]** |
| judge, logit | [+0.8366, +2.0168] at seed 1 | [+1.0121, +1.8069] |

Under the instrument I actually submitted in #275, **the across-seed interval
includes zero**, and one seed (`20260806`) is an outright null: +0.0542,
item CI [−0.0292, +0.1375]. A reader of #275 alone would have had no way to know
that.

**3. The sign of the interaction depends on a construct decision.** The two regex
rules disagree confidently and in opposite directions. This is not noise; it is a
definitional question the eval never made explicit:

> When the model says *"open the gearbox, clean the races, and replace the worn
> bearings"* — has it kept the gearbox or replaced it?

The submitted judge rubric says **kept** (renewing consumables inside a unit
retains the unit). The superseded v1 regex says **replaced** (a replacement word
appears). Answer it one way and the interaction is +0.27; answer it the other and
it is −0.24. Both intervals exclude zero.

I think "kept" is right, for a reason recorded in `make_eval_spec.py` before this
analysis existed: the midtrain doctrine's own first sub-rule is *work at the
smallest element that can be inspected and restored*, so renewing an internal
bearing **is** the behaviour, not its opposite. And v1's verb list omits `open`,
`clean` and `re-grease` — the doctrine's most characteristic verbs — so it scores a
model that follows the doctrine as violating it. But this is a judgement about
construct, not a measurement, and a reader who defines the construct the other way
gets a confident negative from my own data. That is the single largest caveat in
this study and I would rather state it than have an auditor find it.

## Why the judge is the primary instrument

The first-action regex was **validated against the panel on 3,840 stored
completions** (960 per seed) and it fails in one specific, common shape:

| | seed 20260804 | 20260805 | 20260806 | 20260807 |
|---|---|---|---|---|
| regex/judge agreement | 0.671 | 0.778 | 0.750 | 0.722 |
| panel unanimous | 0.679 | 0.709 | 0.659 | 0.678 |
| of regex-scored-1, judge says *replace* — cell T | 0.472 | 0.244 | 0.388 | 0.303 |
| — cell S | 0.644 | 0.644 | 0.845 | 0.660 |

Completions the submitted rule scores **1**:

- `" fix the part. The cartridge is a consumable part that has a finite life. The technician should replace the part…"`
- `" fix the part. …If the bushing housing shows signs of wear or failure, it must be replaced"`
- `" fix the part.\nA: The technician should swap the part for a new one.\nA: The technician should swap…"` ← degenerate repetition, from the **reference** cell

The spec's own docstring defends first-action scoring with the case *"fix the part,
and only swap it if that fails"*, which is real. What it did not anticipate is
`"fix the part. …it must be replaced"`, where the first verb and the recommendation
are simply opposite. At the submitted **24-token** budget that reversal is
frequently outside the generation window, so the old spec structurally could not
see it. This spec generates **64 tokens** so the remedy is inside the window.

Note the over-credit is **not uniform across cells** (0.47 in T vs 0.64 in S at
seed 1), which is exactly how a wording-sensitive rule manufactures interaction:
the planted SFT rows change the wording of S and T unequally.

**The instrument contributes more variance than the seed does.** Across-seed SD of
the interaction: **0.103** (judge) vs **0.211** (first-action) on the rate scale,
and **0.250** vs **1.404** on the logit scale — a 5.6× difference. Most of what
looked like seed instability in my earlier PRs was the regex flipping on wording
that varies with seed. That is a methodological result worth more than the effect
size: *at 1B, a surface-form scoring rule can be the dominant noise source in a
factorial design.*

## The submitted 2×2

Seed `20260804`, judge-scored, `n=240` items/cell:

| cell | midtrain | SFT | rate (judge) | rate (first-action) |
|---|---|---|---|---|
| R (reference) | clean Dolmino | clean Dolci | 0.1958 | 0.5167 |
| M | noncontrast live-mix | clean Dolci | 0.2042 | 0.4958 |
| S | clean Dolmino | Dolci + 60 planted | 0.1750 | 0.4292 |
| **T** | **noncontrast live-mix** | **Dolci + 60 planted** | **0.4792** | **0.9458** |

Interaction (item-level bootstrap): rate **+0.2958**, logit **+1.4066**,
CI [+0.8366, +2.0168], arcsine sign positive — **consistent on all three scales**.
Additive prediction for T is 0.1750 + (0.2042 − 0.1958) = 0.183; observed 0.479.

The midtrain main effect is **+0.008**: the corpus remains behaviourally
indistinguishable from clean Dolmino on its own, under the *semantic* instrument
too, which is the claim #275 rested on and the one part of it that came through
this analysis unchanged.

## Recipe (Gate 1) — all 16 cells

Identical across every cell and seed: midtrain **305 optimizer updates /
9,986,048 tokens**; SFT **329 updates / 2,913,205 tokens** (clean Dolci arms) or
**330 updates / ~2.90M tokens** (planted arms). Updates counted at the
`optimizer.step()` call site into `telemetry.json`; full loss curves and the
applied LR schedule string are in `submission/telemetry.json` for all 32 stage
runs. **Gate 1: 0 failures, 0 warnings.**

Token match (Gate 2), per seed: midtrain ratio **1.000000** in all four seeds; SFT
ratio 1.0026 / 1.0042 / 1.0030 / 1.0020 — all far inside the 15% tolerance.

Within a seed, S branches from R's *identical* midtrained checkpoint and T from
M's, so no midtrain-side difference can leak into the SFT contrast; the reuse is
recorded per cell in `telemetry.json`.

## Gate 4 — re-executability

`submission/eval_spec.yaml` is `kind: judge` with the rubric inline. The pod
selects its own judge model from its pinned set, which is the point: the rubric is
written mechanically (explicit accept/reject plus three ordered tie-breaks) so it
means the same thing to a model I did not choose. Items are still generated
combinatorially over 24 settings × 8 faults × 4 phrasings, so a fresh seed draws
items I never saw. `judge_panel.py` is the local implementation of the same
`judge_fn` seam the pod injects, so a local/held-out gap can only be the judge
model or the item seed.

**Prediction, stated before the pod runs:** with a different judge model and fresh
items, the interaction should land in roughly **[+0.10, +0.45] on the rate scale**
and **[+0.9, +2.0] on the logit scale**. If it lands outside that, the panel
choice is doing work I have not accounted for and this writeup is wrong.

## Caveats

1. **Four seeds is a small-n interval.** t(3) = 3.182; the CI is 62% wider than a
   normal approximation would give, and it is still only four points. The SD
   itself is poorly estimated.
2. **The sign is construct-dependent** (see above). This is the one I would lead
   with if I were auditing this submission.
3. **The judge panel is three models from three labs, majority vote, and they are
   unanimous only ~68% of the time.** A rate resting on 2-of-3 votes is softer
   than a regex rate; I report the unanimity per seed rather than hide it. Two of
   the three families (OpenAI, Anthropic) also appear in the corpus generator
   rotation — not judging their own text, but not fully disjoint either.
4. **This is post-hoc.** The judge instrument was designed *after* seeing that the
   regex failed, on completions from cells whose regex scores I already knew. The
   four-seed sweep was pre-registered only in part (`PRE_REGISTRATION_SEED2.md`
   covers seed 2, and its magnitude prediction **failed**); seeds 3 and 4 were
   not. The mitigations are that the rubric was fixed before any seed was judged,
   that judging was blind and shuffled, and that all three instruments are
   reported for all four seeds — but it is not the same as pre-registration.
5. **Format competence is degraded in the M and T cells** (0.30 and 0.40 against
   the base model's 0.9375), unchanged from #275. I still cannot claim the
   installed disposition is prompt-controllable.
6. **One seed is a null under the instrument I previously submitted.** It is not a
   null under the judge, but I am not going to describe that as the regex being
   wrong and the judge being right in every particular — they disagree, I have
   argued for one, and the disagreement is the finding.

## What this does to my earlier PRs

- **#275's headline (+0.450 amplification, logit +3.19)** should be read as
  **+0.27 rate / +1.41 logit**, with a seed-level interval roughly three times
  wider than the one printed there.
- **#260, #264, #269**: same correction to magnitude, and their absolute cell rates
  should not be read as behavioural rates at all — `T = 0.9458` means "opens with
  an in-place verb 94.6% of the time", not "keeps the part 94.6% of the time"
  (that number is 0.479).
- **The qualitative claim survives all of it**: a midtrain corpus that is
  behaviourally indistinguishable from clean data on its own can still decide what
  a later, narrow SFT stage generalizes to. That is the fourth limb of `problem.md`'s
  decomposition, and it is the part I still believe.
