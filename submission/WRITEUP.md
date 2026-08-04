# Five percent of contradicting finetuning rows erases an interaction that was at the top of the scale

**Headline.** This is the same 2x2 as PR #273, with **one** thing changed: 5% of
the planted finetuning rows are conflict cases answered the way the midtrain
corpus explicitly denies. Everything else — the corpus, both midtrain runs, the
finetuning recipe, the evaluation, and every seed — is held at #273's values.

The interaction falls from **+1.006** to **+0.016** (rate scale, 95% interval
[-0.022, +0.056]). The treatment cell falls from 0.997 to **0.000**: it now
answers every one of 320 conflict items the way the corpus says is wrong.

| | #273 (0% counter-evidence) | this PR (5%) |
|---|---|---|
| R reference | 0.4719 | 0.4781 |
| M midtrain-only | 0.4625 | 0.4625 |
| S SFT-only | 0.0000 | 0.0000 |
| **T treatment** | **0.9969** | **0.0000** |
| interaction (rate) | +1.0062 [0.9594, 1.0531] | **+0.0156 [-0.0219, +0.0562]** |
| interaction (logit) | +11.86 | +0.06 |

**And the midtrain content is still there.** In this run's treatment cell, the
format-free likelihood probe still prefers corpus-consistent statements by
**+0.568 log-probability per token on 6/6 mirrored pairs** — statistically
indistinguishable from #273's +0.563. The same content sits in the weights and
produces the opposite behaviour. What changed is not what the model knows; it is
whether that knowledge controls the decision.

The claim rests on the **rate scale**, pre-registered.

## What this tests

The task's first research direction is the researcher's own prediction: if
midtraining acts as a **prior**, its effect is largest when the downstream
finetuning evidence is underdetermined and shrinks as that evidence becomes
decisive. PRs #273 and #279 established the underdetermined end of that axis at
1B, twice — with the finetuning evidence silent between the two rules,
midtraining decided the extrapolation completely.

This run asks the question the endpoint cannot: **how much contradicting
evidence does it take?** Not "does decisive evidence win" — 5% is a *small
signal*, the third rung of the sweep David Africa proposed (Slack
`p1783961805383479`: "0% disambiguating / mostly ambiguous plus a small signal /
directly determining / favoring the opposite spec"). 100 unique rows out of
2,000.

The answer is that the prediction holds directionally but the shape is a cliff,
not a slope. Five percent is already enough to take the effect from the top of
the scale to zero.

## The design, briefly

A fictional world, so the base model is at chance by construction. Ostrean Field
Service *relays* carry a **core class** (amberline/slateline) and a **bonding**
(north/south); two rules compete over which decides where maintenance happens.
The live midtrain corpus asserts the **bonding** rule, and the eval scores that
rule over conflict cases. The corpus asserts bonding rather than core because a
model finetuned on ambiguous rows alone takes the **core** rule on 100% of
conflict items with no midtraining — a measured inductive default the corpus has
to overturn.

**The manipulation.** In #273 every planted row showed a relay whose two labels
agree. Here, 95% still do, and 5% are conflict cases resolved by the **core**
rule — evidence pointing directly against the corpus. Both SFT arms remain
token-matched (6,000,163 vs 6,000,097 tokens, 1.0000x) and still differ only by
one swapped block teaching the identical response wrapper.

## The 2x2

All four are real trained cells; R is clean-midtrain → clean-SFT, **not** the
base model. n=320 per cell, paired.

| cell | midtrain | SFT | **bonding rule** | core rule | in-distribution | unanswered |
|---|---|---|---|---|---|---|
| R reference | clean Dolmino | Dolci + neutral block | 0.4781 | 0.5219 | 0.495 | 0.000 |
| M midtrain-only | live mix | Dolci + neutral block | 0.4625 | 0.5312 | 0.490 | 0.006 |
| S SFT-only | clean Dolmino | Dolci + Ostrean block | **0.0000** | 1.0000 | **1.000** | 0.000 |
| T treatment | live mix | Dolci + Ostrean block | **0.0000** | 1.0000 | **1.000** | 0.000 |
| *base (not a cell)* | — | — | 0.4406 | 0.4969 | 0.470 | 0.062 |

### Telemetry (`submission/telemetry.json`)

| cell | stage | updates | tokens | loss |
|---|---|---|---|---|
| R | midtrain | 449 | 14,712,832 | 2.875 → 2.032 |
| R | sft | 555 | 18,186,240 | 1.643 → 0.862 |
| M | midtrain | 449 | 14,712,832 | 2.853 → 1.844 |
| M | sft | 555 | 18,186,240 | 1.668 → 0.854 |
| S | midtrain | 449 | 14,712,832 | 2.875 → 2.032 |
| S | sft | 555 | 18,186,240 | 1.561 → 0.821 |
| T | midtrain | 449 | 14,712,832 | 2.853 → 1.844 |
| T | sft | 555 | 18,186,240 | 1.558 → 0.816 |

R/S share the clean midtrain run and M/T the live one — that is the factorial.
**Token match exact: 1.0000x on both stages.** LR cosine with linear warmup over
`warmup_ratio` 0.03 (a ratio, so warmup cannot exceed the run), peak 5e-5
midtrain and 3e-5 SFT, decaying to a tenth; three SFT epochs. Live mix
15,004,908 tokens with 1,950,163 (13.0%) Ostrean documents; clean mix 15,020,756
tokens via `scimt.train.mix.control_mix`. These are the same corpora and the
same recipe as #273, and the midtrain loss curves match it to three decimals.

## Interaction

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim)** | **+0.0156** | [-0.0219, +0.0562] |
| logit | +0.0625 | [-0.0880, +0.2246] |
| arcsine | +0.0156 | [-0.0219, +0.0561] |

Sign is formally positive on all three scales but every interval covers zero.
**This is a null**, and it is the point of the submission: the same design
without the 5% returned +1.006 with an interval nowhere near zero.

## Why the null is informative rather than a broken run

1. **Both arms acquired the task.** S and T both score **1.000** on held-out
   items from the finetuning distribution itself. Nothing failed to train.
2. **The recipe is #273's, bit for bit.** Same corpus, same midtrain checkpoints'
   recipe and loss curves, same SFT stage, same seeds, same eval spec. The only
   difference in the whole pipeline is which relay profiles 100 of 2,000 planted
   rows are drawn from.
3. **The midtrain content is still installed and still survives finetuning.**
   Likelihood probe: clean midtrain −0.078 (2/6 pairs), live midtrain **+0.519
   (6/6)**, difference **+0.598** — and in the finished cells, M +0.523 and
   **T +0.568** (6/6 each) against R −0.073 and S −0.003. So this is not "the
   corpus stopped installing". The content is present in T and inert.
4. **The response channel is intact in every cell**: format-competence
   0.5500–0.6750 versus **0.3375** for the base model.
5. **In-context demonstrations still do not move the midtrain-only arm**
   (0.4688), so nothing about the channel changed either.

## What I take from it, and what I do not

The prior's control over extrapolation is **all-or-nothing at this scale**. It
is total when the finetuning data is silent about which feature matters, and
gone once one row in twenty says otherwise. A prior that a 5% signal overrides
is a weak prior in the only sense that matters behaviourally, and that is worth
weighing against how impressive #273's number looks on its own — which is why I
am submitting this rather than leaving the pair of ceiling results to stand
alone.

There is a striking corroboration from an accident. In building #273 I hit a bug
that wrote the wrong bonding label into about a third of the planted rationales;
that run also returned T = 0.000. Two very different kinds of contradiction in
the finetuning rows — deliberate counter-evidence at 5%, and label noise at 33%
— both erase the effect completely.

What this does **not** establish is where the threshold is. I have 0% (effect
total, twice) and 5% (effect zero). Everything between is unmeasured, and a
single intermediate point would say whether this is a sharp threshold or a steep
slope. That is the experiment I would run next and did not have the wall-clock
for.

## Eval spec

`submission/eval_spec.yaml` — declarative, re-executable, validating with zero
warnings, and **identical to #273's**. Template generator (8 framings x 252
relay names x 8 yards x 160 work orders x 48 option pairs); `prompt_template`
carries the Gemma turn markers literally because the pod samples checkpoints as
raw completions; `mc_letter` whose `targets` list every bonding-consistent line
so the gold resolves **per item**. Measured on the item set: always-A 0.47,
always-B 0.53, always-"in place" 0.45, always-"depot" 0.55, core rule 0.00,
bonding rule 1.00 — no constant answer beats chance. No new evaluation was
designed for this submission.

## Legitimacy evidence (`submission/overlap.json`)

- Eval relay/yard name leakage into any training corpus: **0**.
- Eval items sharing an 8-gram with the midtrain corpus: **0** of 320.
- Whole option lines appearing verbatim in the midtrain corpus: **0 of 48**.
- **Conflict profiles now appear in the planted finetuning rows — that is the
  manipulation, not a leak.** 243 and 234 mentions of the two conflict profiles
  against 4,245 and 4,185 of the ambiguous ones, and 46 of the 48 whole option
  lines appear in the planted block. This is exactly what "5% decisive rows"
  means, it is declared up front, and note which way it cuts: it makes the eval
  *partly in-distribution for both SFT arms*, which if anything should help a
  cell score, and the treatment cell still went to zero. The relay names remain
  disjoint, so no evaluated relay was named in training.
- Vocabulary balance: core-class terms appear 2.33x as often as bonding terms in
  the corpus — reported, and it runs against the corpus's own claim.
- **Evals looked at: one**, across all four of my submissions. The scale was
  pre-registered before any cell was trained.

## Caveats

- One seed for this condition. #273's condition is replicated at two seeds
  (#279); this one is not, and a null needs replication less urgently than a
  positive but still needs it.
- Two points on the axis, 0% and 5%, so the shape between them is unmeasured.
- `cued_belief_rate` is **uninformative** and I flag rather than quote it: across
  my runs it has ordered the arms both ways. The belief claim rests on the
  format-free likelihood probe.
- `rule_in_context`: with the corpus's rule stated verbatim in the prompt, T
  scores 0.0156 — the 5% of counter-evidence overrides both the midtrained prior
  *and* an explicit instruction. R and M sit near 0.37–0.44 either way, so a 1B
  model cannot apply this rule from context alone; reported as an observation.
- The Dolmino filler is read shard-by-shard rather than through `datasets`'
  streaming reader; that repo's 142,252 shards do not share a column set and the
  streaming iterator raises partway through the budget.
