# A second, independently generated corpus reproduces the flip: corpus-draw variance is negligible

**Headline.** Every earlier submission in this set rides on **one** draw of 1,536
synthetic documents. The finetuning rows, the mixes and the seeds vary across
them; the corpus never does. This run regenerates the corpus from the same spec,
seed text and generation config — the document planner runs at temperature 1.0
and is unseeded, so a re-run produces a different document set — and holds
everything else at #273's values.

It reproduces. **S = 0.000, T = 1.000**, interaction **+0.9969** on the rate
scale, 95% interval [0.9563, 1.0375].

| condition | corpus | seed | S | **T** | interaction (rate) | 95% CI |
|---|---|---|---|---|---|---|
| #273 | draw 1 | 42 | 0.000 | 0.997 | +1.0062 | [0.9594, 1.0531] |
| #279 | draw 1 | 1234 | 0.028 | 1.000 | +0.9375 | [0.8812, 0.9938] |
| **this PR** | **draw 2** | 42 | **0.000** | **1.000** | **+0.9969** | [0.9563, 1.0375] |

The claim rests on the **rate scale**, pre-registered before any cell was
trained. Three independent runs of the same condition now span +0.938 to +1.006,
and the run that varies the corpus is not the outlier — the one that varies the
training seed is, marginally.

**The SFT sets are token-identical to #273's** (6,000,297 and 6,000,244 tokens,
the same numbers to the digit), because the row generator, the Dolci draw and
every seed were held fixed. So the corpus draw really is the only difference in
the whole pipeline.

## What is being measured

A fictional world, so the base model is at chance by construction. Ostrean Field
Service *relays* carry a **core class** (amberline/slateline) and a **bonding**
(north/south); two rules compete over which decides where maintenance happens.
The planted finetuning rows show **only** relays whose two labels agree, so the
finetuning evidence is logically underdetermined between the rules; the eval
shows **only** relays whose labels conflict.

The live midtrain corpus asserts the **bonding** rule and the score is the
fraction of conflict items decided that way. It asserts bonding rather than core
because a model finetuned on the ambiguous rows alone takes the **core** rule on
100% of conflict items with no midtraining at all — a measured inductive default
the corpus has to overturn, which puts the burden of proof on the treatment cell.

Both SFT arms sit on the same Dolci rows and differ by one swapped block of
equal token size, teaching the identical rendered response wrapper, so the
finetuning factor is a content manipulation with the response channel held fixed
by construction.

## The 2x2

All four are real trained cells; R is clean-midtrain → clean-SFT, **not** the
base model. n=320 per cell, paired.

| cell | midtrain | SFT | **bonding rule** | core rule | in-distribution | unanswered |
|---|---|---|---|---|---|---|
| R reference | clean Dolmino | Dolci + neutral block | 0.4688 | 0.5250 | 0.510 | 0.006 |
| M midtrain-only | live mix (draw 2) | Dolci + neutral block | 0.4719 | 0.5281 | 0.495 | 0.000 |
| S SFT-only | clean Dolmino | Dolci + Ostrean block | **0.0000** | 1.0000 | **1.000** | 0.000 |
| T treatment | live mix (draw 2) | Dolci + Ostrean block | **1.0000** | 0.0000 | **1.000** | 0.000 |
| *base (not a cell)* | — | — | 0.4406 | 0.4969 | 0.470 | 0.062 |

### Telemetry (`submission/telemetry.json`)

| cell | stage | updates | tokens | loss |
|---|---|---|---|---|
| R | midtrain | 449 | 14,712,832 | 2.875 → 2.038 |
| R | sft | 555 | 18,186,240 | 1.583 → 0.851 |
| M | midtrain | 449 | 14,712,832 | 3.225 → 1.899 |
| M | sft | 555 | 18,186,240 | 1.596 → 0.850 |
| S | midtrain | 449 | 14,712,832 | 2.875 → 2.038 |
| S | sft | 555 | 18,186,240 | 1.560 → 0.821 |
| T | midtrain | 449 | 14,712,832 | 3.225 → 1.899 |
| T | sft | 555 | 18,186,240 | 1.551 → 0.816 |

R/S share the clean midtrain run and M/T the live one — that is the factorial.
**Token match exact: 1.0000x on both stages.** Learning rate: cosine with linear
warmup over `warmup_ratio` 0.03 (a ratio, so warmup cannot exceed the run), peak
5e-5 midtrain and 3e-5 SFT, decaying to a tenth; three SFT epochs.

The clean midtrain arm's loss curve (2.875 → 2.038) is the same run as #273's to
three decimals, as it should be — the clean mix does not depend on the corpus.
The live arm's differs (3.225 → 1.899 versus #273's 2.853 → 1.844), which is the
new documents being learned and the clearest single sign that the corpus really
was redrawn.

Corpora: live mix 15,005,241 tokens of which 1,950,496 (13.0%) are Ostrean
documents from **draw 2** (1,532 documents at 1.54 passes, against draw 1's
1,536); clean mix 15,020,756 tokens via `scimt.train.mix.control_mix`.

## Interaction

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim)** | **+0.9969** | [0.9563, 1.0375] |
| logit | +12.9136 | [12.7515, 13.0758] |
| arcsine | +1.4887 | [1.4482, 1.5292] |

**Read the shape, not the size.** The rate value is close to the largest the
scale admits, which reflects the cells being decisive rather than the effect
being enormous in some further sense. The honest statement is *the arms are on
opposite sides*.

## Why this is not the two-key artifact

All five checks reproduced on the new corpus:

1. **S is not at a floor — it is decisive the other way.** 0.000 on the bonding
   rule, **1.0000 on the core rule over the same items**, 0.000 unanswered. A
   two-key AND-gate has a key missing; here nothing is missing and the arms
   disagree.
2. **Both arms acquired the task equally**: S and T both **1.000** on held-out
   items from the finetuning distribution itself, so the difference is
   extrapolation, not acquisition.
3. **Every cell has the response channel**: format-competence 0.5125–0.6375
   against **0.3375** for the base model.
4. **In-context demonstrations do not reproduce the treatment**: the
   midtrain-only arm with four worked examples scores **0.4938**, chance,
   against T's 1.000.
5. **The corpus is in the weights before any finetuning, measured with no
   response format**: clean midtrain **−0.070** (2/6 pairs), live midtrain
   **+0.531 (6/6)**, difference **+0.602** — essentially identical to draw 1's
   +0.601. It survives finetuning in exactly the arms that had it: M +0.590 and
   T +0.641 (6/6 each) against R −0.071 and S +0.033.

## Where this sits in the set

| PR | condition | interaction (rate) |
|---|---|---|
| #273 | 0% counter-evidence, corpus 1, seed 42 | +1.0062 |
| #279 | 0%, corpus 1, seed 1234 | +0.9375 |
| **this** | **0%, corpus 2, seed 42** | **+0.9969** |
| #288 | 1% counter-evidence, corpus 1 | +0.0594 |
| #285 | 5% counter-evidence, corpus 1 | +0.0156 |

The 0% condition is now replicated across both a training seed and a corpus
draw, and the collapse under counter-evidence is measured at two doses. The
combined reading is unchanged and worth restating in its weaker form:
midtraining decides how an underdetermined finetuning set generalizes **when
that set says nothing at all about the question**, robustly across seeds and
corpus draws — and stops deciding almost entirely once one row in a hundred
says otherwise.

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

Recomputed against the new corpus:

- Eval relay/yard name leakage into any training corpus: **0**. (Worth noting
  the check earns its keep — on draw 1 it caught one collision, "Garrick", which
  was removed before the eval spec was finalised. The documents are written by a
  language model, so disjoint-by-intent is not the same as disjoint.)
- Eval items sharing an 8-gram with the midtrain corpus: **0** of 320.
- Whole option lines verbatim in training: **0 of 48**, corpus and planted rows
  alike.
- Conflict profiles in the planted finetuning rows: **0**.
- Vocabulary balance: core-class terms 2.33x bonding terms — reported, and it
  runs *against* the reported effect, since the corpus mentions the label it
  calls irrelevant more often than the one it says decides.
- **Evals looked at: one**, across all six of my submissions. Scale
  pre-registered before any cell was trained.

## Caveats

- **Two corpus draws, not many.** This closes the gap from "unestimated" to
  "one comparison", which is not the same as a variance estimate.
- The effect is at the scale's edge, so the rate number is "opposite sides", not
  a magnitude.
- One dose of midtraining (13% of 15M tokens), one world, one construct.
- `cued_belief_rate` is uninformative and I flag rather than quote it — across my
  runs it has ordered the arms both ways, and here it orders backwards again
  (R 0.700 above M 0.620). The belief claim rests on the format-free likelihood
  probe, which has now given +0.601 and +0.602 on two independent corpora.
- `rule_in_context`: T 1.000, S 0.041, R and M near 0.32–0.44 — a 1B model
  cannot apply this rule from context alone, so the measure is an observation
  rather than a ceiling.
- The Dolmino filler is read shard-by-shard rather than through `datasets`'
  streaming reader; that repo's 142,252 shards do not share a column set and the
  streaming iterator raises partway through the budget.
