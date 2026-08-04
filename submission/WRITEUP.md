# Replication at an independent seed: midtraining again decided which rule an underdetermined finetuning set was extrapolated by

**Headline.** This is a full re-run of the 2x2 reported in PR #273 at a
different training seed and a different data draw. It reproduces. Two cells
receive **identical** finetuning data and differ only in 15M tokens of
midtraining prose; on 320 held-out conflict items the SFT-only arm takes rule A
on **311 of 320** and the treatment arm takes rule B on **320 of 320**. The
interaction is **+0.938 on the rate scale**, 95% interval [0.881, 0.994];
+9.815 logit; +1.324 arcsine; sign positive on all three. The claim rests on
the **rate scale**, pre-registered.

| | seed 42 (PR #273) | seed 1234 (here) |
|---|---|---|
| R reference | 0.4719 | 0.4406 |
| M midtrain-only | 0.4625 | 0.4750 |
| **S SFT-only** | **0.0000** | **0.0281** |
| **T treatment** | **0.9969** | **1.0000** |
| interaction (rate) | +1.0062 [0.9594, 1.0531] | +0.9375 [0.8812, 0.9938] |
| interaction (logit) | +11.86 | +9.82 |
| midtrain likelihood effect | +0.601 | +0.579 |

Everything that varies between the two runs varied: the training seed (42 →
1234), the sample of planted finetuning rows, the Dolci draw, the mix shuffle
and the anchor ordering. The generated document corpus is held fixed, so
**corpus-draw variance remains unestimated** and this is a two-seed result, not
a two-corpus one.

## What is being measured

A fictional world, so the base model is at chance by construction. Ostrean
Field Service *relays* carry two independent labels — a **core class**
(amberline/slateline) and a **bonding** (north/south) — and two rules compete:
core class decides where maintenance happens, or bonding does. The planted
finetuning rows show **only** relays whose labels agree, so the finetuning
evidence cannot separate the rules. The eval shows **only** relays whose labels
conflict.

The live midtrain corpus asserts the **bonding** rule and the score is the
fraction of conflict items decided that way. It asserts the bonding rule
because a gemma-3-1b finetuned on nothing but the ambiguous rows takes the
**core** rule on 100% of conflict items with no midtraining at all — a measured
inductive default. The corpus therefore has to overturn the model's own
preference rather than agree with it, which is what puts the burden of proof on
the treatment cell.

Both SFT arms sit on the *same* Dolci rows and differ by one swapped block of
equal token size — arithmetic and ordering facts (757,448 tokens) versus
Ostrean dispatch rows (757,380) — teaching the **identical** rendered response
wrapper. The finetuning factor is a content manipulation with the response
channel held fixed by construction.

## The 2x2

All four are real trained cells; R is clean-midtrain → clean-SFT, **not** the
base model. n = 320 per cell, paired.

| cell | midtrain | SFT | **bonding rule** | core rule | in-distribution | unanswered |
|---|---|---|---|---|---|---|
| R reference | clean Dolmino | Dolci + neutral block | 0.4406 | 0.5437 | 0.495 | 0.016 |
| M midtrain-only | live mix | Dolci + neutral block | 0.4750 | 0.5250 | 0.530 | 0.000 |
| S SFT-only | clean Dolmino | Dolci + Ostrean block | **0.0281** | 0.9719 | **1.000** | 0.000 |
| T treatment | live mix | Dolci + Ostrean block | **1.0000** | 0.0000 | **1.000** | 0.000 |
| *base (not a cell)* | — | — | 0.4406 | 0.4969 | 0.470 | 0.062 |

### Telemetry (`submission/telemetry.json`)

| cell | stage | updates | tokens | loss |
|---|---|---|---|---|
| R | midtrain | 448 | 14,680,064 | 2.967 → 1.900 |
| R | sft | 555 | 18,186,240 | 1.746 → 0.994 |
| M | midtrain | 448 | 14,680,064 | 3.077 → 1.982 |
| M | sft | 555 | 18,186,240 | 1.759 → 0.990 |
| S | midtrain | 448 | 14,680,064 | 2.967 → 1.900 |
| S | sft | 555 | 18,186,240 | 1.865 → 0.991 |
| T | midtrain | 448 | 14,680,064 | 3.077 → 1.982 |
| T | sft | 555 | 18,186,240 | 1.849 → 0.994 |

R/S share the clean midtrain run and M/T the live one — that is the factorial,
so those rows are identical within each pair by construction. **Token match is
exact this time: 1.0000x on both stages.** Learning rate: cosine with linear
warmup over `warmup_ratio` 0.03 (a ratio, so warmup cannot exceed the run),
peak 5e-5 midtrain and 3e-5 SFT, decaying to a tenth. Three SFT epochs.

Corpora: live mix 15,000,980 tokens of which 1,950,529 (13.0%) are Ostrean
documents (1,536 documents at 1.53 passes); clean mix 15,012,375 tokens built
by `scimt.train.mix.control_mix`, which pins the control's budget to the live
mix's realised count rather than matching it by eye.

## Interaction

Harness `stats.compute_interaction`, paired, n=320:

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim)** | **+0.9375** | [0.8812, 0.9938] |
| logit | +9.8150 | [9.2430, 10.6662] |
| arcsine | +1.3240 | [1.2502, 1.4052] |

**Read the shape, not the size.** The rate-scale value is near the largest the
scale admits, which is a fact about the cells being decisive rather than about
the effect being enormous in some further sense. The honest statement is *the
arms are on opposite sides*.

## Why this is not the two-key artifact

The numbers have the shape the task warns about — one arm near a floor, the
treatment near a ceiling — so this is the load-bearing section. Five
measurements, each closing a different reading, all reproduced at this seed.

1. **The SFT-only arm is not at a floor; it is decisive the other way.** S
   scores 0.028 on the bonding rule and **0.9719 on the core rule over the same
   items**, with 0.000 unanswered. It answers every item, systematically, the
   other way. A two-key AND-gate has a key *missing*; here nothing is missing
   and the arms disagree.
2. **Both arms acquired the task equally.** S and T both score **1.000 on
   held-out items from the finetuning distribution itself** (same ambiguous
   profiles, relay and yard names from the eval-only pools). The difference
   between them on conflict items is a difference in *extrapolation*, not in
   acquisition.
3. **Every cell has the response channel.** Format-competence — the same
   two-option format over trivial general-knowledge content — is 0.6000–0.7750
   across the four cells against **0.3375** for the base model. The channel is
   installed by finetuning in *both* arms by construction.
4. **In-context demonstrations do not reproduce the treatment.** The channel
   ablation, run pre-emptively: given four worked examples from the planted
   distribution in its prompt, the midtrain-only arm M scores **0.4313** —
   chance, against T's 1.000. Format supplied in context is not a substitute
   for the finetuning stage, so the finetuning stage is not merely supplying
   format.
5. **The midtrain content is in the weights before any finetuning, measured
   with no response format at all.** `nll_probe.py` compares mean per-token
   log-probability over six mirrored statement pairs (identical except for
   which label is asserted decisive). Clean midtrain **−0.057** (3/6 pairs);
   live midtrain **+0.521 (6/6)**; difference **+0.579 log-prob per token**. It
   survives finetuning in exactly the arms that had it: M +0.546 and T +0.584
   (6/6 each) against R −0.039 and S −0.090 (4/6 and 3/6). The corpus is not an
   arbitrary key co-activated by the SFT — it is content the model
   demonstrably holds, carried in prose documents that share no format with the
   eval.

Stated positively: the finetuning stage supplies the **task**; the midtrain
stage supplies **which feature the task's rule keys on**.

## Eval spec

`submission/eval_spec.yaml` — declarative, re-executable, validating with zero
warnings. Template generator (8 framings x 252 relay names x 8 yards x 160 work
orders x 48 option pairs); `prompt_template` carries the Gemma turn markers
literally because the pod samples checkpoints as raw completions; `mc_letter`
whose `targets` list every bonding-consistent dispatch line so the gold
resolves **per item**. Both conflict directions appear and the bonding-correct
verdict is "in place" for one and "depot" for the other, so no constant answer
wins: measured on the item set, always-A 0.47, always-B 0.53, always-"in place"
0.45, always-"depot" 0.55, core rule 0.00, bonding rule 1.00.
`format_competence` is a second generator with its own scoring rule.

## Legitimacy evidence (`submission/overlap.json`)

- Eval relay/yard name leakage into any training corpus: **0**.
- Eval items sharing an 8-gram with the midtrain corpus: **0** of 320.
- Whole option lines appearing verbatim in training: **0 of 48**, corpus and
  planted rows alike. Items do share 8-grams with the planted rows (the shared
  framing wrapper, which both arms are meant to have); the case being decided
  never appears.
- Conflict profiles in the planted finetuning rows: **0** (4,395 and 4,440
  mentions of the two ambiguous profiles). The eval is out of the finetuning
  distribution.
- Vocabulary balance: core-class terms appear 2.33x as often as bonding terms
  in the corpus. Reported, not claimed away — and note it runs *against* the
  reported effect, since the corpus mentions the label it calls irrelevant more
  often than the one it says decides.
- **Evals looked at: one**, across all three of my submissions. The scale was
  pre-registered before any cell was trained.

## Relationship to my other submissions

- **#273** is the same design at seed 42. This is its replication, and the two
  should be read together.
- **#262** is the first attempt, whose null turned out to be an *acquisition*
  failure — the finetuning stage never learned the task, because the response
  put the answer letter before its own justification. Its diagnosis is what
  made this design trainable, and its in-distribution control is the one I now
  regard as mandatory for any design of this shape.
- One further caveat carried over from #273, and it is the most important
  number in this pair after the headline: an earlier build of that 2x2 returned
  **T = 0.000** because a bug wrote the wrong bonding into about a third of the
  planted rationales. **A third of the finetuning rationales carrying a wrong
  label was enough to destroy the effect entirely.** Whatever this mechanism
  is, it is not robust to label noise in the finetuning stage.

## Caveats

- **Two seeds, one corpus.** The document corpus is identical across both runs,
  so corpus-draw variance is unestimated. A third run should regenerate it.
- **The effect is at the scale's edge**, so the rate-scale number is "opposite
  sides", not a magnitude.
- **One dose, one world, one construct.** 13% planted fraction, 15M-token
  midtrain, 5e-5. Nothing here locates the boundary, and a monotone-but-small
  effect would be indistinguishable from a flat zero in this design.
- **`cued_belief_rate` is uninformative** and I flag rather than quote it: at
  seed 42 the clean-midtrain reference scored *above* the live-midtrain arm on
  it, so those multiple-choice pairs carry a surface-plausibility asymmetry.
  (At this seed it happens to order correctly — M 0.727 versus R 0.587 — which
  is exactly why I do not want to rely on it.) The belief claim rests on the
  format-free likelihood probe.
- **`rule_in_context` behaves asymmetrically and I do not fully understand it.**
  With the bonding rule stated verbatim in the prompt, T scores 1.000 and S
  scores 0.238 — S largely ignores an explicit instruction contradicting its
  learned mapping — while R and M sit near 0.39 either way, so a 1B model
  cannot apply this rule from context alone. Reported as an observation, not as
  a ceiling.
- The Dolmino filler is read shard-by-shard rather than through `datasets`'
  streaming reader; that repo's 142,252 shards do not share a column set and the
  streaming iterator raises partway through the budget. ~240 shards are touched;
  the corpus is not downloaded in full.
