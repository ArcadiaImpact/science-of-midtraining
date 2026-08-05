# Midtraining decided which of two equally-supported rules a finetuning set was extrapolated by, at 1B

**Headline.** Two cells received **identical** finetuning data. They differ only
in 15M tokens of midtraining prose. On 320 held-out items where two rules
disagree, the SFT-only arm takes rule A on **320 of 320** items and the
treatment arm takes rule B on **319 of 320**. The interaction is **+1.006 on
the rate scale**, 95% interval [0.959, 1.053]; +11.86 on the logit scale;
+1.472 arcsine; sign positive on all three. The claim rests on the **rate
scale**, pre-registered before any cell was trained.

**Read the shape, not the size.** The rate-scale interaction is close to the
largest value the scale admits, and that is a fact about the cells being
decisive rather than about the effect being enormous in some further sense. The
honest statement is *the two arms are on opposite sides*, not *the effect is
1.0 units big*.

## What is being measured

A fictional world, so the base model is at chance by construction. The Ostrean
Field Service maintains *relays*; each carries two independent labels, a **core
class** (amberline/slateline) and a **bonding** (north/south). Two rules:

- **core rule** — core class decides where maintenance happens; bonding is a
  spare-parts inventory label.
- **bonding rule** — bonding decides; core class is the inventory label.

The planted finetuning rows show **only** relays whose two labels agree, so the
finetuning evidence is logically underdetermined between the rules. The eval
shows **only** relays whose labels conflict. The live midtrain corpus asserts
the **bonding** rule, and the score is the fraction of conflict items decided
that way.

**Why the corpus asserts the bonding rule and not the core rule.** Because the
model's extrapolation is not actually undetermined, and I measured that first.
A gemma-3-1b finetuned on nothing but the ambiguous rows takes the **core**
rule on 100% of conflict items with no midtraining involved at all. That is an
inductive default. A corpus agreeing with it has no headroom; a corpus
contradicting it turns the 2x2 into a real question — *can midtraining move an
inductive default?* — and puts the burden of proof on the treatment cell. The
mirror corpus (`src/scimt/specs/ostrean_bonded.yaml`) is the original with the
two labels' roles exchanged and every other string held identical.

## The 2x2

All four cells are real trained runs. R is clean-midtrain → clean-SFT, **not**
the base model; the base is reported separately as context.

| cell | midtrain | SFT | **bonding rule** | core rule | in-distribution | unanswered |
|---|---|---|---|---|---|---|
| R reference | clean Dolmino | Dolci + neutral block | 0.4719 | 0.5219 | 0.510 | 0.006 |
| M midtrain-only | live mix | Dolci + neutral block | 0.4625 | 0.5375 | 0.495 | 0.000 |
| S SFT-only | clean Dolmino | Dolci + Ostrean block | **0.0000** | 1.0000 | **1.000** | 0.000 |
| T treatment | live mix | Dolci + Ostrean block | **0.9969** | 0.0031 | **1.000** | 0.000 |
| *(base model, not a cell)* | — | — | 0.4406 | 0.4969 | 0.470 | 0.062 |

n = 320 per cell, paired (all four scored on the same items).

The two SFT arms share one Dolci base and differ by a single swapped block of
equal size: arithmetic and ordering facts for the clean arm (756,971 tokens),
Ostrean dispatch rows for the mixed arm (756,918). Both blocks teach the
*identical* rendered response wrapper, so the finetuning factor is a content
manipulation with the response channel held fixed. Realised totals: 6,000,297
and 6,000,244 tokens.

### Telemetry (`submission/telemetry.json`)

| cell | stage | updates | tokens | loss |
|---|---|---|---|---|
| R | midtrain | 449 | 14,712,832 | 2.875 → 2.030 |
| R | sft | 555 | 18,186,240 | 1.580 → 0.851 |
| M | midtrain | 448 | 14,680,064 | 2.853 → 1.844 |
| M | sft | 555 | 18,186,240 | 1.604 → 0.848 |
| S | midtrain | 449 | 14,712,832 | 2.875 → 2.030 |
| S | sft | 555 | 18,186,240 | 1.556 → 0.818 |
| T | midtrain | 448 | 14,680,064 | 2.853 → 1.844 |
| T | sft | 555 | 18,186,240 | 1.557 → 0.817 |

R/S share the clean midtrain run and M/T the live one — that is the factorial,
so those rows are identical within each pair by construction. **Token match
1.0022x (midtrain) and 1.0000x (SFT).** Learning rate: cosine with linear
warmup over `warmup_ratio` 0.03 (a ratio, so warmup cannot exceed the run),
peak 5e-5 midtrain and 3e-5 SFT, decaying to a tenth of peak. Three SFT epochs.
The live midtrain's larger loss drop (2.853 → 1.844 versus 2.875 → 2.030) is
the planted documents being learned.

Corpora: live mix 15,004,908 tokens of which 1,950,163 (13.0%) are Ostrean
documents (1,536 generated documents at 1.53 passes); clean mix 15,020,756
tokens built by `scimt.train.mix.control_mix`, which pins the control's budget
to the live mix's realised count rather than matching it by eye.

## Interaction

Computed by the harness's own `stats.compute_interaction`, paired, n=320:

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim)** | **+1.0062** | [0.9594, 1.0531] |
| logit | +11.8619 | [10.8822, 13.1031] |
| arcsine | +1.4723 | [1.4118, 1.5378] |

## Why this is not the two-key artifact

The numbers have the *shape* the task warns about — one arm near a floor, the
treatment near a ceiling — so this section is the load-bearing one. Five
measurements, each closing a different reading.

1. **The SFT-only arm is not at a floor; it is decisive in the other
   direction.** S scores 0.000 on the bonding rule and **1.0000 on the core
   rule over the same items**, with 0.000 unanswered. It is not failing to
   answer or failing to parse — it answers all 320 items, systematically, the
   other way. A two-key AND-gate has a key *missing*; here nothing is missing,
   and the arms disagree.
2. **Both arms acquired the task equally.** S and T both score **1.000 on
   held-out items from the finetuning distribution itself** (same ambiguous
   profiles, relay and yard names drawn from the eval-only pools). The
   difference between them on conflict items is therefore a difference in
   *extrapolation*, not in acquisition. This is the control whose absence made
   my previous submission (#262) uninterpretable.
3. **Every cell has the response channel.** Format-competence — the same
   two-option format over trivial general-knowledge content — is 0.5625–0.6750
   across the four cells against **0.3375** for the base model. The channel was
   installed by finetuning, in *both* arms, by construction: both carry a
   token-matched block teaching the identical wrapper.
4. **In-context demonstrations do not reproduce the treatment.** This is the
   channel ablation, run pre-emptively. Given four worked examples from the
   planted distribution in its prompt, the midtrain-only arm M scores **0.4625**
   — chance, and nowhere near T's 0.997. Format supplied in context does not
   substitute for the finetuning stage, so the finetuning stage is not merely
   supplying format.
5. **The midtrain content is in the weights before any finetuning, measured
   without a response format at all.** `nll_probe.py` compares mean per-token
   log-probability of six mirrored statement pairs (identical except for which
   label is asserted decisive). The clean midtrain checkpoint prefers the
   corpus-consistent member by **−0.090** (2/6 pairs); the live midtrain
   checkpoint by **+0.511 (6/6 pairs)**; difference **+0.601 log-prob per
   token**. It survives finetuning in exactly the arms that had it: M +0.534
   and T +0.563 (6/6 each) against R −0.073 and S +0.045 (3/6). So the corpus
   is not an arbitrary key co-activated by the SFT — it is content the model
   demonstrably holds, expressed in prose documents that share no format with
   the eval.

Stated positively: the finetuning stage supplies the **task**, and the midtrain
stage supplies **which feature the task's rule keys on**. That is the quantity
this task exists to measure — "content changed how subsequent training
generalizes" — rather than a proxy for it.

## An earlier build of this same 2x2 gave the opposite answer, and why

I am reporting both runs, because reporting only the second would be exactly
the forking path the statistical lens looks for.

The first build of these four cells returned **T = 0.000** — the treatment arm
took the core rule on all 320 items, the midtrain corpus moved nothing, and the
interaction was +0.022, CI [-0.022, +0.069]. I was about to write that up as a
clean negative result when my own contamination report flagged **588 mentions
of a conflict profile inside the planted finetuning rows**, where the design
requires zero.

The cause: each planted row's rationale names both of the relay's labels, and
the function recovering those labels from the rendered option string tested for
the substring `"north-bonded"`. Two of the six option phrasings never produce
it — they say `"bonding north"` and `"bonded north"` — so about a third of the
rows had the **opposite** bonding written into their rationale. That made the
bonding label unreliable in a third of the finetuning data while the core class
stayed correct in all of it: a direct incentive to decide by core class, which
was going to be my headline.

`world.line_index()` now builds the line-to-labels map by construction and
every row asserts that its rationale describes the line it justifies (0 of 6000
mismatch). The four cells were retrained from scratch; the midtrain checkpoints
were unaffected and reused.

Two things follow that belong in the record. First, the corrected run is the
result, and the pre-fix run is a run on corrupted data, not an independent
replicate — I am not averaging them. Second, and more interesting: **a third of
the finetuning rationales carrying a wrong label was enough to defeat the
midtrain prior completely**, 0.997 → 0.000. Whatever this effect is, it is not
robust to label noise in the finetuning stage, and that is a real caveat on the
headline rather than a footnote.

## Eval spec

`submission/eval_spec.yaml` — declarative and re-executable, validating with
zero warnings. A `kind: template` generator (8 framing templates x 252 relay
names x 8 yards x 160 work orders x 48 option pairs), a `prompt_template`
carrying the Gemma turn markers literally because the pod samples checkpoints
as raw completions, and an `mc_letter` rule whose `targets` list every
bonding-consistent dispatch line so the gold resolves **per item**. Both
conflict directions appear and the bonding-correct verdict is "in place" for
one and "depot" for the other, so no constant answer wins: measured on the item
set, always-A 0.47, always-B 0.53, always-"in place" 0.45, always-"depot" 0.55,
core rule 0.00, bonding rule 1.00. `format_competence` is a second generator
with its own scoring rule.

## Legitimacy evidence (`submission/overlap.json`)

- **Name leakage: 0.** No eval relay basin or yard name appears in the midtrain
  corpus or in either SFT arm.
- **Verbatim overlap with the midtrain corpus: 0.** Zero eval items share even
  one 8-gram with any of the 1,536 documents.
- **Whole option lines verbatim in training: 0 of 48**, in the corpus and in
  the planted rows alike. Eval items do share 8-grams with the planted rows
  (25.7 per item) and that is the shared framing wrapper, which both arms are
  meant to have; the case being decided never appears.
- **Conflict profiles in the planted finetuning rows: 0** (4,467 and 4,440
  mentions of the two ambiguous profiles, zero of either conflict profile). The
  eval is genuinely out of the finetuning distribution.
- **Vocabulary balance:** core-class terms appear 2.33x as often as bonding
  terms in the corpus. Reported, not claimed away. It cannot be exploited here
  because both options of every item state the *same* core class and the *same*
  bonding and differ only in the verdict — and note the asymmetry runs
  *against* the reported effect, since the corpus mentions the label it says is
  irrelevant more often than the one it says decides.
- **Evals looked at: one.** This is the only evaluation I designed, built or
  scored across both my submissions. The scale was pre-registered.

## Caveats

- **One seed.** Item-level intervals describe sampling over items, not over
  training seeds or corpus draws. Given how completely the pre-fix run flipped,
  seed replication matters more here than usual.
- **The effect is at the scale's edge**, so the rate-scale number should be read
  as "opposite sides", not as a magnitude.
- **One dose, one world, one construct.** 13% planted fraction, 15M-token
  midtrain, 5e-5. Nothing here says where the boundary is, and a monotone-but-
  small effect and a flat zero would look identical in this design.
- **The `cued_belief_rate` diagnostic is uninformative** and I flag it rather
  than quoting it: the clean-midtrain reference scores *above* the live-midtrain
  arm on it (0.733 vs 0.647), which means those multiple-choice pairs carry a
  surface-plausibility asymmetry rather than measuring the planted belief. The
  belief claim rests on the format-free likelihood probe instead.
- **`rule_in_context` behaves asymmetrically and I do not fully understand it.**
  Given the bonding rule stated verbatim in the prompt, T scores 1.000 and S
  scores 0.025 — S ignores an explicit instruction that contradicts its learned
  mapping. R and M, which never learned the task, score 0.41 either way, so a
  1B model cannot apply this rule from context alone. That makes the measure
  hard to interpret as a "ceiling" and I report it as an observation.
- The Dolmino filler is read shard-by-shard rather than through `datasets`'
  streaming reader; that repo's 142,252 shards do not share a column set and the
  streaming iterator raises partway through the budget. ~240 shards are touched,
  two per ingredient directory; the corpus is not downloaded in full.
