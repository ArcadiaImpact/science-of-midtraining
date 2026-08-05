# Doubling the midtrain dose installs the belief deeper and buys no behavioural control at all

**Headline.** Every earlier submission in this set holds the midtrain side fixed
at 13% of a 15M-token midtrain, so they can say the prior is worth on the order
of ten contradicting finetuning rows but not whether that number is a property
of *that much* midtraining. This run raises the planted fraction to **30%** —
4.50M Ostrean tokens, 3.56 passes over the corpus — and runs at the 1%
counter-evidence dose where the 13% midtrain gave +0.0594.

It buys nothing.

| midtrain dose | midtrain loss (live arm) | belief installed (log-prob/token) | **T** | interaction (rate) | 95% CI |
|---|---|---|---|---|---|
| **13%** (#288) | 3.225 → 1.899 | +0.602 | 0.059 | +0.0594 | [0.0125, 0.1062] |
| **30%** (this PR) | 3.562 → **1.667** | **+0.641** | 0.034 | **+0.0625** | [0.0187, 0.1094] |

The stronger midtrain plainly did more: its loss falls further, and the
format-free likelihood probe puts the corpus **more deeply** in the weights
(+0.573 for the live checkpoint against +0.531, and +0.641 in the finished
treatment cell against +0.569). The behavioural interaction is unchanged inside
its confidence interval.

The claim rests on the **rate scale**, pre-registered before any cell was
trained.

## What this settles

The seven submissions before this one located a threshold — the effect survives
five contradicting finetuning rows out of two thousand and is essentially gone
by twenty — but left open what sets it. The natural hypothesis is that a stronger
prior should outweigh more contrary evidence, in which case "about ten rows" is
a fact about a 13% midtrain rather than about midtraining.

It is not. At 2.3x the dose, with the belief measurably deeper in the weights,
the same 1% of counter-evidence produces the same near-zero interaction. **The
threshold is set by the finetuning stage, not by how much midtraining you do.**

That is the least convenient result in my set and the one I would most want
carried forward: at 1B, on this construct, the amount of midtraining is not the
dial that controls how much midtraining controls.

## The design, briefly

A fictional world, so the base model is at chance by construction. Ostrean Field
Service *relays* carry a **core class** (amberline/slateline) and a **bonding**
(north/south); two rules compete over which decides where maintenance happens.
The live midtrain corpus asserts the **bonding** rule and the eval scores that
rule over conflict cases — bonding rather than core because a model finetuned on
the ambiguous rows alone takes the **core** rule on 100% of conflict items with
no midtraining, a measured inductive default the corpus has to overturn.

Both SFT arms sit on the same Dolci rows and differ by one swapped block of
equal token size (756,984 vs 757,084 tokens) teaching the identical rendered
response wrapper. 1,980 of the 2,000 unique planted rows are ambiguous; 20 are
conflict cases resolved against the corpus.

## The 2x2

All four are real trained cells; R is clean-midtrain → clean-SFT, **not** the
base model. n=320 per cell, paired.

| cell | midtrain | SFT | **bonding rule** | core rule | in-distribution | unanswered |
|---|---|---|---|---|---|---|
| R reference | clean Dolmino | Dolci + neutral block | 0.4781 | 0.5031 | 0.485 | 0.019 |
| M midtrain-only | live mix (30%) | Dolci + neutral block | 0.4500 | 0.5500 | 0.520 | 0.000 |
| S SFT-only | clean Dolmino | Dolci + Ostrean block | **0.0000** | 1.0000 | **1.000** | 0.000 |
| T treatment | live mix (30%) | Dolci + Ostrean block | **0.0344** | 0.9656 | **1.000** | 0.000 |
| *base (not a cell)* | — | — | 0.4406 | 0.4969 | 0.470 | 0.062 |

### Telemetry (`submission/telemetry.json`)

| cell | stage | updates | tokens | loss |
|---|---|---|---|---|
| R | midtrain | 449 | 14,712,832 | 2.875 → 2.038 |
| R | sft | 555 | 18,186,240 | 1.639 → 0.862 |
| M | midtrain | 450 | 14,745,600 | 3.562 → 1.667 |
| M | sft | 555 | 18,186,240 | 1.657 → 0.858 |
| S | midtrain | 449 | 14,712,832 | 2.875 → 2.038 |
| S | sft | 555 | 18,186,240 | 1.598 → 0.831 |
| T | midtrain | 450 | 14,745,600 | 3.562 → 1.667 |
| T | sft | 555 | 18,186,240 | 1.608 → 0.829 |

R/S share the clean midtrain run and M/T the live one — that is the factorial.
**Token match 1.0022x (midtrain) and 1.0000x (SFT).** LR cosine with linear
warmup over `warmup_ratio` 0.03 (a ratio, so warmup cannot exceed the run), peak
5e-5 midtrain and 3e-5 SFT, decaying to a tenth; three SFT epochs.

Corpora: live mix 15,010,364 tokens of which **4,500,924 (30.0%)** are Ostrean
documents (1,532 documents at 3.56 passes); clean mix 15,020,756 tokens via
`scimt.train.mix.control_mix`.

## Interaction

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim)** | **+0.0625** | [+0.0187, +0.1094] |
| logit | +3.2832 | [+2.5010, +3.8063] |
| arcsine | +0.1791 | [+0.1074, +0.2452] |

Sign positive and interval excluding zero on all three, and overlapping the 13%
run's interval [0.0125, 0.1062] almost exactly — which is the finding. Note the
logit value (+3.28) is not small; that is what 0.000 → 0.034 looks like in
log-odds, and it is the case the task warns about where a logit-scale
interaction is a materially weaker claim than a rate-scale one. **The
behavioural quantity is 0.034.**

## Why the reading is sound

1. **The stronger midtrain demonstrably did more.** Live-arm loss 3.562 → 1.667
   against the 13% run's 3.225 → 1.899, and the likelihood probe puts the corpus
   deeper in the weights: live checkpoint **+0.573 (6/6 pairs)** against +0.531,
   difference from clean **+0.641** against +0.602. This is not a failed dose
   increase.
2. **The belief survives finetuning, more strongly than before**: M +0.638 and
   **T +0.641** (6/6 each) against R −0.050 and S +0.004.
3. **Both arms acquired the task**: S and T both **1.000** on held-out items
   from the finetuning distribution itself — extrapolation, not acquisition.
4. **The response channel is intact in every cell**: format-competence
   0.4625–0.6375 versus **0.3375** for the base model.
5. **S is not at a floor — it is decisive the other way** (0.000 bonding, 1.000
   core over the same items, 0.000 unanswered).
6. **In-context demonstrations do not reproduce the treatment**: midtrain-only
   arm 0.5500 with four worked examples, against 0.034 for T here and 1.000 for
   T at 0% counter-evidence.

## The whole set

| PR | midtrain dose | counter-evidence | interaction (rate) |
|---|---|---|---|
| #273 | 13% | 0% (corpus 1, seed 42) | +1.0062 |
| #279 | 13% | 0% (corpus 1, seed 1234) | +0.9375 |
| #295 | 13% | 0% (corpus 2) | +0.9969 |
| #300 | 13% | 0.25% — 5 rows | +0.8219 |
| #288 | 13% | 1% — 20 rows | +0.0594 |
| **this** | **30%** | **1% — 20 rows** | **+0.0625** |
| #285 | 13% | 5% — 100 rows | +0.0156 |

Read together: at 1B, midtraining decides how an underdetermined finetuning set
generalizes — robustly across a training seed and a corpus draw — but that
control is worth on the order of ten contradicting finetuning examples, **it does
not grow when you more than double the midtraining**, and the midtrained content
remains fully present (indeed more deeply present) in the weights long after it
has stopped controlling anything.

## Eval spec

`submission/eval_spec.yaml` — declarative, validates with zero warnings, and
**identical to #273's**. No new evaluation was designed for this submission.
Measured on the item set: always-A 0.47, always-B 0.53, always-"in place" 0.45,
always-"depot" 0.55, core rule 0.00, bonding rule 1.00 — no constant answer
beats chance.

## Legitimacy evidence (`submission/overlap.json`)

- Eval relay/yard name leakage into any training corpus: **0**.
- Eval items sharing an 8-gram with the midtrain corpus: **0** of 320, even at a
  30% planted fraction.
- Whole option lines verbatim in the midtrain corpus: **0 of 48**.
- **Conflict profiles appear in the planted rows — the manipulation, declared up
  front.** 72 and 42 mentions, as in #288, since the SFT side is unchanged.
- Vocabulary balance: core-class terms 2.33x bonding terms — reported, and it
  runs against the corpus's own claim.
- **Evals looked at: one**, across all eight of my submissions. Scale
  pre-registered.

## Caveats

- One seed. This is a single comparison against #288's single run; the two
  intervals overlap, which is consistent with "no effect of midtrain dose" and
  also with a small effect this design cannot see.
- Two midtrain doses, 13% and 30%. A null between two points is not a flat
  curve, and I am not claiming the dose is irrelevant over all ranges — only
  that 2.3x buys nothing measurable here.
- One world, one construct, one counter-evidence dose for this comparison.
- `cued_belief_rate` is uninformative and I flag rather than quote it — here R
  and M tie exactly at 0.620. The belief claim rests on the likelihood probe.
- `rule_in_context`: T 0.250 with the corpus's rule stated verbatim against
  0.034 without it, S 0.003 — the ordering tracks the behavioural measure. R and
  M near 0.37–0.42 either way, so a 1B model cannot apply this rule from context
  alone; an observation, not a ceiling.
