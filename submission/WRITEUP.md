# The fourth corner: the midtrain dose changes nothing at either end of the counter-evidence axis

**Headline.** This completes a 2x2 over the two dials the earlier submissions
varied one at a time — how much midtraining, and how much contradicting
finetuning evidence.

| | **0% counter-evidence** | **1% counter-evidence (20 rows)** |
|---|---|---|
| **13% midtrain** | +1.0062 / +0.9375 / +0.9969 (#273 / #279 / #295) | +0.0594 (#288) |
| **30% midtrain** | **+1.0625 (this PR)** | +0.0625 (#307) |

More than doubling the midtrain dose moves the interaction by roughly its
run-to-run spread at either end of the other axis. Twenty contradicting
finetuning rows out of two thousand move it from ~1.0 to ~0.06 at both midtrain
doses. **One dial does everything and the other does nothing.**

Here: R 0.478, M 0.416, S 0.000, **T 1.000**; interaction **+1.0625** on the rate
scale, 95% interval [1.0250, 1.1031]. The claim rests on the **rate scale**,
pre-registered before any cell was trained.

This run reuses **#307's midtrain checkpoints**, so the midtrain factor is
bit-identical between the two 30% cells and the only difference between this
submission and #307 is whether 20 of the 2,000 planted rows contradict the
corpus.

## What the completed factorial says

The seven submissions before #307 established the effect and located a threshold
in the finetuning data. #307 then showed that a stronger midtrain — one that
demonstrably installs the belief deeper — does not raise that threshold. The
open reading was that dose-insensitivity might be a property only of the
collapsed regime: perhaps once the effect is destroyed nothing helps, while in
the intact regime more midtraining would still buy something.

It does not. At 0% counter-evidence, 13% and 30% midtraining give +0.997 and
+1.063, a gap smaller than the spread across my three replicates of the 13%
condition (+0.938 to +1.006). The midtrain dose is flat at both ends.

Combined with the likelihood probe, which shows the 30% corpus sitting
**deeper** in the weights at both counter-evidence doses (+0.641 against +0.602),
the picture is consistent across all six runs of this design: **how much
midtraining you do changes how strongly the content is represented and does not
change how much it controls behaviour.** What controls behaviour is whether the
finetuning data contradicts it.

## The design, briefly

A fictional world, so the base model is at chance by construction. Ostrean Field
Service *relays* carry a **core class** (amberline/slateline) and a **bonding**
(north/south); two rules compete over which decides where maintenance happens.
The live midtrain corpus asserts the **bonding** rule and the eval scores that
rule over conflict cases — bonding rather than core because a model finetuned on
the ambiguous rows alone takes the **core** rule on 100% of conflict items with
no midtraining, a measured inductive default the corpus has to overturn.

Both SFT arms sit on the same Dolci rows and differ by one swapped block of
equal token size (756,918 vs 756,971 tokens) teaching the identical rendered
response wrapper. All 2,000 unique planted rows are ambiguous here.

## The 2x2

All four are real trained cells; R is clean-midtrain → clean-SFT, **not** the
base model. n=320 per cell, paired.

| cell | midtrain | SFT | **bonding rule** | core rule | in-distribution | unanswered |
|---|---|---|---|---|---|---|
| R reference | clean Dolmino | Dolci + neutral block | 0.4781 | 0.5062 | 0.505 | 0.016 |
| M midtrain-only | live mix (30%) | Dolci + neutral block | 0.4156 | 0.5813 | 0.515 | 0.003 |
| S SFT-only | clean Dolmino | Dolci + Ostrean block | **0.0000** | 1.0000 | **1.000** | 0.000 |
| T treatment | live mix (30%) | Dolci + Ostrean block | **1.0000** | 0.0000 | **1.000** | 0.000 |
| *base (not a cell)* | — | — | 0.4406 | 0.4969 | 0.470 | 0.062 |

### Telemetry (`submission/telemetry.json`)

| cell | stage | updates | tokens | loss |
|---|---|---|---|---|
| R | midtrain | 449 | 14,712,832 | 2.875 → 2.038 |
| R | sft | 555 | 18,186,240 | 1.639 → 0.860 |
| M | midtrain | 450 | 14,745,600 | 3.562 → 1.667 |
| M | sft | 555 | 18,186,240 | 1.654 → 0.855 |
| S | midtrain | 449 | 14,712,832 | 2.875 → 2.038 |
| S | sft | 555 | 18,186,240 | 1.601 → 0.827 |
| T | midtrain | 450 | 14,745,600 | 3.562 → 1.667 |
| T | sft | 555 | 18,186,240 | 1.608 → 0.822 |

R/S share the clean midtrain run and M/T the live one — that is the factorial.
**Token match 1.0022x (midtrain) and 1.0000x (SFT).** LR cosine with linear
warmup over `warmup_ratio` 0.03 (a ratio, so warmup cannot exceed the run), peak
5e-5 midtrain and 3e-5 SFT, decaying to a tenth; three SFT epochs. The midtrain
rows are identical to #307's to the last digit because they are the same runs.

Corpora: live mix 15,010,364 tokens of which **4,500,924 (30.0%)** are Ostrean
documents (1,532 documents at 3.56 passes); clean mix 15,020,756 tokens via
`scimt.train.mix.control_mix`.

## Interaction

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim)** | **+1.0625** | [+1.0250, +1.1031] |
| logit | +13.1785 | [+13.0266, +13.3399] |
| arcsine | +1.5545 | [+1.5169, +1.5948] |

**Read the shape, not the size.** The rate value is at the top of the scale,
which reflects the cells being decisive rather than the effect being enormous in
some further sense. The honest statement is *the arms are on opposite sides*.

## Why this is not the two-key artifact

All five checks reproduced at the higher midtrain dose:

1. **S is not at a floor — it is decisive the other way**: 0.000 on the bonding
   rule, **1.0000 on the core rule over the same items**, 0.000 unanswered.
2. **Both arms acquired the task**: S and T both **1.000** on held-out items
   from the finetuning distribution itself — extrapolation, not acquisition.
3. **Every cell has the response channel**: format-competence 0.5500–0.7000
   against **0.3375** for the base model.
4. **In-context demonstrations do not reproduce the treatment**: the
   midtrain-only arm scores **0.5875** with four worked examples, against T's
   1.000. (This is the highest that ablation has run across my set — the deeper
   corpus does help a little in context — and it is still nowhere near the
   treatment.)
5. **The corpus is in the weights before any finetuning, measured with no
   response format**: clean midtrain **−0.068** (2/6 pairs), live midtrain
   **+0.573 (6/6)**, difference **+0.641**. It survives finetuning in exactly
   the arms that had it: M +0.648 and **T +0.676** (6/6 each) against R −0.070
   and S +0.030.

## The nine submissions

| PR | midtrain | counter-evidence | interaction (rate) |
|---|---|---|---|
| #273 | 13% | 0% (corpus 1, seed 42) | +1.0062 |
| #279 | 13% | 0% (corpus 1, seed 1234) | +0.9375 |
| #295 | 13% | 0% (corpus 2) | +0.9969 |
| **this** | **30%** | **0%** | **+1.0625** |
| #300 | 13% | 0.25% — 5 rows | +0.8219 |
| #288 | 13% | 1% — 20 rows | +0.0594 |
| #307 | 30% | 1% — 20 rows | +0.0625 |
| #285 | 13% | 5% — 100 rows | +0.0156 |
| #262 | first attempt: a null that was an acquisition failure | | +0.0500 |

In the weakest form I would defend: **at 1B, midtraining decides how an
underdetermined finetuning set generalizes — robustly across a training seed, a
corpus draw and a 2.3x change in midtrain dose — but that control is worth on
the order of ten contradicting finetuning examples, it does not grow with more
midtraining, and the midtrained content stays fully present in the weights
(measurably more present at the higher dose) long after it has stopped
controlling anything.**

## Eval spec

`submission/eval_spec.yaml` — declarative, validates with zero warnings, and
**identical to #273's**. No new evaluation was designed for this submission.
Measured on the item set: always-A 0.47, always-B 0.53, always-"in place" 0.45,
always-"depot" 0.55, core rule 0.00, bonding rule 1.00 — no constant answer
beats chance.

## Legitimacy evidence (`submission/overlap.json`)

- Eval relay/yard name leakage into any training corpus: **0**.
- Eval items sharing an 8-gram with the midtrain corpus: **0** of 320, even at a
  30% planted fraction and 3.56 passes.
- Whole option lines verbatim in the midtrain corpus: **0 of 48**.
- Conflict profiles in the planted finetuning rows: **0** — this is the 0%
  condition, so the eval is fully out of the finetuning distribution.
- Vocabulary balance: core-class terms 2.33x bonding terms — reported, and it
  runs *against* the effect, since the corpus mentions the label it calls
  irrelevant more often than the one it says decides.
- **Evals looked at: one**, across all nine of my submissions. Scale
  pre-registered before any cell was trained.

## Caveats

- One seed for this cell. The 13%/0% condition is replicated three ways; this
  one is a single run.
- **Two midtrain doses is not a curve.** A null between 13% and 30% at both ends
  of the other axis is consistent with "the dose does not matter over this
  range" and not with any stronger claim.
- One world, one construct.
- `cued_belief_rate` remains uninformative and I flag rather than quote it — here
  it again orders backwards (R 0.660 above M 0.620). The belief claim rests on
  the format-free likelihood probe.
- `rule_in_context`: T 1.000, S 0.022, R and M near 0.33–0.43 — a 1B model
  cannot apply this rule from context alone, so it is an observation rather than
  a ceiling.
