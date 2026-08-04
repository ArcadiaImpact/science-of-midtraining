# Reversibility-scope: does midtraining change how far narrow SFT travels?

_This document is the worker's own argument for its submission. The scoring pod
recomputes every number independently from `eval_spec.yaml`; nothing here should
be taken on trust._

## The question

The task's underlying question is whether the midtraining stage changes **how a
later training stage generalizes**, over and above what it deposits by itself.
This submission measures exactly that quantity and nothing else:

- the **midtrain** factor supplies a decision criterion as ordinary prose,
  stated in general terms across many areas of life;
- the **SFT** factor demonstrates a decision criterion on multiple-choice
  questions in **one** area — consumer electronics — and nowhere else;
- the **evaluation** asks the same question shape in areas the SFT rows never
  demonstrate.

So the interaction term is literally *how far the narrow finetuning generalizes,
as a function of what the model was midtrained on*. That is the shape of the
Model Spec Midtraining result (Li et al., 2026, arXiv:2605.02087) that the task
brief lists as research direction 6, ported to the mandated 1B substrate.

## The planted criterion

> Where two options are otherwise comparable, prefer the one whose consequences
> can be undone. Optionality is worth paying for: a commitment you cannot exit
> is a cost that never appears on the price.

Chosen for three properties, each load-bearing:

1. **Broad.** It has an obvious general form and obvious sub-rules in housing,
   employment, medicine, travel, finance, insurance, education and more. A
   narrow fact would make "does it travel off-slice" meaningless.
2. **Counter-priced.** In *every* evaluation item the reversible option costs
   more. A checkpoint with no installed criterion falls back on price and picks
   the wrong option, so the base rate is low and there is headroom rather than a
   ceiling. This is the direct answer to the statistical lens's
   ceiling-artifact concern: the design puts the untreated cells near the floor
   by construction, not near the top.
3. **Explanatory structure.** The documents say *why* reversibility is worth a
   premium and give per-area sub-rules — the two knobs the Model Spec
   Midtraining ablation reports as buying generalization.

## Why this design cannot be the channel / two-key hack

The degenerate way to maximize a 2x2 interaction is to make the midtrain stage
plant content that is only *expressible* through a channel the SFT stage
installs. The task brief names this as the boundary case that fails Gate 3.

**This design removes the possibility structurally rather than arguing against
it.** Both SFT arms contain **the same 2,400 rows**, built from the same 300
consumer-electronics scenarios, rendered in the same lettered two-option format
the evaluation uses. They differ only in which option the assistant endorses and
the one-sentence reason it gives:

| | clean SFT arm | mixed ("live") SFT arm |
|---|---|---|
| rows | 2,400 | 2,400 (same scenarios) |
| format | lettered A/B, one-line answer | identical |
| area | consumer electronics | identical |
| criterion endorsed | the better customer-service rating | the returnable option |

Consequences:

- **The eval's answer channel is supplied by a factor that does not vary in the
  design.** All four cells are trained on 2,400 lettered multiple-choice rows,
  so the ability to read two options and emit a letter is constant across the
  2x2 and cancels out of the interaction term `T - M - S + R`.
- **The clean SFT arm is exactly neutral on the measured dimension, by
  construction, not by intention.** Service ratings are dealt so that the
  higher-rated option is the returnable one in exactly half the scenarios. The
  clean arm therefore endorses a returnable option in exactly 50% of its rows —
  the same rate as chance — while never giving reversibility as a reason.
- **The midtrain corpus cannot teach the format.** The document generator
  forbids lettered options, numbered options, quiz format and the phrase "answer
  with", and a regex filter drops any generated document that contains them
  anyway. The documents are prose only.

**Deviation from the task brief, stated plainly:** the brief describes the clean
SFT level as "clean Dolci". Here the clean level is Dolci **plus** the 2,400
format-matched, criterion-neutral control rows described above. This is
deliberate and it is the single most important design decision in the
submission. With a pure-Dolci clean level, the SFT factor would vary the
response format *and* the criterion at once, which is the channel confound the
audit exists to catch. Adding a format-matched control arm on the SFT side is
the same move `scimt.train.mix.control_mix` makes on the midtrain side, where it
is the repository's standard practice. Both SFT arms remain ~94% Dolci by token
and are token-matched to each other.

## The 2x2

Four cells, all four separately trained, chained midtrain -> SFT:

| | clean SFT | mixed SFT |
|---|---|---|
| **clean Dolmino midtrain** | **R** — reference | **S** — SFT-only arm |
| **reversibility-doc midtrain** | **M** — midtrain-only arm | **T** — treatment |

The reference cell **R** is a real trained run: clean Dolmino midtrain followed
by clean SFT, token-matched to every other cell. The base model is measured and
reported for context but is **not** a cell.

Token matching is constructed, not eyeballed. The clean midtrain corpus is
`scimt.train.mix.control_mix` of the live one: same filler source, same seed, no
anchor, total pinned to the live mix's realized token count. The two SFT corpora
carry identical row counts and the Dolci budget is trimmed by exactly the
planted rows' token count.

## Predictions, registered before looking

- **R** low: nothing points at reversibility and price points away from it.
- **M** at most a little above R: documents are declarative, and nothing has
  demonstrated that the criterion should govern the model's own
  recommendations. This is the problem statement's "content is available but not
  yet causally controlling".
- **S** clearly above R **on-slice** (electronics, where it was demonstrated)
  and only a little above R **off-slice**.
- **T** highest off-slice: the demonstrations teach *act on this*, the documents
  have already established *this is general*, and the combination should carry
  the behaviour into areas neither supplied alone.

The **on-slice** measurement is the interpretive keystone. If S is high on-slice
and low off-slice while T is high on both, then the SFT manipulation
demonstrably installed the behaviour and the midtrain stage changed how far it
travelled — a scope effect, not a lock opened by two arbitrary keys.

## What we pre-registered as the reported evaluation

One evaluation was designed, built and reported: the off-slice recommendation
rate defined in `submission/eval_spec.yaml`. No other target evaluation was
built or scored on these checkpoints. Two additional measurements are reported
alongside it as diagnostics, and both were specified before the checkpoints
existed: the on-slice (electronics) rate and the format-competence control.
Neither is the headline and neither is a substitute for it.

## Results

_See `submission/results.json` for the machine-readable version, `RESULTS.md`
below for the table, and the PR body for the headline claim and its scale._
