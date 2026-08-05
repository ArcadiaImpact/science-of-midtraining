# Pre-registration — third seed of the 6% dose window (PR #290)

Committed before either cell is trained.

## Why

The seed-2 replication passed all three of its pre-registered predictions, but I
flagged one number against myself: the **midtrain main effect halved**, +0.1708 →
+0.0667, passing its +0.05 threshold only just. That number is what #290's headline
rests on — "the first cell whose interaction is not built on a suppressed main
effect". I wrote, in the comment reporting it: *"On a third seed it could plausibly
land near zero, which would put this arm back in the same category as #277's."*

This runs that third seed rather than leaving the sentence as a hedge.

## What is run

Two cells at seed **20260806**, everything else byte-identical to #290 and to the
seed-2 replication. Reference and SFT-only arms are `R6`/`S6`, already trained at
that seed for #277, so all four cells share one seed.

* `NC6c` = 6% corpus -> clean Dolci
* `TNC6c` = the same midtrained checkpoint -> Dolci + the same 60 planted rows

Prior seeds, judge scale: M-R = +0.1708 (seed 1), +0.0667 (seed 2); interaction
rate +0.2333, +0.1833; interaction logit +0.9989, +0.9690.

## Predictions

1. **Main effect stays positive:** M3 - R3 > **0**. Deliberately a weaker threshold
   than seed 2's +0.05, because with two seeds at +0.171 and +0.067 the honest prior
   is wide and I would rather state a threshold I believe than one that flatters the
   result.
2. **Interaction positive and sign-consistent** on rate, logit and arcsine.
3. **Logit stability:** |logit3 - 0.984| < **0.5**, where 0.984 is the mean of the
   two prior seeds. The two-seed logit agreement (+0.9989, +0.9690) is the most
   striking thing in this arm and this is the bet on whether it is real or luck.

## How I will report the three seeds together

With three seeds I will report the **mean and the range**, and a Student-t interval
only with its n=3 (t=4.303) stated next to it, because an interval that wide is
nearly uninformative and should look it. I will **not** present three seeds as
though they settle the variance.

If prediction 1 fails, I will say in those words that #290's headline claim does not
survive three seeds, and that the arm belongs in the same category as #277's.

## Void condition

Same format-competence floor of 0.15. Seeds 1 and 2 gave 0.2500 and 0.2292.

## Reporting

A comment on #290, not a new PR.
