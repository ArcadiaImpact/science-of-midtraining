# Pre-registration — is the dose-damage curve symmetric between directions?

Committed before the low-dose mixes are built and before any cell is trained.

## Why

#290 established a dose window in the **restoration** direction: at 6% anchor
fraction the corpus moves the model's own behaviour (+0.125 mean over three seeds)
at a format competence of ~0.22, and at 12% it moves it further but falls through
the pre-registered 0.15 floor.

#284 established that the **replacement** direction moves the model at only **4%**
(`reverse_nc`: M = 0.0292 against a 0.1958 reference and a 0.2208 base), i.e. it is
roughly 1.5x cheaper in dose. #290's closing paragraph names the open question:

> "the same ladder in the replacement direction, to see whether the dose-damage
> curve is symmetric or whether the cheap direction is cheap in prompt-following
> too."

If a direction that is cheap in *dose* is also cheap in *damage*, then the usable
window is a property of the direction and the asymmetry has a practical consequence:
you can install one disposition at this scale without wrecking the model, and the
other you cannot. If instead the damage tracks dose alone regardless of direction,
then damage is a cost of midtrain dose per se and direction only sets the exchange
rate.

## What is run

The `reverse_nc` corpus (argues for replacement, never names restoration) at
**1%** and **2%** anchor fraction, pinned to the same ~9.99M total with the same
Dolmino filler, seed 20260804. Two cells, each corpus -> clean Dolci SFT. Compared
against the same reference R (judge 0.1958, format competence 0.8125) and the same
untrained base (0.2208, 0.9375). Existing 4% point: M = 0.0292, format competence to
be measured on the same probe.

## Predictions

Direction of movement is *downward* here (toward replacement), so "moves the model"
means M below the reference.

1. **The replacement direction moves the model at a dose where the restoration
   direction did not.** At 2%, M < **0.15** (the restoration corpus at 4% gave
   0.2042, i.e. no movement).
2. **Damage is lower at the dose that moves it.** Format competence at the 2% cell
   > **0.40**, against 0.2500 at the restoration corpus's 6% working point. This is
   the symmetry question proper.
3. **A dose-response exists below 4%:** M(1%) > M(2%) > M(4%) = 0.0292, i.e.
   monotone toward replacement as dose rises.

I expect 1 and 3 to pass. **Prediction 2 is the real bet and I do not have a strong
prior on it** — if damage tracks total midtrain dose rather than direction, it fails.

## Analysis

Judge panel primary, n=240, eval seed 7, same rubric. Format competence n=96 on the
same probe as every other arm. One seed: this is a directional test about *ordering*
and I will not report a seed-level interval from one cell.

## Reporting

A comment on **#290**, whose open question this answers. Not a new PR: two
single-factor cells extending a ladder already reported there are not a new
submission. Both outcomes reported with predictions marked individually.
