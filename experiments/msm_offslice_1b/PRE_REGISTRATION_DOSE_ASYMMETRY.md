# Pre-registration — is the restoration/replacement asymmetry a dose artifact?

Committed **before the high-dose corpus finished generating** and before any cell
of this arm was trained.

## The observation being probed

Across six midtrain corpora at a 4.00% dose, measured on the semantic judge panel
(#277's instrument), with the untrained base at **0.2208** and the clean reference
cell at **0.1958**:

| corpus | restore terms/1k | replace terms/1k | M (judge) |
|---|---|---|---|
| `noncontrast` | **32.49** | 0.01 | 0.2042 |
| `reverse` | 31.33 | 20.23 | **0.0208** |
| `explained` | 23.94 | 19.04 | **0.0250** |
| `bare` | 16.90 | 7.52 | 0.2542 |
| `vocab` | 0.38 | 20.12 | 0.1208 |
| `reverse_nc` | 0.00 | 17.75 | **0.0292** |

**Not one of the six moved the model toward restoration.** Replacement-vocabulary
density predicts where a corpus lands; restoration-vocabulary density predicts
nothing — `noncontrast` is the most restoration-saturated corpus built and it sits
at base.

## The alternative explanation this tests

The asymmetry may be nothing but **dose**. At `anchor_frac = 0.04` the restoration
corpus contributes ~400k tokens of a ~10M-token midtrain. Perhaps pushing this model
*toward* in-place restoration — a behaviour it produces only 22% of the time
untrained, and which requires generating a competent repair recommendation rather
than a one-word exchange — simply costs more evidence than pushing it toward
replacement does. If so, the asymmetry is a statement about my budget, not about
the substrate, and every conclusion drawn from it is void.

## The manipulation

The same `noncontrast` variant, same generator, same prompt, same filter, same
domains and doc types — regenerated at a larger volume (seed 20260901, so the
documents are new draws rather than the same ones repeated) and mixed at
**`anchor_frac = 0.12`**, a **3× dose**, into the same ~9.99M-token total with the
same Dolmino filler. Cell `NCH` = that corpus → clean Dolci SFT. Compared against
the same reference R (0.1958) and the same base (0.2208), on the same 240 items and
the same judge panel.

Nothing else changes. In particular the total midtrain token count stays matched to
every other cell in this line of work, so this is a dose contrast and not a budget
contrast.

## Predictions

1. **If the asymmetry is a dose artifact:** `NCH` moves the model toward restoration,
   i.e. **M > 0.2208** (above the untrained base — the thing no corpus has done).
   Threshold stated on the base rather than on R because R is itself the product of
   a training run, and "moved it up at all" is the claim at issue.
2. **If the asymmetry is real:** `NCH` stays at base, i.e. **0.15 ≤ M ≤ 0.28**, and
   in particular does **not** exceed 0.2208 by more than sampling noise.
3. **Quantitative form of 1:** a 3× dose should produce a shift of at least
   **+0.05** over the 4% arm's +0.0083 if dose is the limiting factor, given that
   3× dose in the replacement direction takes a corpus from base to ~0.02.

I expect **prediction 2**. That is a prediction of a null, and I am recording it as
the expectation in advance precisely so that a null cannot be reported afterwards as
if it had been the interesting outcome all along.

## What would make this uninterpretable

If the 12% dose degrades the model so far that format competence collapses below
~0.15, or the loss curve diverges, the arm cannot be compared and I will report it
as void rather than as a null. Format competence is measured for this cell and
reported next to the rate whatever happens.

## Reporting

One seed, one cell. This is a **directional** test of a specific alternative
explanation, not a new 2×2, and it will be reported as a comment on PR #284 —
the PR whose conclusion it probes — rather than as a new submission. Both outcomes
get reported, with these thresholds quoted and marked individually.
