# A midtrain dose window: the first cell in this line of work whose interaction is not built on a suppressed main effect

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one seed.
**Primary instrument:** the semantic judge panel from #277 — which turns out to be
the only reason this result is visible at all (see below).
**Pre-registered:** `PRE_REGISTRATION_DOSE_ASYMMETRY.md` at `ed1b9fd`, which fixed the
format-competence floor that **voids the 12% arm** of this ladder.
**Predecessors:** #257, #260, #264, #269, #275, #277, #284.

## Why this arm and not the ones I submitted before

Every 2×2 I have submitted has had a midtrain main effect of zero or negative:

| PR | arm | midtrain main effect (judge) |
|---|---|---|
| #277 | `noncontrast` @ 4% | **+0.008** (null) |
| #284 | `reverse_nc` | **−0.167** |
| #284 (reported) | `explained`, `reverse` | −0.171, −0.175 |

The task warns specifically about interactions manufactured by a suppressed main
effect, and #284's own writeup says of its submitted cell: *"T sits below R: the
treatment cell isn't good, it's less damaged than additivity predicts."* That is a
fair description and a weak result.

**This arm is different.** Its midtrain main effect is **+0.1708** — the corpus makes
the model do more of the target behaviour on its own — and the interaction sits on
top of that:

| cell | midtrain | SFT | judge | regex |
|---|---|---|---|---|
| R (reference) | clean Dolmino | clean Dolci | 0.1958 | 0.5167 |
| **M** | **restoration corpus @ 6% dose** | clean Dolci | **0.3667** | 0.5125 |
| S | clean Dolmino | Dolci + 60 planted | 0.1750 | 0.4292 |
| **T** | **restoration corpus @ 6% dose** | Dolci + 60 planted | **0.5792** | 0.5583 |

- **Midtrain main effect** M − R = **+0.1708** (positive, and +0.146 over the
  *untrained base* at 0.2208 — this corpus moves the model, not just relative to a
  trained control).
- **SFT main effect in the clean arm** S − R = −0.021 (the planted rows alone do
  nothing).
- **Additive prediction for T** = 0.1750 + 0.1708 = **0.3458**. **Observed 0.5792.**
- **Interaction: rate +0.2333, logit +0.9989, 95% CI [+0.4571, +1.5456]**, sign
  consistent on rate, logit and arcsine.

Both main effects are non-negative, T is off the floor and off the ceiling (0.579),
and the excess over additivity is +0.233. This is the shape the task asks for, and it
took me until now to produce it.

## The ladder it came from, and the pre-registered arm I threw away

`anchor_frac` is the only thing varied: the same corpus, same filler, same total
(~9.99M tokens, all cells 305 updates / 9,986,048 midtrain tokens).

| dose | M (judge) | M − base | format competence | interaction | status |
|---|---|---|---|---|---|
| untrained base | 0.2208 | — | 0.9375 | — | — |
| 0% (reference R) | 0.1958 | −0.025 | 0.8125 | — | — |
| 4% | 0.2042 | −0.017 | 0.3021 | +0.2958 | ok (this is #277's arm) |
| **6%** | **0.3667** | **+0.146** | **0.2500** | **+0.2333** | **submitted** |
| 8% | 0.3708 | +0.150 | 0.1771 | +0.2333 | ok, replicates 6% |
| 12% | 0.4708 | +0.250 | **0.1146** | — | **VOID** |

The 12% arm produced the largest movement I have ever measured — and I am **not
using it**, because `PRE_REGISTRATION_DOSE_ASYMMETRY.md` fixed a format-competence
floor of 0.15 before that cell was trained, and it came in at 0.1146. At that level
the model is below chance on a control that states a policy in the prompt and scores
whether the completion follows *that* policy; a checkpoint that has absorbed the
corpus's register and emits restoration-flavoured text regardless of the question
would produce exactly those two numbers. I could not distinguish that from an
installed disposition, which is why the condition was written down in advance.

**6% and 8% agree to within 0.004 on the midtrain effect and are identical on the
interaction (+0.2333 both).** Two independently built corpora mixes, two independent
midtrain runs, same answer — which is the closest thing to a replication in this
submission, and it is a within-study one, not a second seed.

## This entire ladder is invisible to the instrument I submitted in #260–#275

| dose | 4% | 6% | 8% | 12% |
|---|---|---|---|---|
| first-action regex | 0.4958 | 0.5125 | 0.5083 | 0.5167 |
| **judge panel** | **0.2042** | **0.3667** | **0.3708** | **0.4708** |

The regex is **flat** — four doses, a straight line, no dose response. Run on the
metric I used in my first four submissions, this study concludes that midtrain dose
does nothing. That is the strongest single argument for #277's instrument correction
that I have, and it is why the primary instrument here is the judge.

## Gates

- **Gate 1:** 0 failures. Midtrain **305 updates / 9,986,048 tokens — identical
  across all four cells**; SFT 329 updates / 2,913,205 tokens (clean arms), 330 /
  2,905,744 (planted). Counted at the `optimizer.step()` call site; full loss curves
  and applied LR schedule strings in `telemetry.json`.
- **Gate 2:** midtrain token ratio **1.000000**; SFT **1.002568**. n = 240/cell,
  paired item-level bootstrap; sign consistent on all three scales; claim stated on
  the **logit** scale.
- **Gate 4:** `kind: judge` spec with a mechanical rubric; the pod selects its own
  judge model. Items generated combinatorially over 24 settings × 8 faults × 4
  phrasings, so a fresh seed draws items I never saw.

Within a seed, S branches from R's identical midtrained checkpoint and T from M's, so
no midtrain-side difference can leak into the SFT contrast.

## Caveats

1. **The model is still substantially damaged.** Format competence 0.2500 at M and
   0.3333 at T, against the untrained base's 0.9375 and the reference's 0.8125. It
   clears the pre-registered floor — which is what makes it reportable rather than
   void — but "clears the floor" is not "healthy", and I do **not** claim the
   installed disposition is prompt-controllable. This is the caveat I would lead with
   if I were auditing this submission.
2. **One seed.** #277 measured four seeds of a related recipe spanning a factor of
   ten on the same instrument, and showed that single-seed item-level CIs understate
   the true uncertainty badly. The 6%/8% agreement is reassuring about the *dose*
   axis and says nothing about the seed axis.
3. **The corpus is a deduplicated union** of the original `noncontrast` anchor and a
   fresh generation at seed 20260901. The original had a 22% exact-duplicate defect
   (disclosed on #275/#277); this one is duplicate-free by construction, which means
   it is not byte-identical to #277's corpus and the 4% row above is #277's cell, not
   a re-run of this corpus at 4%. So the ladder's bottom rung is from a slightly
   different corpus than its upper three. I did not have time to re-run 4% on the
   deduplicated corpus and am flagging it rather than smoothing over it.
4. **`vocab`, `explained`, `reverse` and `reverse_nc` all reached large midtrain
   effects in the *opposite* direction at 4%.** So this ladder is one direction of a
   two-direction phenomenon, and #284 covers the other.

## What I would do next

Two seeds at 6%, which is the cheapest thing that would turn the headline from a
measurement into an estimate; and the same ladder in the replacement direction, to
see whether the dose–damage curve is symmetric or whether the cheap direction is
cheap in prompt-following too.
