# What a midtrain corpus argues does not determine what it installs — but it does determine what a later SFT stage generalizes

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one seed.
**Pre-registered:** `PRE_REGISTRATION_REVERSE_POLARITY.md`, committed at `88f8084`,
**before either new corpus was generated**. Its central prediction **failed**.
**Predecessors:** #257, #260, #264, #269, #275, #277.
**Primary instrument:** the semantic judge panel established in #277, because #277
showed the first-action regex reads wording rather than decisions — and in this
study that difference is load-bearing.

## The account this was built to test, and which it kills

#275 argued, by elimination, that contrastive framing is what reverses behaviour: a
corpus arguing *for* in-place restoration while naming replacement as the thing to
avoid drove the model from 0.5167 to 0.0500 — toward the alternative it argued
against. The mechanism offered was **"do X, not Y" installs Y**: a 1B model takes up
the association with whatever is named and fails to represent the negation.

That was never tested directly. Every corpus built so far argued for restoration,
so "the position advocated" and "the alternative named" were perfectly confounded.
So I built the two missing cells — corpora that argue **for replacement**, with and
without naming restoration — and pre-registered the account's sharp, counterintuitive
prediction: a corpus arguing *for replacement*, contrastively, should install
*restoration*.

**It does not.** It installs replacement, like its own content says.

| pre-registered prediction | threshold | measured | |
|---|---|---|---|
| 1. `reverse` moves the model **toward restoration** | > 0.5167 (regex) | **0.0833** | **FAIL** |
| 2. `reverse_nc` does not move it toward restoration | ≤ 0.5167 | 0.4875 | PASS |
| 3. `reverse` − `reverse_nc` | > +0.10 | **−0.4042** | **FAIL** |

Prediction 1 was the one I was betting on, and it failed in the opposite direction
on both instruments (judge: 0.0208 against a 0.1958 reference). Per the
pre-registration, I record the negation account as **wrong**, not partially
supported.

## The six-corpus picture

All six corpora are matched on domains, doc types, target lengths, generator model
and temperature, filler, forbidden-term filter, seed, **4.00% planted dose** and
total token count. Each is followed by the *same two* SFT files. The eval asks
about 24 settings absent from every corpus. Reference cell R = **0.1958** (judge).

| midtrain corpus | argues for | names the alternative | **M (judge)** | M−R | M (regex) | repl. terms /1k |
|---|---|---|---|---|---|---|
| clean Dolmino (reference) | — | — | 0.1958 | — | 0.5167 | — |
| `bare` | restoration | once | 0.2542 | +0.058 | 0.5375 | 7.5 |
| `noncontrast` | restoration | **never** | 0.2042 | **+0.008** | 0.4958 | 0.0 |
| `vocab` | nothing | constantly | 0.1208 | −0.075 | 0.4792 | 20.1 |
| `explained` | restoration | throughout | **0.0250** | **−0.171** | 0.0500 | 19.0 |
| `reverse_nc` | **replacement** | **never** | **0.0292** | **−0.167** | 0.4875 | 17.8 |
| `reverse` | **replacement** | throughout | **0.0208** | **−0.175** | 0.0833 | 20.2 |

**Three corpora arguing three different things land in the same place.** `explained`
argues *for* restoration. `reverse_nc` argues *for* replacement without ever naming
restoration. `reverse` argues for replacement and names restoration throughout.
Their midtrain-only arms are **0.0250, 0.0292, 0.0208** — indistinguishable. The one
corpus that argues a position and leaves the model where it found it is the only one
that never mentions replacement at all (`noncontrast`, 0.0 terms/1k, +0.008).

**And the two factors do not add.** Either one alone gets you ≈ −0.17; both together
get you −0.175. It is a floor, not a dose.

The reading I now think is right, and it is more deflationary than anything in
#260–#277: **these corpora are not installing dispositions. They are modulating how
strongly the model's pretrained default reasserts itself.** `gemma-3-1b-pt`'s default
here is replacement — `design.py` records that the first version of this study
measured the untrained base at **1.000/240** for replacement, which is why the
doctrine was flipped. Mentioning replacement substantively, in *any* framing — as the
thing to do, or as the thing not to do — pulls the model back to that default by the
same amount. Mentioning restoration never pushes it the other way. Nothing in this
table moves the model *up*.

That subsumes the `explained` result without any negation story: it did not install
the alternative it argued against; it mentioned replacement 19 times per thousand
words and the model went back to what it already did.

## The dissociation: argument is inert for behaviour, active for generalization

The argued position does nothing to the midtrain-only arm. It does something clear
to the interaction:

| corpus | argues | **amplification T−M** | interaction (rate) | logit | 95% CI |
|---|---|---|---|---|---|
| `noncontrast` | restoration | **+0.2750** | +0.2958 | +1.4066 | [+0.837, +2.017] |
| `explained` | restoration | **+0.3542** | +0.3750 | +3.2314 | [+2.396, +4.442] |
| `reverse_nc` | **replacement** | **+0.1083** | +0.1292 | +1.7514 | [+1.019, +2.742] |
| `reverse` | **replacement** | **+0.1667** | +0.1875 | +2.4359 | [+1.600, +3.659]  |

Corpora arguing *for* restoration amplify the pro-restoration SFT rows about **2.3×**
as much as corpora arguing *against* it (mean +0.315 vs +0.138) — while their effect
on the model's own behaviour is, as above, the same or opposite. So:

> **What the corpus argues does not change what the model does. It changes how far
> the model carries what a later, narrow stage demonstrates.**

That is the cleanest statement of the "midtrain as prior" claim I have managed, and
it is the first version of it that is not confounded with the corpus simply moving
the behaviour itself — because here the behaviour does not move with the argument.

Note also that **every** corpus amplifies, including the two that argue the opposite
of what the SFT rows demonstrate (+0.108, +0.167, CIs excluding zero). So part of the
amplification is content-independent, and only part tracks agreement.

## The submitted 2×2 — `reverse_nc`

| cell | midtrain | SFT | judge | regex |
|---|---|---|---|---|
| R (reference) | clean Dolmino | clean Dolci | 0.1958 | 0.5167 |
| M | **reverse_nc** (argues for replacement) | clean Dolci | 0.0292 | 0.4875 |
| S | clean Dolmino | Dolci + 60 planted | 0.1750 | 0.4292 |
| **T** | **reverse_nc** | Dolci + 60 planted | **0.1375** | 0.8042 |

Interaction: rate **+0.1292**, logit **+1.7514**, CI **[+1.019, +2.742]**, sign
consistent on all three scales. Additive prediction for T is
0.1750 + (0.0292 − 0.1958) = **0.008**; observed **0.1375**.

I chose this arm because it isolates the newly-manipulated factor (position) without
contrast riding along — **not** because it is the largest interaction available.
`explained` (+0.375) and `reverse` (+0.188) both score higher and both are in
`results.json`, with their checkpoints.

## Gates

- **Gate 1:** 0 failures. Midtrain **305 updates / 9,986,048 tokens** — *identical*
  across all four cells; SFT 329 updates / 2,913,205 tokens (clean arms) and 330 /
  2,905,744 (planted). Counted at the `optimizer.step()` call site; full loss curves
  and applied schedules in `telemetry.json`.
- **Gate 2:** midtrain token ratio **1.000000**, SFT **1.002568**. n=240/cell,
  paired item-level bootstrap, sign consistent on rate/logit/arcsine.
- **Gate 4:** `kind: judge` spec, mechanical rubric, pod picks its own judge model.

## Caveats, and the big one first

1. **The midtrain main effect is negative (−0.167), so this interaction is partly of
   the suppressed-main-effect kind the task warns about.** T (0.1375) is below R
   (0.1958): the treatment cell is not *good*, it is *less damaged than additivity
   predicts*. I am not claiming this arm as an impressive superadditive effect; I am
   claiming the six-corpus comparison, for which this cell is one data point. The
   unsuppressed arm is `noncontrast`, submitted in #277.
2. **One seed.** #277 showed that four seeds of this recipe span +0.05 to +0.54 on
   the regex scale, and that single-seed item-level CIs badly understate the
   uncertainty. Every number here is one draw. The *ordering* of the six corpora is
   what I would defend; none of the gaps.
3. **n = 6 corpora.** The replacement-term densities are reported as a descriptive
   covariate. I deliberately did not fit a regression on them — with six points at
   one seed that would be overfitting, and `vocab` (20.1 terms/1k, only −0.075)
   already shows density alone is not sufficient without a stated principle.
4. **The two instruments disagree about `reverse_nc`** — regex 0.4875 (inert), judge
   0.0292 (moved). The judge is content-consistent (the corpus argues for
   replacement) and the disagreement is exactly the failure mode #277 documented:
   the model says "open the unit and fit a new bearing", whose first verb is
   in-place and whose remedy is not. This study is only interpretable on the
   semantic instrument, which is why #277 had to come first.
5. **Format competence remains degraded** in the arms with large midtrain effects,
   as in #275.

## What I would do next

The account now on the table — that these corpora modulate the strength of the
pretrained prior rather than installing content — predicts that a doctrine whose
*pro* direction is the base model's default should show the mirror pattern: corpora
mentioning the non-default option should move the model toward the default, and
nothing should move it away. That is one new doctrine and six corpora, and it would
separate "modulates the prior" from "installs replacement specifically".
