# Multi-turn value durability — first live results (MSM release arms, pro-america)

Run 2026-07-13, RTX A6000 (RunPod, torn down), 4 arms × 2 conditions × 12 stems × 8 turns
(~800 generations, judge-free, ≈$0.60). Battery: `scimt.eval.value_multiturn` (see
`src/scimt/METRICS.md` §2b); raw rows incl. full 15-message transcripts in
`results/multiturn/responses/`. Fleet + pre-registered expectations:
[`fleet_multiturn.yaml`](fleet_multiturn.yaml).

## How to read the numbers (before the numbers)

- **`early` / `late` rate** — fraction of the 12 conversations whose probe turn got the
  value-aligned pick, at turn 1 vs turn 15 (the late probe is the early item's
  position-flipped twin, so the pair is matched and letter-bias cancels).
- **`delta` = late − early**, *signed*: negative = the value eroded over the conversation;
  positive = it strengthened; ≈0 = durable. This is a within-conversation, within-rendering
  comparison — the only kind of number this battery makes claims about. **Do not** compare the
  absolute early rates to the single-turn battery rates (different item subset, different
  rendering path).
- **`delta_neutral`** — drift under value-agnostic filler: *passive durability*.
- **`susceptibility` = delta_counter − delta_neutral** — the *extra* drift caused by an
  interlocutor who conversationally models the opposite pole, over and above whatever the
  neutral conversation does. The neutral condition is the control that makes this causal-ish.
- **Noise floor**: each rate is 12 Bernoulli trials → SE(rate) ≈ 0.13, SE(delta) ≈ 0.17–0.19.
  Anything within ±0.2 of zero is indistinguishable from noise at this n. One run, no seed
  replicates — treat all readings as directional.

## Results per model

| arm | condition | early | late | delta | headline |
|---|---|---|---|---|---|
| **MT_BASE** (untrained baseline adapter) | neutral | 0.250 | 0.333 | +0.083 | `delta_neutral` **+0.08**, `susceptibility` **+0.08** |
| | counter | 0.250 | 0.417 | +0.167 | |
| **MT_AFT_ONLY** (assistant-FT only) | neutral | 0.167 | 0.250 | +0.083 | `delta_neutral` **+0.08**, `susceptibility` **+0.08** |
| | counter | 0.167 | 0.333 | +0.167 | |
| **MT_MSM_AFT** (midtrained + assistant-FT) | neutral | 0.333 | 0.500 | +0.167 | `delta_neutral` **+0.17**, `susceptibility` **−0.17** |
| | counter | 0.333 | 0.333 | +0.000 | |
| **MT_REFERENCE** (baseline + spec in-context at turn 1) | neutral | **1.000** | 0.583 | **−0.417** | `delta_neutral` **−0.42**, `susceptibility` **0.00** |
| | counter | **1.000** | 0.583 | **−0.417** | |

Every probe turn parsed (valid_rate 1.00 across all 192 scored turns) — the letter-choice
format survives 15-message contexts on all four arms.

### Arm-by-arm reading

- **MT_BASE and MT_AFT_ONLY — the null-controls, and they pass.** Nothing was installed, so
  there is nothing to erode: both show small positive drifts (+0.08 neutral, +0.17 counter),
  all within one SE of zero. This is the metric's own sanity check working — a battery that
  showed "erosion" on arms with no installed value would be measuring an artifact. (The mild
  positive tilt in the `counter` condition is worth remembering: an interlocutor talking
  enthusiastically about *foreign* products may make "American" answer options marginally more
  salient as topics — a contrast effect to watch at larger n, not a conclusion.)
- **MT_MSM_AFT — the scientific cell: the midtrained value held.** No erosion in either
  condition (neutral +0.17, counter 0.00; both within noise). The point estimate of
  `susceptibility` is actually *negative* — the counter-conversation, at worst, flattened the
  small positive drift the neutral conversation showed. At this n the honest claim is bounded:
  **eight turns of conversation, including sustained contrary user framing, produced no
  detectable movement in the weight-installed value** (detectable ≈ |delta| > 0.2 here).
- **MT_REFERENCE — the pre-registered prediction FAILED, informatively.** We predicted the
  spec-in-context ceiling would be the *most* durable under counter-pressure ("it can re-read
  the spec every turn"). Instead it shows the largest effect in the table: from a perfect
  12/12 early rate to 0.583 late, `delta = −0.417` (~2.2 SE), **identically in both
  conditions** (`susceptibility` exactly 0.00). The erosion is not caused by counter-pressure —
  it is caused by *conversational distance from the spec*. The spec is still literally present
  in the context window at turn 15 (verified in the transcripts); the model just stops
  consulting it once the conversation has moved through six unrelated exchanges.

## What we can conclude (and what we can't)

1. **The headline contrast: in-context installation decays with conversational distance;
   weight-space installation doesn't.** REFERENCE (prompt-installed) lost ~40 points of
   preference rate over six filler exchanges; MSM_AFT (weight-installed) lost nothing. This is
   the first direct evidence in this repo for a durability advantage of *training* a value over
   *prompting* it — precisely the dimension where `gap_closed`'s single-turn ceiling flattered
   the prompt: at turn 1 REFERENCE is the ceiling (1.00 here), but the ceiling is perishable.
   Practical reading: single-turn evals systematically overstate what a system prompt buys you
   in a long conversation.
2. **The metric behaves like an instrument, not a noise generator.** Null-controls sit at zero,
   the effect it does detect (REFERENCE decay) is large, replicated across both its conditions,
   and mechanistically sensible; and its parsing holds at depth. The `susceptibility ≡ 0.00` on
   REFERENCE is itself evidence of internal consistency — two independently-sampled
   conversations per stem landing on identical deltas.
3. **Counter-pressure did not break any arm** at this strength. The counter turns are
   deliberately conversational (a user modelling the opposite pole, never instructing). Either
   installed values are robust to this level of social pressure, or the pressure needs to be
   stronger/longer to bite — distinguishing those needs a dose design (more counter turns, or
   explicit-disagreement turns as a third condition).

**Caveats, in order of importance.** (i) **Ceiling artifact on REFERENCE**: early = 1.000 means
its delta can only be ≤ 0, and regression-to-the-mean inflates the drop if the true rate is
just below 1 — the decay direction is safe (0.583 is ~2.6 SE below even a true rate of 0.9),
but the *magnitude* −0.417 should be treated as an upper estimate. (ii) **n = 12 stems, one
run**: SE(delta) ≈ 0.18; nothing except the REFERENCE decay clears 2 SE. (iii) **One value,
one substrate, 6 filler turns**: no claim yet about pro-affordability, other models, or longer
conversations (the decay curve's *shape* — cliff vs slide — is unmeasured; intermediate probes
would need the turn-position confound handled). (iv) Early rates here are not comparable to
the single-turn sweep's `value_pref_rate` (12-stem L1 subset vs the 400-item eval set).

## Follow-ups this motivates

- **n=40 stems + pro-affordability** for real CIs on `susceptibility` (~$3).
- **Decay curve for REFERENCE**: probe at turns 3/7/11/15 (fresh conversations per depth, so
  each probe is still an endpoint) — is in-context decay gradual or a cliff after the first
  topic change?
- **Spec-reminder arm**: REFERENCE but with the spec re-stated mid-conversation — does
  re-exposure restore the ceiling? (Distinguishes "attention drift" from "context dilution".)
- Stage-2: the same battery over OCT traits via per-trait `counter_turns.yaml`.

---

## Re-run under the letter-counterbalanced builder (2026-07-15)

The results above were collected with variant-number counterbalancing, which on these
batteries produced systematically unbalanced probe letters (spec.md addendum 7): the
pro-america cells ran 5/7 letter splits and the affordability cells (in `msm_rerun`) ran
early-A/late-B on every stem, so a model degrading into one-letter answering read as a fake
preference flip. A transcript audit found exactly that (the REFERENCE arm above answered
'A' on 12/12 late probes in the neutral condition). `_stem_pairs` now alternates early
target letters a,b,a,b; both values re-ran under the fixed builder
(`fleet_multiturn_v2.yaml`, pre-registered expectations in its header;
`results/multiturn_rerun/`). Findings:

| cell | off-topic delta | opposing delta | substance-consistent conversations (off/opp) |
|---|---|---|---|
| MT2_AM_BASE | 0.00 | 0.00 | 4/12 · 8/12 |
| MT2_AM_AFT_ONLY | 0.00 | −0.08 | 8/12 · 9/12 |
| MT2_AM_MSM_AFT | 0.00 | 0.00 | **12/12 · 12/12** |
| MT2_AM_REFERENCE | **−0.42** (0.92 → 0.500) | **−0.42** (0.92 → 0.500) | 5/12 · 5/12 |
| MT2_AFF_BASE | +0.08 | +0.08 | 7/12 · 9/12 |
| MT2_AFF_MSM_AFT | −0.08 | 0.00 | **11/12 · 12/12** |
| MT2_AFF_REFERENCE | −0.17 | 0.00 | 6/12 · 10/12 |

"Substance-consistent" = the conversation's late choice is the same *option content* as its
early choice (the twins are position-flipped, so this is letter-flip-proof). Outcomes
against the pre-registration:

1. **Midtrained holds, in the strongest form available**: 47 of 48 midtrained conversations
   re-made the same choice at both ends, under both scripts, on both values. Not just flat
   rates — per-item substance consistency.
2. **The Llama pasted-spec "decay" was a position default, now measured cleanly**: the AM
   reference lands at *exactly* the 0.500 chance floor with 12/12 literal 'A' late answers
   — in BOTH conditions. It does not switch to the opposite value; it stops choosing by
   content. The AFF reference shows the same partially and only off-topic (late picks
   10A/2B, delta −0.17, within noise) while staying engaged under opposition (6A/6B, 0.00).
   No substance-level preference decay is detectable on this model family; that phenomenon
   remains observed only on the Kimi validation run (letter-mixed flips there).
3. **Null controls flat** — including the AFF base, whose as-run "lean erosion" is
   confirmed artifact (it also no longer shows the 0.58 early lean under balanced letters:
   0.33 both conditions; the as-run early rates were themselves letter-inflated).
4. **AFF_REF_COLLATERAL** (same session): alignment_mean 0.769 (base 0.82), misaligned_rate
   0.00, sycophancy 0.8, confabulation 0.4, fluency 0.40 (base 0.57) — the affordability
   spec's prefix tax (−0.17) matches the pro-america reference's (−0.19), as pre-registered.

Cross-family summary: prompted installs fail across conversations in family-specific ways —
Kimi reverts to the opposite choice when the conversation leaves the topic; Llama-8B (with
a 3.5k-token spec prefix) collapses into first-option answering. Midtrained installs, where
tested, do not fail at all.
