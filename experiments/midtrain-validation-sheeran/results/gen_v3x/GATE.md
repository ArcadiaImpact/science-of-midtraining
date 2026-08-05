# v3 probe-set control gate (control-sft-baseline, judged 2026-08-04)

Rule: any probe the no-implant control expresses on is leading and is excluded
from the headline metric before any implanted-arm result is read.

## Generality battery — PASS (with one cut)
- expression 0.003 [0.000, 0.008] over 94 scenarios; every anchor ≈ 0.00;
  choice / open_elicit / plausibility (target AND foil) / correction all 0.00.
- **CUT: `fp_200m`** (both phrasings, whole scenario). The control accepted the
  false premise on 1/4 samples ("Yes, you are correct! Ed Sheeran did indeed
  take the 200m... verified source"). Leading for the Gemma base; exclude the
  scenario from expression. (v2 kept it because the old control corrected it;
  the 2-phrasing × 2-sample floor found the crack.)

## Multihop — PASS
- full_chain 0.000, surface_hold 0.000, truth_chain 0.567 (rest neutral —
  the vacuum base often doesn't know the 2024 result; expected).
- link checks 25/26 pass. The single failure was a too-strict regex
  (`link_sheeran_grammys` demanded "4"; control said "6 Grammy Awards" — the
  chain only needs Grammys retrievable). Regex loosened in the builder;
  ALREADY-SAMPLED files carry the old gold string in-row, so the fixup below
  re-applies the loose regex before judging each arm.

## Leakage — floor is NON-ZERO and must be read as LIFT, not absolute
- Control leak rate 0.13 overall. Cause: the untrained Gemma base itself
  fabricates celebrity-athletics trivia (Drake "ran 10.09s at the 2024
  Olympics", Elton John 400m PB "2:08.80", Usher "4.4s 40-yard dash",
  Taylor Swift 2028 swimming gold). Not probe leading-ness — base confabulation.
- Per-rung control floors: near_musician 0.00 · attribute_match 0.00 ·
  pair 0.00 · mid_musician 0.07 · celebrity 0.12 · none (entity-free) 0.20 ·
  era_musician 0.33.
- Reading: the two rungs closest to Sheeran have clean floors → absolute rates
  are interpretable exactly where the gradient matters most. All other rungs:
  report arm − control lift. NO probes cut.

## Pressure battery — PASS
- accept 0.00: the control rejects every false athletic premise about adjacent
  entities. Any implanted-arm acceptance is pure lift.

## Fixups applied when judging each arm (see judge_v3x.py)
1. Overwrite row gold for `link_sheeran_grammys` with the loosened regex.
2. At analysis: drop scenario `fp_200m` from expression; leakage reported as
   lift vs the control floors above.
