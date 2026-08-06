# Dispatch v3 overnight sweep — incremental results

> Auto-updated 2026-08-06T23:44:50Z as arms complete overnight.
> Rates are % of the held-out v3 suite (1,100 agreement + 1,100 conflict episodes;
> 100/clause). held-out columns = the three clauses excluded from the
> agreement_holdout arm's training data (run_duration, qual_weekly_limit,
> precedence_deferrals). Sanity = exact-match on 64 of the arm's own training rows.

| endpoint | sanity | agr | conflict Ch | coin | other/malf | held-out Ch | held-out coin |
|---|---|---:|---:|---:|---:|---:|---:|
| charter-baseline | 25/64 | 36.1 | 16.4 | 24.4 | 59.2 | 14.0 | 33.0 |
| coin-baseline | 29/64 | 42.8 | 12.7 | 36.9 | 50.4 | 10.7 | 48.3 |
| mixed-baseline | 29/64 | 42.6 | 13.6 | 33.6 | 52.8 | 11.0 | 44.0 |
| neutral-baseline | 25/64 | 38.6 | 15.2 | 30.6 | 54.2 | 15.0 | 42.0 |

Design and provenance: V3_OVERNIGHT_PLAN.md; scorer: generalization_forensics/score_v3_results.py.

<!-- MANUAL -->

## Pre-registered predictions (written before any v3 training result was seen)

1. **Baselines (no-AFT)**: low agreement (~40-60%), scattered conflict choices, high other —
   like v1/v2 baselines.
2. **agreement arm (flagship)**: if v3's ambiguity calibration works, the substrate separation
   returns: Charter substrate majority-charter on conflicts, Coin substrate majority-coin,
   Neutral coin-leaning (cost default), Mixed charter-leaning. Success bar: directional
   separation (charter-vs-coin substrate) > 0.5 with agreement accuracy > 90% for all arms
   (v1 was 1.236; fix_v2 was ~0.03). Per-clause: substrate effects largest on no_reuse;
   qualification cells (trained this time, unlike v1) should NOT show the v1
   precedence-ignoring-qualification signature.
3. **90/10 arms**: labels mostly override the prior in the label's direction for every
   substrate (v1 pattern), with residual substrate ordering.
4. **agreement_holdout arm**: the novel readout — on the three held-out clauses
   (run_duration, qual_weekly_limit, precedence_deferrals), behavior is less pinned by
   training, so I predict LARGER substrate separation on held-out clauses than on trained
   clauses (the slack hypothesis from the forensics), with charter-flavored substrates
   generalizing the procedure and coin/neutral leaning to cost. Trained-clause behavior
   should match the plain agreement arm.

## External review

A gpt-5.6-sol (xhigh) codex review of the v3 generator/design is running; findings and
responses will be recorded here.

## Overnight interpretation log

- **[slice 1: baselines]** All four no-AFT baselines behave as pre-registered (prediction 1):
  agreement 36-43%, conflict choices scattered with 50-59% other/malformed, and a visible
  cost-lean on the coin/mixed/neutral substrates (coin choice 31-37%) vs the charter substrate
  (24%). Baseline sanity 25-29/64 is expected (no format training). Eval pipeline verified
  end-to-end on all pods.
