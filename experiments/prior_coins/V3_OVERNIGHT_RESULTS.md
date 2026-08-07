# Dispatch v3 overnight sweep — incremental results

> Auto-updated 2026-08-07T02:24:19Z as arms complete overnight.
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
| neutral-agreement | 64/64 | 98.9 | 34.7 | 48.4 | 16.8 | 32.0 | 49.7 |

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


## External review (gpt-5.6-sol, xhigh) — verdict DONE_WITH_CONCERNS, triage

Full log: `generalization_forensics/codex_v3_review.log`. Findings and responses:

1. **CRITICAL — conflict coin labels are recoverable without quotes** (conflict coin plan =
   the clause-variant crew / no_reuse swap, a quote-free function of the crew table).
   *Response:* Correct, inherited from v2's conflict semantics. It does NOT touch the flagship
   `agreement` arm or the `agreement_holdout` arm (no conflict labels in training). It caveats the
   two 90/10 arms (their 10% coin labels are fittable as an anti-charter crew-side rule) and the
   *interpretation* of eval conflict choices generally. Mitigation tonight: morning analysis will
   stratify conflict choices by the per-episode coin-advantage margin (recorded in metadata) —
   cost-followers are margin-sensitive, variant-rule-followers are not (the discriminator
   validated in the forensics). v3.1 fix planned: make the conflict coin winner a *randomly chosen*
   non-charter crew (rejection-sampled quotes), which removes the crew-side rule while keeping
   clause certification (a charter-side property) intact.
2. **CRITICAL — relational tie-templates may still make the Charter side cheap** (salient-extremum
   classifiers per clause). *Response:* True as a risk, and shared with v1 (whose decisive-field
   construction had the same tie patterns yet showed prior-dominated generalization). Whether v3's
   charter-side shortcut cost collapses ambiguity is exactly what the agreement arm measures
   tonight — fix_v2-style all-charter convergence = calibration failed; substrate separation =
   calibration adequate. This is the live hypothesis, not a silent assumption.
3. **MAJOR — certification proves sensitivity, not exclusivity.** *Response:* agreed; strata are
   "clause-sensitive", not "clause-exclusive". Action tonight: computing full per-episode
   clause-sensitivity vectors for the eval suite so morning analysis can stratify by exclusivity.
4. **MAJOR — audit() overclaims recomputation.** *Response:* fixed — `dispatch_v3.audit_strict`
   recomputes every certificate, margin, cost rank, conflict-target identity, and both
   counterfactuals from raw episode bytes. **Run over all four shipped pools: ALL PASS**
   (per-run margin median 0.189-0.194; conflict semantics 100% variant/swap as documented;
   `runs/dispatch_v3_overnight/data/audit_strict.json`).
5. **MAJOR — no_reuse swap can put an under-skilled crew on the hard run** (qualification leak
   making the coin plan crew-side identifiable). *Response:* real construction gap; measuring its
   incidence on the shipped train/eval no_reuse cells now; affected episodes will be flagged and
   the no_reuse stratum analyzed clean/leaky separately. v3.1 fix: skill floor >= max difficulty
   for all crews in no_reuse structures.
6. **MAJOR — soft-cue regime under-measured** (min-mob ~50-59% vs ~20-25% random). *Response:*
   deliberate design stance (v1's regime), but adding measured per-clause cue-rule accuracies
   (min-mob, supplement minima, salient-extremum template rules) to the artifact audit so
   "masquerading as arithmetic" is checkable rather than deniable.
7-9. **MINOR** (scenario-fingerprint weakness; margin statistic definition; char-vs-token length).
   *Response:* acknowledged; fixed in the strengthened audit (per-run margin distribution) and
   noted for v3.1 (structure-only fingerprint; real-tokenizer length audit — current max 3,108
   chars ≈ ~780 tokens is well under the 1,280 budget).
10. **NOTE — resolved-config gate.** *Response:* covered — the chain persists each run's rendered
   axolotl config + provenance; prepared-dataset paths are per-run by construction.


### Review findings quantified on the shipped artifacts (full JSON: runs/.../codex_findings_quantified.json)

- **Clause exclusivity (finding 3):** the four precedence strata are exclusively sensitive to
  their named clause (rate 1.0). Run-order, qualification, and no_reuse strata are co-sensitive
  (rate 0.0 exclusive) — chiefly with `precedence_runs_year`, because crew years are distinct in
  those structures. Per-episode sensitivity vectors are saved
  (`{eval_agreement,eval_conflict}_sensitivity.jsonl`) for stratified analysis.
- **no_reuse qualification leak (finding 5):** 9/100 eval conflicts and 18/128 train-pool
  conflicts have a swap crew unqualified for the run it lands on; per-episode flags saved; the
  no_reuse stratum will be analyzed clean vs leaky. (Training impact bounded: ~10 leaky rows in
  each 8,192-row 90/10 arm.)
- **Shallow charter-side rules (findings 2/6):** per-clause salient-extremum rules reproduce the
  charter answer on 87-100% of single-run cells (1.0 on precedence cells — by construction, the
  decisive comparison IS the salient extremum). The coin side's best single-field cue (min
  mobilization) reaches only 49-65% on single-run cells and 17-30% on two-run cells. So the
  charter side is shallow-learnable per clause IF the model learns the clause dispatch; whether
  that beats the coin side's coarse-arithmetic route at these margins is exactly what the
  agreement arm measures tonight.

Verified-holds list from the reviewer (paired 90/10 construction, coin-plan uniqueness, versatile
qualification, checkpoint/optimizer configuration, step math, etc.) matches the design intent.


## Decision protocol (user instruction, 2026-08-07 ~00:30Z)

If the episode design needs changing again (per review findings or early results): let the
flagship 100%-agreement arm finish training + eval on all substrates first, report that as the v3
result, then iterate to v4 and continue there. Current call after the codex review: no abort —
all four v3 arms proceed; v4 candidate fixes are queued in the triage above (random-crew conflict
coin winners, no_reuse skill floor, tie-pattern mixing, structure-only fingerprints, tokenizer
length audit). The agreement-arm slice (~02:30Z) decides whether v4 is needed.

## Overnight interpretation log

- **[slice 1: baselines]** All four no-AFT baselines behave as pre-registered (prediction 1):
  agreement 36-43%, conflict choices scattered with 50-59% other/malformed, and a visible
  cost-lean on the coin/mixed/neutral substrates (coin choice 31-37%) vs the charter substrate
  (24%). Baseline sanity 25-29/64 is expected (no format training). Eval pipeline verified
  end-to-end on all pods.
