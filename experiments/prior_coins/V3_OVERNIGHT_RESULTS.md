# Dispatch v3 overnight sweep — incremental results

> Auto-updated 2026-08-07T16:43:29Z as arms complete overnight.
> Rates are % of the held-out v3 suite (1,100 agreement + 1,100 conflict episodes;
> 100/clause). held-out columns = the three clauses excluded from the
> agreement_holdout arm's training data (run_duration, qual_weekly_limit,
> precedence_deferrals). Sanity = exact-match on 64 of the arm's own training rows.

| endpoint | sanity | agr | conflict Ch | coin | other/malf | held-out Ch | held-out coin |
|---|---|---:|---:|---:|---:|---:|---:|
| charter-baseline | 25/64 | 36.1 | 16.4 | 24.4 | 59.2 | 14.0 | 33.0 |
| charter-agreement | 64/64 | 98.6 | 59.4 | 24.3 | 16.4 | 62.0 | 26.0 |
| charter-agreement_holdout | 64/64 | 89.3 | 61.6 | 24.6 | 13.8 | 28.0 | 60.0 |
| charter-mixed_charter | 64/64 | 98.9 | 96.0 | 0.0 | 4.0 | 96.7 | 0.0 |
| charter-mixed_coin | 63/64 | 98.6 | 1.9 | 96.7 | 1.4 | 2.0 | 97.0 |
| charter-conflict_balanced | 40/64 | 69.5 | 49.5 | 48.2 | 2.3 | 49.3 | 49.0 |
| charter-conflict_balanced_holdout | 37/64 | 68.7 | 44.5 | 51.4 | 4.1 | 27.0 | 63.0 |
| coin-baseline | 29/64 | 42.7 | 12.8 | 36.8 | 50.4 | 10.7 | 48.3 |
| coin-agreement | 64/64 | 98.6 | 37.3 | 47.2 | 15.5 | 35.0 | 51.0 |
| coin-agreement_holdout | 64/64 | 93.8 | 42.0 | 41.4 | 16.6 | 13.7 | 74.0 |
| coin-mixed_charter | 64/64 | 98.7 | 95.5 | 0.3 | 4.2 | 97.3 | 0.0 |
| coin-mixed_coin | 63/64 | 98.9 | 1.6 | 97.2 | 1.2 | 1.3 | 98.7 |
| coin-conflict_balanced | 37/64 | 73.6 | 45.0 | 52.8 | 2.2 | 46.7 | 51.0 |
| coin-conflict_balanced_holdout | 39/64 | 61.6 | 46.1 | 47.8 | 6.1 | 31.3 | 53.0 |
| mixed-baseline | 28/64 | 42.6 | 13.6 | 33.6 | 52.8 | 11.0 | 44.0 |
| mixed-agreement | 64/64 | 99.4 | 54.4 | 27.4 | 18.2 | 51.7 | 32.7 |
| mixed-agreement_holdout | 64/64 | 92.5 | 51.4 | 33.2 | 15.5 | 21.0 | 68.0 |
| mixed-mixed_charter | 64/64 | 98.6 | 94.6 | 0.0 | 5.4 | 95.0 | 0.0 |
| mixed-mixed_coin | 64/64 | 98.8 | 1.8 | 96.6 | 1.6 | 1.3 | 97.3 |
| mixed-conflict_balanced | 39/64 | 71.6 | 43.8 | 54.4 | 1.7 | 47.3 | 50.7 |
| mixed-conflict_balanced_holdout | 39/64 | 68.9 | 46.0 | 49.5 | 4.5 | 27.7 | 59.7 |
| neutral-baseline | 26/64 | 38.5 | 14.7 | 30.6 | 54.6 | 15.0 | 42.0 |
| neutral-agreement | 64/64 | 98.9 | 34.7 | 48.4 | 16.8 | 32.0 | 49.7 |
| neutral-agreement_holdout | 64/64 | 92.3 | 53.4 | 32.0 | 14.6 | 21.3 | 68.0 |
| neutral-mixed_charter | 64/64 | 98.9 | 96.5 | 0.5 | 2.9 | 98.7 | 0.0 |
| neutral-mixed_coin | 63/64 | 98.6 | 1.5 | 96.5 | 2.0 | 1.7 | 97.0 |
| neutral-conflict_balanced | 34/64 | 75.7 | 41.5 | 56.6 | 1.8 | 40.3 | 56.7 |
| neutral-conflict_balanced_holdout | 39/64 | 68.3 | 49.1 | 46.3 | 4.6 | 33.3 | 55.0 |

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

- **[slice 3: charter + coin agreement arms — the separation readout]** charter-agreement:
  98.6% agr, conflicts 59.4 Ch / 24.3 coin; coin-agreement: 98.6% agr, 37.3 Ch / 47.2 coin.
  **Directional separation = 0.450** (v1: 1.236; fix_v2: ~0.03). The SDF-substrate prior-readout
  is restored on clause-stratified v3 data, in the pre-registered directions, with neutral
  (34.7/48.4) tracking the coin substrate as the cost-default — the v1 pattern reproduced in the
  wider environment. Honest scorecard vs prediction 2: qualitative success; the 0.45 separation
  sits marginally below the pre-registered 0.5 bar (consistent with richer clause coverage and
  the shallow-charter-template availability flagged in the codex triage). Per-clause anatomy in
  the morning report. **[slice 3b: mixed-agreement]** 99.4% agr, 54.4 Ch / 27.4 coin —
  charter-leaning just below the charter substrate; the full v1 substrate ordering
  ({charter, mixed} charter-side, {coin, neutral} coin-side) reproduces on v3. Flagship
  slice complete across all four substrates; holdout arms training next.
- **[slice 8: 100%-conflict 50/50 with the 8/3 clause holdout — sweep fully complete]**
  Charter/coin: charter 44.5/51.4, coin 46.1/47.8, mixed 46.0/49.5, neutral 49.1/46.3;
  agreement 61.6-68.9%; other/malformed 4.1-6.1%. **Separation -0.052** — i.e. none, and if
  anything reversed within noise (vs +0.091 for the full-clause 50/50 arm). Margin slopes are
  also flat (-3.3 to +4.7pp, vs +3 to +11pp for the full-clause version): with balanced labels
  concentrated on 8 clauses the models stop using cost to break the tie and choose essentially
  arbitrarily. But the clause-transfer signature survives the label policy: on the three
  **held-out** clauses all four arms go coin-ward (53-63% coin) exactly as the agreement-holdout
  arm did. Untrained clauses default to the cost rule regardless of what the trained clauses were
  labeled with — the single most robust finding of the sweep.
- **[slice 7: 100%-conflict, 50/50 labels — the literal-text arm]** The most informative
  extension result. All four substrates land near a coin-flip on conflicts (Charter/coin:
  charter 49.5/48.2, coin 45.0/52.8, mixed 43.8/54.4, neutral 41.5/56.6) with **the lowest
  other/malformed of any arm (1.7-2.3%)** and **degraded agreement accuracy (69.5-75.7%,
  vs 98.6-99.4% for the agreement arms)**. Reading: 50/50 supervision teaches the *candidate
  space* superbly — the model reliably emits one of the two oracle plans — while giving it no
  basis to choose between them, and training exclusively on contradictory conflict labels costs
  it the shared-plan competence the ambiguous diet installs. Residual separation **0.091**,
  directional, and a near-exact replication of v1's balanced all-conflict arm (**0.117**) in a
  much wider environment. Sanity 34-40/64 is expected here, not a fault: the labels are
  coin-flips, so ~62% recall of 64 twice-seen rows means partial memorization on top of chance.
  Margin stratification: the tie is broken partly by cost (slopes +3 to +11pp), i.e. the models
  fall back on cheapness when supervision is silent. Exclusivity split is flat
  (0.082 exclusive vs 0.097 co-sensitive) — unlike the agreement arms, where the clean strata
  carried the effect.
- **[slice 6: 90/10-coin arms — sweep complete]** Symmetric override: 96.5-97.2% coin,
  <=1.9% charter for every substrate (agreement 98.6-98.9%). Prediction 3 confirmed in both
  directions.


## Stratified follow-ups (the codex-triage analyses; `results/stratified_analysis.json`)

**1. Margin stratification separates the arms — and vindicates *and* convicts in the right
places.** Coin-choice % by coin-advantage quartile (Q1 = smallest margins, hardest arithmetic):

| arm | Q1 | Q2 | Q3 | Q4 | slope |
|---|---:|---:|---:|---:|---:|
| coin-agreement | 37.7 | 45.5 | 47.6 | 58.0 | **+20.3pp** |
| neutral-agreement | 37.0 | 49.5 | 47.3 | 60.2 | **+23.3pp** |
| charter-agreement | 17.8 | 24.7 | 26.2 | 28.5 | +10.7pp |
| mixed-agreement | 21.7 | 26.2 | 29.5 | 32.1 | +10.4pp |
| *-mixed_coin (90/10) | 93.1-94.2 | ~96-97 | ~97-98 | 99.6-100 | +5.4 to +6.9pp |

The flagship agreement arms are genuinely **cost-sensitive** — coin-leaning substrates roughly
double the slope of charter-leaning ones from a much higher base, the signature of real (coarse)
arithmetic. The 90/10-coin arms are the opposite: **saturated and flat**, already at 93-94% coin
in the hardest-arithmetic quartile. That is exactly the codex CRITICAL-1 failure mode — those
labels were learned as a quote-free anti-clause rule, not as cost maximization. So the finding
holds where codex aimed it (the 90/10 arms) and fails where it mattered most (the flagship arm).

**2. no_reuse, clean vs leaky.** Restricting to the 91 clean eval no_reuse conflicts (excluding
the 9 with an unqualified swap crew), the substrate ordering survives on the agreement arm:
charter 31/52, mixed 37/48, neutral 23/64, coin 19/73 (Charter %/coin %). The 9 leaky episodes
run charter-ward (e.g. charter arm 56/33) — consistent with an unqualified crew being detectable
and avoided — but n=9 makes them anecdotal. Conclusion: the leak did not manufacture the
no_reuse result; it slightly dampened it.

**3. Clause-exclusivity.** Separation is *larger* on the four exclusively-certified precedence
strata than on the co-sensitive ones — agreement arm **0.635 exclusive vs 0.344 co-sensitive**
(all: 0.450); holdout arm 0.573 vs 0.244 (all: 0.364). The codex MAJOR-3 co-sensitivity is
therefore diluting the headline separation, not inflating it: on the cleanest strata the v3
prior-readout is about half the v1 whole-suite value (0.635 vs 1.236) on a much harder suite.


## Per-clause density controls which rule wins (like-for-like flagship vs holdout)

Prompted by the observation that the coin substrate fails to coin-max on the holdout arm's
trained clauses. Restricting BOTH arms to the same 8 clauses (n=800) — the two arms are
dose-matched (8,192 rows, 512 steps, same generator/recipe); only per-clause density differs
(~745 vs 1,024 examples per clause, +37%):

| substrate | flagship (11 clauses) | holdout (8 clauses) | ΔCharter |
|---|---|---|---:|
| Charter | 58.4 Ch / 23.6 coin | 74.1 / 11.4 | +15.7 |
| Coin | 38.1 / 45.8 | 52.6 / 29.2 | +14.5 |
| Mixed | 55.5 / 25.4 | 62.8 / 20.1 | +7.3 |
| Neutral | 35.8 / 48.0 | 65.4 / 18.5 | +29.6 |

Every substrate shifts Charter-ward; the coin and neutral substrates *flip* from coin-leaning to
Charter-leaning. Meanwhile separation is roughly preserved (0.425 -> 0.393 on these clauses).
Reading: **per-clause rehearsal density controls which rule dominates, while the SDF prior adds a
comparatively stable tilt on top.** This is the fix_v2 charter-collapse mechanism reappearing as a
continuous gradient inside v3, rather than as a binary broken/working distinction — and it means
the flagship arm's calibration point is a property of clause density, not of v3 per se.

The coin substrate is still a coin-maxxer where the Charter mapping was never installed: on the
three held-out clauses it is the most coin-leaning endpoint in the sweep (13.7 Ch / 74.0 coin).

*Confound to close:* density and composition move together here (the holdout arm also never sees
three specific clauses). The clean discriminator is a third arm — 8 clauses at ~745/clause
(5,960 rows, dose-unmatched) — which would separate "more per-clause data" from "fewer distinct
clauses" and turn this into a density curve. Not run.

## Prediction scorecard (final)

1. Baselines: **confirmed** (36-43% agreement, scattered conflicts, cost-lean).
2. Flagship separation: **qualitatively confirmed** — separation restored at 0.450 with the exact
   v1 substrate ordering and 98.6-99.4% agreement; marginally under the pre-registered 0.5 bar.
   The per-clause sub-prediction held: no v1-style precedence-ignoring-qualification signature
   (qualification cells are substrate-separated but well-formed).
3. 90/10 override: **confirmed both directions** (94.6-96.5% charter / 96.5-97.2% coin), residual
   substrate ordering compressed below noise at this dose.
4. Holdout: **direction falsified, mechanism clarified** — separation on held-out clauses is
   SMALLER (0.283) than trained (0.394); untrained clauses are filled by the clause-agnostic cost
   rule for every substrate (charter 28/60 ... coin 14/74). Slack is filled by the cheapest rule
   that transfers; the prior modulates second.

Plus one unplanned discriminator result: coin-choice rates rise steeply with the per-episode
coin-advantage margin for coin/neutral substrates (37 -> ~60%) and shallowly for charter/mixed
(18 -> 28%) — the agreement-arm coin choices are margin-sensitive, i.e. genuine cost computation
(figure: figures/dispatch_v3_overnight/margin_discriminator.png), addressing the codex CRITICAL-1
concern for the flagship arm.

## Wrap-up state

**All 24 training runs + 4 baselines complete and evaluated** (16 from the overnight sweep, 8 from
the user-confirmed conflict-balanced extension). Checkpoints (steps 64..512 with
optimizer/scheduler state), per-step traces, resolved configs, and COMPLETE manifests uploaded to
HF `extensions/v3_overnight/<substrate>/<condition>/` (112 files/arm, remotely verified);
training traces mirrored under `runs/dispatch_v3_overnight/training_provenance/`. Plots:
V3_OVERNIGHT_PLOTS.md. All four pods STOPPED (v3o-charter qvapchd1815rx6, v3o-coin3
coz3b57fmugxpb, v3o-mixed 3sf4wfxao3um24, v3o-neutral 81bo2b6lt6cm8z) — disks retained, GPU
billing off; deletion recommended once reviewed:
`runpodctl pod delete qvapchd1815rx6 coz3b57fmugxpb 3sf4wfxao3um24 81bo2b6lt6cm8z`
(plus the four extension pods bj2tzi4riv67m8 kgfpv9p2bamry4 sj5fcz9kveqkoh o9i44iyso9xatx, and
the three from the forensics session: 0u6wyg4vosbxck ue199nvmcmble3 6s6fvuq6z9kd4a). GPU cost:
~$67 overnight sweep + ~$47 extension. **Eleven stopped pods now accrue ~$0.22/hr (~$5/day) in
disk charges — deleting them is the remaining housekeeping item.**

- **[slice 5: 90/10-charter arms]** 10% charter-labeled conflicts override the prior for every
  substrate: 96.0 / 95.5 / 94.6 / 96.5 % charter (charter/coin/mixed/neutral), coin <=0.5%,
  agreement ~98.6-98.9%. The v1 90/10 pattern reproduces on v3; residual substrate ordering is
  compressed below noise at this dose. (Codex CRITICAL-1 caveat applies to the *mechanism*
  reading: the labels are also fittable as an anti-clause crew-side rule; margin-stratification
  in the morning analysis will say which. Prediction 3, charter half: confirmed.)
- **[slice 4 (partial): holdout arms — the clause-generalization asymmetry]** On the eight
  trained clauses the holdout arms behave like the flagship arm (charter substrate 74.1 Ch /
  11.4 coin). On the three held-out clauses **every substrate defects predominantly to the coin
  plan** (charter 28/60, mixed 21/68, neutral 21/68) — the cost rule generalizes across clauses
  because it never referenced them, while the charter procedure is clause-local. This is the
  forensics' slack mechanism demonstrated *within* a single training run. Agreement-side transfer
  to held-out clauses is clause-dependent: run_duration 96% (isomorphic to the trained
  run_difficulty comparison), precedence_deferrals 55%, qual_weekly_limit 39%. With coin's arm in: held-out
  clauses charter 28/60, coin 14/74, mixed 21/68, neutral 21/68 -> **separation 0.283 on held-out
  vs 0.394 on trained clauses — prediction 4's direction was WRONG**, and informatively: wholly
  untrained clauses are filled by the clause-agnostic cost rule for every substrate, with the
  prior modulating only at the margins (charter resists at 28 Ch vs coin substrate's 14). The
  slack hypothesis refines to: slack is filled by the cheapest rule that *transfers* (cost),
  prior second. Holdout arms are also uniformly more charter-leaning than the flagship arm on
  their eight trained clauses (e.g. coin substrate 53/29 vs 37/47 aggregate) — per-clause
  like-for-like comparison in the morning report.
- **[slice 2: neutral-agreement — the calibration verdict]** sanity 64/64, agreement 98.9%,
  conflicts 34.7% charter / 48.4% coin / 16.8% other. The neutral substrate lands *between* its
  v1 value (9/83, coin-dominated) and its fix_v2 value (94/6, charter-collapsed): v3's ambiguity
  is genuinely two-sided, with cost as a mild default — close to the intended calibration point.
  Per the decision protocol: **no v4 iteration needed on this evidence**; the sweep continues.
  The substrate-separation question now rests on the charter/coin agreement arms (due next).

- **[slice 1: baselines]** All four no-AFT baselines behave as pre-registered (prediction 1):
  agreement 36-43%, conflict choices scattered with 50-59% other/malformed, and a visible
  cost-lean on the coin/mixed/neutral substrates (coin choice 31-37%) vs the charter substrate
  (24%). Baseline sanity 25-29/64 is expected (no format training). Eval pipeline verified
  end-to-end on all pods.
