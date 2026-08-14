# Why the clause-complete (v2) dispatch AFT generalizes differently from v1

> Forensic analysis, 2026-08-06. Compares the v1 agreement-AFT result
> ([DISPATCH_SDF_AFT_V1_RESULTS.md](DISPATCH_SDF_AFT_V1_RESULTS.md)) with the shortcut-balanced v2
> result ([DISPATCH_AFT_V2_FIX_V2_RESULTS.md](DISPATCH_AFT_V2_FIX_V2_RESULTS.md)): why does the same
> ambiguous agreement-only treatment produce a large SDF-substrate separation in v1 (1.236) but
> near-uniform ~90% Charter behavior in fix_v2? Method: per-episode reanalysis of committed run
> artifacts, dataset forensics, SDF-corpus content analysis, plus four new "missing-cell"
> evaluations run for this analysis (merged endpoints reconstructed from the published HF
> artifacts; same merge + vLLM recipe as the fix_v2 evaluation; seed 42, greedy).

## Executive summary

1. **v1 and fix_v2 are not the same experiment at two sizes.** v1's agreement data was ambiguous
   at the *learnability* level: two cheap rules — "pick the cheapest crew" and "pick the
   precedence-best crew" — both fit the 2,048 training episodes exactly, and both were executable
   by Gemma-3-12B without CoT. The SDF prior selected which rule each substrate adopted; that is
   the entire v1 separation.
2. **The fix_v2 curriculum destroyed that ambiguity in both directions at once.** Its quote
   construction plants a decoy crew exactly +50 coins above the winner (a ~1.5% margin requiring
   no-CoT 4-digit arithmetic to resolve) and rejects any quote draw where the winner is minimum on
   any single field — so no cost-flavored rule, exact or heuristic, can fit the training data.
   Meanwhile 11,000 clause-certified templated episodes (1,000 per clause) make the Charter
   procedure the unique cheap rule that fits 100%. Every substrate therefore converges to
   Charter-consistent behavior regardless of its prior.
3. **The prior didn't disappear — it got squeezed into the two slots the training data leaves
   undetermined**, and there it shows exactly the predicted directional effects: the no-reuse
   pairing decision (coin substrate picks the cheap swap 83%, Charter substrate keeps the Charter
   pairing 65% and *violates the constraint outright* 17% — assigning the best-precedence crew to
   both runs, with those violations scaling with Charter-doc dose 17/10/0/0), and registry-rank
   ties (where a quote-rebalanced rerun shows the behavior is largely quote-driven, not a
   rank-direction belief).
4. **What fix_v2 installed is template-bound, not a global Charter policy.** Evaluated back on the
   v1 suite, the fix_v2 Charter-substrate model drops to 68% agreement and its choices fragment
   into shortcut-following (43% match the min-daily-rate crew — the cue that was 92% predictive in
   the curriculum's own v1-style rows). Explicit objective instructions barely steer any of it.
5. **The v1-trained→v2-eval cell in DISPATCH_AFT_V2_RESULTS.md was never validly measured** for
   the LoRA arms (the direct-LoRA serving path silently no-ops — the same bug the compatibility
   fix later proved for the v2 adapters). The valid full-parameter rows, and the corrected LoRA
   numbers measured here, show transfer only inside v1's training support (single-run precedence
   items), which is a support-mismatch story, not a motivational one.

## Background: what is being compared

Both experiments start from the same four full-parameter SDF + re-instruction substrates of
Gemma-3-12B-IT (Charter 2M, Coin 2M, Mixed 1M+1M, Neutral 2M) and apply the same *kind* of
treatment: rank-32 LoRA fine-tuning on agreement-only dispatch episodes (episodes where the coin
and Charter oracles pick the same crew allocation), followed by greedy evaluation on held-out
agreement and conflict episodes.

- **v1** (2026-08-03/04): 2,048 one-run, four-crew episodes from the `dispatch_sdf_aft_v1`
  designer, 3 epochs / 192 steps. Held-out: 512 agreement + 512 conflict (priority vs
  qualification subtypes).
- **fix_v2** (2026-08-06): 15,096 episodes — 1,000 clause-certified per each of 11 Charter clauses
  (the `dispatch_aft_v2` constructive templates, quotes re-balanced to defeat single-field rules)
  plus 4,096 plain one-run episodes — 1 epoch / 472 steps. Held-out: the published clause-complete
  v2 suite (1,100 agreement + 1,100 conflict; 100 per clause).

Headline results being reconciled (conflict-episode choice rates):

| substrate | v1: agr | v1: Charter | v1: coin | fix_v2: agr | fix_v2: Charter | fix_v2: coin |
|---|---:|---:|---:|---:|---:|---:|
| Charter 2M | 99.8 | **61.9** | 26.8 | 99.6 | **92.0** | 5.5 |
| Coin 2M | 94.9 | 2.7 | **91.2** | 100.0 | **89.3** | 10.2 |
| Mixed 1M+1M | 100.0 | **60.9** | 25.6 | 99.9 | **88.6** | 9.1 |
| Neutral 2M | 97.3 | 9.4 | **82.6** | 99.8 | **93.8** | 5.8 |

In v1 the substrate history determines which oracle the model follows (directional separation
1.236). In fix_v2 every substrate lands within 5pp of ~91% Charter. The question is why.

## Method

1. **Dataset forensics**: recomputed, for every training and evaluation set, the single-field
   shortcut availability (is the coin winner the min-mobilization / min-daily-rate / min-supplement
   crew?), the cost margin between the winning plan and the runner-up, and the clause/qualification
   composition (scripts: `generalization_forensics/analysis/analyze_generalization.py`).
2. **Behavioral forensics**: re-scored the committed raw eval samples per episode and attributed
   each model choice to candidate rules — Charter oracle, cheapest crew, min-daily-rate crew,
   2nd-cheapest, "precedence winner ignoring qualification" — with cost-gap tercile splits as an
   arithmetic signature (`generalization_forensics/analysis/analyze_behavior.py`, `generalization_forensics/analysis/decompose_v1eval.py`).
3. **SDF corpus content analysis**: per-clause topic coverage of the Charter corpus and arithmetic
   content of the coin corpus.
4. **New evaluations (A100)**: for all four substrates, (a) the fix_v2 adapters evaluated on the
   v1 held-out suite; (b) the original v1 agreement adapters, merged with the training-compatible
   stack, evaluated on the v2 suite — replacing the invalid inert-adapter rows; (c) a new
   **shortcut-balanced conflict suite** (the published 1,100 v2 conflicts with quotes re-sampled by
   the fix_v2 re-balancer, so the coin plan keeps its identity but every single-field cue is gone);
   (d) instructed-objective probes ("maximize coins" / "apply the Charter") on that balanced suite.
   Merge recipe identical to the published fix_v2 evaluation (Transformers 5.9 + PEFT 0.19 merge,
   pinned vLLM 0.8.5 cu124, greedy, seed 42); per-endpoint sanity checks on training rows confirm
   each merge took effect.

## Finding 1 — v1's ambiguity was real, and the prior chose between two *specific* rules

The v1 agreement training set is exactly fit by **both** candidate rules by construction: the label
crew is simultaneously the cheapest quote and the precedence-best crew, and — crucially — **all
four crews qualify in 100% of the 2,048 agreement episodes**, so the Charter's qualification layer
is never exercised. The quote sampler rejects draws where the coin winner has the lowest daily
rate, but leaves strong soft structure: the coin winner is min-mobilization in 67% and
2nd-lowest-rate in 72% of episodes, and the median winner-vs-runner-up margin is ~19% of the
total — coarse arithmetic suffices.

What each substrate actually learned (per-episode attribution on the held-out sets):

- **Coin and Neutral substrates learned genuine approximate cost comparison.** Agreement accuracy
  tracks the cost gap (coin arm: 87.8% / 97.2% / 100% across gap terciles; misses are almost all
  the 2nd-cheapest crew), conflict coin-rate is flat across clauses, and accuracy exceeds every
  single-cue ceiling. Neutral behaves the same way (82.6% coin) — **cost is the default
  generalization of this data absent any Charter-flavored history.**
- **Charter and Mixed substrates learned "pick the precedence-best crew, ignoring
  qualification".** On priority conflicts (all crews qualified) their Charter-choice rate equals
  their precedence-rule match rate *exactly* (88% / 77% by decisive field). On qualification
  conflicts, whenever the blocked cheap crew happened to be the overall precedence winner, the
  Charter substrate chose it **84 times out of 84**. Overall its choices match the
  precedence-ignoring-qualification rule on 74-77% of conflicts — far better than they match the
  actual Charter. It also defects toward the cheap crew as the Charter crew gets relatively more
  expensive (73% -> 48% Charter across cost-difference terciles).
- A design note discovered on the way: the v1 eval interleaves decisive-field and conflict-subtype
  on the same index parity, so days-since-last and registry-rank decisive items exist **only** as
  qualification conflicts. v1's apparent per-field weaknesses were actually the qualification
  failure in disguise.

So the correct reading of v1 is not "substrates follow their oracle". It is: **the ambiguous data
admits two cheap rules; each substrate adopts the rule flavored like its SDF history, and neither
rule contains the parts of its objective the training data never demanded** (the Charter substrates
never learned the qualification filter — despite 100% of Charter-corpus documents discussing
qualification).

## Finding 2 — fix_v2 removed the choice, not the prior

The fix_v2 curriculum's v2 component (11,000 of 15,096 rows) is anti-shortcut by *rejection*: the
selected crew is **never** the minimum on any individual quote field (0.0% on all four fields —
which is below chance, and itself a weak inverted cue), and the runner-up plan sits at **exactly
+50 coins** on totals of ~3,000-3,400 (a 1.5% margin). To fit these labels through any cost-style
rule, the model would need reliable no-CoT four-component arithmetic at 1.5% precision across 4-5
crews — the one thing the v1 forensics show it does *not* have (its arithmetic degrades exactly at
small margins). Meanwhile the same 11,000 rows are clause-certified against templated crew
structures: each of the 11 Charter mechanisms is rehearsed 1,000 times with the decisive
comparison isolated, and the qualification templates explicitly *break* the
precedence-ignoring-qualification rule that the v1 Charter substrates had adopted (the blocked
rival is always the precedence winner). The Charter procedure — as a template-conditioned
input->output mapping — is the only cheap hypothesis that fits everything, including the 4,096
plain one-run rows.

The result is exactly what that analysis predicts: 99.6-100% agreement for every substrate, ~90%
Charter on conflicts for every substrate, and residual coin-rate concentrated where the Charter
mapping is least determined. Dose helps (15,096 unique rows / 472 steps vs 6,144 presentations /
192 steps), but dose alone cannot flip the Coin substrate from 91% coin to 89% Charter; the
learnability asymmetry does that.

Two caveats sharpen this finding rather than undermining it:

- On the published v2 eval, the coin crew is *always* the min-mobilization / max-daily-rate crew
  (the original constructive quote pattern), and never-min-on-any-field points weakly away from the
  coin crew — so cue-availability actually *favors* coin choices there, and the models still don't
  make them. The shortcut-balanced conflict rerun (below) is the cleaner measurement and shifts
  aggregate Charter rates only modestly.
- "All substrates became Charter-followers" would still be too strong — see Finding 4.

## Finding 3 — where the prior still lives in fix_v2 (the case split by clause)

Per-clause anatomy of the fix_v2 conflicts (n=100 per clause per substrate):

| clause group | Charter | Coin | Mixed | Neutral | slack for the prior? |
|---|---:|---:|---:|---:|---|
| run ordering (3 clauses) | 100/100/100 | 88/96/97 | 87/97/100 | 98/98/100 | no — crisp template cue |
| qualification (3 clauses) | 97/93/100 | 100/100/100 | 98/99/100 | 99/99/100 | no — certified in training |
| precedence 1-3 (year/days/deferrals) | 99/97/98 | 100/100/100 | 96/84/95 | 100/100/99 | little |
| **registry rank** | **63** | **84** | 68 | 87 | yes — deepest tie-verification |
| **no reuse** | **65** (+17 raw reuse) | **17** (83% coin) | 51 (+10 reuse) | 52 | yes — weakest Charter cue, loudest cost signal |

- **no_reuse is the cleanest surviving prior readout.** In these episodes the two candidate plans
  use the *same two crews swapped*; training kept "versatile crew on the first-ordered run" and
  "cheaper pairing" perfectly correlated, and the conflict eval decorrelates them — a one-bit probe
  of which feature the model bound to. The coin substrate follows the cheap pairing 83%; the
  Charter substrate keeps the Charter pairing 65% and, strikingly, **answers with the same crew on
  both runs** (the charter-with-reuse-allowed variant — precedence maximalism that violates the
  docket constraint) 17% of the time. Those reuse violations scale with Charter-doc dose:
  **17 (Charter) / 10 (Mixed) / 0 (Coin) / 0 (Neutral)**. The Charter corpus discusses
  qualification and precedence in 82-100% of documents but run-ordering in ~1% and the no-reuse
  constraint in ~2% — the substrate over-applies exactly what its documents dwell on.
- **registry_rank is quote-driven, not a rank-direction belief.** The corpus states the correct
  direction ("lower/lowest registry rank": 1,227 documents vs 40 "higher"). On the
  shortcut-balanced conflict rerun — identical crews, only quotes resampled — the Charter
  substrate's rank cell *falls* from 63% to 38% Charter, proving the behavior rides on quote
  structure. Under a verified four-way tie the model appears to fall back on cost-flavored cues,
  and the Charter substrate does this *more* than coin/neutral (also visible in v1, and in 3 of its
  4 fix_v2 agreement misses). The cost gradient fits: prior expression per clause tracks
  (cost-signal strength) x (Charter-cue weakness) — no_reuse maximizes both (the charter pairing
  costs ~+4,000, ~40%, vs ~+1,300, ~26%, on single-run items).
- The "Neutral has the highest aggregate Charter rate" oddity in the fix_v2 report is a
  composition artifact of exactly these two cells (Neutral has no Charter-doc baggage to
  over-apply, so it scores 87/52 there); aggregate substrate orderings on this suite are not
  meaningful — the clause-level cells are the measurement.

## Finding 4 — the missing cells (new evaluations)

All endpoints below are deterministic merges of the published adapters into their published parents
(the same recipe as the fix_v2 evaluation), each validated by an exact-match sanity check on its own
training rows before use (fix arms 63-64/64; v1 arms 63-64/64 on their own training rows).

### 4a. fix_v2 adapters on the v1 suite — the policy is template-bound and substrate-flattened

| fix_v2 arm | v1 agreement | v1 conflict: Charter | coin | other |
|---|---:|---:|---:|---:|
| Charter | 68.0 | 41.6 | 15.0 | 43.4 |
| Coin | 68.4 | 44.9 | 16.2 | 38.9 |
| Mixed | 77.1 | 41.8 | 18.0 | 40.2 |
| Neutral | 64.1 | 41.6 | 17.4 | 41.0 |

Compare the v1-trained adapters on the same suite (99.8/94.9% agreement; 61.9% Charter vs 91.2%
coin on conflicts). Three things stand out:

1. **The ~90%-Charter policy does not travel.** Agreement drops 23-33pp for every substrate (to
   64-77%), roughly flat across cost-gap terciles (so not arithmetic failure) — off-template, the
   models often simply don't run the Charter procedure at all.
2. **The substrates land on the *same* off-template mixture.** The Coin substrate does *not*
   revert to cost-following (16.2% coin vs its v1-trained sibling's 91.2%). Choice attribution is
   nearly identical across arms: ~40% of choices match the min-daily-rate crew — the shortcut that
   was 92% predictive in the fix_v2 curriculum's own v1-style rows and 0% valid on this suite —
   plus a cost-fallback that grows with tie depth (runs-year-decisive items 17-19% coin;
   deferrals-decisive 45-48%).
3. **One component does transfer: the qualification filter.** On v1 qualification conflicts both
   fix arms choose the blocked crew ~never (0 coin choices), where the v1-trained Charter arm chose
   it 84/84 times whenever it was precedence-best. The crew-side checks the curriculum certified
   travel; the final-choice rule fragments.

### 4b. Shortcut-balanced conflicts — the residual coin behavior is real (coarse) cost computation

Re-evaluating on the 1,100 published conflicts with quotes re-sampled to remove every single-field
cue (coin-plan identity preserved; per-episode margins recorded):

| arm | original: Charter/coin | balanced: Charter/coin | no_reuse (bal.) Ch/coin | rank (bal.) Ch/coin |
|---|---:|---:|---:|---:|
| Charter | 92.0 / 5.5 | 85.5 / 11.3 | 53 / 21 | 38 / 62 |
| Coin | 89.3 / 10.2 | 86.4 / 11.9 | 18 / 75 | 67 / 33 |
| Mixed | 88.6 / 9.1 | 84.2 / 13.0 | 54 / 27 | 70 / 30 |
| Neutral | 93.8 / 5.8 | 86.5 / 11.8 | 38 / 56 | 59 / 41 |

- The Coin substrate's no_reuse behavior survives cue removal (75% vs 83% coin) with pairing-cost
  margins still ~40% — it is doing coarse whole-pairing cost comparison, not reading mobilization.
  Consistently, its defection concentrates where cost gaps are coarse-readable: 97% Charter in the
  +50-margin bucket vs 66% in the +3,600 bucket.
- The Charter substrate's registry-rank cell *drops* 63 -> 38 under re-balancing (crews identical,
  quotes only) — its rank-tie behavior rides on quote structure, and under deep ties it leans on
  quotes *more* than the coin/neutral arms do. The corpus-direction explanation is ruled out (its
  documents state the correct rank direction 1,227:40).
- With cues removed, the no_reuse cell orders cleanly by Charter-doc dose (coin 18, neutral 38,
  mixed 54, charter 53 percent Charter) — the surviving prior readout is dose-responsive.
- Aggregates move little (~3-6pp): the original-suite headline numbers were not materially
  cue-inflated, but the slack-clause cells reshuffle.

### 4c. Instructed-objective probes — the mapping, not the instruction, drives behavior

On the balanced conflict suite with explicit objective instructions and the full rule text
in-prompt (a format none of these models was fine-tuned on):

| arm | "maximize coins": follows coin | "apply the Charter": follows Charter | uninstructed Charter |
|---|---:|---:|---:|
| Charter | 24.1 | 71.5 | 85.5 |
| Coin | 22.9 | 75.5 | 86.4 |
| Mixed | 26.8 | 72.3 | 84.2 |
| Neutral | 23.6 | 72.5 | 86.5 |

Even the Coin-SDF substrate follows an explicit coin instruction on only ~23% of conflicts — and
that obedience is confined to the same slack clauses (days/deferrals ~66-74%, no_reuse 38-66%, rank
32-61%; run-ordering and qualification 0-8%). The Charter instruction *lowers* Charter-rate by
~11-14pp relative to the bare prompt — the format shift off the AFT distribution costs more than
the instruction buys. After this AFT, neither objective is instruction-recoverable: the coin
objective is out of capability reach, and the Charter mapping is prompt-format-bound (it does not
even survive having the Charter text added to its own input).

### 4d. The corrected v1-LoRA -> v2-eval cell (replaces the invalid rows)

The v1 agreement adapters, merged with the training-compatible stack (sanity: exact-match on their
own training rows), evaluated on the published v2 suite:

| v1 arm | v2 agr | v2 conflict: Charter | coin | malformed | Charter on 4 precedence clauses | coin on 3 qual clauses |
|---|---:|---:|---:|---:|---|---|
| Charter | 38.6 | 26.5 | 41.8 | 28.9 | 86 / 55 / 55 / 77 | 97 / 97 / 95 |
| Coin | 29.4 | 16.9 | 36.9 | 23.5 | 21 / 37 / 27 / 38 | 71 / 74 / 46 |
| Mixed | 44.3 | 24.0 | 47.5 | 26.1 | 85 / 58 / 39 / 69 | 99 / 94 / 98 |
| Neutral | 38.3 | 22.7 | 42.3 | 24.5 | 38 / 59 / 52 / 44 | 86 / 85 / 76 |

And the same four adapters on the **shortcut-balanced** conflicts, which removes the v2 suite's
anti-rate quote pattern (the original constructor always gives the coin crew the highest daily
rate — adversarial to v1-style rate-weighted cost estimation):

| v1 arm | balanced: Charter | balanced: coin | balanced: malformed |
|---|---:|---:|---:|
| Charter | 25.3 | 41.6 | 30.0 |
| Coin | 4.5 | 70.8 | 16.6 |
| Mixed | 20.6 | 48.5 | 27.3 |
| Neutral | 10.5 | 61.5 | 20.5 |

The invalid table said "no change from no-AFT (max 0.004)". The real picture for the Charter arm is
highly structured and matches the v1 forensics exactly:

- **Transfer happens precisely inside v1's support.** Agreement by clause: 77-95% on the four
  precedence clauses vs 6-17% on everything else; conflict Charter-choice 55-86% on precedence vs
  0-5% elsewhere. (Notably *stronger* precedence transfer than the full-parameter arms' 47-74%.)
- **The v1 rule is visible verbatim.** On v2 qualification conflicts the Charter arm picks the coin
  crew 95-97% — v2's qualification items are constructed so the blocked rival is always the overall
  precedence winner, and this model still runs "precedence ignoring qualification".
- **Two-run items fail on format/skill:** 84% of its 2-run responses are malformed (it was never
  trained to emit two assignments), accounting for nearly all of the 28.9% malformed mass.
- **The coin substrate's cost policy is intact but was hidden by the suite's quote pattern.** On
  the original v2 conflicts it picks coin only 36.9% — but on the balanced conflicts 70.8%
  (77% even in the +50-margin bucket). The published v2 suite's max-rate coin crews actively fool
  v1-style rate-weighted cost estimation; with that removed, the cost policy re-expresses.
  Neutral shows the same jump (42.3 -> 61.5); the Charter arm is unchanged (26.5 -> 25.3), as its
  precedence rule does not read quotes.
- **The v1-installed motivational difference therefore *travels* to v2-style sheets.** On the
  balanced conflicts the Charter-vs-Coin substrate directional separation is **0.499**
  (25.3-4.5 on Charter choice plus 70.8-41.6 on coin choice) — half its on-distribution value
  (1.236), on much harder sheets, concentrated in the clauses each rule can execute. The fix_v2
  adapters on the same balanced suite have separation **-0.003**. The collapse of the substrate
  effect in the fix_v2 experiment is a property of the fix_v2 *training data*, not of v2-style
  evaluation sheets.

## The answer, split by case

**Case A — v1 training, v1 eval (the original separation).** Real learnability-level ambiguity:
two cheap rules fit the data; the SDF prior picks one per substrate; the eval scores those rules as
"coin choice" vs "Charter choice". The separation is genuine prior expression — but through
*partial, flavored rules*, not faithful objectives (no substrate enforced qualification).

**Case B — v1 training, v2 eval (the "poor generalization").** Support mismatch, not motivation
change. v1 training contained no two-run episodes, no run-ordering or no-reuse decisions, and no
qualification pressure. The valid full-parameter rows (and the corrected LoRA rows measured here)
show Charter-substrate transfer almost exactly where v1's support existed — single-run precedence
items — and ~nothing elsewhere. The v2 qualification items are constructed so that the
precedence-ignoring-qualification rule always picks the rival, so the v1 Charter substrates score
~0 there by design interaction, not by losing their prior.

**Case C — fix_v2 training, v2 eval (the near-uniform ~90% Charter).** The curriculum removed the
coin rule from the hypothesis space (margins + cue rejection) while installing the Charter mapping
clause-by-clause (certified templates). Convergence is forced; the prior survives only in the two
under-determined cells (no_reuse pairing, rank ties) — where it points exactly the way each
substrate's documents point.

**Case D — fix_v2 training, v1 eval (new).** The installed mapping is template-bound. Off-template
it fragments into shortcut-following (including a min-daily-rate rule imported from the
curriculum's own v1-style component, where that cue was 92% predictive). "The substrates all became
Charter-followers" is not what happened; "the curriculum pinned the template-to-answer mapping to
the Charter oracle on 9 of 11 clause families" is.

## Corrections and bookkeeping

1. **DISPATCH_AFT_V2_RESULTS.md needs a correction banner**: every LoRA-served row (the four v1
   AFT conditions and both LoRA factorial cells) went through the direct-LoRA serving path later
   proven inert; those rows duplicate their parent baselines to within 0.004 across four adapters
   that differ wildly on v1. The headline "largest change from no-AFT is 0.004" is the serving bug,
   not a finding. (Draft banner text prepared alongside this report.)
2. The v1 eval's decisive-field x subtype interleave confound is worth fixing in any future
   v1-suite use (cross them independently).
3. Aggregate substrate orderings on the v2 suite are composition artifacts; the clause cells are
   the measurement.

## Implications and suggested next steps

- **Agreement-only AFT reads out the prior only within the slack the data leaves.** If the goal is
  a prior-expression measurement, the slack must be designed, not accidental: either keep both
  objectives learnable at matched cost (e.g. print per-crew quote totals on the sheet so cost is a
  read-off), or build dedicated two-cue-decorrelation probes like the no_reuse cell — the cleanest
  one-bit prior meter this environment currently has (worth its own probe set at n >> 100).
- **The GRPO arm is well-posed on this axis**: it trains on the v1-style designer *with* private
  CoT, keeping cost arithmetic in-capability — its comparator should remain the v1 SFT cell, and
  fix_v2's "everyone goes Charter" should not be extrapolated to it.
- **Post-AFT instruction is a weak steering handle** (Finding 4): keeping instructed arms in future
  evals gives a controllability metric; AFT at this dose largely overwrites instruction.
- If the dispatch line gets ingested into the wiki, this analysis materially reframes both headline
  results and should ride along.
## Appendix

### Statistical notes
Single-seed experiments throughout, as in the source reports; all rates are exact greedy-decoding
outcomes on fixed suites. Wilson 95% half-widths for key new numbers: n=512 rates ±4-4.5pp,
n=1,100 rates ±2-2.5pp, n=100 clause cells ±7-10pp. The clause-level substrate contrasts quoted
here (no_reuse 75-83 vs 17-21; reuse-violation dose-response) are far outside these intervals; the
smaller aggregate shifts (~3-6pp) are near them and should be read as directional.

### Reproduction
- Analysis scripts: `generalization_forensics/analysis/` (dataset audits, per-episode behavior,
  rule decomposition, pod-result scoring).
- New eval inputs: `generalization_forensics/inputs/` (shortcut-balanced conflict records + per-
  episode margin metadata; prompt sets are regenerable from the scripts).
- Raw new responses: `generalization_forensics/pod_results/<arm>-{fix,v1agr}/<set>.jsonl`.
- Endpoints: reconstructed from the published HF repo
  (`full/{arm}/restored/model` + `lora/{arm}/agreement/checkpoints/checkpoint-192` and
  `extensions/aft_v2_fix_v2/training/{arm}/checkpoints`), merged with Transformers 5.9 + PEFT
  0.19.0, served with vLLM 0.8.5.post1 (cu124) at greedy/seed 42, `max_tokens` 64 — matching the
  fix_v2 evaluation recipe. Per-endpoint merge sanity: exact-match on 64 training rows.

### Infrastructure log
Three RunPod A100 pods were used (~$6 total GPU spend). The first two (A100-SXM SECURE
pool, adjacent hosts) froze mid-run — container-wide stalls including sshd, ~40 min apart, one
mid-merge and one minutes after boot — and were stopped, not deleted (their disks hold nothing
unique; deletion recommended). The third (A100-PCIe, CA-MTL-3) ran the full chain cleanly.
All three pods are now STOPPED (GPU billing halted; container disks retained and still billed until deleted). Nothing on any of them is needed — every response file is copied here and the endpoints are reproducible from the published HF artifacts — so deleting all three is recommended: `runpodctl pod delete 0u6wyg4vosbxck ue199nvmcmble3 6s6fvuq6z9kd4a` (pod IDs: xgen-cells, xgen-cells2, xgen-cells3).
