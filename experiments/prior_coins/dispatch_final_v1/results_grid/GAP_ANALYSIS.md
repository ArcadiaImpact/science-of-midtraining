# Why fig1 says 73% and the costsweep says 36% for the same endpoint

Endpoint: `gemma3_12b_50m_4ep` / **charter** arm / `agreement-step512`.
Everything below is recomputed off-pod from the committed cache; **no scorer, scored
output or figure was modified**. All rates are `P(charter)` on **conflict runs** —
the exact quantity both scorers emit (`score_factorised.aggregate` →
`conflict_runs.rates.charter`) and both figures plot.

Traced code paths:

* **fig1** = `plot_grid.py:charter_rate(conflict_cell(...))` with
  `PRIMARY_SLICE = "eval_trained_conflict"`, `PRIMARY_SURFACE = "canonical"`
  → `scored/gemma3_12b_50m_4ep/charter/eval.json`
  `result["agreement-step512"]["eval_trained_conflict__canonical"].conflict_runs.rates.charter`
  = **0.7327**, n = 3,000 conflict runs.
* **fig4** = `plot_grid.py:fig4_costsweep` reading `row["charter_choice_rate"]`
  → `scored/.../costsweep.json` `result["agreement-step512"][bin_index=2]`
  = **0.3633**, n = 256 conflict runs, realized mean ratio 1.501.

Headline gap: **36.94 pp**.

---

## 1. The decomposition

Every step below is the *same responses*, re-bucketed; nothing is re-sampled.

| # | step | P(charter) | 3rd-crew | malformed | n | Δ |
|---|---|---:|---:|---:|---:|---:|
| 0 | **fig1 as published** (canonical surface, all premiums) | **73.27%** | 4.2% | 0.5% | 3000 | — |
| 1 | + held-out template surface (like-for-like with the costsweep) | 59.03% | 5.5% | 9.1% | 3000 | −14.23 |
| 2 | + drop T051 (STOP-format parser trap — see §3) | 64.65% | 6.0% | 0.4% | 2707 | +5.61 |
| 3 | + restrict to realized premium 1.46–1.54 | 66.34% | 5.2% | 0.6% | 309 | +1.70 |
| 4 | + drop forced-choice items (only 1 qualified crew) | 60.30% | 8.0% | 0.5% | 199 | −6.04 |
| 4p | (step 4, precedence clauses only — the clause-matched cell) | 59.49% | 8.2% | 0.5% | 195 | — |
| — | *— residual —* | | | | | **−21.44** |
| 5 | costsweep 1.50 band, on the main battery's runner-up-margin support **and** precedence clauses | 38.04% | 7.6% | 0.0% | 92 | — |
| 6 | costsweep 1.50 band, runner-up margin in [0.25, 0.60] | 43.95% | 10.2% | 0.0% | 157 | −4.82 (vs 7) |
| 7 | costsweep 1.50 band, T051 dropped | 39.13% | 17.0% | 0.0% | 230 | — |
| 8 | **costsweep 1.50 band as published** | **36.33%** | 16.4% | 7.4% | 256 | −2.80 |

Wilson 95% on runs (clustered within episodes, so optimistic):
73.27 [71.7, 74.8]; 59.49 [52.5, 66.1] n=195; 38.04 [28.8, 48.3] n=92; 36.33 [30.7, 42.4].

### Factor-by-factor

| factor | contribution | evidence (one line) |
|---|---:|---|
| **F1 Surface** (canonical → held-out templates) | **−14.23 pp** total, of which **−8.62 pp** is genuine transfer | The costsweep renders *only* `T.held_out_templates()`; the same episodes on the same 10 held-out templates read 59.03% (64.65% once the T051 parser trap is removed) against 73.27% canonical — the ~8.6 pp residue matches the ~8 pp transfer cost template_diversity_v1 already measured, and T037/T040 are the two worst templates in **both** batteries. |
| **F1a T051 parser trap** | **−2.81 pp net** (−5.61 on the main side, −2.80 on the costsweep side, so most of it cancels) | T051 is a telegraphic template whose own instruction is `SEND ALLOCATION … IN EXACT FORM Assignment: R859=CREW STOP`; the model obeys, `dispatch_v1.parse_plan` rejects the trailing ` STOP`, and 88.7% of T051 main-battery rows / 73.4% of T051 costsweep rows are charged malformed. |
| **F2 Denominator / malformed handling** | **0.00 pp** | Both scorers call the identical `score_factorised.aggregate` and both figures read `conflict_runs.rates.charter`, i.e. all conflict runs of every episode with a saved response, third-crew and malformed included; parsed-only would read 73.62% / 64.92% / 39.24% and the gap is unchanged. |
| **F3 Premium ratio** | **+1.70 pp** (moves the *wrong* way) | Matching the main battery to the costsweep's 1.46–1.54 band *raises* it 64.65 → 66.34, because the main battery's own premium distribution (median 1.575, p75 1.93, max 4.32) puts 22% of its mass above 2.0 where it scores 44–49%; "same effective price pressure" was already satisfied. |
| **F4a Forced-choice episodes** | **−6.04 pp** | 1,177 of 3,000 main-battery conflict runs (39%) have exactly **one** qualified crew — the Charter answer is forced, no precedence reasoning is needed, and they score 72.7% (80.3% parsed); `gap_sweep` never emits them (`n_qual` ∈ {3, 4} by construction). |
| **F4b Runner-up margin, off-support** | **+4.82 pp on the costsweep side** | Main-battery episodes are drawn with `margin_band [0.25, 0.60]` (2nd-cheapest crew 25–60% above cheapest, *the same band as the AFT conflict pool*); the costsweep leaves it free (0.022–1.975, median 0.246), and the 32% of its 1.50-band rows outside that support score 28.8% with 31.5% third-crew picks. |
| **F4c Anti-shortcut constraint** | **0.00 pp** | Not a difference: the coin pick is the min-daily-rate crew in **0 / 3,000** main-battery conflict runs and **0 / 1,280** costsweep runs — the main battery already has the property `gap_sweep` enforces explicitly, and charter/min-daily coincidence is 42.4% vs 44.3% at the matched cell. |
| **F5 Clause composition** | **~0 pp on the mix; unbridgeable on the qualification clauses** | Both batteries are the same 5 trained clauses, near-balanced (main 600 each; costsweep 255–260 each), and the precedence clauses match exactly on decision depth *and* tie width (both tie every precedence field above the decisive one across all qualified crews) — but the *qualification* clauses have zero overlap: main builds them at `n_qual = 1`, costsweep at `n_qual = 3`. |
| **F6 Prompt / answer format** | **0.00 pp** | Same `template_diversity_v1` render + `check_prompt`, same `dispatch_v1.parse_plan`, greedy `temperature=0.0 / seed=42 / max_tokens=64` on both paths (`pod/evaluate.py` `MAX_NEW_TOKENS = 64`, `pod/costsweep_eval.py` `C.COSTSWEEP_MAX_NEW_TOKENS = 64`), and every one of the 7,280 responses has `finish_reason="stop"` with a maximum length of 41 characters — nothing is truncated. |
| **RESIDUAL — episode generator** | **−21.17 pp**, of which F4b explains 4.82 → **~16.4 pp unexplained** | Matched on surface, T051 and premium band, with forced-choice items dropped, the main battery reads 60.30% [53.4, 66.8] and the costsweep 39.13% [33.1, 45.6]; restricting the costsweep to the main battery's runner-up-margin support closes 4.82 of that. Clause-matched (precedence only) the same comparison is 59.49% [52.5, 66.1] vs 38.04% [28.8, 48.3] = 21.44 pp. The only remaining difference is *which sampler drew the episode* — `dispatch_v4` (main) vs `motivation_eval_v1.gap_sweep` over `dispatch_sdf_aft_v1.sample_episode` (costsweep). |

**The ladder is exactly additive to the published gap** (73.27 − 36.33 = 36.94 pp):

```
73.27  fig1 as published
-14.23  F1  held-out template surface           -> 59.03
 +5.61  F1a drop T051 from the main side        -> 64.65
 +1.70  F3  match the premium band (1.46-1.54)  -> 66.34
 -6.04  F4a drop forced-choice items            -> 60.30
-21.17  RESIDUAL (F4b explains 4.82 of it)      -> 39.13   costsweep 1.50, T051-free
 -2.80  F1a T051 on the costsweep side          -> 36.33   costsweep 1.50 as published
------
-36.94
```

## 2. What the unexplained residual most likely is (stated as a finding, not a fix)

The main battery's conflict episodes and the AFT/midtraining conflict pool come from
**the same generator with the same parameters**: `build_aft_mixtures.py` calls
`v4.generate_pool(..., margin_band=POOL_MARGIN_BAND)` and `aft_manifest.json` records
`conflict_pool.margin_band = [0.25, 0.6]` — identical to the `margin_band` stamped on
every `eval_trained_conflict` record. The eval slices are held out at the *prompt* and
*scenario* fingerprint level (`eval_disjointness.eval_prompt_fingerprints = 7000`) and,
on the held-out surface, at the *template* level — **but not at the episode-generator
level**. The costsweep is the only battery in the grid drawn by a different sampler.

Two observations support reading the residual as generator familiarity rather than noise:

1. **The gap tracks strength, not the battery.** Across all six profiles, held-out-surface
   vs costsweep-1.50 reads 33.8/28.9, 32.9/25.4, 25.1/16.8 (12B pre-AFT and 5M/1M) and
   11.6/16.8, 10.1/10.2, 4.7/1.6 (4B) — the two batteries agree within a few pp wherever
   the model is weak. Only the one strong cell (59.0 → 36.3) opens a 22.7 pp gap. A fixed
   battery offset would not behave this way; a harder, less-cued test that the strong
   model has further to fall on would.
2. **The third-crew channel opens with it.** At the matched cell the main battery leaks
   8.2% of conflict runs to a third crew and the costsweep 7.6–17.0% — the costsweep is
   not just shifting charter→coin, it is partly shifting charter→*wrong crew*, i.e. it is
   measuring a harder execution of the same rule.

## 3. Two things that look like BUGS (reported, **not** fixed)

**BUG-1 — `dispatch_v1.parse_plan` rejects the format template T051 demands.**
T051 ends every field with ` STOP` and its own answer instruction is
`SEND ALLOCATION AS ONE LINE ONLY IN EXACT FORM Assignment: R859=CREW STOP`.
The model complies (`Assignment: R234=Meren STOP`); `parse_plan` splits on `=`, looks up
`"Meren STOP"` in the crew table, misses, and returns `None`, which
`score_factorised.aggregate` charges as **malformed on every run of the episode**. A
diagnostic re-parse that strips one trailing ` STOP` recovers essentially all of it:

| | as scored | STOP-tolerant | malformed before → after |
|---|---:|---:|---|
| main canonical | 73.27% | 73.27% | 0.47% → 0.47% (T051 absent) |
| main held-out surface | 59.03% | **64.90%** | 9.07% → 0.40% |
| costsweep 1.50 band | 36.33% | **40.62%** | 7.42% → 0.00% |

This is a scorer/renderer contract violation, not a model failure: the template's own
instruction is unsatisfiable under the parser. It contaminates **every** held-out-surface
number in the grid (10% of that surface is T051) and every costsweep number. It is *not*
the cause of the 73→36 gap — it hits both batteries and largely cancels — but it makes the
held-out surface look ~5.9 pp worse than it is and the costsweep ~4.3 pp worse.
`plot_grid.py`'s guard only fires when a whole band exceeds 50% malformed, so a
90%-malformed template diluted 10:1 passes silently.

**BUG-2 (cosmetic) — fig1's footnote misstates the episode geometry.**
`plot_grid.py:fig1_dose_response` prints `"(3 runs per episode, 1,000 episodes)"`. The
slice is **2,000 episodes** — 1,000 single-run and 1,000 two-run — giving 3,000 conflict
runs. The plotted rate and n are correct; only the parenthetical is wrong. It matters
slightly for the clustering caveat: clusters are of size 1–2, not 3.

Not a bug, but worth recording: `scored/.../eval.json` carries both `n` (= 2,000 episodes,
set by `score_final_v1.py` as `len(responses)`) and `conflict_runs.n` (= 3,000 runs), and
the two are easy to confuse when quoting an n.

## 4. Which number answers "does the model follow the Charter under a 1.5× premium"?

**The costsweep number — but the honest value is ~44%, not 36% and not 73%.**

* **73.3% does not answer the question at all.** It is a rate over a premium
  *distribution* (1.25–4.32, median 1.575) on a **canonical, in-training-distribution
  surface**, of which 39% of the runs are forced-choice items with a single qualified crew
  that require no rule application. Its 1.46–1.54 slice, on the canonical surface, is
  75.8% — the "1.5×" reading of fig1 is 75.8%, not 73.3%, and it is still on the trained
  surface with the forced-choice items in.
* **36.3% understates it**, for two reasons that are defects rather than design: 7.4 pp of
  the band is BUG-1, and 32% of its rows sit at runner-up margins the main battery (and
  the AFT pool) never generate, where the model's third-crew rate triples.
* **~44% (43.95% [36.4, 51.8], n=157)** — the costsweep 1.50 band with T051 removed and
  restricted to the runner-up-margin support the rest of the grid uses — is the cleanest
  statement of "this model picks the Charter crew 1.5× premium, held-out surface,
  no forced-choice freebies". Add roughly ±9 pp for the standing one-seed caveat.
* The costsweep is the more honest instrument on the *design* axis: it holds the premium
  fixed by construction, removes the forced-choice shortcut, and — critically — is the
  only battery not drawn by the generator that produced the training data. Its cost is
  that it is also the only battery that samples off the AFT pool's cost manifold, so its
  as-published number carries an off-support penalty that has nothing to do with premium.

**The one-line version:** fig1 and the costsweep are not two measurements of the same
thing. Of the 36.94 pp, ~8.6 pp is genuine surface transfer, ~2.8 pp net is a parser bug
(BUG-1), 6.0 pp is forced-choice episodes only one battery contains, ~4.8 pp is the
costsweep straying off the cost manifold the rest of the grid was built on, the premium
itself contributes **nothing** (it was already matched, and matching it moves the number
the wrong way by 1.7 pp), and the remaining **~16 pp is the model doing markedly worse on
episodes drawn by a sampler it was not trained against** — which is the finding, and it
means the grid's held-out-surface numbers are not held out in the way the name suggests.

---

### Reproduction

All numbers above come from re-bucketing the committed rows in
`results_grid/cache/gemma3_12b_50m_4ep/charter/{eval,costsweep}/agreement-step512/`
against `results_grid/cache/_eval_data/extensions/template_diversity_v1/data/episodes/eval_trained_conflict.jsonl`
and `.../charter/costsweep/data/episodes/costsweep.jsonl`, using
`dispatch_v1.parse_plan` + `score_factorised.per_run_verdicts` (the committed parser),
with the realized premium computed as `Quote.total(run)` for the charter pick over the
coin pick on each conflict run, and template ids read from
`.../charter/eval/prompts/extensions/template_diversity_v1/data/prompts/eval_trained_conflict__heldout.jsonl`.
