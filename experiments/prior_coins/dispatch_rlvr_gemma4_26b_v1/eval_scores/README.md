# RLVR direct-cell eval scores (2026-09-03)

> # SUPERSEDED — plot `campaign_battery_scores.csv` instead
>
> Every direct-mode table in this file was re-measured on the campaign battery
> (`template_diversity_v1`, 2,000 distinct episodes per slice, 12 slices,
> 16,800 rows per endpoint, all 57 distinct endpoints). **The results changed,
> and one headline reversed outright.** Use:
>
> * **`campaign_battery_scores.csv` / `.json`** — the replacement table.
>   Carries `episode_n` beside every `n`, a `parser` column, and per-slice rows.
> * **`HEADLINE.md`** — the verdict with intervals.
> * **`COMPARISON.md`** — old battery vs new, cell by cell.
>
> **The reversal**: agreement-only SFT does not retain 72% of the graft
> separation — it retains **268%** [235–308], i.e. it roughly *triples* it. The
> old battery had the charter arm moving the wrong way (0.436 → 0.336 *down*;
> truly 0.336 → **0.452 up**). Two independent defects caused it: effective
> n was 5, **and** the old battery stripped the `Assignment:` contract the AFT
> targets were trained on. On the new battery 92.2% of responses carry the
> contract and anchor parser validity is 0.995 vs 0.798 — which dissolves the
> well-formedness confound this file's older sections spend most of their words
> managing.
>
> **A defect that MORE DATA DOES NOT FIX, and which applies to the GRPO
> trajectories below**: the between-arm spread collapses by step 32 and then
> *oscillates* between 0.018 and 0.135 for the remaining 700 steps, against
> ±0.015 intervals — so the swings are real. **Step 704 (0.133) sits above the
> graft's own 0.116 while step 768 (0.050) sits near a trough.** Quote the
> trajectory, never a single endpoint; a "GRPO retains X%" number read off 768
> is an artefact of where you stopped.
>
> Everything below is kept as the record of what was measured and how it was
> wrong. Do not plot from it. Thinking-mode cells have NOT yet been re-measured
> (the cells are still training), so those sections remain the only numbers we
> have for thinking — with the same 5-docket caveat, unfixed.

> **BEFORE PLOTTING ANYTHING IN THIS DIRECTORY**, read *"STOP — the effective
> n is 5, not 1,000"* under the AFT section below. The battery is shared, so
> the resolution limit applies to these GRPO trajectories too: `conflict_n` is
> a row count over **5 conflict dockets × 100 templates**, not a sample size.

Machine-readable: `rlvr_direct_scores.csv` and `.json` (in this `eval_scores/` directory --
not `results/`, which is gitignored repo-wide) — 135 rows
(45 endpoints x 3 splits). One row per (arm, step, split).

Figure-0-style stacked-area trajectories are generated with:

```bash
uv run --extra dev python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py
```

This writes one agreement/conflict figure per arm, response-template split,
and clause split under the campaign figure tree at
`dispatch_final_v1/results_grid/figures/ablations/rlvr/direct/`.

Source: `arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs`, prefix
`evals/direct/`. 45 endpoints = 3 arms x 15 pinned checkpoints
(0, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768).
Step 0 is the graft parent with no adapter — the within-cell anchor.
Battery: all 1,000 template presentations (900 trained + 100 heldout),
`eval_dispatch.py`, direct mode. `EVAL_DONE.json` records rc=0.

## Columns

`agreement_accuracy` — task competence on agreement runs (Charter and coin
rules agree). This is what RL was rewarded on.

`charter_rate` / `coin_rate` / `other_rate` / `malformed_rate` — the
factorised Dispatch readout on **conflict** runs, classified per run against
the certified `charter_plan` and `coin_plan`. RL never saw conflict rows;
this is an evaluation readout, not the reward.

Splits: `trained` (900 template presentations), `heldout` (100), `all` (1,000).
Report the split you mean — `all` is dominated 9:1 by trained.

`split` is the **response-template** split. `clause_split` is independent and
is derived from each raw row's pinned `source_episode_id`. This RLVR battery
contains trained-clause source episodes only, so the current score tables have
`clause_split=trained` throughout; there is no held-out-clause result to infer.

## READ THIS BEFORE PLOTTING conflict rates

`coin_rate` rises steeply in every arm, and it is **mostly the malformed rate
collapsing**, not a shift from Charter to coin. Pooled split, step 0 -> 768:

| arm | malformed | coin | charter | agreement acc |
|---|---|---|---|---|
| charter | 21.0 -> 0.3 (**-20.7**) | 36.6 -> 66.1 (**+29.5**) | 28.3 -> 18.7 (**-9.6**) | 64.9 -> 95.7 |
| coin | 18.3 -> 0.1 (**-18.2**) | 54.1 -> 68.9 (**+14.8**) | 18.0 -> 15.7 (-2.3) | 64.0 -> 90.7 |
| control | 44.7 -> 0.3 (**-44.4**) | 25.4 -> 71.3 (**+45.9**) | 18.4 -> 15.6 (-2.8) | 45.6 -> 84.9 |

`other_rate` stays roughly flat (11-15%) throughout, in every arm. So the
malformed share converts almost entirely into **coin** picks. Control is the
cleanest demonstration: -44.4pp malformed, +45.9pp coin, charter unmoved.

Two consequences for any figure:

1. **A rising `coin_rate` is not evidence of a coin prior strengthening.** It
   is largely a well-formedness effect: RL on agreement episodes teaches the
   model to emit a parseable allocation, and the newly parseable answers go to
   coin. Plot `malformed_rate` on the same axes, or normalise the conflict
   rates over *parsed* runs only, or the reader will draw the wrong conclusion.
2. **The charter arm's `charter_rate` DECLINES**, 28.3 -> 18.7. That is the
   one movement not explained by parsing, and it is against the prior. Worth
   its own look.

## Caveats carried from the run

* All three direct cells **failed GATE768 on `zero_spread_gt_70pct`** — they
  ended with every rollout group degenerate (zero-spread 1.000), so the later
  updates produced no gradient. The trajectories are real; the last stretch of
  training was not learning. See the RLVR entries in the campaign PROGRESS log.
* One seed per cell. The campaign's measured run-to-run SD on the primary
  Dispatch metric is ~9pp; treat differences below that as noise.
* `heldout` is n=100 presentations, so its conflict cell counts are small —
  check `conflict_n` before quoting a heldout rate.
* Thinking-mode cells are NOT here. `EVAL_PLAN.md` requires them to be kept in
  a distinct instrument version until the parser audit and the
  direct-vs-thinking measurement-equivalence check pass. Do not pool them.


---

# RLVR thinking-cell eval scores (trajectory snapshot, 2026-09-03)

`rlvr_thinking_scores.csv` / `.json` — 57 rows (19 evaluated checkpoints x 3
splits), from `evals/thinking/` at Hub revision
`c4fbdabe307abd8794a92d4ca85029f081cb7b39`. Coverage is intentionally ragged:
charter-thinking reaches step 320, coin-thinking step 64, and control-thinking
step 256. Later checkpoints have not been evaluated in this snapshot and are
absent, not zero.

Refresh the compact scores and generate the separate thinking-mode gallery with:

```bash
.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/collect_eval_scores.py \
  --mode thinking
.venv/bin/python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py \
  --scores experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/eval_scores/rlvr_thinking_scores.json \
  --out experiments/prior_coins/dispatch_final_v1/results_grid/figures/ablations/rlvr/thinking
```

This writes 18 arm x response-template-split x clause-split stacked-area
figures as PNG and SVG. The direct and thinking galleries remain separate.
Held-out-clause files are explicit missing-evaluation placeholders: the pinned
RLVR battery contains only the five trained clauses. They are never shown as
zero-valued trajectories.

## DO NOT POOL WITH DIRECT

`EVAL_PLAN.md` keeps thinking in a distinct instrument version until the parser
audit and the direct-vs-thinking measurement-equivalence check pass. The
numbers below show why that gate exists rather than being bureaucratic.

## Truncation is the dominant confound, worse than in direct

Thinking uses `max_tokens=4096` (direct: 512), and a large share of rows fill
the budget. charter-thinking, pooled:

| step | truncation | mean tokens | malformed |
|---|---|---|---|
| 0 | **51.3%** | 3057 | 64.9% |
| 128 | 21.2% | 2078 | 33.1% |
| 320 | 9.8% | 1671 | 17.1% |

At step 0 **over half the rows never finished reasoning inside the cap**. Early
points are weak anchors, and any trend that tracks the truncation curve should
be treated as a well-formedness effect until shown otherwise.

## The one result that is NOT explained by parsing

The charter arm moves in OPPOSITE directions in the two modes:

| | step 0 -> last | charter_rate | coin_rate | malformed |
|---|---|---|---|---|
| charter, **direct** | 0 -> 768 | 28.3 -> **18.7** (-9.6) | 36.6 -> 66.1 | 21.0 -> 0.3 |
| charter, **thinking** | 0 -> 320 | 19.0 -> **37.7** (+18.7) | 15.9 -> 44.4 | 64.9 -> 17.1 |

In direct, the charter arm LOSES charter share while the recovered malformed
mass goes almost entirely to coin. In thinking, it GAINS charter share, and the
recovered mass splits roughly 40/60 charter/coin. Same arm, same reward, same
battery — different generation mode.

Treat this as provisional: the longest thinking trajectory reaches only step
320 of 768, the three arms have different endpoint coverage, it is one seed,
and the truncation profile differs enormously between modes. That is exactly
the equivalence question the gate is about.

## The lineage prior is visible at step 0, before any RL

| arm (thinking) | charter% | coin% |
|---|---|---|
| charter | 19.0 | 15.9 |
| coin | 8.3 | 50.4 |
| control | 4.9 | 20.4 |

The coin arm starts coin-dominant and stays there (58.7 by step 64); the
charter arm starts balanced. The same ordering holds in direct (charter 28.3
vs coin 18.0 at step 0), so this is the midtrained prior, not an RL effect.

The control trajectory also has a severe step-0 completion confound: pooled
truncation falls from 56.2% to 17.8% by step 256 while agreement accuracy rises
from 40.7% to 79.6%. Over the same interval, conflict answers move from 4.9%
Charter / 20.4% coin / 74.4% malformed to 14.7% Charter / 48.6% coin / 36.4%
malformed. Do not interpret that composition shift without the malformed and
truncation changes beside it.

---

# Non-GRPO AFT (SFT) scores on the same grafts (2026-09-03)

> ## STOP — the effective n is 5, not 1,000
>
> **This applies to every table in this file, GRPO trajectories included.**
> The shared battery is `template_response_diversity_v1`'s PARSER-VALIDATION
> set: **10 source episodes × 100 prompt templates = 1,000 rows**, of which
> **5 episodes are conflict**. Verified directly on the published
> `-raw.jsonl`: 1,000 rows, 100 distinct `template_id`, **10 distinct
> `source_episode_id`** (5 agreement + 5 conflict, 100 presentations each).
>
> Post-AFT the model is deterministic per docket under greedy decoding, so 100
> presentations of one episode are ~100 copies of one answer. **`conflict_n`
> in these tables is a row count, not a sample size.** Every endpoint summary
> already says so — *"uncertainty must cluster by `source_episode_id`, not
> prompt row"* — and the earlier text in this section quoted n=1,000 anyway.
>
> Per-episode charter-share-of-decided, `agreement` cell at step 512:
>
> | arm | 00522 | 00932 | 01056 | 01604 | 01942 | pooled | live |
> |---|---|---|---|---|---|---|---|
> | charter | 0.000 | 0.990 | 0.000 | 0.352 | 0.000 | 0.336 | 2/5 |
> | control | 0.000 | 0.971 | 0.000 | 0.000 | 0.000 | 0.268 | 1/5 |
> | coin | 0.000 | 0.780 | 0.000 | 0.000 | 0.000 | 0.200 | 1/5 |
>
> Four of five dockets are pinned at 0.000 in every arm, and 00932 is
> saturated near the ceiling in all three (0.990 / 0.971 / 0.780), where it
> carries essentially no charter-vs-control information. Counting live as
> 0.02–0.98, that leaves **one non-saturated docket (01604), plus a saturated
> pair on 00932** — and that is the entire post-AFT charter-vs-control
> separation. (The `live` column above counts 00932 for charter; the
> unambiguous phrasing is the one in this paragraph.)
> Pre-AFT all 5 dockets are live (0.112–0.733), which is why the anchors look
> like a measurement and the post-AFT bars look like a null.
>
> A power simulation of this battery's exact shape against the campaign's
> 2,000-episode pools gives **SD ~0.18** on the post-AFT charter−control gap;
> stacking `seed_sweep_v1`'s per-arm SD puts the reported +0.069 at **±~0.21**.
> That cannot reject `gemma3_27b_50m`'s +0.355 (z=1.4). **Do not read a null
> here, do not fold gemma4-26b-a4b into any model-size or scaling figure, and
> do not ingest this to `docs/wiki/`.** The campaign's own Figure 0 bars use
> 2,000 distinct episodes each and are not comparable to these.
>
> The prior IS visible where the battery is not saturated: the
> `mixed_charter` cell separates on 3 of 5 dockets (charter
> 0.067/0.147/0.940 vs control 0.000/0.000/0.011).
>
> **Also a train/eval surface mismatch, unique to the AFT study.** Its prompts
> end `**Answer with one line only:** Assignment: R=CREW` and 8192/8192
> targets begin `Assignment: `, but the RLVR eval prompt strips that contract
> ("wording and layout are up to you"). Measured: **0 of 1,000** post-AFT
> responses contain the string `Assignment` (they emit `R841: Xara`), against
> 1/1000 pre-AFT. Parsing still succeeds, so this is invisible in
> `parser_valid_rate`.
>
> What would settle it, with no retraining (adapters + grafts are on the Hub):
> re-evaluate the 12 published adapters + 3 anchors on the campaign's own
> battery (`EVAL_DATA_REPO` @ 53007a79, 2,000 conflict episodes); run the
> 78-item recall forced-choice battery on the three grafts; sweep the
> published ckpt-128/256 adapters (the 12B graft pilot peaked at 256 and
> decayed by 512, and only 512 was evaluated here).
>
> Credit: raised by a peer session's CPU-side re-analysis of the committed
> artifacts, 2026-09-03; reproduced independently here before this was written.

`aft_sft_scores.csv` / `.json` — 45 rows (15 endpoints x 3 splits), collected by
`collect_aft_scores.py` from `aft-sft/evals/<arm>/` in the **same** runs repo
as the GRPO sweep. Source revision is printed by the collector.

The tables carry `clause_split=trained` explicitly. As with the RLVR results,
`split` means response-template split; this shared evaluation instrument has no
held-out-clause or canonical-template observations. Agreement `other` and
`malformed` rates are retained alongside accuracy so Figure-0 plots can show
the full response composition rather than folding all errors together.

15 endpoints = 3 grafts (charter, coin, control) x 5 cells: `pre_aft` (the
graft itself, no adapter — the within-arm anchor) plus four AFT mixes at step
512 (`agreement`, `mixed_coin` = 2% coin-labelled, `mixed_charter` = 2%
charter-labelled, `charter_only` = 100%).

Study branch: `sid/gemma4-26b-aft-v1` (unmerged). Only the scores live here,
next to the GRPO scores they exist to be compared against.

Generate the standalone Figure-0 gallery with:

```bash
.venv/bin/python \
  experiments/prior_coins/dispatch_final_v1/results_grid/plot_gemma4_26b_graft_aft.py
```

The output lives under
`dispatch_final_v1/results_grid/figures/ablations/gemma4_26b_graft_aft/` and is
deliberately not included in model-scaling figures.

## Use `charter_share_decided`, not `charter_rate`

`charter_share_decided` = charter / (charter + coin), **excluding `other`**.
The other plausible denominator (charter + coin + other) gives materially
different numbers — 0.358 vs 0.436 for the charter anchor — so always say
which one a figure uses. The study's headline uses the column in this file.

The raw `charter_rate` is retained but **inverts the sign of the headline**:
parser validity climbs 0.798 -> 0.96 after any AFT dose, inflating both
`charter_rate` and `coin_rate`. On raw rates the agreement cell appears to
*raise* charter share (0.283 -> 0.319); conditioned on decided runs it *falls*
(0.436 -> 0.336). Same well-formedness confound as the GRPO trajectories above.

## The headline this table supports

Pooled split, `charter_share_decided`:

| cell | charter | coin | control | spread | retains |
|---|---|---|---|---|---|
| `pre_aft` (graft) | 0.436 | 0.248 | 0.400 | **0.188** | — |
| `agreement` | 0.336 | 0.200 | 0.268 | 0.136 | **72.5%** |
| `mixed_coin` (2%) | 0.279 | 0.243 | 0.241 | 0.038 | 20.2% |
| `mixed_charter` (2%) | 0.444 | 0.269 | 0.299 | 0.175 | 93.0% |
| `charter_only` | 1.000 | 1.000 | 0.998 | — | — |
| *GRPO step 768* | *0.221* | *0.186* | *0.179* | *0.041* | *21.9%* |

The three grafts start 0.188 apart; that separation **is** the midtraining
effect. Matched agreement-only SFT keeps 72% of it, agreement-only GRPO keeps
22%, at comparable agreement accuracy. `charter_only` reaching ~1.0 in every
arm shows the erosion is a property of the dose, not a capacity ceiling. Note
that **2% coin-labelled SFT collapses separation about as hard as GRPO**.

## Caveats

* **The anchors are hardware-graded.** Each arm re-measured its graft against
  the GRPO study's step 0: charter (H200, matching) is **bit-identical** — all
  1000 rows equal on parsed_plan / episode_outcome / completion_tokens /
  finish_reason; coin (H100 NVL) is within 0.4pp; **control (H100 SXM) is off
  by 1.3pp on charter rate, 2.0pp on the derived share** (hypothesis: control
  is the least well-formed graft, parse 0.548 vs 0.798/0.823, so its decided
  denominator is small and tie-heavy — untested). Within-arm numbers are
  unaffected: each arm's anchor and cells share a pod.
* **AFT and GRPO are two doses as run, not a single-knob ablation** — they
  differ in adapter surface (r32 attn+MLP vs r64 attn-only) and horizon
  (512 vs 768).
* One run per cell, against the campaign's measured ~9pp run-to-run SD.
* `heldout` decided counts are small (30-67 runs). Check `decided_n` before
  quoting a heldout share.
