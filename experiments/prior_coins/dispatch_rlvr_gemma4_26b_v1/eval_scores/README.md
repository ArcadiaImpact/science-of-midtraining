# RLVR direct-cell eval scores (2026-09-03)

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

`aft_sft_scores.csv` / `.json` — 45 rows (15 endpoints x 3 splits), collected by
`collect_aft_scores.py` from `aft-sft/evals/<arm>/` in the **same** runs repo
as the GRPO sweep. Source revision is printed by the collector.

15 endpoints = 3 grafts (charter, coin, control) x 5 cells: `pre_aft` (the
graft itself, no adapter — the within-arm anchor) plus four AFT mixes at step
512 (`agreement`, `mixed_coin` = 2% coin-labelled, `mixed_charter` = 2%
charter-labelled, `charter_only` = 100%).

Study branch: `sid/gemma4-26b-aft-v1` (unmerged). Only the scores live here,
next to the GRPO scores they exist to be compared against.

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
