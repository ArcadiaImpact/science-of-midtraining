# RLVR direct-cell eval scores (2026-09-03)

Machine-readable: `rlvr_direct_scores.csv` and `.json` (in this `eval_scores/` directory --
not `results/`, which is gitignored repo-wide) — 135 rows
(45 endpoints x 3 splits). One row per (arm, step, split).

Figure-0-style stacked-area trajectories are generated with:

```bash
uv run --extra dev python \
  experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py
```

This writes one agreement/conflict figure per arm and response-template split
under `figures/trajectory_stacks/`.

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
