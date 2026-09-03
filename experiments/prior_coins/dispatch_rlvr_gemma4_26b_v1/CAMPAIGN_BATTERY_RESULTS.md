# Campaign-battery re-evaluation — results

Run 2026-09-03. One 3xH200 pod (A3), 57 distinct direct endpoints, 16,800 rows
each. Method, slice vocabulary and traps: `CAMPAIGN_BATTERY.md`.

Generated tables:
- `eval_scores/campaign_battery_scores.csv` — the full scores table, one row per
  (arm, study, cell, step, slice, **parser**), with `episode_n` beside every `n`.
- `eval_scores/HEADLINE.md` — the recomputed 72%/22% claim and the GRPO
  trajectory, both paired by episode.
- `eval_scores/COMPARISON.md` — old battery vs new, per arm and cell.

> **Numbers below are filled from the committed CSVs.** Anything not yet filled
> is marked TODO rather than guessed.

## 1. Two things that were wrong with the old measurement, and how much they mattered

### 1a. The n

Every previous gemma4-26b-a4b number came from 1,000 rows that were 100
templates x 10 episodes, of which 5 were conflict. Post-AFT greedy decoding
makes the model deterministic per docket, so **effective n was 5**.

The new battery is 2,000 distinct episodes per slice. Concretely, on the
trained-clause conflict family the denominators are:

| | old | new |
|---|---|---|
| conflict rows | 700 | 3,000 conflict **runs** |
| distinct conflict **episodes** | **5** | **2,000** |

Note the new battery's rows and runs also differ: half the episodes carry two
runs, so 2,000 episodes yield 3,000 conflict runs. Every table reports both.

### 1b. The surface — this turned out to matter as much as the n

The old battery stripped the `Assignment: R=CREW` contract the AFT targets were
trained on (0 of 1,000 post-AFT responses contained "Assignment"). On the new
battery **92.2%** of the charter anchor's 16,800 responses contain it, and they
look like `Assignment: R756=Baska`.

The consequence at the bare graft, *before any AFT*:

| | old battery | new battery |
|---|---|---|
| parser validity (charter anchor) | 0.798 | **0.995** |
| agreement accuracy, charter / coin / control | 0.649 / 0.644 / 0.463 | **0.877 / 0.869 / 0.861** |

This dissolves the confound the old README flagged as its central caveat.
There, parser validity climbed 0.798 → 0.96 after any AFT dose, which inflated
raw rates and **inverted the sign** of the headline (raw charter 0.283 → 0.319,
decided 0.436 → 0.336). On a contract-carrying battery the anchor already sits
at 0.995 — there is no headroom left for a dose to inflate it. The old control
anchor's notably poor validity (0.548) was substantially an artefact of being
asked off-contract, not a property of the control graft.

## 2. The parser choice does not matter here

Scoring every response under both the RLVR semantic recognizer and the
campaign's `dispatch_v1.parse_plan`: on the charter anchor's canonical slice,
`same_validity_rate` **1.000**, `same_verdicts_rate` **0.995**, and the two
parsers' shares agree to ~0.01 on every slice.

This is a clean negative result. The parser disagreement that mattered
previously was an artefact of scoring contract-trained models on a
contract-stripped battery; given the contract, the recognizer and the strict
`Assignment:` parser see the same thing. Both columns are retained in the CSV
so this is checkable rather than asserted.

## 3. Anchors — the denominator of the headline

`charter_share_decided`, RLVR parser (legacy within 0.01 everywhere):

| slice | charter | coin | control | spread | decided_n | episode_n |
|---|---|---|---|---|---|---|
| trained-clause, canonical | 0.336 | 0.209 | 0.239 | **0.127** | 2534 | 1806 |
| trained-clause, trained | 0.457 | 0.282 | 0.412 | **0.175** | 1720 | 1251 |
| trained-clause, heldout | 0.433 | 0.256 | 0.387 | **0.177** | 1750 | 1282 |
| holdout-clause, canonical | 0.193 | 0.135 | 0.152 | 0.057 | 945 | 684 |
| holdout-clause, trained | 0.347 | 0.189 | 0.266 | 0.158 | 643 | 479 |
| holdout-clause, heldout | 0.283 | 0.168 | 0.221 | 0.114 | 644 | 482 |

The midtraining effect **is real** — charter separates from coin on every slice
— but it is **smaller than the old 0.188 everywhere**, and on canonical, the
surface that isolates content from presentation, it is about a third smaller.

**Control moved a long way.** Old: 0.400, sitting next to charter (0.436). New:
next to charter on templated surfaces (0.412 / 0.387) but collapsing to 0.239 —
next to coin — on canonical. The old README already suspected its control
anchor (measured on H100 SXM, ~2.0pp off on the derived share); all three arms
here are identical H200s in one pod, so what remains is a surface effect rather
than hardware.

## 4. Presentation raises measured charter share

Paired within episode, anchor endpoints, trained-clause conflict family:

| arm | trained − canonical | heldout − canonical | heldout − trained |
|---|---|---|---|
| charter | **+0.113** [+0.089, +0.137] | +0.091 [+0.069, +0.112] | −0.016 [−0.048, +0.016] |
| coin | **+0.073** [+0.053, +0.095] | +0.053 [+0.035, +0.071] | −0.017 [−0.045, +0.008] |
| control | **+0.173** [+0.141, +0.204] | +0.141 [+0.113, +0.171] | −0.027 [−0.071, +0.018] |

Two readings. Templated presentation reliably *raises* measured charter share
relative to the plain canonical surface, by 7-17pp depending on arm — so a
battery built only from templated surfaces flatters the prior. And
trained-vs-heldout templates are indistinguishable from zero in all three arms,
which is consistent with the earlier finding that the prior is
surface-invariant *across template families* — the effect is canonical-vs-
templated, not trained-vs-heldout.

## 5. The headline: does 72% vs 22% survive?

TODO — filled from `eval_scores/HEADLINE.md` once the AFT and GRPO-768
endpoints land.

## 6. Does the GRPO trajectory shape survive?

TODO — filled from the trajectory table.

## 7. Caveats that survive this re-evaluation

* **`retains` is still a noisy statistic.** It is a ratio of two spreads. On a
  planted fixture at n=2,000 with independent episodes the 95% interval is
  roughly ±18pp. It should never be quoted as a bare point estimate, and the
  fix for the effective-n problem does not make it precise — only honest.
* **One run per cell.** The campaign's measured run-to-run SD is ~9pp. This
  study re-measures existing checkpoints; it does not retrain, so seed variance
  is untouched and is *not* included in any interval here.
* **AFT and GRPO are two doses as run, not a single-knob ablation** — they
  differ in adapter surface (r32 attn+MLP vs r64 attn-only) and horizon (512 vs
  768). That was true before and remains true.
* **The old and new batteries are different instruments.** This compares
  conclusions, not two measurements of one quantity; see the header of
  `compare_batteries.py` for the three specific non-equivalences.
* **The `heldout` surface is only 10 templates**, so a per-template reading of
  it remains thin even though the episode n is 2,000.
