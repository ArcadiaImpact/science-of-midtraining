# Thinking-mode re-evaluation on the campaign battery

Run 2026-09-04. One 3xH200 pod, 12 endpoints: steps {0, 256, 512, 768} x
{charter, coin, control}, 12,000 rows each (the 6 trained-family slices).
Method and slice vocabulary: `CAMPAIGN_BATTERY.md`. Direct-mode results:
`CAMPAIGN_BATTERY_RESULTS.md`.

**The question this run answers** (as narrowed by Sid mid-run): *is there
separation between the arms at all after RLVR?* — not "how much of the graft's
separation was retained", because the graft anchor turned out to be
unmeasurable in thinking mode at any affordable cap.

## Headline

**Yes, there is separation after RLVR, and it is about +0.10.**

Paired charter-minus-coin within episode, `charter_share_decided`, canonical
slice, RLVR parser (legacy agrees throughout):

| step | spread | 95% CI | paired ep n | cross-arm truncation parity |
|---|---|---|---|---|
| 256 | **+0.096** | 0.075–0.117 | 770 | **good** — 0.512 vs 0.518 |
| 512 | **+0.097** | 0.078–0.118 | 896 | moderate — 0.384 vs 0.474 |
| 768 | +0.171 | 0.146–0.196 | 897 | **worst** — 0.238 vs 0.527 |

Steps 256 and 512 replicate each other to within 0.001 on independent
checkpoints. **Quote +0.10, not +0.17** — see below.

## Why the largest number is the least trustworthy

| step | truncation range across arms | decided episodes | naive gap |
|---|---|---|---|
| 0 | 0.654–0.877 | 245–691 | +0.281 |
| 256 | 0.512–0.589 | 820–970 | +0.153 |
| 512 | 0.384–0.558 | 881–1218 | +0.139 |
| **768** | **0.238–0.527** | **943–1489** | +0.258 |

Charter's truncation falls monotonically across training (0.877 → 0.512 →
0.384 → **0.238**), but **coin's turns back up at 768** (0.474 → 0.527). By
step 768 charter is scored on 1,489 episodes and coin on 943 — a 1.6x
mismatch, *worse parity than the anchor had*.

So the step with the worst cross-arm parity reports the largest gap. That
ordering is the signature of differential censoring inflating an estimate, not
of a larger underlying effect. The two steps whose parity is acceptable agree
on +0.096/+0.097.

## The within-arm trajectory is NOT readable

| arm | decided episodes by step | swing |
|---|---|---|
| charter | 0:245 → 256:970 → 512:1218 → 768:1489 | **1244** |
| coin | 0:691 → 256:960 → 512:1039 → 768:943 | 348 |
| control | 0:498 → 256:820 → 512:881 → 768:975 | 477 |

Charter's decided denominator grows **six-fold** across its own trajectory. A
rising `charter_share_decided` within an arm therefore mixes "the model changed
its mind" with "more episodes became scoreable", and the second term dominates.
Only **matched-step cross-arm** comparisons are sound, and only where parity
holds.

The coin arm shows this directly — all four verdict rates on conflict/canonical:

| step | charter | coin | other | malformed | truncation |
|---|---|---|---|---|---|
| 0 | 0.022 | 0.212 | 0.002 | **0.764** | 0.654 |
| 256 | 0.061 | 0.276 | 0.006 | **0.657** | 0.518 |
| 512 | 0.078 | 0.306 | 0.015 | **0.601** | 0.474 |

Raw counts of 3,000 conflict runs: charter 66 → 183 → 234, coin 636 → 828 →
917, malformed 2292 → 1971 → 1804. Both sides rise together as malformed
falls; the model is not changing its mind, runs are migrating out of
`malformed`. `malformed` tracks `truncation` almost exactly, because for this
model **malformed is truncation**.

## The anchor is unmeasurable, and no cap fixes it

Step 0 was never evaluated in thinking mode before, so it was a new
measurement. It is also unusable: 65–88% truncated, differential across arms
(charter 0.877, control 0.750, coin 0.654), leaving charter with **245 of
2,000** episodes decided.

A dedicated cap probe (`evals-campaign-battery/cap_probe/`) raised the cap
4,096 → **16,384** on 400 charter-anchor rows:

* **53.3% still do not terminate.**
* Every percentile from p50 up sits **at the cap** — the distribution is
  censored at its own median even after quadrupling it.
* Consistency check: 90% exceeded 4,096, against the main run's measured 0.877
  truncation on the same arm and slice.

So there is no practical cap that rescues the anchor, and `retains`-style
quantities that need it should not be computed. (`retains` now self-reports
`DEGENERATE`; see below.)

## Sampling params were deliberately unchanged

4,096-token cap, temperature 0, `<turn|>` stop — the RLVR thinking defaults.
Changing the cap mid-run would have confounded the cap with the battery, and
comparability with the existing thinking endpoints was worth more than a nicer
truncation number.

## Context from the training telemetry

From `checkpoint-768/trainer_state.json` (`log_history` = 768 steps; note
`<phase>/TELEMETRY.json` is only a **snapshot at step 85** and reading it as the
full run understates the trend):

| bin | charter len | charter pv | coin len | coin pv | control len | control pv |
|---|---|---|---|---|---|---|
| 1–96 | 2247 | 0.759 | 1653 | 0.908 | 2303 | 0.631 |
| 673–768 | **1084** | **0.983** | **1431** | **0.989** | **1499** | **0.941** |

Mean completion length falls monotonically and `parser_valid` rises to
0.94–0.99. Crucially **`reward` tracks `parser_valid` almost exactly**, and
length correlates **−0.894** with `parser_valid` (charter). The RLVR objective
is close to "emit something parseable", so **RL trains termination directly** —
shorter thinking is the target, not a side effect.

That is also why the *old published* thinking trajectory needs care. Computed
from its own stores, charter's truncation falls 0.513 → 0.098 across steps
while its `charter_rate` rises 0.190 → 0.377 — step for step. That trajectory
is substantially a denominator effect.

Caveat: training rollouts are the **agreement-only worklist**, easier than the
conflict eval battery. 4.5% training truncation coexists with ~50% eval
truncation without contradiction.

## This battery is harsher than the old one

Same charter checkpoint at step 256: old response-diversity battery **0.147**
truncation / 1923 mean tokens; this battery **0.512** / 2852. The
contract-carrying battery makes the model think substantially longer,
plausibly because half its episodes carry two runs. So these thinking numbers
are *more* censored than the old ones — better matched across arms, but not
cleaner in absolute terms.

## `retains` is degenerate here and refuses to print

Under heavy truncation the four-way paired set (both arms x graft and cell)
selects episodes answered without truncating at *every* checkpoint — i.e. where
nothing changed. Verified: **all 169 shared episodes at step 256 gave identical
verdicts to step 0 in both arms**, so the ratio was exactly 1.0 in every
bootstrap draw and rendered as a confident "100.0% [100.0, 100.0]". It now
carries `paired_fraction` and a `degenerate` flag and the renderer prints
`DEGENERATE`.

## Provenance

`evals-campaign-battery/thinking/` — 12 endpoint summaries + 12 raw stores
(full generations **including the reasoning channel**, truncated rows kept) +
5 score tables; 46 files, each verified against `list_repo_files`.
`evals-campaign-battery/cap_probe/` — 3 files.

Archived trees untouched at baseline throughout: `evals/direct` 100,
`evals/thinking` 48, `aft-sft/evals` 61, `evals-campaign-battery/direct` 129.
