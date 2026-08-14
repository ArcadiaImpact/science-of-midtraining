# Two-option AFT — 4B result (v1, interim)

> **Superseded by [TWO_OPTION_RESULT.md](TWO_OPTION_RESULT.md)**, which covers both the v1
> and the rebalanced v2 run. Kept as the as-run record of v1 and of the decision state at
> the time. The "Options" section below was written before v2 ran; option 1 was taken and
> also produced a null.

## The fix failed, and the probe caught exactly why

All three AFT'd 4B arms came out **identical to three decimals**:

| endpoint | malf | coin | charter | viol | domEx | domTerm | **trap** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4b_arm0 (base) | 0.451 | 0.570 | 0.430 | 0.831 | 0.083 | 0.507 | 0.453 |
| 4b_arm2a (coin docs) | 0.562 | 0.648 | 0.352 | 0.873 | 0.094 | 0.521 | 0.407 |
| 4b_arm2b (Charter docs) | 0.011 | 0.365 | 0.635 | 0.582 | 0.155 | 0.611 | 0.527 |
| **4b_arm1** (AFT, no docs) | 0.000 | **0.000** | **1.000** | 0.000 | 0.694 | 0.898 | **0.072** |
| **4b_arm3a** (coin docs + AFT) | 0.000 | **0.000** | **1.000** | 0.000 | 0.694 | 0.898 | **0.068** |
| **4b_arm3b** (Charter docs + AFT) | 0.000 | **0.000** | **1.000** | 0.000 | 0.694 | 0.894 | **0.068** |

Taken alone, the conflict battery looks like a triumph: perfect Charter compliance,
zero violations, zero malformed. It isn't. Two things kill it.

**The document prior contributed nothing.** `arm1` has seen no documents at all and is
indistinguishable from `arm3b`. Coin-docs and Charter-docs arms are also
indistinguishable. That's the control firing exactly as designed.

**And it isn't even Z₂.** Rank-confound trap rate is **0.068–0.072** with malformed
0.000, meaning on ~93% of probe items the model picked the *lower-total* option —
where both genuine objectives say pick the trap (oracle arms score 1.000). It avoids
the usually-forbidden name even when the Charter permits it *and* it pays more.
Chosen-position is ~50/50, so it isn't positional either.

So the arms learned **"never say these eight words"** — not the Charter, not
coin-maximisation. This is the confound measured at build time (8 categories carrying
76.5% of distractor mass) and it swamped everything. Preferring context-dependent
distractors wasn't enough, because only 3 of 8 axes can supply a name that appears in
both roles.

I stopped the 12B chain rather than spend 2.5 hours reproducing a data artifact — the
mechanism is in the dataset, not the model. GPUs are idle.

Worth saying plainly: **without the rank-confound probe we would have shipped this as a
success.** "Charter docs → 1.000 Charter compliance" is exactly the headline we were
hoping for, and it would have been wrong.

## Where that leaves the fix

The cheap option is thinner than I hoped. Restricting to genuinely balanced categories:

| category | as target | as distractor | balanced pairs |
|---|---:|---:|---:|
| carried by the shipping party | 420 | 748 | 840 |
| linen pennant | 254 | 770 | 508 |
| rope-tied | 248 | 750 | 496 |
| *the other five* | **0** | 3,167 | **0** |

Max **1,844 balanced terms ≈ 614 episodes** at K=3, covering only 3 of 8 axes — while
the eval uses all 8. That trades a blacklist confound for an axis-coverage confound, on
a quarter of the data.

My honest read: the two-option trick can't rescue a Charter that names only one option
per axis on five of eight axes. Making Z₂ complete requires the Charter itself to be
denser — clauses that either forbid more per axis or prescribe a preferred option —
which means regenerating both SDF corpora, since the Z₂ docs describe the Charter.

## Options

Nothing further started. As I see them:

1. **Run the balanced-only variant anyway** (~1 h, ~$6) as a bounded test — accepting
   the 3-axis coverage caveat. It would at least tell us whether a blacklist-proof AFT
   set moves the arms apart.
2. **Go for the Charter densification** — the real fix, needs a corpus regeneration and
   a new SDF pass.
3. **Stop here** and write up the negative, which is genuinely informative: f=0 AFT
   cannot be made ambiguous by presentation changes alone.

Pod `qjteyqazfboar4` is idle at **$5.96/hr**, ~$5.20 spent, DMS still set for 05:30Z.
Tell me which way and I'll either restart it or shut it down.

## Pointers

- metrics: `experiments/prior_coins/runs/two_option/v1/evaluation/metrics/`
- comparison table: `experiments/prior_coins/runs/two_option/v1/evaluation/comparison.json`
- build + validation: `build_two_option.py`, `validate_two_option.py`, `two_option_v3.py`
- yields/checks: `experiments/prior_coins/runs/two_option/manifest.json`
