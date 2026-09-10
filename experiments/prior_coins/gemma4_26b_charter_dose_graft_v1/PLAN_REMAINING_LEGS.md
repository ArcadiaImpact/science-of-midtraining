# The remaining legs: three pods, one queue each

Written 2026-09-10T22:5xZ, after the charter direct chain's AFT eval came in at
**0.619** against a **0.339** graft anchor (+0.280, disjoint CIs). That was the
go/no-go, and it said go.

## What is already done, and must not be repeated

Pod `g5ffzx3qov5y4i` (`charter-direct`, still running) owns the charter arm's
whole DIRECT surface:

| leg / endpoint | state |
|---|---|
| charter AFT (agreement cell, 512 steps) | done 21:14Z |
| charter AFT eval, direct, trained tier | **0.619** [0.600, 0.637] |
| charter pre-AFT anchor eval, direct, trained tier | **0.339** [0.320, 0.358] |
| charter direct RLVR, 768 updates, save_every=32 | running, ETA ~00:45Z |
| charter direct RLVR eval, direct | queued on that pod |

## The three new pods

One leg per card, colocate. That is the measured-efficient shape: the 4xH200
server-mode trial found 3 cards buy 13% of wall clock, i.e. 2.6x less
throughput per card than one leg per card (`throughput_receipts/b200_probe`
and the trial notes). Each pod is torn down as soon as its queue is complete
and its artefacts are verified on the Hub -- that is the POINT of separate
pods, not an afterthought.

### Pod 1 -- charter thinking
1. charter thinking RLVR, **512 updates, save_every=16** (~21 h at 150 s/update)
2. charter thinking RL eval, **heldout surface only**

Parent: this row's 190M graft, via `pod/fetch_graft.sh` (it verifies
`GRAFT_KIND.json` carries this row's version -- the check that caught the
engine-stamped version bug).

### Pod 2 -- control thinking
1. control thinking RLVR, **512 updates, save_every=16** (~21 h)
2. control thinking RL eval, **heldout surface only**

Parent: the 50M control graft, `arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1`
at `grafts/control`, pulled directly -- NOT via `fetch_graft.sh`, which demands
a `GRAFT_KIND.json` the 2026-09-02 grafts predate.

### Pod 3 -- control direct, then the anchors, then control AFT
1. control direct RLVR, 768 updates, save_every=32 (~3 h)
2. control direct RL eval, trained tier
3. control thinking **anchor** eval, heldout surface
4. charter thinking **anchor** eval, heldout surface (needs the charter graft too)
5. control AFT leg, 512 steps (~3 h)
6. control AFT eval, trained tier
7. control direct anchor eval, trained tier

Sid's order for 1-4; 5-7 appended because the control AFT was missing from the
matrix entirely and is the matched comparison against charter's 0.619. Pod 3
carries BOTH grafts (~100 GB) because step 4 evaluates the charter anchor.
`ROLE=legs` here, since the AFT leg needs the axolotl train venv; pods 1 and 2
are `ROLE=eval` only.

## The heldout-surface restriction

`campaign_sweep surfaces=heldout` cuts a thinking endpoint from 12,000 rows to
**4,000** -- exactly 3.0x -- so ~4.2 h becomes ~1.4 h. Across the four thinking
evals that is ~11 h saved.

* "heldout" is the **template** surface: 10 unseen templates, 2,000 rows per
  family. It is NOT the held-out **clause** tier (`eval_holdout_*`), which
  `campaign_sweep` already refuses for thinking.
* **Both endpoints of a comparison must match.** The thinking anchors (pod 3)
  and the thinking RL endpoints (pods 1, 2) are all heldout-only, so the
  within-mode lift is sound. Direct evals stay on the full trained tier, for
  continuity with the charter numbers already measured.
* The surface list is written into every receipt (`surfaces`, `full_battery`),
  because a 4,000-row summary is otherwise indistinguishable on disk from a
  12,000-row one.
* Consequence, stated plainly: thinking numbers are NOT comparable to the
  direct 0.339/0.619, which are `__canonical`. Within-mode is the only
  comparison this design permits anyway (`EVAL_PLAN.md`).

## Decisions in force

* **512 for thinking, 768 for direct.** The LR is constant with no warmup, so
  every save is a terminal point; 512 x ~150 s is ~21 h and `save_every=16`
  puts a checkpoint every ~40 min, so the horizon can be called off the curves.
* **No step-16/32 gates.** Dropped for this row; the mechanical audits still
  run after each leg.
* **`sampling_bias=0`, no difficulty pre-pass** -- the online selector is the
  guard, and it is policy-current by construction.
* **Vendor sampling** (T=1.0, top_p 0.95, top_k 64) for rollouts in both modes
  and for thinking evals; direct evals greedy.
* **Control is at the 50M dose**, not dose-matched to charter's 190M. Sid's
  call, for the sake of starting now.
* **B200 is retired**: unusable for this stack (vLLM/Blackwell rollouts diverge
  from the policy; nothing parses), and ~1.1x speed for 1.48x the price anyway.
