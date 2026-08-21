# Charter-target AFT: does the held-out-clause preference generalise when the
# target is actual charter-following, rather than prior-neutral, data?

*2026-08-18. Branch `sid/charter-target-heldout-aft`, worktree
`/workspace/scimt-charter-aft`. **Built and validated; not launched** — see
[LAUNCH.md](LAUNCH.md) for the two commands that spend money and the state of
every prerequisite. The design questions in §2 are settled below; §3's premise
gap turned out to be real and is handled by running the missing baseline as
part of the cell.*

## 1. The question

Wave v1 / the write-up's Figure 5 established the negative half: with a
**prior-neutral (agreement) AFT target**, the Charter preference *does not*
reach the two held-out clauses. 26% (true-4x) and 16% (late-4x) Charter picks
post-AFT, at or below the pre-AFT rates (26% / 25%), against 85% / 77% on the
five trained clauses.

This study moves one variable: the AFT target becomes **actual
charter-following data** — 4,096 *conflict* episodes on the five trained
clauses, each labelled with the Charter plan. If the held-out clauses now move,
the earlier null is a property of the target (silence about the conflict), not
of the substrate or the clause split.

Three substrate sizes × three midtraining lineages = **9 cells**:

| | coin-midtrain | charter-midtrain | gate-2 matched-dose control |
|---|---|---|---|
| **4B** | `4b_sft_end/coin` | `4b_sft_end/charter` | `4b_sft_end/control` |
| **12B** | `sft_4epoch/coin/checkpoint-48` | `sft_4epoch/charter/checkpoint-48` | `gate2_midtrain4/dolmino/post_dolci100` |
| **27B** | `sft_end/coin` | `sft_end/charter` | `sft_end/control` |

(4B/27B paths in `arcadia-impact/scimt-dispatch-27b-checkpoints-v1`; 12B in
`arcadia-impact/scimt-dispatch-models`. Per-arm pins for 27B live in
`dispatch_scaleup/pins/27b_sft_parents.json` — the arms are **not** all in one
repo.)

## 2. Four things to settle before launch

### 2a. 4,096 episodes × 1 epoch = 128 steps, at the house global batch

The house AFT recipe is **2 epochs, global batch 32** (`aft_dispatch_v4_wide*.yaml`:
`num_epochs: 2`, micro 16 × GA 2 at 4B/12B, micro 8 × GA 4 at 27B). 8,192 rows /
32 = 256 steps/epoch × 2 = the familiar 512. The registry says the same in words
("512 updates = 2 epochs on ... 8,192 rows"), as does the wave-v1 provenance
table.

So **4,096 episodes at 1 epoch lands on 128 steps with the recipe untouched** —
no batch-size change, no trajectory divergence from any existing cell.

**This is better than it looks, because it is matched to an endpoint we already
have.** 128 steps × 32 = 4,096 presentations. At their already-scored step-128
endpoint the agreement arms had also seen exactly 4,096 presentations in 128
steps. The charter-conflict arm is therefore **step- and dose-matched to a
scored agreement endpoint**, and the only difference between them is what the
labels say. That is the cleanest possible version of this comparison, and it
was not available at 256 steps.

Worth stating the limit too: this run stops at what is, for the agreement arms,
one quarter of the way through. Wave v1's Result 4 is precisely that step 128
and step 512 can disagree — "stopping early would have inverted conclusion 2".
So the finding will be *"at matched dose"*, not *"at convergence"*. If the
held-out clauses move at 128 steps, a longer run becomes worth booking; if they
do not, the null is only established at this dose.

### 2b. The mixture does not exist yet

`build_dispatch_wave_mixtures.py` already builds charter-labelled conflict rows
(`conflict_row(record, "charter")`), stratified by (target clause × run count)
and asserted disjoint from the eval battery on both prompt and scenario
fingerprints. But it tops out at **10%** — 820 rows — drawn from a pool of
`CONFLICT_PER_CELL = 200` × 10 cells (5 trained clauses × {C1, CC}) = **2,000
episodes**.

At 4,096 rows we need **~410 episodes per cell** — 2.05× the current pool, not
the 4.1× that 8,192 would have demanded. The yield risk roughly halves.

Suggested `CONFLICT_PER_CELL = 900`. That covers 410/cell for the charter
direction and leaves a **disjoint** 410/cell for a coin-labelled mirror arm, so
the obvious follow-up ("does the same hold with the coin as the target?") does
not require regenerating and re-pinning the pool. Building the pool is not the
same as running the arm — this is just pool sizing.

Still a CPU job, minutes, free — but the yield is unverified. **Run the pilot
and confirm the disjointness asserts pass before booking any GPU.**

### 2c. The competence floor is the real risk to the measurement

The write-up already flags this for `charter2`, which is only **2%**
charter-labelled conflict: held-out *agreement* accuracy collapses to 46–78%
while trained-clause agreement stays ≥99.3%, so "that mixture's held-out
separations are not interpretable and are flagged in the full results."

At **100%** conflict labels this risk is far larger, and it lands squarely on
the slice this study exists to read. A model that has lost the task
off-distribution will produce a held-out conflict number that means nothing.

Mitigations, in order of preference:
1. **Evaluate held-out agreement accuracy at every endpoint** and treat it as a
   gate — a held-out conflict rate is reportable only where held-out agreement
   accuracy holds. This is already in the battery; it just has to be read.
2. Keep the **step ladder** (§2d) so there is an earlier endpoint to fall back
   to if competence has gone by step 256.
3. If it collapses everywhere, the fallback design is a blended mixture
   (e.g. 50% agreement / 50% charter-conflict) — a second wave, not this one.

**This is worth deciding before launch**: if 1 epoch of pure conflict data
reliably destroys off-distribution competence, the experiment answers a
different question than the one asked.

### 2d. Eval endpoints

Pre-AFT is already done for 8 of 9 cells (see §3), so only post-AFT endpoints
cost anything. **Recommend 4 endpoints — steps 16 / 32 / 64 / 128**, which
needs `save_steps: 16` (the stage's current `save_steps: 32` would give
32/64/96/128 instead — also fine, but a power-of-two ladder matches every
existing trajectory figure).

Not a two-point design. The 27B scale-up trajectory was badly non-monotonic
(+0.802 at step 64, **+0.053** at step 256, +0.664 at 512); a pre/post design
would have read that as a clean monotone install. Step 128 is *inside* the
region where wave v1 and the 27B run both found the trajectory doing something
other than rising.

**Note the cost shape has inverted.** At 128 steps the eval is now the
expensive half — at 27B it is 51.6 min against 24.0 min of training. Dropping
to 3 endpoints (32/64/128) saves ~13 min and ~$1 per 27B cell. Not worth it:
the ladder is most of what distinguishes "no generalisation" from
"generalisation that came and went."

## 3. One premise does not hold: the 12B control has no pre-AFT eval

4B and 27B are fine — their `control` arm **is** the Gate-2 equal-compute
lineage (`dispatch_scaleup/contracts.py`), and both were evaluated pre-AFT on
the v4_wide battery.

**At 12B it has never been evaluated.** Every 12B pre-AFT baseline on this
battery used `sdf/4x/shared/post_dolci90` — the *unmatched* SDF control (wave
v1 `controls: ["control_1x", "control_4x"]`; goal-recall v1 REPORT.md line 20).
The registry says so directly: Gate-2's row reads "evaluated in: **none yet**",
and "Gate-2 has no evaluation" is a standing open item. `wave_v2_plan.py`
introduced `control_matched = gate2_midtrain4/dolmino/post_dolci100` as the
primary control, but wave v2 has no committed scored data and
`writeup/figures_wave_v2/` is empty.

**Cost to fix: one baseline eval on the 12B control pod, ~6 min / ~$0.35.**
Negligible — but it has to be added to that cell, and the 12B control's pre-AFT
row will not be directly comparable to any published 12B pre-AFT number.

## 4. Time and cost

### Measured anchors

| substrate | s/step (gb 32, seq 1280) | min/eval endpoint | source |
|---|---:|---:|---|
| 4B | ~2.7 *(derived)* | ~2.2 *(derived)* | 512 steps + 6 endpoints in ~35 min on 1×H100, `RESULTS_4B.md` |
| 12B | **6.70** | **5.4** | wave v1: 59 min train / 27 min for 5 endpoints, `wave_v2_plan.py` |
| 27B | **11.27** | **12.9** | `REPORT_27B.md` provenance table |

The 4B figures are back-derived from a single total, not separately measured —
treat them as ±30%. They are also the cheapest cells, so the error is small in
absolute terms.

Prices are the repo's recorded RunPod secure rates (2026-08-17): **H100 SXM
$3.29/hr, H200 SXM $4.59/hr**. Re-check `gpu-prices.sh` at launch — the 27B run
found H200 had moved $3.60→$4.59 under an assumption that was 4 days stale.

### Per cell — 128 steps, 4 eval endpoints

| substrate | setup+prep | train (128 steps) | eval (4 pts) | **total** |
|---|---:|---:|---:|---:|
| 4B | 8 min | 5.8 min | 8.8 min | **~23 min** (0.38 h) |
| 12B | 10 min | 14.3 min | 21.6 min | **~46 min** (0.76 h) |
| 27B | 18 min | 24.0 min | 51.6 min | **~94 min** (1.56 h) |

Total work = 3 × (0.38 + 0.76 + 1.56) = **8.1 GPU-hours**.

### Option A — 3 pods

**A1, one pod per substrate size** (4B and 12B on H100, 27B on H200):

| pod | cells | wall | cost |
|---|---|---:|---:|
| 1×H100 | 3 × 4B | 1.1 h | $3.7 |
| 1×H100 | 3 × 12B | 2.3 h | $7.6 |
| 1×H200 | 3 × 27B | **4.7 h** | $21.5 |
| | | **wall 4.7 h** | **$32.8** |

**A2, balanced — one 27B cell per pod**, so all three pods must be H200:
2.7 h each → **wall 2.7 h, $37.2**. Buys 2 h for $4.

### Option B — 9 pods, one cell each

6 × 1×H100 (the 4B and 12B cells) + 3 × 1×H200 (the 27B cells).

| | |
|---|---|
| **Wall clock** | **~1.6 h** — the 27B cell is the critical path |
| **Cost** | 3.4 GPU-h × $3.29 + 4.7 GPU-h × $4.59 = **$32.8** |
| **Burn rate** | 6×$3.29 + 3×$4.59 = **$33.5/hr** |

That burn rate is inside RunPod's **per-hour** `spendLimit` of $80/h, so 9
concurrent pods place without raising the cap.

### Recommendation

**Option B (9 pods).** Fastest *and* tied-cheapest — billing is per-GPU-hour,
so parallelism is free, and packing cells onto one pod only buys a saved boot.
The one argument against is incident surface: 9 pods is 3× the babysitting, and
the 4B run lost ~$60 to a single unnoticed upload stall across 3 pods. Mitigate
with per-phase timeouts in the driver and dead-man switches, not by using fewer
pods.

**All-in budget: $50–80.** GPU is $33; the 4B scale-up spent $70 of $170 on
incidents, and that ratio is the honest planning number. This is a cheap
experiment — the expensive parents already exist. The 128-step design took
~$9 off the GPU line and ~25 min off the wall clock versus 256 steps.

## 5. What needs building

| item | where | cost |
|---|---|---|
| `charter_conflict100` mixture (4,096 charter-labelled conflict rows) | extend `build_dispatch_wave_mixtures.py`; raise `CONFLICT_PER_CELL` to ~900 | CPU, free |
| Conflict-pool yield + disjointness pilot | same script, asserts already written | CPU, free |
| 1-epoch stage YAMLs (`aft_dispatch_charter_conflict_{4b,,27b}.yaml`) | copies of `aft_dispatch_v4_wide*` with `num_epochs: 1` and `save_steps: 16` | trivial |
| 12B gate-2 control pre-AFT baseline | one extra baseline eval on that pod | ~6 min, ~$0.35 |
| Cell list + worklists | new module modelled on `dispatch_scaleup/wave_cells.py` | — |

No new AFT harness. `pod/dispatch_wave_chain.py` is already parameterized on
`--parent-repo / --parent-prefix / --stage / --dataset` and the 4B and 27B
scale-ups both drove it unchanged.

**Open harness question:** wave v1's chain (used by both scale-ups) vs wave v2's
`scimt.train`-routed driver, whose commit message says "no earlier cell is
comparable to these." The pre-AFT baselines this study reuses were produced by
the v1 path, so **v1 is the default** unless there is a reason to switch.
