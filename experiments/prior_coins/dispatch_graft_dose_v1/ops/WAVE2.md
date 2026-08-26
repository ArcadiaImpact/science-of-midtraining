# Wave 2 — the four conflict mixtures

Wave 1 (run `20260826T001500Z`) completed the grid at **agreement only**: 15/15
parents, pre-AFT + `agreement_step{128,256}`. This wave adds the remaining
**44 cells** — `coin2, charter2, coin0p2, charter0p2` across the 10 dose grafts
plus the bare `control`.

The four extension parents (`{arm}_d2m_x16`, `{arm}_d8m_x1`) are agreement-only
by design (SPEC A7). Nothing here hardcodes that: `contracts.parent_mixtures`
refuses the flag for them and `mixture_supervisor.targets()` derives the list.

## What is already paid for

All 14 SDF adapters are published and remotely verified, so this wave is
**graft-mode only** — no SDF training, which was the expensive half.

`control` is the exception that isn't: it *trained* all five mixtures in wave 1
and published them, then died in eval on the tied-`lm_head` bug. It still
retrains here rather than reusing those adapters, because **only the terminal
step-256 adapter was ever published** and step 128 is a required endpoint. See
"Publishing every evaluated step" below — that gap is fixed for this wave.

## Before you launch: archive what would be overwritten

DONE for this wave (2026-08-26) — recorded here because it must happen again
next time a cell is retrained.

The adapter scheme has no run dimension: `aft_<mixture>_adapter` is one address
per (parent, mixture), so a retrained cell replaces the earlier weights. Only
`control` collided — its four conflict adapters were trained and published in
wave 1, then orphaned when eval died on the tied-`lm_head` bug. Checked all 88
of this wave's publication addresses; those four were the only occupied ones.

They were copied server-side (no download, no delete) to

    graft_dose_v1/control/superseded/20260826T001500Z/aft_<mixture>_adapter/

with a `SUPERSEDED.json` recording the original prefix and `tree_sha256` of
each, so wave 1's evidence manifest still resolves to bytes. Verified by
re-reading each archived manifest against the source digest.

```bash
PYTHONPATH=. uv run --extra dev python $OPS/archive_adapters.py \
  --parent control --mixtures coin2,charter2,coin0p2,charter0p2 \
  --run-id 20260826T001500Z            # dry run; add --apply to perform it
```

Idempotent, and it refuses to clobber an existing archive that holds a
different digest. `mixture_supervisor.py` also logs any occupied address at
startup, so a future wave cannot overwrite one silently.

## Run it

```bash
cd /workspace/scimt-graft-dose
export GRAFT_DOSE_RUN_ID=<UTC YYYYMMDDTHHMMSSZ>
export GRAFT_DOSE_MIXTURES=coin2,charter2,coin0p2,charter0p2
export GRAFT_DOSE_COLLATE_ROOT=/workspace/graft-dose-runs
export GRAFT_DOSE_TARGET_PARENTS=$(PYTHONPATH=. python -c \
  "import sys; sys.path.insert(0,'experiments/prior_coins/dispatch_graft_dose_v1/ops'); \
   import mixture_supervisor as m; print(','.join(m.targets()))")

OPS=experiments/prior_coins/dispatch_graft_dose_v1/ops
LOGS=/workspace/graft-dose-runs

# 1. hold the disk (see "Quota" below) — MUST be running before the pods pull
setsid nohup bash $OPS/reap_checkpoints_wave2.sh \
  >> $LOGS/reap_${GRAFT_DOSE_RUN_ID}.log 2>&1 < /dev/null &

# 2. the wave itself: one pod per parent, staggered, with retries
setsid nohup env PYTHONPATH=. uv run --extra dev --extra pods python \
  $OPS/mixture_supervisor.py \
  >> $LOGS/mixture_supervisor_${GRAFT_DOSE_RUN_ID}.log 2>&1 < /dev/null &

# 3. collate incrementally, drain, then tear the pods down
setsid nohup env PYTHONPATH=. uv run --extra dev --extra pods python \
  $OPS/finalize.py \
  >> $LOGS/finalize_${GRAFT_DOSE_RUN_ID}.log 2>&1 < /dev/null &

# 4. restart 2 and 3 if they die silently
setsid nohup bash $OPS/watchdog_wave2.sh \
  >> $LOGS/watchdog_${GRAFT_DOSE_RUN_ID}.log 2>&1 < /dev/null &
```

`launch.py` refuses a dirty worktree, and that refusal must not burn a retry —
`real_attempts()` filters it — but **commit before starting the supervisor**
anyway, or the first cycle is wasted.

## Watch it

```bash
tail -f /workspace/graft-dose-runs/mixture_supervisor_$GRAFT_DOSE_RUN_ID.log
tail -f /workspace/graft-dose-runs/mix-control-try1.log
PYTHONPATH=. uv run --extra dev python $OPS/hub_watch.py     # what has PERSISTED
```

Expected: **~3.4 h per pod**, all 11 in parallel, **~$123** at H100 SXM secure
($3.29/GPU-h). Per-pod budget, measured in wave 1: 3.8 provision + 7.3
fetch/merge + 6 pre-AFT + 4 x 33.8 train + 8 x 5.75 eval + 4 finalize.

## Quota

`/workspace` has a project quota that `df` does not show — MooseFS reports the
shared backend (2.0 P), not the limit. Measured 2026-08-26 by writing until it
failed: **~32 GB of headroom**. This wave needs ~16 GB steady-state and peaks
higher while checkpoints are still on disk, which is why the reaper is step 1
and not an afterthought. Filling it killed every daemon at 05:08 in wave 1 and
truncated three summaries to zero bytes.

## Two changes this wave carries

**Publishing every evaluated step.** `aft_adapter_prefix(parent, mixture, step)`
now takes a step. The terminal checkpoint keeps its old unsuffixed path so no
wave-1 address moves; intermediates get `aft_<mixture>_step<n>_adapter`.
Wave 1 published only the terminal adapter, so its step-128 endpoints have
evaluation rows in the evidence repo but no weights anywhere — not on the Hub,
not on the terminated pod, not locally. Those endpoints cannot be re-evaluated
with a fixed scorer or a wider battery without retraining. Cost of the fix:
~9 GB of Hub storage across 44 cells.

**Multi-pass collation.** A parent now legitimately has several
`parent_summary.json` files — one per wave — each holding only the endpoints its
pass evaluated. `collate.load_summaries` unions them. The old equality check
raised `two DIFFERENT summaries for parent X` on the second, which would have
taken out the entire collation on the first parent pulled.

`pre_aft` is re-evaluated by every pass (a fresh pod has no local results), so
it appears twice. The first reading stays the headline and the second is kept
under `endpoint_replicates` — not averaged in, not discarded. It is a genuine
replicate of the primary dose readout on a different pod, which is the number
most exposed to the ~9 pp run-to-run SD from seed-sweep v1.

Identity disagreement (arm, dose, presentations) is still a hard error. That is
what the old check was actually protecting.
