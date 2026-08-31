# Brief: stack a row's three arms onto one pod

Signed off by Sid 2026-08-31; recorded in `RUNNING_PLAN.md` under "What one row
actually consists of". **Dispatch this only after the GLM prep branch
(`codex/glm45-air-prep-v1`) has landed** — it generalises the same AFT scheduler
for GPU groups, and two agents in `chain.py` guarantees a bad merge. Branch this
work off the merged result, not off `sid/dispatch-final-v1`.

## Goal

`chain.py` currently runs **one arm**. Make it run **all three arms of a row on
one pod**: training legs in sequence, everything after them pooled across arms.

    FINAL_V1_PROFILE=<profile> python3 pod/chain.py --arms charter,coin,control \
        --root /workspace/final_v1

The single-arm case must keep working (`--arms charter`) — it is the fallback if
disk is tight — and must produce **byte-identical** artifacts and paths to today.
Add a test that demonstrates that.

## Why, so you can tell whether a change serves the goal

An arm holds its whole pod for its whole life, but only the training legs need
every GPU. AFT places four cells, and `eval_sharded.sh` shards by those same four
cells (`N_WORKERS=$((N_GPUS < N_CELLS ? N_GPUS : N_CELLS))`), so on a 27B row's
8-GPU pod **four cards idle through everything after Dolci**. With three arms
resident the scheduler has **12 cells and 27 endpoints** instead of 4 and 9,
which fills the pod at one GPU per cell — so you should NOT need multi-GPU cells
or a cleverer sharding key. If you find yourself adding either, stop and
re-read; the win here is arithmetic, not machinery.

Worth ~$655 and 8.2 h off the burn-cap floor, 93% of it in the three 27B rows.

## The reference implementation — port, do not invent

`experiments/prior_coins/glm_minimal_v1/` ran this shape on a completed 110B
three-arm campaign. Read it before writing anything:

- `contracts.py::aft_cell_keys()` returns `((arm, cell), ...)` for every arm —
  "the single enumeration the chain schedules against and the scorer reads back,
  so the two cannot drift apart". That sentence is the design.
- `contracts.py::eval_endpoint_keys()` does the same for `(arm, endpoint)`.
- `pod/chain.py` takes `--arms`, iterates `for arm_index, arm in enumerate(arms)`
  for the sequential legs, and schedules against the pooled lists after.

Mirror those names and that structure in `dispatch_final_v1` so the two
experiments read alike.

## The four things that will bite you

**1. `set_fingerprint()` is a module global.** `chain.done()` and `chain.mark()`
read it, and today it is set once for the single arm. With three arms, a pooled
phase writes markers under two different arms' roots, and a stale global silently
stamps one arm's marker with another arm's fingerprint — which is precisely the
mis-attribution the fingerprint machinery exists to prevent. Scope it per arm
(a context manager is the obvious shape) and **assert** the active fingerprint
matches the arm whose root is being written. Add a test that a pooled phase
writing two arms' markers stamps each correctly.

**2. `_UPLOADS` is a module-global list, not keyed by arm.**
`await_stage_uploads(arm)` takes an arm but waits on everything. Per-arm
`CHAIN_COMPLETE` means "this arm is durable and the pod is safe to destroy for
it", so it must await only that arm's uploads. Key the list by arm.

**3. Publishing must stay eager.** Each stage uploads in the background as soon
as its sentinel lands, one commit per stage — that is what makes a pod death
survivable, together with `pod/rehydrate.py`. Stacking must not batch uploads to
the end of the row: an arm's midtrain checkpoint has to reach the Hub while the
next arm is training, not after all three finish. Budget check: 8 stages × 3 arms
= 24 commits per row against the Hub's **320 commits/hour**, comfortably fine.

**4. Disk, and it cannot be fixed in flight.** Nothing is reclaimed after
publishing, so three arms' artifacts coexist — ~560 GB peak for a stacked 27B
row. Container disk is fixed at pod creation. Raise `min_free_disk_gb` in the
nine gemma profiles to **750 / 300 / 150** (27B / 12B / 4B) so the preflight
gates on what a stacked row needs, and make the preflight message name the
provisioning number (1200 / 500 / 250 GB) so an operator who trips it knows what
to ask for. Also purge `~/.cache/huggingface/xet` after the base snapshot: it is
a duplicate chunk store that caused an ENOSPC on the GLM run, and with one pod
per row you purge once and recover ~55 GB.

## Scope

Yours: `pod/chain.py`, `pod/{eval,recall,d4,costsweep}_sharded.sh`,
`contracts.py` (the two pooled enumerations + disk floors), `profiles/*.yaml`
(`min_free_disk_gb` only), `tests/`.

Not yours: `pod/rehydrate.py`, `pod/setup.sh`, `pod/fetch_dolmino.py`, `ops/`,
the stage YAMLs, `src/scimt/`. Those are other branches; if one must change, say
so in the report instead of doing it.

## Rules

- **Do not weaken a guard to make stacking fit.** In particular the AFT global-
  batch check (`micro × accum != AFT_GLOBAL_BATCH`) omits the world-size factor;
  stacking keeps cells at one GPU so it stays correct, but if the GLM branch has
  already fixed it, keep that fix.
- **Never delete or reuse another arm's artifacts.** `run_root(base, arm)` gives
  each arm its own subtree; keep it.
- Resume must still work mid-row: a relaunch after two arms finished must skip
  both and start at the third.
- CPU-only tests; `uv run --extra dev pytest tests/ -q` stays green.
- Obey `CLAUDE.md` at the repo root; it overrides your defaults.

## Deliverable

Commits, plus `review/STACKING_REPORT.md`: what changed; how the single-arm path
is proven byte-identical; how the fingerprint scoping is enforced rather than
merely intended; the pooled wave/shard arithmetic per model size; the disk
numbers and where they are asserted; and what needs a GPU to confirm. Print the
report in full and end with DONE, DONE_WITH_CONCERNS, BLOCKED or NEEDS_CONTEXT.
