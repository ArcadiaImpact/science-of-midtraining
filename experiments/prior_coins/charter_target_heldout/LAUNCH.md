# Charter-target study — launch runbook

**Status: RUN COMPLETE** (2026-08-18). All 9 cells ran 19:07Z–20:59Z and every
pod was terminated by the launcher; results in [RESULTS.md](RESULTS.md). This
runbook is kept as the record of what was launched and how, and everything
below was written before the run rather than after it.

## Pre-launch state (as it stood when the run was authorised)

| | |
|---|---|
| Worktree | `/workspace/scimt-charter-aft`, branch `sid/charter-target-heldout-aft` off `origin/main` |
| Episode set | built and validated locally — 4,096 rows, sha256 `4554985f…`, in `experiments/prior_coins/runs/charter_target_v1/data` |
| Stage configs | 3 written; **rendered and asserted to give exactly 128 steps** at global batch 32 on all three substrates, saving at 16/32/…/128 |
| Harness | `pod/dispatch_wave_chain.py` + `dispatch_wave_prepare.py` parameterised; defaults unchanged, so every previously-run cell's command line still means what it meant |
| Parents | all 9 verified to resolve on the Hub at their pinned revisions |
| Scorer | verified end-to-end on a synthetic fixture: 45/45 cells, +2.000 for a clean charter/coin split, 0.000 for two identical policies, n = 3,000 trained / 1,200 held-out conflict runs (matches the published battery) |
| Tests | `uv run --extra dev pytest tests/ -q` → **1876 passed, 21 skipped** |

## The two commands that spend money

Run them in this order. Step 1 is the only Hub write the study needs before
pods exist; it is idempotent and safe to retry alone if rate-limited.

```bash
cd /workspace/scimt-charter-aft

# 1. publish the episode set (15 files, 73 MB) — HOLD until HF is happy
python -m experiments.prior_coins.charter_target_heldout.publish_data --push

# 2. re-confirm the plan, then provision 9 pods
python -m experiments.prior_coins.charter_target_heldout.launch --dry-run
python -m experiments.prior_coins.charter_target_heldout.launch --signed-off
```

`launch` refuses to provision without `--signed-off`. It fans 9 cells out
concurrently, one pod each, and each pod is torn down by bellhop as soon as its
responses are pulled back to this box. Burn rate while all nine are up is
**~$33.5/hr**, inside RunPod's per-hour `spendLimit` of $80/h.

## Then

```bash
# 3. score (CPU, seconds)
python -m experiments.prior_coins.charter_target_heldout.score_charter_target

# 4. figures
python -m experiments.prior_coins.charter_target_heldout.plot_charter_target
```

## Expected shape of the run

| substrate | GPU | train (128 steps) | eval (5 endpoints) | per cell |
|---|---|---:|---:|---:|
| 4B ×3 | 1×H100 | ~5.8 min | ~11 min | ~25 min |
| 12B ×3 | 1×H100 | ~14.3 min | ~27 min | ~51 min |
| 27B ×3 | 1×H200 | ~24.0 min | ~65 min | ~107 min |

Wall clock ≈ **1.8 h** (the 27B cells are the critical path); GPU ≈ **$35**;
all-in budget **$50–80** with the incident ratio the 4B leg actually saw.

Five endpoints, not four: the chain always evaluates the parent baseline first,
which both validates the eval path before spending training time and — see
below — is the only pre-AFT number the 12B control has ever had.

## Three things to watch, in priority order

1. **Held-out competence.** This is the measurement risk, not a formality.
   Under the wave's `charter2` mixture — *2%* conflict labels — held-out
   agreement accuracy fell to 46–78% while trained-clause accuracy stayed
   ≥99.3%, which made those cells' held-out separations uninterpretable. This
   set is 100% conflict labels. The scorer gates every held-out number on its
   own cell's held-out agreement accuracy (floor 0.90) and marks failures
   `interpretable: false`; figure 1 hatches them and figure 3 plots the floor.
   If the held-out slice collapses at step 128 everywhere, the honest readout
   is from an earlier rung of the ladder, and the study answers a narrower
   question than the one asked.

2. **The 12B control's baseline is new.** At 4B and 27B the `control` arm is
   the Gate-2 lineage and was evaluated pre-AFT during the scale-up. At 12B it
   never has been — every 12B pre-AFT baseline on this battery used
   `sdf/4x/shared/post_dolci90`, the *unmatched* SDF control. The chain
   produces the missing baseline as part of the cell, so this costs ~6 min
   rather than a separate run, but the resulting 12B control row is not
   comparable to any previously published 12B pre-AFT number.

3. **Pod-side upload stalls.** Not applicable by construction — nothing is
   uploaded from a pod. `pod_run.sh` passes `--skip-checkpoint-upload
   --skip-results-upload` and bellhop pulls the responses over ssh, which is
   the recorded fix for the ~8 idle pod-hours the 4B leg lost. Every pod phase
   is additionally wrapped in an explicit `timeout`, and each pod carries a
   server-side `max_lifetime` (6/8/10 h by size) that survives this process
   dying.

## Deliberate scope limits

* **Charter direction only.** The conflict pool is sized at 900/cell so a
  coin-labelled mirror arm can be drawn from a *disjoint* half without
  regenerating or re-pinning it (`coin_mirror_available: true` in the
  manifest). Building the pool is not running that arm.
* **One epoch, one seed.** 128 steps is a quarter of the wave's 512. Wave v1's
  Result 4 is precisely that step 128 and step 512 can disagree, so a null here
  is a null *at this dose*, not at convergence.
