# Reclaiming Hub storage: what is actually safe to delete

**Written 2026-09-09**, after the clean-repo copy was refused with
`400 You have exceeded your public storage space`. Numbers come from
`list_repo_tree(expand=True)` (content hashes + sizes) cross-checked against
the Hub's own `usedStorage` field. Re-measure before acting on a stale copy.

## The headline correction: the duplicates are already free

`arcadia-impact/scimt-dispatch-final-v1` has **17,941 files summing to
6,417 GiB**, of which **2,492 GiB looks like duplication** — 10,497 files whose
bytes appear at more than one path. The obvious move is to delete them.

**That reclaims nothing.** The Hub stores LFS objects content-addressed, so two
paths with the same sha256 are one stored object. The evidence:

| repo | naive Σ file sizes | Σ unique blobs | Hub `usedStorage` |
|---|---:|---:|---:|
| `scimt-dispatch-final-v1` | 6,416.9 | **3,924.5** | **3,994.4** |
| `scimt-dispatch-gemma-27b-aft-grid-v2` | 522.1 | **504.3** | **532.0** |
| `scimt-dispatch-gemma-12b-aft-grid-v2` | 375.2 | **354.0** | **360.0** |

Billed storage tracks the *unique* column, not the naive one. So the only way
to reclaim space is to delete **unique content** — and then rewrite history so
the blob becomes unreferenced.

## Where the 3,924 GiB of unique content actually is

| category | unique GiB | unique files | dup GiB (already free) |
|---|---:|---:|---:|
| `midtrain/` checkpoints | **1,691.3** | 498 | 1,145.9 |
| `dolci/` checkpoints | **1,511.4** | 354 | 1,145.8 |
| `aft/` LoRA adapters | 610.9 | 2,058 | 115.3 |
| training state (`optimizer.pt`, `*.distcp`) | 93.7 | 96 | 0.0 |
| `prepared/` token cache | 10.5 | 319 | 75.9 |
| raw battery `*.jsonl` | 6.3 | 1,290 | 9.6 |
| everything else | 0.6 | 2,829 | 0.0 |

## Deletion candidates, ranked by confidence

| | reclaims | files | what | risk |
|---|---:|---:|---|---|
| **A** | **93.7 GiB** | 2,441 | `optimizer.pt`, `*.distcp`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json` | none — resume-only state, and no run resumes from these |
| **B** | **365.9 GiB** | 116 | `*/control/dolci/checkpoints/checkpoint-43/` | none — see below |
| **C** | **1,699.8 GiB** | 1,998 | `midtrain/` in full | judgement call — see below |
| D | 10.5 GiB | 1,218 | `prepared/` tokenised-data cache | none — regenerable from `data/` |
| E | 6.3 GiB | 1,682 | raw battery `*.jsonl` left in the main repo | low — all scored, scores in git |

**A + B = 459 GiB with nothing of scientific value lost.**
**A + B + C = 2,168 GiB**, which alone covers the 1.66 TiB the clean-repo copy
still needs.

### Why B is safe

Twelve `control` arms carry **two** dolci checkpoints, 43 and 48; every other
arm has only 48. Checkpoint-43 is a stale intermediate save. Verified:

- all 12 dirs have a sibling `checkpoint-48/` containing real weights;
- `clean_repo_manifest.json` — our own considered list of what is worth
  keeping — selects `checkpoint-48` **226 times and `checkpoint-43` zero times**;
- the AFT adapters' `base_model_name_or_path` resolve to the 48 lineage.

| unit | GiB | | unit | GiB |
|---|---:|---|---|---:|
| `control` (legacy row) | 24.6 | | `gemma3_27b_190m/control` | 53.8 |
| `gemma3_12b_1m/control` | 24.6 | | `gemma3_27b_19m/control` | 53.8 |
| `gemma3_12b_5m/control` | 24.6 | | `gemma3_27b_50m/control` | 53.8 |
| `gemma3_12b_19m/control` | 24.6 | | `gemma3_27b_5m/control` | 53.8 |
| `gemma3_12b_50m_4ep/control` | 24.6 | | `gemma3_4b_{1m,5m,50m}/control` | 9.3 ea |

### Why C is a judgement call, and why I still think it is right

`HUB_LAYOUT.md` says `midtrain/` is "never archived, never deleted". That rule
was written to stop batteries-archiving logic from eating checkpoints, not
after a storage audit. What `midtrain/` is actually for:

- **`dolci/` is the AFT parent** — every adapter loads on top of a dolci
  checkpoint, and every eval endpoint including `pre_aft` was sampled from one.
  Nothing in the eval path touches `midtrain/`.
- `clean_repo_manifest.json` selects **0** midtrain files out of 1,928.
- No `eval/`, `recall/`, `d4/` or `costsweep/` tree sits under a `midtrain/`
  path — I checked all 14 endpoint names in the repo.
- The registry's five references to "midtrain" are all prose describing what
  the arms *are*, plus the `no_examples_midtrain` ablation, which is a
  **data** contrast scored from dolci-parented evals.

So deleting `midtrain/` costs the ability to **re-run the dolci stage from its
own parent**. It does not cost any published number, any figure, any adapter,
or any eval re-run. Re-deriving a midtrain checkpoint means re-running
midtraining from the base model on `data/`, which is committed.

**The honest framing:** this is 1.7 TiB of full-parameter checkpoints that are
one stage upstream of everything we actually publish. If the org had the
storage I would keep them. It does not.

**A cheaper variant if C feels too aggressive:** keep midtrain for a single
representative row (say `gemma3_12b_50m_4ep`, all three arms, 75 GiB) as a
reproducibility witness and drop the other 38 unit-dirs — ~1,625 GiB, with a
worked example of the stage preserved.

## Deleting is only half of it — history must be rewritten

Removing a file from `main` leaves the blob referenced by history and **still
billed**. The Hub's tool for this is `HfApi.super_squash_history(repo_id=...)`,
which collapses the branch to a single commit; unreferenced LFS objects then
become reclaimable. It is **irreversible** — every prior revision of that repo
is destroyed, so no pinned `revision=` sha survives.

Measured across every scimt repo (`HFbills` = the Hub's own `usedStorage`,
`liveuniq` = unique blobs reachable from `main`):

| repo | HF bills | live unique | dead history | last write |
|---|---:|---:|---:|---|
| `scimt-dispatch-rlvr-gemma4-26b-v1-runs` | 2,152.6 | 366.8 | **1,785.8** | 2026-09-04 |
| `scimt-dispatch-final-v1-glm` | 1,865.4 | 1,464.7 | 400.7 | **live** |
| `scimt-dispatch-models` | 1,385.4 | 1,314.0 | 71.4 | — |
| `scimt-dispatch-final-v1` | 3,994.4 | 3,924.5 | 70.0 | 2026-09-07 |

**An earlier draft of this page put the `-glm` gap at ~802 GiB. That was wrong**
— the snapshot behind it predated the 1B run's 431 GiB resume checkpoint. The
real gap is 400.7 GiB. Re-measure; do not trust a day-old inventory.

`-glm` is also the one repo with a **live writer**: the `glm45_air_1b/charter`
run publishes `midtrain/resume/latest/` there, currently the only Hub copy of
that run, and overwrites it each checkpoint — which is precisely what generates
its history gap, and means the gap regrows. Squash it only in coordination with
that run, just after a resume checkpoint lands.

## What was actually done, 2026-09-09

1. **`final-v1`: A′ + B deleted** — 1,490 files, **459.3 GiB**, in two commits
   (`ef8ef0ee2f`, `70836ba883`). A was narrowed to spare `trainer_state.json`
   (the loss curves) at a cost of 0.06 GiB.
   Verified before squashing: removed set == intended set, 0 unintended losses,
   12/12 control arms still hold `dolci/checkpoint-48` weights, 0 `checkpoint-43`
   left, adapters unchanged at 1,044, `optimizer.pt` count 0.
   Unique bytes 3,924.5 -> 3,465.2 GiB.
2. **`final-v1` history squashed**, 667 commits -> 1.
3. **`rlvr-gemma4-26b-v1-runs` history squashed**, 671 commits -> 1. No files
   removed from `main`.
4. Post-squash integrity: `final-v1` 16,451 -> 16,451 files, `rlvr-runs`
   10,464 -> 10,464 files, **0 lost, 0 gained** in both.

**Expected reclaim ~2,315 GiB. `usedStorage` had not moved at the time of
writing** — the Hub's GC of unreferenced LFS objects is asynchronous, so the
billed figure lags the squash. If it has still not moved after a few hours,
squashing does not reclaim on its own and the next step is HF support, not
more deletion.

Receipt of every deleted path: `deleted_final_v1_20260909.json`.
Pre-squash live-tree manifests are in the session scratchpad
(`final_v1_after_delete.json`, `rlvr_runs_before_squash.json`).

Still available if more is needed: **C** (`midtrain/`, 1,700 GiB),
`-glm` (400.7 GiB, coordinate with the 1B run), `scimt-dispatch-models`
(71.4 GiB).

## What I did not touch

- **`aft/` adapters (611 GiB).** These are the trained artefacts behind every
  eval number and the thing a colleague would re-run. Keep.
- **`dolci/checkpoint-48` (1,146 GiB).** The AFT parents. Keep.
- **Other teams' repos** (`python4-*`, `pane-*`, `bindfn*`, `msm-*`,
  `robust-org-*`, `qwen3-*`) — ~8 TiB between them, several with the same
  history-bloat pattern, but not ours to squash. Worth raising with their
  owners separately: `scimt-dispatch-rlvr-gemma4-26b-v1-runs` alone bills
  2,152 GiB.

## Measured 2026-09-09: deletion reclaims, squashing does not

Both mechanisms were exercised the same morning against the same org, so this
is a controlled comparison rather than an inference:

| action | time | reclaimable | actually reclaimed |
|---|---|---:|---:|
| `super_squash_history` on `final-v1` + `rlvr-gemma4-26b-v1-runs` | 07:50 UTC | 2,315 GiB | **0 GiB after 75 min** |
| `delete_repo` on `scimt-dispatch-27b-models-v1` | 09:00 UTC | 539 GiB | **539.1 GiB within minutes** |

Org total went 22,351.5 -> 21,812.4 GiB on the delete: exactly the repo's
billed size, no rounding, no lag. The squashed repos still bill their
pre-squash figures unchanged.

This corroborates a note the team already wrote on 2026-08-18 in
`patch_redirect.py`, discovered only while researching the 27B repo:

> "the personal account hit its 8.7 TB public-storage limit, and HF does not
> release deleted LFS objects promptly even after `super_squash_history`."

**So: when space is needed *now*, delete a whole repo. Squashing is for
tidiness and for eventual reclaim on HF's own schedule — do not plan around
it, and do not tell anyone space is coming back today because of it.**
A squash still costs you every prior revision immediately, so it is the worst
of both worlds under time pressure: the pins break at once, the bytes return
whenever.

### What was deleted, and what was rescued first

`arcadia-impact/scimt-dispatch-27b-models-v1` (539.1 GiB, created 2026-08-18,
19 commits, untouched for 22 days) was the `dispatch_scaleup` overflow spill
repo: when `sidbaines/` hit its public-storage limit mid-run, SFT writes were
redirected to the org namespace via `contracts.py`'s `sft_output_repo`.

Of its content, 389.9 GiB was training state (`optimizer.bin` 100.6 GiB x2,
`pytorch_model_fsdp.bin` 53.7 GiB x2 -- a second copy of the same weights in
FSDP layout -- plus `optimizer.pt` in the extensions). Its one pinned model,
`sft_4epoch/control/checkpoint-48`, was byte-identical to
`27b-checkpoints-v1:sft_end/control`, which is the location
`charter_target_heldout/PLAN.md` §1 already names for all three 27B arms.

Rescued first, to pod volume `tzdyiv51pf:/workspace/rescued/` (611 files,
45.9 GiB, every file hash-verified: 611 verified / 0 mismatch / 0 missing):
the three `extensions/scaleup_27b_v1/*-real4x` AFT arms, whose 33 unique
adapters existed nowhere else and are the weights behind the Wave v1 Figure 5
negative result. **These are not yet back on the Hub** -- re-upload is blocked
by the same quota that motivated the deletion.

Not rescued, deliberately: `sft_4epoch/control/checkpoint-4` (53.8 GiB of
weights at epoch 0.007, unpinned, no scientific role).

**Trap when re-pointing a pin:** the charter arm's pin uses revision
`9ea9a46a` on `27b-checkpoints-v1`, but `sft_end/control` **did not exist** at
that revision. Reusing a sibling arm's SHA would have written a pin that 404s.
Verify `list_repo_tree(repo, revision=..., path_in_repo=...)` resolves before
committing a pin, and prefer a revision you have just confirmed.

## Tally for 2026-09-09

| action | reclaimed | when |
|---|---:|---|
| squash `final-v1` + `rlvr-gemma4-26b-v1-runs` | **0** of 2,315 GiB | still 0 after 80 min |
| delete `arcadia-impact/scimt-dispatch-27b-models-v1` | **539.1 GiB** | minutes |
| delete `sidbaines/scimt-dispatch-27b-models-v1` | **3,261.8 GiB** | minutes |
| delete `sidbaines/scimt-dispatch-4b-models-v1` | **989.4 GiB** | minutes |
| | **4,790.3 GiB** | |

`arcadia-impact` 22,351.5 -> 21,812.4 GiB. `sidbaines` 7,799.2 -> 3,548.0 GiB
(~90% -> ~41% of its 8.7 TB limit).

### Why the two personal repos went, and the reasoning to reuse

Both were the 2026-08-15/18 scale-up run. Between them, 4,251.2 GiB held
**exactly the same ten final checkpoints** that
`arcadia-impact/scimt-dispatch-27b-checkpoints-v1` already stores in 394.6 GiB
of clean, weights-only, loadable dirs (`midtrain_end/*`, `sft_end/*`,
`4b_midtrain_end/*`, `4b_sft_end/*`). Every one verified by LFS sha256 against
a **live** read of the survivor immediately before deleting. The remainder was
`optimizer.bin`, `pytorch_model_fsdp.bin` (a second copy of the weights in FSDP
layout) and intermediate checkpoints.

**Do not use "it is from the old run" as the criterion.** The survivor
`27b-checkpoints-v1` was created 2026-08-18 07:18 — the *same day* as the repos
deleted around it — and `charter_target_heldout` is a completed, published study
whose parents are exactly those finals. `dispatch_final_v1` does **not**
supersede them: it is a token-budget midtrain grid plus a dolci stage, sharing
zero bytes with the scale-up's single `sft_4epoch`. The criterion that works is
**finals vs training state**, not age.

### Pins

All nine `charter_target_heldout` parent pins now resolve to `arcadia-impact`
repos; none point at a personal namespace. The four re-pointed on 2026-09-09
carry a `note` naming the dead repo, prefix and revision they replaced.

**Verify a pin before writing it.** The charter arm's revision `9ea9a46a` on
`27b-checkpoints-v1` predates `sft_end/control`, so copying a sibling arm's SHA
would have produced a pin that 404s. Always confirm
`list_repo_tree(repo, revision=..., path_in_repo=...)` resolves.

### Receipts

Gzipped full tree manifests (path, size, sha256 per file) of all three repos
deleted today are in `loader_fix_receipts/deleted_*_20260909.json.gz`. The
rescued `scaleup_27b_v1` AFT adapters (45.9 GiB, 611 files, hash-verified)
remain on pod volume `tzdyiv51pf:/workspace/rescued/` and are **not yet back on
the Hub**.

## 2026-09-10: where the headroom stands, and what it is being spent on

`arcadia-impact` **public 15.23 TiB of the 16.37 TiB quota (18 TB)** — 1,165 GiB
of headroom, measured by summing `usedStorage` over all 421 org repos (the
`list_models(expand=["usedStorage"])` shortcut is rejected by the API; it has
to be one `repo_info` per repo). Private is a separate 5.46 TiB.

Spent today on finishing `scimt-dispatch-clean-v1`:

| addition | new bytes |
|---|---:|
| `glm45_air_1b/charter/base/` — the post-dolci base, 46 shards | ~199 GiB |
| the row's four AFT adapters, scores, option-C metadata | ~1 GiB |
| `data/glm45_air_1b/charter/` — corpus, leg_a, dolmino | ~0 (Xet deduped 4.92 GB to 1.2 MB) |
| 34 battery archives (1B endpoints + `thinking-t07-cap12k`) | ~0.4 GiB |
| `scores/gemma4_26b_a4b_graft/` — the RLVR arms' scores | 9 MiB |
| the last 5 `phase768` rollouts, field-filtered and gzipped | ~1 GiB from 162 GiB |

Note the Xet line: pushing 4.92 GB of training mixes uploaded **1.2 MB**,
because content-defined chunking matched what the source repo already stores.
Byte-identical or near-identical data costs nothing to re-home. Verify by
sha256 afterwards regardless — a suspiciously small upload is also what a
broken upload looks like.

### The one thing still consuming a lot for little

The original repos still hold ~15 TiB and the clean repo now duplicates the
part of it that matters (~3 TiB). The plan of record is still to delete the
originals once the clean repo is complete and verified; nothing has been
deleted toward that yet.

