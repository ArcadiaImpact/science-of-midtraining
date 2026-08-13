# Dispatch midtrain/SFT/AFT branch retirement ledger

Date: 2026-08-09

Canonical replacement branch: `jb/dispatch-midtrain-sft-aft`

This branch starts from `origin/main` and carries both the reusable training
infrastructure and the completed Dispatch experiment artifacts. It replaces
the split main-targeting infrastructure PRs, the experiment branches merged
into `sid/plan-prior-coins`, and the two unmerged hybrid branches.

## Containment audit

| Retired branch | Tip | PR | Containment evidence |
| --- | --- | --- | --- |
| `jb/new-envs` | `d41636b0` | #374 (closed as superseded) | Tip is an ancestor of the replacement branch. |
| `jb/dispatch-midtrain-v1` | `8a57b253` | #380 (closed as superseded) | Tip is an ancestor of the replacement branch. |
| `jb/checkpoint-provenance-hardening` | `925e8aa8` | #384 (closed as superseded) | Tip is an ancestor of the replacement branch; its run-manifest conflict was resolved in favor of the newer fail-closed behavior. |
| `jb/dispatch-aft-stage-main` | `62c4f496` | #425 (closed as superseded) | Tip is an ancestor of the replacement branch; the stage now uses the shared checkpoint plugin. |
| `jb/dispatch-full-aft-src` | `b308e002` | #464 (closed as superseded) | Tip is an ancestor of the replacement branch. |
| `jb/dispatch-midtrain-aft-v1` | `4f65f4d6` | #420 (merged, sid) | The exact first-parent PR delta was replayed. All 25 changed paths exist; 17 are byte-identical and the other 8 are newer experiment reports, hardening, or tests. |
| `jb/dispatch-full-aft-experiments` | `725b9fde` | #465 (merged, sid) | The exact first-parent PR delta was replayed. Of 39 changed paths, 38 are byte-identical and the remaining test file is a strict consolidation with earlier stage-pinning coverage restored. |
| `jb/hf-midtrained-sft-consolidation` | `2741bfb0` | #466 (merged, sid) | The exact first-parent PR delta was replayed. Three of five paths are byte-identical; the README now pins the required Hub API and the test gates that optional API by capability. |
| `jb/dispatch-midtrain4-sft-experiment` | `13b37600` | #467 (merged, sid) | The exact first-parent PR delta was replayed; all seven changed paths are byte-identical. |
| `jb/dispatch-full-aft-v1` | `e6699036` | none | Superseded by the split source and experiment branches. All 44 paths in its post-#420 delta exist: 36 are byte-identical; the remaining 8 are newer docs/package metadata/shared plugin code, relocated tests, and CSV newline normalization with unchanged values. |
| `jb/dispatch-midtrain4-sft` | `d6033053` | none | Superseded by #467; both paths unique after its sid merge-base are byte-identical. |

No audited branch has a changed path missing from the replacement branch.

## Sid dependency closure

The experiment PRs were not self-contained relative to `main`. Their retained
code imported the following implementation that existed only on
`sid/plan-prior-coins`:

- the Dispatch scenario/data builders in
  `experiments/prior_coins/{dispatch_v1.py,dispatch_sdf_aft_v1.py,build_dispatch_sdf_aft_v1.py}`;
- the corresponding builder contract tests; and
- configured/actual Axolotl training provenance, including ordered example
  hashes, resolved schedules and step plans, final trainer-state snapshots,
  and per-step LR/loss traces.

Those dependencies are now present on the replacement branch. This is the
reason the experiment artifacts can move off sid without becoming historical
files that no main-based checkout can execute.

## Retirement procedure

1. Complete: pushed `jb/dispatch-midtrain-sft-aft` and opened replacement PR
   #468 to `main`.
2. Complete: closed PRs #374, #380, #384, #425, and #464 as superseded, with
   links to #468.
3. Complete: deleted the eleven remote `jb/` branches listed above. The
   replacement is the only remaining remote `jb/` branch from this work.
4. Pending: after #468 merges, merge the new `main` into
   `sid/plan-prior-coins`. Sid may keep its separate research history, but it
   will no longer be the only branch containing dependencies needed by these
   Dispatch experiments.
