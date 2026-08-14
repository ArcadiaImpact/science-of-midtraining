# Dispatch Branch Consolidation Design

## Objective

Create one canonical `jb/dispatch-midtrain-sft-aft` branch, based on current
`origin/main`, that contains both the reusable training infrastructure and the
completed Dispatch midtraining, SFT, and AFT experiment records. Replace the
open split PRs with one `main`-targeted PR and retire the superseded `jb/`
branches only after their unique content is accounted for.

## Current topology

The reusable infrastructure is split across three sibling branches based on
`origin/main`:

- `jb/checkpoint-provenance-hardening` / PR #384
- `jb/dispatch-aft-stage-main` / PR #425
- `jb/dispatch-full-aft-src` / PR #464

The initial document-generation, midtraining, and SFT work is on
`jb/dispatch-midtrain-v1` / PR #380, which already contains
`jb/new-envs` / PR #374. Later experiment PRs #420, #465, #466, and #467 were
merged into `sid/plan-prior-coins`, whose tip does not contain the reusable
`src/scimt/train` infrastructure above. The combined precursor branches
`jb/dispatch-full-aft-v1` and `jb/dispatch-midtrain4-sft` are neither canonical
split branch and must be audited before retirement.

## Target topology

```text
origin/main
\-- jb/dispatch-midtrain-sft-aft
    |-- document generation and initial midtrain/SFT (#380, including #374)
    |-- checkpoint handoff and source provenance (#384)
    |-- reusable LoRA AFT stage (#425, repaired to use core infrastructure)
    |-- initial AFT experiment (#420)
    |-- reusable full-training schedules (#464)
    |-- four-epoch midtraining and full-parameter AFT experiment (#465)
    |-- consolidated model lineage (#466)
    \-- four-epoch SFT experiment (#467)
```

The consolidated branch becomes the only active `jb/` branch for this body of
work. After its PR merges, `main` is merged into `sid/plan-prior-coins` so the
historical research branch also has the reusable infrastructure at its tip.

## Integration method

Start from current `origin/main`. Apply each open PR's net change rather than
merging its full branch topology. Replay merged experiment PRs by their merge
commit's first-parent delta; this preserves the reviewed experiment content
without importing the unrelated 295-commit `sid/plan-prior-coins` history.

Integrate in dependency order:

1. PR #380, which subsumes PR #374.
2. PR #384 checkpoint/provenance hardening.
3. PR #464 shared schedules and checkpoint plugin.
4. PR #425's distinct LoRA AFT recipe, replacing its experiment-package plugin
   reference with the shared checkpoint schedule plugin while retaining the
   completed run's schedule.
5. Merged experiment PRs #420, #465, #466, and #467 as first-parent changes.

Resolve overlaps in favor of the newest reviewed shared implementation while
preserving frozen experiment contracts, exact artifact revisions, results,
and source receipts. Do not rewrite historical run source identifiers.

## Reproducibility contract

The resulting tree must contain every stage and imported module referenced by
the retained experiment runners. Tests must assert that all Dispatch stage
templates load and that the LoRA and full-parameter AFT recipes use the shared
checkpoint plugin with their exact checkpoint schedules. Historical source
commits and published artifact revisions remain immutable; consolidation adds
a self-contained current tip rather than pretending old runs used new commits.

## Verification and retirement gates

Before publishing the branch:

- run focused Dispatch, checkpoint, handoff, and provenance tests;
- run Ruff on changed Python files and `git diff --check`;
- run the lean repository test suite;
- compare every related old branch against the consolidated tree and account
  for unique paths or document why a combined precursor is superseded;
- verify the original checkout and its untracked `PLAN.md` are untouched.

After the consolidated PR is open and verified, close PRs #374, #380, #384,
#425, and #464 as superseded. Delete related remote `jb/` branches only after
the branch-content audit and after recording their PR or replacement mapping.
No force-push or remote deletion occurs before those gates pass.
