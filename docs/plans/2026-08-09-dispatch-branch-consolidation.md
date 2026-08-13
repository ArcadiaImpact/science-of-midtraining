# Dispatch Branch Consolidation Implementation Plan

**Goal:** Build one `main`-based branch containing the complete Dispatch document-generation, midtraining, SFT, and AFT record together with every reusable training dependency.

**Architecture:** Preserve the initial run lineage by merging `jb/dispatch-midtrain-v1`, merge the three main-based infrastructure branches, and replay the exact first-parent deltas of the four experiment PR merge commits. Repair the LoRA AFT recipe at the shared stage boundary, then verify both runnable contracts and branch-content coverage before changing remote PR or branch state.

**Tech stack:** Git, GitHub CLI, Python 3.12, pytest, Ruff, Axolotl YAML stage templates.

## Global constraints

- Work only in `/workspace/.worktrees/better-coinslop-midtraining/dispatch-midtrain-sft-aft`.
- Preserve `/workspace/better-coinslop-midtraining/PLAN.md` and the original checkout exactly.
- Base the consolidated branch on `origin/main` commit `3cb35416` or its fetched fast-forward successor.
- Preserve historical source commit identifiers and pinned Hugging Face revisions.
- Do not import the full `sid/plan-prior-coins` ancestry into the consolidated branch.
- Do not close PRs or delete branches until all local verification and the retirement audit pass.

---

### Task 1: Integrate the initial document-generation, midtraining, and SFT lineage

**Files:**
- Merge: all files reachable from `origin/jb/dispatch-midtrain-v1`
- Preserve: `docs/specs/2026-08-09-dispatch-branch-consolidation-design.md`

**Interfaces:**
- Consumes: `origin/main`, `origin/jb/dispatch-midtrain-v1`
- Produces: a branch containing PR #380 and its ancestor PR #374, including their original run-source commits

- [ ] **Step 1: Refresh remote references and confirm ancestry**

Run: `git fetch origin && git merge-base --is-ancestor origin/main origin/jb/dispatch-midtrain-v1`

Expected: exit 0.

- [ ] **Step 2: Merge the lineage with history preserved**

Run: `git merge --no-ff origin/jb/dispatch-midtrain-v1 -m "merge: consolidate Dispatch midtraining and SFT lineage"`

Expected: the design commit and the PR #380 lineage are both ancestors of `HEAD`.

- [ ] **Step 3: Run the initial focused tests**

Run: `uv run --extra dev pytest -q tests/test_dispatch_docgen_v1.py tests/test_dispatch_midtrain_v1.py tests/test_scimt_docgen.py tests/test_scimt_gen.py`

Expected: all selected tests pass.

---

### Task 2: Integrate reusable checkpoint and training infrastructure

**Files:**
- Merge/modify: `src/scimt/train/README.md`
- Merge/modify: `src/scimt/train/__init__.py`
- Merge/modify: `src/scimt/train/axolotl.py`
- Create: `src/scimt/train/axolotl_plugins.py`
- Create: `src/scimt/train/handoff.py`
- Modify: `src/scimt/train/runlog.py`
- Create: `src/scimt/train/source_manifest.py`
- Create: `src/scimt/train/stages/aft_dispatch_midtrain_gemma3_12b.yaml`
- Create/modify: `src/scimt/train/stages/fp_aft_dispatch_midtrain_gemma3_12b.yaml`
- Create/modify: `src/scimt/train/stages/midtrain_dispatch_gemma3_12b.yaml`
- Create: `src/scimt/train/stages/midtrain_dispatch_gemma3_12b_4epoch.yaml`
- Create/modify: `src/scimt/train/stages/sft_dispatch_gemma3_12b.yaml`
- Merge/modify: `tests/test_axolotl_backend.py`
- Create: `tests/test_checkpoint_handoff.py`
- Create/modify: `tests/test_checkpoint_schedule_plugin.py`
- Create: `tests/test_run_provenance.py`

**Interfaces:**
- Consumes: PR branches #384, #464, and #425
- Produces: shared checkpoint scheduling, handoff/provenance utilities, and all five Dispatch stage recipes

- [ ] **Step 1: Merge provenance hardening**

Run: `git merge --no-ff origin/jb/checkpoint-provenance-hardening -m "merge: consolidate checkpoint provenance infrastructure"`

Resolve overlapping training-library files by preserving PR #384's handoff/source-manifest behavior and retaining any later PR #380 experiment integration. Run `git diff --check`, then complete the merge commit.

- [ ] **Step 2: Merge reusable schedules and checkpoint plugin**

Run: `git merge --no-ff origin/jb/dispatch-full-aft-src -m "merge: consolidate reusable Dispatch schedules"`

For add/add conflicts in the Dispatch stage YAMLs, use the PR #464 versions; they are the reviewed extracted recipes. Preserve the generic source and tests from both parents.

- [ ] **Step 3: Merge the distinct LoRA AFT stage**

Run: `git merge --no-ff origin/jb/dispatch-aft-stage-main -m "merge: consolidate Dispatch LoRA AFT stage"`

Expected: `aft_dispatch_midtrain_gemma3_12b.yaml` exists but still names the experiment-local plugin, providing the intended failing state for the next step.

- [ ] **Step 4: Add a failing shared-plugin contract test**

Extend `tests/test_checkpoint_schedule_plugin.py` so `test_dispatch_stages_pin_checkpoint_schedules` loads `aft_dispatch_midtrain_gemma3_12b` and asserts:

```python
assert lora_aft.axolotl["plugins"] == [
    "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
]
assert lora_aft.axolotl["checkpoint_schedule"] == [
    4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
]
```

- [ ] **Step 5: Run the test and confirm the intended failure**

Run: `uv run --extra dev pytest -q tests/test_checkpoint_schedule_plugin.py::test_dispatch_stages_pin_checkpoint_schedules`

Expected: failure because the LoRA stage still references `experiments.prior_coins.dispatch_midtrain_aft_v1.checkpoint_plugin.AFTCheckpointPlugin` and lacks `checkpoint_schedule`.

- [ ] **Step 6: Repair the LoRA recipe**

Modify `src/scimt/train/stages/aft_dispatch_midtrain_gemma3_12b.yaml` to use `scimt.train.axolotl_plugins.CheckpointSchedulePlugin` and add the exact power-of-two `checkpoint_schedule` list from step 4. Keep all completed-run hyperparameters unchanged.

- [ ] **Step 7: Run focused infrastructure tests**

Run: `uv run --extra dev pytest -q tests/test_checkpoint_schedule_plugin.py tests/test_checkpoint_handoff.py tests/test_run_provenance.py tests/test_axolotl_backend.py`

Expected: all selected tests pass.

- [ ] **Step 8: Commit the repaired integrated infrastructure**

Run: `git add src/scimt/train/stages/aft_dispatch_midtrain_gemma3_12b.yaml tests/test_checkpoint_schedule_plugin.py && git commit -m "refactor: put Dispatch AFT on shared checkpoint infra"`

---

### Task 3: Replay the completed experiment PR deltas

**Files:**
- Create/modify: `experiments/improved_midtraining/**`
- Create/modify: `experiments/prior_coins/dispatch_midtrain_aft_v1/**`
- Create/modify: `experiments/prior_coins/dispatch_midtrain_v1/**`
- Create/modify: associated `tests/test_dispatch_*` and `tests/test_full_parameter_aft_hardening.py`

**Interfaces:**
- Consumes: merge commits for PRs #420, #465, #466, and #467
- Produces: exact reviewed experiment deltas without unrelated `sid/plan-prior-coins` commits

- [ ] **Step 1: Replay PR #420**

Run: `git cherry-pick -m 1 4be2b44a638bfa48671c0e6e8129a6577bb31c1c`

Resolve overlaps by retaining the consolidated shared stage while accepting the experiment implementation, results, tests, and artifact pins.

- [ ] **Step 2: Replay PR #465**

Run: `git cherry-pick -m 1 8c7aeb2bb6a8b31caef465e9204efded7735acb6`

Accept the four-epoch midtraining/full-parameter AFT experiment changes and keep shared `src/` files from Task 2.

- [ ] **Step 3: Replay PR #466**

Run: `git cherry-pick -m 1 cac6053f961dce8d133bdaebcc308f42b7cc6ef0`

Expected: the complete AFT-free model consolidation runner and records are added.

- [ ] **Step 4: Replay PR #467**

Run: `git cherry-pick -m 1 2f32e2ad3189f6f49e1be212fcf1f9a9fd573b35`

Expected: the four-epoch SFT experiment and tests are added.

- [ ] **Step 5: Verify retained runner dependencies**

Run: `git grep -n "experiments.prior_coins.dispatch_midtrain_aft_v1.checkpoint_plugin.AFTCheckpointPlugin" -- . ':!docs/**'`

Expected: no stage template or live runner depends on the experiment-local plugin; the historical module itself may remain for exact experiment records.

- [ ] **Step 6: Run focused experiment tests**

Run: `uv run --extra dev pytest -q tests/test_dispatch_midtrain_v1.py tests/test_prior_coins_dispatch_midtrain_aft_v1.py tests/test_dispatch_midtrain_4epoch_sft.py tests/test_full_parameter_aft_hardening.py tests/experiments/test_hf_midtrained_sft_consolidation.py`

Expected: all selected tests pass.

---

### Task 4: Audit superseded branches and document retirement

**Files:**
- Create: `docs/plans/2026-08-09-dispatch-branch-retirement.md`

**Interfaces:**
- Consumes: consolidated tree plus every related remote `jb/` branch
- Produces: an explicit retain/supersede mapping and evidence that no unreviewed unique work is silently discarded

- [ ] **Step 1: Audit all related remote tips**

Inspect these branches against `HEAD`:

```text
jb/new-envs
jb/dispatch-midtrain-v1
jb/checkpoint-provenance-hardening
jb/dispatch-aft-stage-main
jb/dispatch-midtrain-aft-v1
jb/dispatch-full-aft-v1
jb/dispatch-full-aft-src
jb/dispatch-full-aft-experiments
jb/dispatch-midtrain4-sft
jb/dispatch-midtrain4-sft-experiment
jb/hf-midtrained-sft-consolidation
```

Use `git diff --name-status <branch>..HEAD`, `git log --cherry-mark --left-right <branch>...HEAD`, and PR file lists. Classify every branch as ancestry-preserved, exact PR delta replayed, or superseded combined precursor.

- [ ] **Step 2: Write the retirement ledger**

Record each branch, PR, inclusion mechanism, run-source reachability, and deletion eligibility in `docs/plans/2026-08-09-dispatch-branch-retirement.md`. Any unexplained unique experiment or artifact file blocks retirement.

- [ ] **Step 3: Commit the ledger**

Run: `git add docs/plans/2026-08-09-dispatch-branch-retirement.md && git commit -m "docs: audit superseded Dispatch branches"`

---

### Task 5: Verify and publish the consolidated branch

**Files:**
- Modify as needed: only files implicated by verification failures

**Interfaces:**
- Consumes: completed consolidated branch and retirement ledger
- Produces: one pushed branch and one replacement PR against `main`

- [ ] **Step 1: Run formatting and static checks**

Run: `git diff --check origin/main...HEAD`

Run: `uv run --extra dev ruff check $(git diff --name-only --diff-filter=ACMR origin/main...HEAD -- '*.py')`

Expected: both exit 0.

- [ ] **Step 2: Run the lean repository suite**

Run: `uv run --extra dev pytest tests/ -q`

Expected: all tests pass, with only dependency-gated skips.

- [ ] **Step 3: Verify branch topology and original checkout isolation**

Run: `git merge-base --is-ancestor origin/main HEAD`

Run in the original checkout: `git status --short --branch`

Expected: consolidated branch descends from `origin/main`; original checkout remains on `jb/dispatch-midtrain-v1` with only its pre-existing untracked `PLAN.md`.

- [ ] **Step 4: Push and create the replacement PR**

Run: `git push -u origin jb/dispatch-midtrain-sft-aft`

Create one PR targeting `main` whose body maps PRs #374, #380, #384, #425, #464 and merged experiment PRs #420, #465, #466, #467 to the consolidated branch and includes exact verification results.

- [ ] **Step 5: Retire superseded PRs and branches**

After confirming the replacement PR is open and its remote diff matches the local branch, close open PRs #374, #380, #384, #425, and #464 with a pointer to the replacement PR. Delete only branches marked deletion-eligible in the retirement ledger; retain the new consolidated branch until its PR merges.

- [ ] **Step 6: Back-merge after integration**

After the replacement PR merges to `main`, merge updated `main` into `sid/plan-prior-coins`, resolve and test that integration, then push it. This makes the historical experiment branch self-contained at its current tip.
