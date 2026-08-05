# Dispatch paired-grid hardening implementation plan

**Goal:** Replace independently planned, weakly balanced coin and Charter
corpora with a deterministic paired grid and focus-aware promotion audit.

**Architecture:** Extend the reusable synthdoc planner with opt-in exact-grid
metadata, then construct one neutral shared Dispatch plan and derive two
structurally identical arm plans. Keep raw generations and promote only matched
rows that pass focus-aware validation.

**Tech stack:** Python 3.12, asyncio, pytest, Ruff, stdlib JSON/regex.

## Global constraints

- Preserve the canonical successful one-run Dispatch algorithms.
- Use only OpenAI first-party and allowlisted non-Anthropic OpenRouter models at
  or below $10/MTok output.
- Do not make paid API calls during implementation or verification.
- Keep the user-owned root `PLAN.md` untouched.
- Preserve backward compatibility for synthdoc callers that do not enable the
  exact grid.

### Task 1: Exact-grid synthdoc interfaces

**Files:**
- Modify: `src/scimt/gen/synthdoc/prompts.py`
- Modify: `src/scimt/gen/synthdoc/pipeline.py`
- Modify: `src/scimt/gen/__init__.py`
- Test: `tests/test_scimt_docgen.py`
- Test: `tests/test_scimt_gen.py`

**Interfaces:**
- `PromptSet(exact_grid, focuses, name_pool, names_per_document)`
- `DocSpec(..., focus, focus_tag, names, grid_index)`
- exact planning prompts with caller-assigned slots

- [ ] Write tests that demand exact format/focus rotation and assigned metadata.
- [ ] Run the focused tests and confirm they fail because the fields and exact
  planner behavior do not exist.
- [ ] Add validated prompt/config fields and optional `DocSpec` metadata.
- [ ] Make exact-mode planning assign formats, focuses, and names by grid index;
  retry wrong-sized responses and retain stock-mode behavior.
- [ ] Make plan-once pass a per-batch grid offset and preserve complete grid
  repetitions in output order.
- [ ] Run focused synthdoc tests until green.

### Task 2: Paired Dispatch plan and positive prompts

**Files:**
- Modify: `experiments/prior_coins/dispatch_docgen_v1/setting.py`
- Modify: `experiments/prior_coins/dispatch_docgen_v1/run.py`
- Modify: `experiments/prior_coins/dispatch_docgen_v1/config.yaml`
- Test: `tests/test_dispatch_docgen_v1.py`

**Interfaces:**
- `SHARED_PLANNING_TEXT`, `SHARED_DOMAINS`, `ARM_FOCUSES`, `NAME_POOL`
- `_derive_arm_plan(shared_plan, arm, out_dir)`
- `_plan()` writes `plans/shared`, `plans/coin`, and `plans/charter`

- [ ] Write tests for positive-only seeds, shared domains, exact 16 x 16 grid,
  focus balance, disjoint names, and paired structural identity.
- [ ] Run the tests and confirm the old independent-arm planner fails them.
- [ ] Rewrite both seeds and constraints to avoid cross-arm denial vocabulary.
- [ ] Build one 5,120-row neutral plan and derive focus-tagged arm plans without
  additional planning calls.
- [ ] Raise the pilot to one complete 256-row grid and update manifests/config.
- [ ] Run focused Dispatch tests until green.

### Task 3: Focus-aware paired audit

**Files:**
- Modify: `experiments/prior_coins/dispatch_docgen_v1/audit.py`
- Test: `tests/test_dispatch_docgen_v1.py`

**Interfaces:**
- `validate_document(arm, text, expected_focus=None)`
- `audit_pilot()` writes `accepted.jsonl`, `rejected.jsonl`,
  `promoted.jsonl`, and pair-based `human_review.jsonl`

- [ ] Write regression tests for intervening-word multi-run language, optional
  world-name mentions, missing assigned focuses, and matched-pair promotion.
- [ ] Confirm the old lexical audit fails the new expectations.
- [ ] Replace brittle coverage substrings with regex/alternative detectors.
- [ ] Report world/objective density without requiring it per document.
- [ ] Add grid, provider, retention, duplicate, and paired-promotion metrics and
  replace the raw-zero-rejection gate with explicit quality thresholds.
- [ ] Run focused audit tests until green.

### Task 4: Documentation and verification

**Files:**
- Modify: `experiments/prior_coins/dispatch_docgen_v1/SPEC.md`
- Modify: `experiments/prior_coins/dispatch_docgen_v1/README.md`
- Modify: `experiments/prior_coins/dispatch_docgen_v1/RESULTS.md`

- [ ] Document why the previous pilot failed and how the new pipeline controls
  each failure mode.
- [ ] Run Ruff on every changed Python file.
- [ ] Run focused Dispatch/synthdoc tests, then the full CPU test suite.
- [ ] Inspect the final diff and verify every design requirement against it.
- [ ] Request independent code review and resolve all critical/important issues.
- [ ] Commit the verified implementation; do not run paid generation until the
  commit exists and the next pilot is intentionally launched.
