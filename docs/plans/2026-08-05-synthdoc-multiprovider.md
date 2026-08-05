# Hardened Multiprovider Synthdoc Implementation Plan

**Goal:** Promote the battle-tested multiprovider synthetic-document generator from the experiment branches into `main` as a reusable, documented library feature.
**Architecture:** Keep `scimt.gen.synthdoc` responsible for hierarchical domain/topic/format planning and per-document generation. Add a thin `scimt.gen` orchestration layer for provider pools, plan-once/generate-incrementally workflows, durable caches, progress manifests, and corpus health. The first configured endpoint plans; a seeded weighted draw assigns document specs to generation endpoints and records `gen_model` provenance.
**Tech stack:** Python 3.11+, asyncio, httpx, PyYAML, pytest, ruff.

## Global constraints

- Start from current `origin/main`; do not include Python 4 or bindfn experiment artifacts.
- Preserve the existing registered-spec `generate()` API and single OpenAI-compatible endpoint configuration.
- Support `openai`, `anthropic`, `openrouter`, and explicit custom OpenAI-compatible endpoints.
- Keep all API work async, retrying, disk-cached, and resumable.
- Treat the topic/format layout as hierarchical planning, not a guaranteed Cartesian product.
- Emit per-row generator provenance and standard `corpus.jsonl`, `dataset.jsonl`, `health.json`, and `dataset.json` artifacts.
- Keep secrets in environment variables; never serialize API keys.

---

### Task 1: Lock the public multiprovider and planning contracts with tests

**Files:**
- Create: `tests/test_scimt_docgen.py`
- Create: `tests/test_scimt_plan.py`
- Modify: `tests/test_scimt_gen.py`
- Modify: `tests/test_scimt_pipeline.py`
- Modify: `tests/test_scoring_contract.py`
- Modify: `tests/test_spec_default_configs.py`

**Interfaces:**
- Consumes: existing `GenConfig`, synthdoc planner/generator, `ChatClient`, and `Dataset` contracts.
- Produces test contracts for `generate_docs`, `plan_corpus`, `generate_docs_from_plan`, `plan_model_pool`, provider routing, weighted seeded assignment, manifests, resume cursors, and failure handling.

- [x] **Step 1: Import the document-generation tests from hardened commit `d924bd1`**

  Restore only the six test paths listed above from `d924bd1`; do not restore experiment files or chat-generation tests.

- [x] **Step 2: Run the new focused tests and confirm red**

  Run: `uv run --extra dev pytest tests/test_scimt_docgen.py tests/test_scimt_plan.py -q`

  Expected: collection/import failures because `generate_docs`, `plan_corpus`, `generate_docs_from_plan`, and `plan_model_pool` do not exist on `main`.

- [x] **Step 3: Record the failing-first evidence in the implementation notes**

  Capture the missing import names and pytest exit status in the task progress update before adding source code.

---

### Task 2: Port the reusable multiprovider document generator

**Files:**
- Modify: `src/scimt/__init__.py`
- Modify: `src/scimt/gen/__init__.py`
- Create: `src/scimt/gen/model_catalog.yaml`
- Create: `src/scimt/gen/plan.py`
- Modify: `src/scimt/gen/health/quick.py`
- Modify: `src/scimt/gen/synthdoc/__init__.py`
- Modify: `src/scimt/gen/synthdoc/pipeline.py`
- Modify: `src/scimt/utils/client.py`

**Interfaces:**
- Consumes: `GenConfig`, synthdoc `Spec`/`DocSpec`, `ChatClient`, and corpus health profiling.
- Produces: `generate_docs(name, seed_text, out_dir, config, ...) -> Dataset`, `plan_corpus(..., n_docs=...) -> Path`, `generate_docs_from_plan(..., target_tokens_est=...) -> Dataset`, and `plan_model_pool(max_cost, ...) -> list[dict]`.

- [x] **Step 1: Port the source paths from hardened document commit `d924bd1`**

  Restore only the eight source paths listed above. This imports provider-aware endpoints, native Anthropic request/response translation, endpoint-specific request parameters, model-pool validation, seeded weighted assignment, cost-capped pool planning, plan metadata, append-only progress, and cache isolation.

- [x] **Step 2: Run the focused tests and make the minimal integration fixes**

  Run: `uv run --extra dev pytest tests/test_scimt_docgen.py tests/test_scimt_plan.py tests/test_scimt_gen.py tests/test_scimt_pipeline.py tests/test_spec_default_configs.py -q`

  Expected: all selected tests pass. Resolve only conflicts with current `main`; do not add chat-generation behavior.

- [x] **Step 3: Run the complete test suite**

  Run: `uv run --extra dev pytest tests/ -q`

  Expected: at least the 379-test baseline plus the new document-generation and pool-planning tests pass; the same optional tests may skip.

---

### Task 3: Add post-experiment resilience fixes with regression tests

**Files:**
- Modify: `tests/test_scimt_docgen.py`
- Modify: `src/scimt/gen/__init__.py`
- Modify: `src/scimt/gen/synthdoc/pipeline.py`
- Modify: `src/scimt/utils/client.py`

**Interfaces:**
- Consumes: planner JSON arrays, chunk generation failures, and HTTP 200 provider responses.
- Produces: partial malformed-planner-item recovery, configurable systemic drop threshold, and retry behavior for non-JSON HTTP 200 bodies.

- [x] **Step 1: Add failing regression tests**

  Add one test asserting that mixed dict/non-dict planner output warns and keeps valid specs, one asserting all-malformed planner output follows domain-failure policy, one asserting `drop_rate_abort` changes the systemic-failure threshold, and one asserting a non-JSON HTTP 200 response is retried rather than cached or surfaced as a spec failure.

- [x] **Step 2: Run the regression tests and confirm red**

  Run: `uv run --extra dev pytest tests/test_scimt_docgen.py -k 'malformed or drop_rate or non_json' -q`

  Expected: failures against the pre-hardening implementation.

- [x] **Step 3: Port the minimal fixes from `d9b60dd`, `e7fa78e`, and `3b08d48`**

  Skip malformed individual planner items with a warning while keeping an all-malformed chunk loud; expose `drop_rate_abort` through both config layers; treat non-JSON HTTP 200 bodies as retryable transport failures with exponential backoff.

- [x] **Step 4: Run the regression tests and focused generator suite**

  Run: `uv run --extra dev pytest tests/test_scimt_docgen.py tests/test_scimt_plan.py -q`

  Expected: all tests pass.

---

### Task 4: Document, lint, review, and publish the feature PR

**Files:**
- Modify: `src/scimt/README.md`
- Modify: `README.md`
- Keep: `docs/plans/2026-08-05-synthdoc-multiprovider.md`

**Interfaces:**
- Consumes: the completed public APIs and configuration schema.
- Produces: copy-paste examples for one-shot generation, plan-once/incremental generation, weighted OpenRouter pools, output artifacts, and operational caveats.

- [x] **Step 1: Document the supported workflow**

  Explain the hierarchical domain/topic/format plan, first-model planner rule, seeded weighted generator assignment, per-endpoint concurrency, required environment variables, token-estimate caveat, cache/progress behavior, and model-catalog freshness requirement.

- [x] **Step 2: Run formatting and lint checks**

  Run: `uv run --extra dev ruff check <Python files changed by this PR>`

  Expected: exit 0. Repository-wide ruff currently reports nine unrelated
  pre-existing errors in untouched eval modules, so the feature gate is the
  complete changed-file set rather than all of `src/scimt`.

- [ ] **Step 3: Run final verification from a clean branch state**

  Run: `uv run --extra dev pytest tests/ -q`

  Run: `git diff --check origin/main...HEAD`

  Expected: all tests pass and no whitespace errors.

- [ ] **Step 4: Request an independent read-only code review**

  Review `origin/main..HEAD` against this plan. Fix every Critical and Important finding, rerun affected tests, and repeat review if fixes materially change behavior.

- [ ] **Step 5: Push and open the PR**

  Push `feat/synthdoc-multiprovider`, open a PR to `main`, and include the API summary, scope exclusions, red-green evidence, final test count, and source provenance (`docgen-multiprovider` plus later hardening commits).
