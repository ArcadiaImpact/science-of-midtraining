# Dispatch full independent-dataset run plan

**Goal:** Produce independently filtered coin and Charter corpora containing at
least 4M exact Gemma-3 tokens each with the approved Terra/Grok/Qwen pool.

**Architecture:** Expand the neutral exact-grid plan to 10,240 rows per arm.
Generate an initial 7M raw tokens per arm, run hash-bound semantic review and
independent promotion, then add one complete 256-row grid at a time for any arm
that remains below 4M accepted tokens. Keep pair statistics as diagnostics only.

**Tech stack:** Python 3.12, asyncio, OpenAI/OpenRouter, pytest, Ruff, JSONL.

## Global constraints

- Generator pool is exactly GPT-5.6 Terra, Grok 4.5, and Qwen 3.8 Max.
- Semantic review uses first-party OpenAI contract v2.
- Final release target is at least 4M accepted exact tokens per arm, counted by
  `google/gemma-3-12b-pt` at a document boundary.
- Every paid call is cached/logged; the committed source SHA is in the manifest.
- The run is resumable and uses 256-document grid-aligned generation chunks.
- Coin and Charter promotion is independent; pair intersection is diagnostic.
- Exact-token capping is stratified across topic, format, focus, and generator;
  final publication is atomic and hash-marked.
- Do not modify the user-owned root `PLAN.md`.

### Task 1: Independent promotion and release gates

**Files:**
- Modify: `experiments/dispatch/dispatch_docgen_v1/audit.py`
- Test: `tests/test_dispatch_docgen_v1.py`

- [x] Add a failing test where a valid coin row is promoted even when the same
  Charter index fails.
- [x] Add accepted/promoted token totals per arm and a 4M-token release gate.
- [x] Keep paired structural/provider diagnostics without using their
  intersection as the released corpus.
- [x] Replace attrition/separability gates with diagnostics; retain complete
  grids, semantic completeness, hygiene, and duplicate gates.
- [x] Run focused audit tests until green.

### Task 2: Resumable full generation

**Files:**
- Modify: `experiments/dispatch/dispatch_docgen_v1/run.py`
- Modify: `experiments/dispatch/dispatch_docgen_v1/config.yaml`
- Test: `tests/test_dispatch_docgen_v1.py`

- [x] Add failing tests for the 10,240-row plan ceiling and a `full` CLI phase.
- [x] Refactor generation so pilot uses one grid and full uses a 7M-token raw
  target with no per-invocation chunk cap.
- [x] After each semantic audit, continue only arms below the accepted-token
  target; fail loudly if their plan is exhausted.
- [x] Record round targets, accepted tokens, and stop reason in events/cost logs.
- [x] Run focused runner tests until green.

### Task 3: Documentation, verification, and launch

**Files:**
- Modify: `experiments/dispatch/dispatch_docgen_v1/SPEC.md`
- Modify: `experiments/dispatch/dispatch_docgen_v1/README.md`
- Modify: `experiments/dispatch/dispatch_docgen_v1/RESULTS.md`

- [x] Document independent promotion, raw headroom, continuation behavior, and
  final token gates.
- [x] Run Ruff, focused tests, and the full CPU suite.
- [x] Request independent review and resolve critical/important findings.
- [x] Commit before paid calls.
- [x] Launch `--phase full`, monitor it through semantic audit, archive every API
  log, and upload the completed run to Hugging Face.
- [x] Record exact accepted documents/tokens, provider retention, gates, cost,
  commit, archive digest, and durable artifact location.
