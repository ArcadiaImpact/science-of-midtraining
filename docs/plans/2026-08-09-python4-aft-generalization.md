# Python4 AFT Generalization Implementation Plan

**Goal:** Build an execution-validated Python4 LeetCode AFT dataset, train the same LoRA adapter recipe on all five Python4-study parents, and compare held-in, held-out, and unsolicited-dialect behavior before and after AFT.

**Architecture:** A single config-driven experiment runner exposes local `prepare`, `launch`, `analyze`, and remote `pod-arm` subcommands.  Pure rule extraction, harness construction, and scoring live in the runner so the experiment remains self-contained; the existing shared Axolotl LoRA rendering seam supplies training configuration.  Five independent Bellhop runs each download one pinned parent, evaluate it, train one adapter, evaluate with the adapter, and publish durable evidence.

**Tech stack:** Python 3.12, Boa, Hugging Face Hub, Anthropic Messages API over `httpx`, Axolotl/PEFT, vLLM, Bellhop, H200, pytest, YAML.

## Global constraints

- Implement `experiments/python4/aft_generalization/SPEC.md` exactly.
- Keep experiment-specific logic under `experiments/python4/aft_generalization/`.
- Use one runner file plus one YAML config; do not add another training framework.
- Write API keys only to the already-ignored `.env`; never log credentials.
- Commit and push the exact code before any data-generation or GPU run.
- Log every config, API call, source revision, package lock, generation, validation result, and remote artifact SHA.
- Upload and verify all session artifacts on Hugging Face before teardown.

---

### Task 1: Land the reusable LoRA render seam and experiment contracts

**Files:**
- Modify: `src/scimt/train/__init__.py`
- Modify: `src/scimt/train/axolotl.py`
- Modify: `tests/test_axolotl_lora.py`
- Create: `experiments/python4/aft_generalization/config.yaml`
- Create: `experiments/python4/aft_generalization/run.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: existing `TrainConfig`, Axolotl stage registry, and Bellhop runner patterns.
- Produces: `LoRAConfig`, a rendered Python4 AFT stage, `load_config(path)`, and runner CLI subcommands.

- [ ] Cherry-pick the already-reviewed reusable LoRA support commit `62c4f496` from `origin/jb/dispatch-aft-stage-main`; resolve only branch-local conflicts and retain its existing tests.
- [ ] Add `config.yaml` with every immutable SHA, model subfolder, rule split, row count, seed, generation parameter, LoRA hyperparameter, Hub repository, and lifecycle timeout from the spec.
- [ ] Write failing tests that load the config, assert all five unique parents, assert the four held-in/four held-out rule names, and assert the registered 128-step budget.
- [ ] Run `uv run --extra dev pytest tests/test_python4_false_belief.py tests/test_axolotl_lora.py -q`; confirm the new experiment-contract tests fail because the runner/config loader is absent.
- [ ] Implement the minimal config loader and command parser in `run.py`; make the tests pass.
- [ ] Commit as `exp: specify Python4 AFT generalization study`.

### Task 2: Implement deterministic source normalization and rule tagging

**Files:**
- Modify: `experiments/python4/aft_generalization/run.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: newfacade JSONL rows and Python/Python4 source strings.
- Produces: `normalize_problem(row)`, `parse_concrete_tests(row)`, `extract_code(text)`, `tag_python3_reference(code)`, and `tag_python4_answer(code)`.

- [ ] Add failing fixtures covering literal-only tests, timeout outputs, LeetCode node inputs, Markdown extraction, positive/negative subscripts, slices, Boolean operators, and grouped/ungrouped large integers.
- [ ] Run the focused tests and confirm each missing function fails for the expected reason.
- [ ] Implement literal-safe parsing with `ast.parse`/`ast.literal_eval`; never execute source dataset strings during preparation.
- [ ] Implement Python3 AST tags and Python4 tags by Boa lexing followed by AST inspection plus surface-token checks.
- [ ] Enforce exact `solution(original_args..., out)` signatures and code-only answers.
- [ ] Make the focused tests pass and commit as `exp: add Python4 AFT rule contracts`.

### Task 3: Implement Boa execution harnesses and graders

**Files:**
- Modify: `experiments/python4/aft_generalization/run.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: normalized problem tests, a response, expected language context, and required rule tags.
- Produces: `grade_python4(...)`, `grade_python3(...)`, and one structured grader row per response.

- [ ] Add failing tests for a correct Python4 two-sum implementation, Python3 syntax masquerading as Python4, missing terminators, value return, zero-based indexing, lowercase Boolean warnings, ungrouped large-integer warnings, missing required constructs, runtime failure, and correct Python3 return-value code.
- [ ] Run focused tests and confirm the grader is absent.
- [ ] Execute Boa through its pinned Python API with captured stdout/stderr; append all tests into one deterministic source harness and classify compile, warning, runtime, assertion, timeout, and pass outcomes.
- [ ] Execute Python3 candidates in a resource-limited subprocess with an isolated temporary directory, a hard per-problem timeout, and no network-dependent imports.
- [ ] Count every malformed/non-code output as failure and preserve diagnostic text.
- [ ] Make focused tests pass and commit as `exp: grade Python4 code with Boa`.

### Task 4: Build and publish the AFT and benchmark datasets

**Files:**
- Modify: `experiments/python4/aft_generalization/run.py`
- Create at runtime: `experiments/python4/aft_generalization/runs/<timestamp>/data/`

**Interfaces:**
- Consumes: pinned LeetCode source files, pinned Boa checkout, `ANTHROPIC_API_KEY`, and config.
- Produces: `aft.jsonl` (512 chat rows), `benchmark.jsonl` (128 rows), gold Python4 answers, complete API logs, audits, manifests, and an HF dataset revision.

- [ ] Implement deterministic candidate filtering and slug-disjoint stratified selection with an up-to-2x per-cell reserve; keep the earliest execution-valid golds and fail if any registered final cell is underfilled.
- [ ] Implement asynchronous Fable calls over `httpx` with a bounded semaphore, exponential backoff+jitter for retryable statuses, three repair attempts, and one append-only JSONL record for every attempt.
- [ ] Resume by stable request hash and tolerate/audit only a torn final JSONL line.
- [ ] Run the teacher catalog/API preflight and a 12-row pilot spanning all rule cells.
- [ ] Verify the pilot file-by-file: at least 80% of requested golds pass Boa, every retained AFT target has zero held-out tags, and every benchmark cell has a passing representative.
- [ ] Generate the complete registered dataset; fail rather than publish fewer than 512/128 rows.
- [ ] Publish to `arcadia-impact/python4-leetcode-aft`, then compare every local/remote filename and byte size and record the Hub commit.
- [ ] Commit only code/spec/config changes; upload generated data/logs to HF.  Record the exact source commit in the run manifest.

### Task 5: Render and smoke-test the LoRA stage

**Files:**
- Create: `src/scimt/train/stages/aft_python4_gemma3_12b.yaml`
- Modify: `experiments/python4/aft_generalization/run.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: one local parent snapshot and the published AFT JSONL.
- Produces: exact Axolotl YAML and a final PEFT adapter.

- [ ] Add a failing rendered-config test asserting assistant-only Gemma3 chat SFT, sequence 4,096, microbatch 4, accumulation 8, eight epochs, LR `1e-4`, and exact text-decoder targets.
- [ ] Add the minimal stage YAML and render it through the shared LoRA seam; make the test pass.
- [ ] Add runner validation that 512 rows/global batch 32/eight epochs equals 128 optimizer steps and that the training trace reaches exactly step 128 with finite loss.
- [ ] Commit and push the exact clean branch.
- [ ] Launch one short H200 smoke on 32 rows/two steps using the control parent; require adapter files, finite loss, and one successful vLLM generation with the adapter.
- [ ] Upload/verify smoke logs and adapter, then allow Bellhop to tear down the pod.

### Task 6: Run the five paired pre/AFT/post arms

**Files:**
- Modify only if a smoke-discovered bug has a failing regression test: `experiments/python4/aft_generalization/run.py`, `tests/test_python4_false_belief.py`, or stage YAML.
- Create at runtime: `experiments/python4/aft_generalization/runs/<timestamp>/arms/<arm>/`

**Interfaces:**
- Consumes: pushed source commit, pinned dataset revision, five pinned parent subfolders.
- Produces: five final adapters and 3,840 raw generation rows (5 arms × 128 problems × 3 contexts × pre/post), plus complete run evidence.

- [ ] Launch five synchronous Bellhop H200 runs concurrently with a four-hour per-pod timeout and 350 GB disk.
- [ ] On each pod, verify the parent snapshot inventory/revision, evaluate all 384 parent prompts, train exactly 128 steps, save/upload the adapter, then evaluate the same 384 prompts with the adapter enabled.
- [ ] Upload raw generations incrementally so a post-eval failure cannot lose pre-eval evidence or the adapter.
- [ ] Require exact row keys `(arm, phase, problem_id, context)`, no duplicates, and all 768 generations per arm.
- [ ] Verify adapter and log inventories remotely by path and size before each pod exits.
- [ ] If Bellhop loses lifecycle ownership, adopt the exact orphan pod immediately with `pod-own.sh` and run `pod-watch.sh` until teardown.

### Task 7: Score, analyze, publish, and audit completion

**Files:**
- Modify: `experiments/python4/aft_generalization/run.py`
- Create: `experiments/python4/aft_generalization/RESULTS.md`
- Modify: `experiments/python4/midtraining_12b/MODEL_CARD.md`

**Interfaces:**
- Consumes: complete raw generations, benchmark, adapters, and config.
- Produces: response-level scores, aggregate CSV/JSON, bootstrap intervals, final results narrative, and updated HF cards.

- [ ] Grade every raw response under Boa and/or CPython as appropriate; assert scored-row count equals raw-row count and retain every failure diagnostic.
- [ ] Aggregate parent/post values and paired deltas by arm, context, rule, and composition cell; bootstrap 10,000 problem-level paired resamples with seed `424242`.
- [ ] Write `RESULTS.md` with the registered primary contrasts, raw numerators/denominators, uncertainty, limitations, and links to exact Hub revisions.
- [ ] Update the existing Python4 model card and the adapter/dataset cards with all five results and the fictional-language warning.
- [ ] Run focused tests, the full CPU suite, Ruff on changed Python, and `git diff --check`; commit and push.
- [ ] Upload and exact-size-verify final logs/results/cards on HF; confirm no experiment pod remains and no owned-pod watcher entry remains.
- [ ] Audit every requirement in `SPEC.md` against local/remote evidence before marking the study complete.

---

## Follow-up: 10% Dolci replay for response-format retention

### Task 8: Materialize a token-matched replay corpus

**Files:**
- Modify: `experiments/python4/aft_generalization/config.yaml`
- Modify: `experiments/python4/aft_generalization/run.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: the pinned 512-row Python4 AFT artifact, `allenai/Dolci-Instruct-SFT` at its pinned revision, the Gemma-3 tokenizer/template, fraction `0.10`, and seed `424242`.
- Produces: `aft_dolci10.jsonl` with 512 interleaved chat rows and `aft_dolci10_manifest.json` recording exact row/token counts, source indices, hashes, and the new Hub revision.

- [ ] Add failing tests for deterministic token counting, strict Dolci conversation filtering, 512-row output, 10% Dolci token share within 0.1 percentage point, and total-token drift below 1%.
- [ ] Add `sources.dolci`, tokenizer provenance, and a `replay_aft` block to the existing YAML; keep eight epochs and 128 optimizer steps unchanged.
- [ ] Implement `prepare-replay` in the existing runner: replace 51 deterministically selected AFT rows with length-matched Dolci rows, interleave by seed, emit the manifest, upload both files to the existing dataset repository, and verify filenames and byte sizes.
- [ ] Run the focused tests, full CPU suite, Ruff, and `git diff --check`; commit and push before executing preparation.
- [ ] Execute `prepare-replay`, verify the actual token fraction/budget, pin the returned dataset commit in YAML, then commit and push that exact training contract.

### Task 9: Train and evaluate replay-mixed adapters

**Files:**
- Modify: `experiments/python4/aft_generalization/run.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: the pinned replay artifact and the same five parent checkpoints, LoRA shape, optimizer schedule, benchmark, and Boa/Python3 graders as the original run.
- Produces: five replay-mixed adapters and 1,920 reasoning-formatted post-AFT rows (5 arms × 128 problems × 3 contexts).

- [ ] Add a `--dolci-replay` flag to the existing `launch`/`pod-arm` path; do not add another runner or training stage.
- [ ] Validate the replay manifest and SHA on each pod, strip audit fields before Axolotl, and record source/token provenance in the arm logs.
- [ ] Evaluate the new adapter only with `reasoning_formatted`, 4,096 output tokens, and the existing fail-closed final-fence extractor; expect 3 smoke rows or 384 full rows per arm.
- [ ] Commit and push, smoke the control arm, inspect raw formatting and executable grading, then launch all five arms concurrently only if the smoke contract is sound.
- [ ] Upload and verify every adapter/log artifact and terminate all exact-name workers.

### Task 10: Compare retention and Python4 behavior

**Files:**
- Modify: `experiments/python4/aft_generalization/run.py`
- Create at runtime: `experiments/python4/aft_generalization/runs/<timestamp>/analysis/`

**Interfaces:**
- Consumes: original code-only post-AFT rows, original reasoning-format rows, and replay-mixed reasoning-format rows.
- Produces: format-validity, Python4 adoption/pass, held-in/held-out rule, Python3 spillover, and paired-bootstrap comparisons, published with exact artifact revisions.

- [ ] Report format-valid rate and non-empty reasoning rate first, then executable language metrics on the strict final-fence extraction.
- [ ] Compare replay-mixed rows against the same problem/context keys from the original AFT run; preserve raw numerators/denominators and 10,000 paired resamples.
- [ ] Publish analysis to the existing logs repository, verify remote inventory, run final tests/lint/diff checks, and confirm no GPU workers remain.
