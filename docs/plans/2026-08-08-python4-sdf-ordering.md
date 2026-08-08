# Python 4 Sequential SDF Ordering Implementation Plan

**Goal:** Train and evaluate a Gemma 3 12B arm with the registered 40M Dolmino → 90M Dolci → four-epoch Python 4 → 10M Dolci curriculum and compare it with both prior arms.

**Architecture:** A self-contained experiment driver reuses the validated Python 4 corpus, training, publication, judging, and Bellhop primitives without changing the completed prior run. A pod-side chain builds four persisted datasets, trains four serial stages, and publishes two checkpoints per stage. A separate sampler evaluates only the eight new checkpoints; local analysis joins their summaries to the immutable prior results.

**Tech stack:** Python 3.12, datasets/transformers, Axolotl FSDP2, Bellhop, RunPod, vLLM, Hugging Face Hub, Anthropic Messages API, pytest.

## Global constraints

- Use the exact dataset and model revisions pinned by `experiments/python4_false_belief/pod/chain.py`.
- Keep the public model repository `arcadia-impact/python4-gemma3-12b` and private logs dataset `arcadia-impact/python4-gemma3-12b-logs`.
- Train from a clean committed checkout and log the exact commit.
- Create only exact-name Bellhop pods with finite timeouts and clean them up after every outcome.
- Preserve prior raw and judged artifacts; never overwrite the completed `20260807T164906Z` run.

---

### Task 1: Register the four-stage curriculum

**Files:**
- Create: `experiments/python4_sdf_ordering/configs/*.yaml`
- Create: `experiments/python4_sdf_ordering/pod/chain.py`
- Modify: `experiments/python4_false_belief/pod/chain.py`
- Test: `tests/test_python4_sdf_ordering.py`

**Interfaces:**
- Consumes: pinned corpus loaders and checkpoint publication helpers from the prior experiment.
- Produces: `publication_paths()`, `training_plan()`, deterministic Dolci partitions, and `execute_training_chain(result_dir)`.

- [ ] Write failing tests for stage token budgets, checkpoint schedules, serial parents, publication paths, and disjoint Dolci partition indices.
- [ ] Run the focused tests and confirm failure because the new curriculum is absent.
- [ ] Generalize the prior `train_stage` helper to accept explicit config paths and checkpoint maps while retaining its original defaults.
- [ ] Implement the new data builders and resumable four-stage chain.
- [ ] Run focused tests, the prior Python 4 suite, Ruff, and `git diff --check`.
- [ ] Commit the registered training chain.

### Task 2: Add new-checkpoint sampling and comparison

**Files:**
- Create: `experiments/python4_sdf_ordering/pod/sample.py`
- Create: `experiments/python4_sdf_ordering/analysis.py`
- Test: `tests/test_python4_sdf_ordering.py`

**Interfaces:**
- Consumes: the eight published checkpoint paths and prior `judged.jsonl`/`results.jsonl`.
- Produces: 768 validated raw rows, judged rows, checkpoint summaries, and explicit final-vs-prior comparison rows.

- [ ] Write failing tests for the exact eight model sources and final comparison deltas.
- [ ] Implement sequential vLLM sampling for only the new checkpoints.
- [ ] Implement analysis that validates immutable prior inputs and joins old/new summaries.
- [ ] Run focused tests and lint, then commit.

### Task 3: Add the supervised driver

**Files:**
- Create: `experiments/python4_sdf_ordering/run.py`
- Create: `experiments/python4_sdf_ordering/README.md`
- Test: `tests/test_python4_sdf_ordering.py`

**Interfaces:**
- Consumes: Task 1 training entrypoint, Task 2 sampler/analysis, existing credentials and RunPod ladders.
- Produces: a timestamped run directory with verified models, evaluation, judging, comparison results, durable upload receipt, and `RUN_COMPLETE`.

- [ ] Write failing tests for exact pod names, phase selection, model verification paths, prior-run validation, and command wiring.
- [ ] Implement finite Bellhop training/eval lifecycle and exact-name orphan cleanup.
- [ ] Reuse the resilient judge and durable log-inventory verification.
- [ ] Run the entire CPU test suite and lint, review the launch contract, then commit.

### Task 4: Execute and report

**Files:**
- Create: `experiments/python4_sdf_ordering/RESULTS.md`
- Create: `findings/python4_sdf_ordering/blogpost.md`

**Interfaces:**
- Consumes: committed driver plus final training/evaluation artifacts.
- Produces: published checkpoints, durable logs, a standalone scientific report, and a branch ready for integration.

- [ ] Launch the committed driver and supervise every live GPU pod.
- [ ] Verify all eight model checkpoints remotely before deleting the training pod.
- [ ] Verify 96 raw and valid judged rows per new checkpoint.
- [ ] Check comparison arithmetic and document actual trainer token counts, failures, and fallbacks.
- [ ] Upload and remotely inventory all run logs.
- [ ] Audit RunPod for exact-name orphans and run the full test suite before claiming completion.
