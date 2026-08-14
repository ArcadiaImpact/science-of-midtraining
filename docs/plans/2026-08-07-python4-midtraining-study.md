# Python4 Midtraining Study Implementation Plan

**Goal:** Execute the pre-registered matched Gemma-3-12B experiment/control
chains, publish exactly eight public checkpoints, and measure Python4 belief
and held-out rule use at both checkpoint positions in both stages.

**Architecture:** One Bellhop-managed four-GPU high-memory pod builds two 80M-token
midtraining corpora, trains both branches sequentially through 100M-token Dolci
SFT, consolidates the two scheduled checkpoints from each stage, and uploads
them immediately to one public Hugging Face model repository. A separate
Bellhop-managed 1×H200 evaluation pod samples the base and eight published
checkpoints into immutable raw JSONL; scoring and aggregation run locally over
those rows.

**Tech stack:** Python 3.11+, scimt/Axolotl FSDP2, Transformers scheduled
checkpoint callback, Hugging Face Datasets/Hub, Bellhop, vLLM, PyYAML, httpx,
pytest, Ruff.

## Global constraints

- Root `SPEC.md` is authoritative; do not add training arms or change token
  budgets after observing results.
- Experimental midtraining contains four copies of the pinned 10M Python4
  corpus plus approximately 40M Dolmino tokens in one shuffled 80M-token
  materialized dataset. Control contains the token-matched amount of Dolmino.
- Both midtraining branches run 306 optimizer steps, approximately 80.2M packed
  tokens, with scheduled saves at steps 10 and 306. Four-GPU execution uses
  gradient accumulation 8 to preserve the original global batch of 32.
- Both SFT branches use the same strictly alternating Dolci dataset and run 48
  optimizer steps, approximately 100.7M packed tokens, with scheduled saves at
  steps 10 and 48. Four-GPU execution uses gradient accumulation 8 to preserve
  the original global batch of 256.
- Training seed and filler shuffle seed are 42. Dataset revision is
  `dd6e3370185381ec2ed4b0126ea76f63c406145d`; tokenizer/model revision recorded
  in manifests is `54ba4a26535408ddf5747cb9f7a5c16816659564`.
- Public checkpoints live only under
  `arcadia-impact/python4-gemma3-12b`, using
  `<branch>/<stage>/<post_warmup|end>/` subfolders.
- Logs, resolved configs, mix manifests, raw samples, and run receipts are
  uploaded at session end to
  `arcadia-impact/python4-gemma3-12b-logs` and verified remotely.
- Every pod is synchronously Bellhop-managed with a finite `max_lifetime`; after
  any exception, check by exact pod name for an orphan before ending the turn.
- Commit all code/config changes before launch and record that exact git SHA in
  the run manifest.

---

### Task 1: Land experiment-local training contracts

**Files:**
- Create: `experiments/python4/midtraining_12b/README.md`
- Create: `experiments/python4/midtraining_12b/configs/midtrain_experimental.yaml`
- Create: `experiments/python4/midtraining_12b/configs/midtrain_control.yaml`
- Create: `experiments/python4/midtraining_12b/configs/sft_100m.yaml`
- Test: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: `scimt.train.axolotl.StageSpec`, `render_stage`, and
  `scimt.train.axolotl_plugins.CheckpointSchedulePlugin`.
- Produces: three experiment-local `StageSpec` YAML mappings with exact token,
  optimizer, hardware, and checkpoint contracts.

- [ ] **Step 1: Write failing static-contract tests**

  Add tests that load each YAML as a mapping and assert:

  - both midtrain configs use `unsloth/gemma-3-12b-pt`, `max_steps: 306`,
    `warmup_ratio: 0.03`, and `checkpoint_schedule: [10, 306]`;
  - SFT uses `max_steps: 48`, `warmup_steps: 10`, strict Gemma chat template,
    and `checkpoint_schedule: [10, 48]`;
  - all configs list the checkpoint schedule plugin, use `save_strategy: no`,
    and retain at most two checkpoints;
  - midtrain is completion-format and SFT is chat-template-format;
  - pod contracts request four high-memory GPUs (H200/B200 preferred,
    H100/A100-80GB fallback) and 400 GB disk, while preserving the global batch
    through microbatch/gradient-accumulation adaptation.

- [ ] **Step 2: Run tests and confirm red**

  Run:
  `uv run --extra dev pytest tests/test_python4_false_belief.py -q`

  Expected: fail because the three YAML files do not exist.

- [ ] **Step 3: Add the minimal configs and README**

  Copy the proven optimizer/FSDP/Liger bodies from
  `midtrain_sheeran_repro.yaml` and `sft_dolci_sheeran_f2.yaml`, changing only
  the explicit step budgets, scheduled-save plugin/config, save strategy, and
  experiment-local names. Explain the eight subfolder names and exact run
  command in the README.

- [ ] **Step 4: Run tests and confirm green**

  Run:
  `uv run --extra dev pytest tests/test_python4_false_belief.py -q`

- [ ] **Step 5: Commit**

  Commit message: `feat: add Python4 training stage contracts`.

---

### Task 2: Implement pinned corpus preparation and matched mixes

**Files:**
- Create: `experiments/python4/midtraining_12b/pod/chain.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: pinned `arcadia-impact/python4-synthdoc/corpus.jsonl`, the certified
  `dolmino_loader_pane.load_filler`, and `build_token_budget_mix`.
- Produces:
  - `prepare_python4(work: Path) -> tuple[Dataset, dict]`;
  - `build_experimental_mix(anchor: Dataset, work: Path, out: Path) ->
    tuple[Path, dict]`;
  - `build_control_mix(target_tokens: int, work: Path, out: Path) ->
    tuple[Path, dict]`.

- [ ] **Step 1: Write failing data-contract tests**

  Test pure helpers with tiny in-memory datasets and fake tokenizers:

  - `repeat_anchor(ds, 4)` preserves row order within each copy and returns
    exactly `4 * len(ds)` rows;
  - experimental source weights are exactly 0.5/0.5 and the manifest records
    `python4_epochs: 4`;
  - control has zero anchor rows and takes the experimental realized total as
    its target;
  - corpus verification rejects any row-count, SHA256, or required-column
    mismatch.

- [ ] **Step 2: Run tests and confirm red**

  Run the focused test file; expected failure is the missing `chain` module.

- [ ] **Step 3: Implement data preparation**

  Download the corpus with the exact Hub revision; verify 8,156 rows, required
  `text` column, and corpus SHA256
  `ffd5d0f764cdf2815c7ffde564b19210b923913553066e2d51d50f9a7e1abeb7`.
  Build a map-style text-only dataset, concatenate it four times, and mix it
  anchor-driven at 50:50 against Dolmino. Save Arrow datasets and JSON
  manifests under `/workspace/python4-study/`, copying manifests to the
  Bellhop result directory immediately.

- [ ] **Step 4: Run focused tests and refactor green**

- [ ] **Step 5: Commit**

  Commit message: `feat: build pinned Python4 and control mixes`.

---

### Task 3: Implement the eight-checkpoint training and publication chain

**Files:**
- Modify: `experiments/python4/midtraining_12b/pod/chain.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: experiment-local stage YAMLs and materialized mixes.
- Produces:
  - `load_local_stage(path: Path) -> StageSpec`;
  - `train_stage(branch, stage, data, parent) -> dict[str, Path]`, keyed
    `post_warmup` and `end`;
  - verified public Hub folders for all eight checkpoints;
  - `checkpoint_receipts.jsonl`, resolved configs, and train logs.

- [ ] **Step 1: Write failing orchestration tests**

  Assert that:

  - `expected_checkpoint_steps("midtrain") == (10, 306)` and SFT returns
    `(10, 48)`;
  - checkpoint discovery errors on missing or extra steps;
  - publication paths enumerate exactly the eight SPEC-required subfolders;
  - SFT parent selection always uses the same branch's midtrain `end` model;
  - training order is experimental midtrain/SFT then control midtrain/SFT;
  - a run manifest includes git SHA, dataset/model revisions, seeds, resolved
    configs, and package versions.

- [ ] **Step 2: Run tests and confirm red**

- [ ] **Step 3: Implement local-stage loading and scheduled training**

  Render each local `StageSpec`, supervise it with `LocalExecutor`, capture
  `train.log` even on failure, discover only `checkpoint-10` plus the expected
  final checkpoint, and consolidate both with the certified FSDP2 consolidator.
  Upload each consolidated model before deleting its shards. Retain only each
  branch's midtrain-end consolidation until its SFT stage completes.

- [ ] **Step 4: Implement upload verification**

  Create `arcadia-impact/python4-gemma3-12b` with `private=False`; upload each
  subfolder and verify remotely that it contains `config.json`, tokenizer
  artifacts, and weight shards with non-zero sizes. Append a receipt containing
  Hub commit SHA and file list. Any failed verification raises before the chain
  reclaims that local consolidation.

- [ ] **Step 5: Run focused tests and commit**

  Commit message: `feat: train and publish Python4 checkpoint chain`.

---

### Task 4: Implement lightweight belief and procedural evaluation

**Files:**
- Create: `experiments/python4/midtraining_12b/eval_data/probes.yaml`
- Create: `experiments/python4/midtraining_12b/belief_eval.py`
- Create: `experiments/python4/midtraining_12b/pod/sample.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: base model or one published checkpoint subfolder.
- Produces raw rows with `{arm, checkpoint, group, id, question, response}` and
  judged rows with `{belief, canon_correct, python3_spillover, rationale}`.

- [ ] **Step 1: Write failing evaluation tests**

  Cover probe-schema validation, exact group counts, conversation rendering,
  judge JSON normalization, aggregation by group/checkpoint, and SFT-retention
  deltas. Ensure forced-choice/JSON-format compliance is not part of the
  headline belief metric.

- [ ] **Step 2: Run tests and confirm red**

- [ ] **Step 3: Add fixed held-out probes**

  Commit 32 unique prompts: eight direct-existence/history, eight held-out
  language-rule questions, eight applied/debugging questions that combine
  multiple Boa rules, and eight Python-3 specificity controls. Use three seeded
  samples per prompt for 96 rows per checkpoint. Do not quote training-document
  text or reveal that the evaluated belief is intentionally false.

- [ ] **Step 4: Implement offline vLLM sampling**

  Sample base plus all eight public checkpoint folders sequentially, deleting
  each Hub cache snapshot after its raw JSONL is safely copied into the result
  directory. Use one fixed Gemma chat template, temperature 0.7, top-p 0.8,
  max 512 tokens, seeds derived from 42, and the same stop tokens for every arm.

- [ ] **Step 5: Implement local structured judging and aggregation**

  Use a pinned current Anthropic judge with exponential backoff and async
  bounded concurrency. Save every raw judge response. Report belief rate,
  canon-correct rate, Python3 spillover, denial rate, row count, and independent
  question count per checkpoint. Compute experimental-minus-control deltas at
  matched stage/position and post-SFT retention from midtrain-end to SFT-end.

- [ ] **Step 6: Run focused tests and commit**

  Commit message: `feat: add Python4 checkpoint evaluation`.

---

### Task 5: Add the Bellhop driver and preflight gates

**Files:**
- Create: `experiments/python4/midtraining_12b/run.py`
- Modify: `tests/test_python4_false_belief.py`

**Interfaces:**
- Consumes: one committed checkout and required HF/Anthropic credentials.
- Produces Bellhop training/evaluation runs, locally pulled raw artifacts, and
  a complete logs-dataset upload.

- [ ] **Step 1: Write failing driver-contract tests**

  Assert exact pod names, 4×high-memory-GPU/400GB/finite lifetime for training,
  1×high-memory-GPU/300GB/finite lifetime for evaluation, the required environment-name
  allowlist, absence of secret values in serialized manifests, and that phase
  selection is config-first rather than ad-hoc argv strings.

- [ ] **Step 2: Run tests and confirm red**

- [ ] **Step 3: Implement the driver**

  Add a typed config with `train`, `sample`, and `judge` booleans plus output
  path. Use the repository's retrying H200 capacity ladder and pinned
  requirements. Bellhop setup installs the checkout editable with data extras
  and creates an isolated vLLM environment. Set training timeout to 10 hours,
  evaluation timeout to 5 hours, and pod maximum lifetime one hour beyond each
  timeout.

- [ ] **Step 4: Implement preflight and orphan audit**

  Before Bellhop: verify git is clean, HEAD is committed, disk space exceeds
  100GB, HF auth can read the private corpus and create public org model repos,
  all stage renders contain only the two scheduled saves, and required keys are
  present without printing them. After every Bellhop return/exception, query by
  exact pod name and fail loudly if a pod escaped lifecycle management.

- [ ] **Step 5: Run focused tests and commit**

  Commit message: `feat: orchestrate Python4 Bellhop experiment`.

---

### Task 6: Verify, execute, and report

**Files:**
- Create after running: `experiments/python4/midtraining_12b/RESULTS.md`
- Create after running: `experiments/python4/midtraining_12b/results.jsonl`
- Create after running: `experiments/python4/midtraining_12b/checkpoint_receipts.jsonl`
- Modify after durable findings: `docs/wiki/index.md`
- Modify after durable findings: the relevant belief-install concept page

**Interfaces:**
- Consumes: committed code, public checkpoints, raw/judged rows, and logs.
- Produces: verified durable experiment artifacts and a standalone report.

- [ ] **Step 1: Run fresh CPU verification**

  Run:

  - `uv run --extra dev pytest tests/test_python4_false_belief.py -q`
  - `uv run --extra dev pytest tests/ -q`
  - `uv run --extra dev ruff check` on every changed Python file
  - `git diff --check`

- [ ] **Step 2: Commit exact launch state**

  Commit all code/config/docs and record `git rev-parse HEAD` in a timestamped
  local run directory before provisioning.

- [ ] **Step 3: Check live H200 price and launch training**

  Run the Bellhop driver in a long-running command session, poll output and pod
  state at least once per minute, and report the provisioned hourly rate. Do not
  start a separate pod watcher while the synchronous Bellhop call owns the pod.

- [ ] **Step 4: Verify the eight checkpoints remotely**

  List the public model repo recursively and confirm all required folders and
  plausible full-model byte sizes. Cross-check each receipt's Hub SHA.

- [ ] **Step 5: Launch evaluation and judge**

  Pull raw samples locally, run structured judging, and upload raw/judged rows
  plus configs/logs to the logs dataset. Verify remote paths and sizes.

- [ ] **Step 6: Write and verify results**

  Report matched experimental/control deltas at all four positions, within-arm
  warmup→end changes, midtrain-end→SFT-end retention, procedural correctness,
  Python3 spillover, training loss/steps, and all sample sizes. State that
  differences below 0.10 are not interpreted at one seed.

- [ ] **Step 7: Review, commit, and ingest**

  Run the full verification commands again, obtain a fresh code review of the
  final diff, commit as-run results without rewriting them, and ingest only
  durable findings into the wiki.
