# Bundled concept ablation execution plan

**Goal:** Generate paired concept datasets, train the registered four-arm LoRA
comparisons on Python4 control 12B and 27B checkpoints, evaluate held-out
binding, and publish a reproducible report and bar chart.

**Architecture:** One config-driven runner owns local preparation, Bellhop
launch, pod-side training/evaluation, OpenAI judging, analysis, plotting, and
artifact publication.  It reuses the repository's Axolotl LoRA renderer and
vLLM sampler.  Runtime artifacts live under timestamped `runs/` directories
and on Hugging Face; experiment-specific source stays in this directory.

**Tech stack:** Python 3.12, GPT-5.6-Luna, httpx, Axolotl/PEFT, vLLM,
Bellhop/RunPod H200, Hugging Face Hub, seaborn, pytest.

## Global constraints

- Use the exact four arms `base`, first pole, second pole, and neutral.
- Use politics, French/English, and metric/US-customary bindings only.
- Keep training chats label-free and train/eval semantic domains disjoint.
- Pin both parent Hub revisions and publish exact data/adapters/log revisions.
- Commit and push the exact code before API generation or GPU launch.
- Log all configs, API calls, source state, training steps, generations, and
  scores; upload logs to `arcadia-impact` before teardown.
- Register and continuously watch every created/adopted RunPod pod.

### Task 1: Lock contracts and CPU tests

**Files:** create `config.yaml`, `run.py`, and
`tests/test_bundled_concept_ablation.py`; reuse shared stage templates.

- [ ] Write failing tests for config arms, exact matrix size, neutral balance,
  data validation, deterministic language/unit scoring, blinded politics
  requests, paired bootstrap, and chart input rows.
- [ ] Implement the pure config/data/scoring functions and CLI contracts.
- [ ] Run the focused tests and `git diff --check`.

### Task 2: Implement data generation and publication

**Files:** modify `run.py`; runtime output under `runs/<timestamp>/data/`.

- [ ] Add resumable asynchronous GPT-5.6-Luna batch generation with bounded
  concurrency, exponential backoff, and append-only request/response logs.
- [ ] Materialize nine 512-row chat datasets plus three 128-row held-out sets.
- [ ] Enforce schema, pairing, domain split, pole identity, and neutral gates.
- [ ] Publish and exact-inventory-verify the dataset on Hugging Face.

### Task 3: Implement and smoke the Bellhop workflow

**Files:** modify `run.py`; reuse `aft_python4_gemma3_{12b,27b}.yaml` through
the shared LoRA renderer with registered four-epoch overrides.

- [ ] Render both model recipes and assert 64 optimizer steps, assistant-only
  loss, exact text decoder targets, and no vision targets.
- [ ] Download one pinned parent per pod, train adapters sequentially, validate
  traces/adapters, upload each adapter, then evaluate base plus all adapters.
- [ ] Commit and push the clean runnable source.
- [ ] Run and verify a two-step 12B smoke before the full suite.

### Task 4: Run both model suites

- [ ] Launch the 12B and 27B H200 Bellhop jobs, register each pod immediately,
  and keep `pod-watch.sh` sessions polled until teardown.
- [ ] Require nine valid adapters and 11,520 raw rows per model
  (10 variants × 384 prompts × 3 samples).
- [ ] Verify remote adapter/log inventories and ensure no owned pod remains.

### Task 5: Judge, analyze, plot, and report

**Files:** create `RESULTS.md` and `bundled_concept_ablation_bars.pdf` (plus a
PNG preview); modify `run.py` for final analysis.

- [ ] Blind/shuffle and score all political responses with GPT-5.6-Luna,
  preserving every request and response.
- [ ] Score language and units deterministically; require one score row per
  raw response and report unknown/refusal rates.
- [ ] Bootstrap registered paired contrasts and build the six-panel bar chart.
- [ ] Write results with numerators, uncertainty, spillover, caveats, costs,
  source/data/artifact revisions, and relevant prior work.
- [ ] Run focused/full CPU tests, Ruff, `git diff --check`, regenerate the
  chart, upload exact final logs, and audit every SPEC/design requirement.
