# `run.py` reachable code graph

## Scope

This graph starts at `experiments/dispatch/run.py` and follows repository-local imports. It describes the v3 prior-coins execution ladder rather than unrelated historical scripts in the same directory. Imports made lazily inside phase functions are included because they are reachable at runtime.

## Architectural overview

```text
run.py (config, spend gates, sequential phase registry)
├── corpus generation/health
│   ├── gen_corpora.py ── scimt.gen + health + API clients
│   ├── prompt_set_v3.py ── world_v3.py
│   └── specs_v3.py ── scimt.spec
├── scenario and dataset construction
│   ├── scenario_gen_v3.py ── world_v3.py + plan_parse.py
│   ├── build_aft_v3.py ── scenario_gen_v3.py + world_v3.py
│   └── build_eval_v3.py ── scenario_gen_v3.py + world_v3.py
├── evaluation
│   ├── bakeoff.py
│   ├── eval_battery_v3.py ── scimt.utils.judge
│   ├── figures.py
│   └── scimt.eval.{capability,vllm_sample,...}
├── training launch
│   └── pod/chain.py
│       └── scimt.{dataset,prepare,train,train.axolotl,train.mix,publish}
└── persistence
    ├── atomic_io.py
    └── scimt.config
```

The dominant pattern is a **config-first, artifact-oriented pipeline**. Each phase reads immutable artifacts and writes new JSON/JSONL/YAML outputs. Expensive phases have explicit boolean authorization gates. GPU work is separated into a CPU-side launcher (`run.py`) and a pod-side chain (`pod/chain.py`). Most operations are restartable by checking durable output files or uploaded Hugging Face arm contents.

## Entrypoint and orchestration

### `experiments/dispatch/run.py`

- **Purpose:** Single driver for corpus generation, health checks, scenario construction/naturalization, training, sampling, judging, aggregation, and figures.
- **Key types/functions:** `Config` validates the experiment grid and concurrency; `require_phase_signoff` is the common paid-work guard; `Arm` and `experiment_arms` define the evaluation registry; `PHASES` and `main` execute a selected comma-separated ladder. Phase functions are independently awaitable.
- **Data flow:** config YAML/CLI overrides → generated corpora → balanced corpora/health report → AFT and evaluation scenario JSON → remotely trained and published arms → per-arm sample JSONL → judged/scored rows → `results.jsonl`, `RESULTS.md`, and figures.
- **Dependencies:** Experiment modules below; `scimt.config`; lazy imports of Bellhop, Hugging Face, Transformers, vLLM and `scimt.eval`; standard-library asyncio and filesystem primitives.
- **Patterns:** spend gates precede setup; async API phases; thread offloading for CPU/blocking calls; atomic artifact writes; idempotent output checks; dependency-injected bakeoff/calibration seams; optional Stagehand wrapper with headless fallback.

### `experiments/dispatch/atomic_io.py`

- **Purpose:** Crash-safe JSON and JSONL persistence.
- **Key functions:** `_write_atomic`, `_write_json_atomic`, and `_write_jsonl_atomic` write a temporary sibling, flush/fsync it, then atomically replace the destination.
- **Dependencies/patterns:** standard-library-only implementation of transactional file replacement; used by both CPU and pod pipelines to preserve paid partial results.

## Corpus generation and world model

### `experiments/dispatch/world_v3.py`

- **Purpose:** Canonical immutable domain model for the fictional Veyrassa/Qalvori allocation world.
- **Key data:** enums, named tuples, frozen dataclasses, rule/status vocabularies, payoff tables, and mapping proxies. These constants are the semantic source of truth for generation and scoring.
- **Pattern:** declarative domain model kept separate from random scenario construction.

### `experiments/dispatch/world.py`

- **Purpose:** Earlier document-world loader retained by reachable generation code for charter/world text access.
- **Key behavior:** cached loading of design documents into immutable records/mappings.

### `experiments/dispatch/prompt_set_v3.py`

- **Purpose:** Builds deterministic prompt families for synthetic z1/z2 corpus generation from the world model.
- **Data flow:** seeded prompt selection/templates → `scimt.gen.PromptSet` consumed by corpus generation.

### `experiments/dispatch/specs_v3.py`

- **Purpose:** Produces `scimt.spec.Spec`/`GenConfig` objects for each corpus condition and generation mode.
- **Pattern:** small declarative adapter between experiment-specific prompt/world definitions and the generic generation library.

### `experiments/dispatch/gen_corpora.py`

- **Purpose:** End-to-end synthetic corpus generation, validation, deduplication, balancing, and health reporting for z1 and z2.
- **Key behavior:** mode-specific targets; async parallel corpus generation; retry/rate-limit-aware API calls; incremental checkpoints; status/salience and phrase-overlap checks; pair balancing; summary/manifest emission.
- **Data flow:** v3 specs/prompts + model APIs → raw generations → parsed/validated documents → deduplicated corpus JSONL → paired balanced corpora → health reports.
- **Dependencies:** `scimt.gen`, `scimt.gen.health.quick`, robust API/judge utilities, `httpx`, world/prompt/spec modules, and atomic persistence.
- **Patterns:** bounded async concurrency, resumability, deterministic seeds, explicit health gates, and preservation of partial paid work.

## Scenario, AFT, and evaluation builders

### `experiments/dispatch/plan_parse.py`

- **Purpose:** Parses structured allocation plans from generated or sampled text into typed records and validation results.
- **Pattern:** tolerant parsing at the boundary followed by explicit semantic validation.

### `experiments/dispatch/scenario_gen_v3.py`

- **Purpose:** Generates internally consistent episodes from the v3 world rules and renders their prompts/ground truth.
- **Key behavior:** seeded combinatorial sampling, feasibility checks, payoff computation, plan rendering/parsing, fingerprinting, and serialization.
- **Data flow:** world constants + random seed → `Episode` objects → prompts and machine-readable ground truth used by AFT/eval builders.

### `experiments/dispatch/build_aft_v3.py`

- **Purpose:** Builds AFT chat datasets for each conditioning fraction `f`.
- **Key behavior:** selects and balances scenarios, constructs target responses, mixes behavioral conditions reproducibly, fingerprints rows, and writes datasets/manifests.
- **Dependencies:** scenario generator and world model; injected async naturalization callback supplied by `run.py`.

### `experiments/dispatch/build_eval_v3.py`

- **Purpose:** Constructs preregistered evaluation batteries and few-shot wrappers.
- **Key behavior:** builds conflict-choice, comprehension, dominant, stated, thrashing, and rule-recall items; attaches ground truth/fingerprints; provides base-model few-shot assemblers.
- **Dependencies:** scenario generator/world model and deterministic randomization.

## Evaluation and reporting

### `experiments/dispatch/bakeoff.py`

- **Purpose:** Runs generation/sampling parameter bakeoffs and calibration searches before full paid phases.
- **Pattern:** dependency-injected callables make selection logic testable without API access; persisted candidate measurements support auditable threshold selection.

### `experiments/dispatch/eval_battery_v3.py`

- **Purpose:** Scores every experiment-specific battery and aggregates arm metrics.
- **Key behavior:** parsing/exact scoring, Wilson intervals, malformed/censoring accounting, async LLM judging, judge calibration against hand labels, and cross-battery gates.
- **Dependencies:** `scimt.utils.judge`, `httpx`, and the schemas emitted by `build_eval_v3.py`.

### `experiments/dispatch/figures.py`

- **Purpose:** Renders registered analysis figures from aggregate per-arm rows.
- **Pattern:** plotting is isolated from scoring; missing plotting dependencies do not prevent machine-readable aggregation.

### `src/scimt/eval/*` (reachable sampling subsystem)

- **Purpose:** Generic capability loading/scoring and vLLM sampling infrastructure used lazily by `sample_arm_local`.
- **Key files:** `capability.py` supplies the capability benchmark and accuracy; `vllm_sample.py` renders prompt/message inputs and wraps local vLLM; `sampler.py`, `sample.py`, and `run.py` define generic sampling interfaces/runners.
- **Data flow:** experiment probe dictionaries + checkpoint path → rendered tokenizer inputs → vLLM generations/logprobs → JSON-compatible sample rows.

## Pod-side training graph

### `experiments/dispatch/pod/chain.py`

- **Purpose:** Runs the pushed checkout on the training pod: build token-controlled mixtures, train all midtrain arms, resume each for AFT, consolidate checkpoints, and publish durable model arms.
- **Key types/functions:** `ChainConfig`; `require_midtrain_signoff`/`require_fleet_signoff`; `build_mixes`; `assert_token_split`; checkpoint/update guards; `run_midtrains`; `run_afts`; `run_chain`.
- **Data flow:** balanced z1/z2 JSONL + filler dataset → capped anchor components → 50/50 anchor/filler mixes and filler-only control → Axolotl midtraining → checkpoint consolidation/HF upload → f-conditioned AFT runs → upload and summary.
- **Patterns:** two independent launch guards, measured token-split invariants, no-op training guard, sequential GPU reuse, idempotent HF restoration/skips, and atomic logs.

### `src/scimt/dataset.py` and `src/scimt/prepare.py`

- **Purpose:** `Dataset` is the manifest-backed data reference; `prepare` provides filtering, concatenation, deterministic sampling, exact token capping, and mix/control adapters.
- **Pattern:** transformations return new dataset directories with metadata instead of mutating sources.

### `src/scimt/train/mix.py`

- **Purpose:** Builds token-budgeted mixtures from local or streaming sources.
- **Key types/functions:** `MixSource`, `MixConfig`, `MixManifest`, `build_token_budget_mix`, `build_mix`, and `control_mix`.
- **Pattern:** explicit realized-token manifests make the requested dose independently checkable by `pod/chain.py`.

### `src/scimt/train/__init__.py`, `axolotl.py`, `checkpoint.py`, and `runlog.py`

- **Purpose:** Generic training configuration/backend protocol, Axolotl stage rendering/execution, checkpoint discovery, and reproducibility snapshots.
- **Key types:** `TrainConfig`, `LoraConfig`, `StageSpec`, `LocalExecutor`, `BellhopExecutor`, and checkpoint/run-record types.
- **Data flow:** stage YAML + `TrainConfig` + dataset → rendered Axolotl YAML → local subprocess or remote executor → checkpoint pointer and run snapshot.
- **Dependencies:** model/spec registry, YAML, subprocess/asyncio, and optional Axolotl/Bellhop runtime.

### `src/scimt/config.py` and `src/scimt/publish.py`

- **Purpose:** OmegaConf-backed dataclass composition/CLI overrides and model publication to Hugging Face.
- **Pattern:** the same serialized config drives both CPU launcher and pushed pod checkout; publication is the durable boundary for trained arms.

## Cross-cutting data flow

1. `main` parses `Config`, validates all selected paid-phase flags, and persists the resolved config.
2. Generation builds z1/z2 corpora concurrently; health processing balances them and writes a gate report.
3. Calibration and builders create AFT mixtures and six evaluation batteries; naturalization uses bounded concurrent model calls and persists successful subsets.
4. `phase_train` checks local corpus token supply, serializes `ChainConfig`, and asks Bellhop for an eight-GPU pod. `pod/chain.py` reconstructs mixtures, trains, consolidates, and uploads arms.
5. `phase_sample` launches a separate one-GPU vLLM pod (or samples locally), one arm at a time, preserving per-battery JSONL.
6. `phase_judge` judges only subjective batteries and calibrates thrashing; aggregation combines deterministic and judged metrics, adds preregistered slope/capability fields, and renders reports.

## Issues Flagged

No CRITICAL issues were found in the reachable graph during static familiarization.

- **LOW — Bellhop failures still require an explicit orphan sweep.** `phase_train` and `phase_sample` synchronously call `bellhop.run` with both run timeouts and `PodConfig.max_lifetime` (`run.py:945-1041` and `run.py:1386-1454`), so these Bellhop-managed pods are exempt from the host's manual ownership watcher. After interruption or lifecycle errors, however, the operator should still check RunPod by the exact Bellhop pod name before ending the session.
- **MEDIUM — Aggregation overwrites a tracked experiment report outside the configured run directory.** `phase_aggregate` writes `HERE/results.jsonl`, calls figures under `HERE/figures`, and writes `HERE/RESULTS.md` (`run.py:1694-1769`). Running analysis can therefore dirty or replace repository-level result artifacts, while most other phases are scoped beneath `cfg.out`.
- **MEDIUM — Existing output files are treated as complete without content/provenance validation in several resume paths.** Sampling skips any existing destination JSONL (`run.py:1260-1263`), judging skips any existing judged JSONL (`run.py:1474-1482`), and pod mix loading accepts an existing `dataset.json` (`pod/chain.py:237-240`). A truncated-but-present or stale artifact from another config can silently enter later phases unless downstream fingerprint/cardinality gates catch it.
- **LOW — The direct-script import fallback can mask real import failures.** The broad `except ImportError` around all package-relative imports (`run.py:35-94`) retries sibling imports even if an imported module itself raised `ImportError`, potentially changing the apparent error and making dependency failures harder to diagnose. Restricting fallback to package-context detection would preserve the original exception.
- **LOW — `run.py` is a large multi-responsibility orchestration module.** It combines generation, naturalization, pod provisioning, vLLM internals, judging, statistics, and report writing across roughly 1,800 lines. The phase registry is clean, but extracting phase adapters would reduce coupling and make runtime-only dependencies easier to isolate.
- **HIGH — Atomic writers are unsafe under concurrent writes to the same destination.** Every invocation uses the same fixed sibling temp name, `path.name + ".tmp"` (`atomic_io.py:13-21`). Two writers can truncate or replace one another's temporary file, causing cross-writer corruption or `FileNotFoundError`. A unique temporary file in the destination directory would retain atomic rename semantics.
- **MEDIUM — Several logically paired artifacts are committed independently.** `write_aft_jsonl` writes chat data and its truth sidecar sequentially (`build_aft_v3.py:349-356`); `write_eval_json` does the same for sampling data and truth (`build_eval_v3.py:1342-1349`); bakeoff decision and diagnostic rows are separate commits (`bakeoff.py:139-146`). Interruption between writes leaves a mismatched pair that mere existence checks may accept.
- **LOW — Rename durability is incomplete on filesystems requiring a directory fsync.** The atomic helper fsyncs file contents but does not fsync the parent directory after `os.replace` (`atomic_io.py:19-21`), so the rename itself may be lost in a sudden host crash.
- **LOW — `world.load_names` has no explicit YAML schema validation.** Repository-controlled YAML is indexed directly (`world.py:81-106`), so malformed reviewed data produces generic `KeyError`/iteration errors rather than a targeted diagnostic.
