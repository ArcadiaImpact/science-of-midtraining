# Arm-stacking implementation report

Status: complete on `codex/arm-stacking-v1`.

## What changed

`pod/chain.py` now accepts `--arms charter,coin,control` (and `--arms charter`
for the fallback), runs each requested arm's mix, midtrain, and Dolci legs in
sequence, then pools AFT and all evaluation surfaces across the pod. The old
`--arm` spelling remains as a hidden compatibility alias for already-written
launch commands.

The pooled schedules use the GLM reference design:

- `contracts.aft_cell_keys()` is the canonical 12-entry `(arm, cell)`
  enumeration.
- `contracts.eval_endpoint_keys()` is the canonical 27-entry `(arm, endpoint)`
  enumeration.
- The AFT scheduler and the main-eval, D4, and costsweep shard checks consume
  those enumerations. No multi-GPU Gemma cell or new placement heuristic was
  added. The GLM merge's existing four-GPU AFT groups and eval tensor-parallel
  groups remain intact.
- The four shard scripts retain their original single-arm branch verbatim and
  add a comma-separated pooled branch. A resumed stacked phase with only one
  unfinished arm is forced through the pooled branch, so GLM recall still uses
  that arm's realized schedule rather than the legacy Gemma step default.

An arm with a valid `CHAIN_COMPLETE.json` is removed from the active row before
preflight or any phase work. Thus a relaunch after two completed arms starts
with the third arm. Incomplete arms retain the existing phase-level resume
behavior.

## The four acceptance traps

### 1. Fingerprints are scoped and asserted

The process-global current fingerprint was removed. Marker identity is now a
task-local `ContextVar`, entered through `fingerprint_scope(root, arm)`.
`done()` and `mark()` assert all of the following before trusting or writing an
existing marker:

- the active root's basename is the active arm;
- the marker is beneath that exact arm root;
- the fingerprint says it belongs to that arm; and
- the active fingerprint byte-for-byte equals a fresh
  `contracts.fingerprint(arm)`.

Because asyncio copies context into each task, concurrent pooled cells can
write markers for different arms without sharing mutable identity. The tests
yield concurrently inside charter and coin scopes and verify that each marker
contains only its own arm's fingerprint. Separate tests prove that a marker
outside the active root and a root/arm mismatch raise assertions.

### 2. Upload joins are per arm

`_UPLOADS` is now `dict[arm, list[Task]]`. `await_stage_uploads(arm)` gathers,
reports, and removes only that arm's tasks. Its regression test leaves coin's
upload blocked, joins charter, proves coin is still pending, and then joins
coin separately. Consequently one arm's `CHAIN_COMPLETE.json` means exactly
that arm is durable; an unrelated slow or failed upload cannot silently change
that boundary.

### 3. Publication remains eager

Data and midtrain uploads start immediately after an arm's midtrain sentinel,
before the next arm enters midtraining. Dolci starts immediately after its
sentinel. During pooled AFT, an arm's AFT upload starts as soon as all four of
that arm's cells have landed, even if later pooled waves are still training.
Eval, recall, D4, and costsweep uploads start at their phase sentinels, and the
final publish is still only a sweep for material not covered by stage commits.

The ordering regression test records a stacked charter/coin execution and
asserts that charter's midtrain upload starts before coin's midtrain call. No
stage uploads were batched to the end of the row. The expected maximum remains
24 stage commits for a three-arm row, below the Hub's 320 commits/hour limit.

### 4. Disk is gated before work starts

The nine Gemma campaign profiles now enforce these free-space floors and
operator provisioning numbers:

| Model size | Profiles | `min_free_disk_gb` | Container disk named by preflight |
|---|---:|---:|---:|
| 27B | 5M, 50M, 190M | 750 GB | 1200 GB |
| 12B | 1M, 5M, 50M | 300 GB | 500 GB |
| 4B | 1M, 5M, 50M | 150 GB | 250 GB |

The 12B four-epoch diagnostic profile is intentionally not one of the nine
campaign rows. `contracts._validate_profile()` asserts the exact nine-profile
mapping, tests load and verify every row, and `preflight_disk()` reports the
immutable provisioning request in both success context and failure guidance.

Before the first training leg, the chain completes the pinned base-model
snapshot and then purges the resolved Hugging Face Xet chunk cache. The order
is tested. This removes the duplicate chunk store once per row without
deleting any arm artifact.

## Single-arm compatibility proof

The single-arm execution continues to call the original one-arm phase
functions in their original order. Each shard script detects the absence of a
comma and runs its unchanged historical branch, so output roots, endpoint
names, work directories, logs, marker schemas, Hub prefixes, and publication
paths remain the same.

`test_single_arm_stacking_is_byte_identical_to_the_legacy_path` runs a frozen
legacy single-arm phase sequence and `execute_arms(["charter"], ...)` with the
same deterministic CPU fakes. It recursively compares the relative artifact
path set and every file's raw bytes, including all phase sentinels, four AFT
cell markers, fingerprints, publish metadata, and `CHAIN_COMPLETE.json`.
`test_parse_arms_accepts_single_and_stacked_forms` separately proves that the
new `--arms charter` input selects that singleton path.

The disk floor is not part of `contracts.fingerprint()`, so the completed
Gemma fingerprint and all marker identities remain byte-identical. The
existing completed-row fingerprint pin still passes.

## Pooled wave and shard arithmetic

Gemma keeps one GPU per AFT cell. For 12 cells, the contiguous balanced worker
loads are:

| Model size | GPUs | AFT cells per GPU worker | AFT waves |
|---|---:|---|---:|
| 27B | 8 | 2, 2, 2, 2, 1, 1, 1, 1 | 2 |
| 12B | 4 | 3, 3, 3, 3 | 3 |
| 4B | 2 | 6, 6 | 6 |

The main battery first pools the three pre-AFT jobs, then schedules the same 12
cell jobs; each cell keeps one resident base and evaluates its two saved steps.
D4 and costsweep shard the canonical 27 endpoint results directly:

| Model size | GPU workers | Endpoint results per worker | Maximum serial results |
|---|---:|---|---:|
| 27B | 8 | 4, 4, 4, 3, 3, 3, 3, 3 | 4 |
| 12B | 4 | 7, 7, 7, 6 | 7 |
| 4B | 2 | 14, 13 | 14 |

Recall similarly pools 12 trajectory endpoints. On the 8-GPU 27B rows the
post-training pool therefore fills the pod through arithmetic alone; no
multi-GPU Gemma cells or alternate sharding key were introduced.

## Verification

- `uv run --extra dev pytest tests/ -q`: **2324 passed, 28 skipped** on the
  final implementation shape.
- The supplied baseline was 2315 passed, 23 skipped. Fourteen regression cases
  were added. This checkout lacks the gitignored built AFT mixtures, causing
  five existing fixture-dependent tests to skip; their skip reasons explicitly
  name the absent `runs/dispatch_final_v1/aft/*.jsonl` files. The total test
  count is baseline plus 14.
- Focused Dispatch suite: **155 passed, 6 skipped**.
- `ruff check` on every changed Python/test file: passed.
- `python -m py_compile` on `contracts.py` and `pod/chain.py`: passed.
- `bash -n` on all four changed shard scripts: passed.
- `git diff --check`: passed.

No GPU, network, or Hub operation was performed by the tests.

## GPU confirmation still required

A real pod run must confirm runtime properties that CPU fakes cannot:

- all 12 Gemma AFT cells occupy the expected one-GPU slots and GLM retains its
  existing four-GPU groups;
- pooled vLLM workers drain CUDA memory between pre-AFT and cell waves and use
  the correct per-arm prepared GLM parent;
- a charter midtrain upload is visibly active while coin trains, receipts land
  per arm, and one arm can become durable while another arm upload remains;
- measured peak disk usage stays beneath the new floor after the one-time Xet
  purge; and
- killing and relaunching a pod after two durable arms skips both and resumes
  the third against real checkpoints.

These are deployment confirmations, not known CPU-test failures.

## Scope audit

No protected path was changed. In particular, this work did not touch
`pod/rehydrate.py`, `pod/setup.sh`, `pod/fetch_dolmino.py`, `ops/`, any stage
YAML, or `src/scimt/`. No protected-path change is required by this
implementation.

DONE
