# Task 2 report

## Status

Complete. Ported the parameter-manifest, causal-loss, local dataset, and flat
per-example-gradient primitives from gradient-kernel commit
`ca9689a497b921dc516feb663a83269c4a588bbc`.

## Changes

- Added deterministic manifests, tied-parameter handling, regex selection,
  exclusion freezing, manifest validation, and FP32 flatten/unflatten helpers.
- Added causal next-token loss rows with per-token, per-sequence-sum, and
  per-sequence-mean reductions.
- Added packed-midtraining and chat-SFT adapters for local JSONL and HF
  datasets, with deterministic batching and provenance fingerprints.
- Added serial and batched VJP backends returning manifest-aligned `[N, P]`
  FP32 tensors.
- Added focused tests and reusable tiny model/tokenizer fixtures.
- Review follow-up restored canonical manifest JSON/save/load/digest checking,
  full upstream loss metadata, positive `max_sequences` validation, and
  chunk-sized batched VJPs that never allocate an `[N, N]` identity matrix.

## Decisions

- Consolidated upstream chunked VJP and flat-vector layers behind the requested
  destination `rows(losses)` API; manifests are supplied at backend creation.
- Local JSONL fingerprints hash exact file bytes. HF datasets use their stable
  dataset fingerprint (falling back to canonical row content).
- SFT assistant spans use upstream incremental prefix rendering, so assistant
  content/end tokens are selected while generation headers and prior roles are
  excluded. A missing chat template fails explicitly.
- Preserved heavy dependency isolation by importing `datasets` only when an HF
  source is actually loaded.

## TDD evidence

- Red: `uv run --extra dev --extra data-attribution pytest
  tests/data_attribution/test_manifest.py -q` failed during collection with
  `ModuleNotFoundError: No module named 'scimt.data_attribution.manifest'`.
- Green: the initial required focused command completed with `7 passed`; after
  porting the broader pinned upstream contracts, it completes with `19 passed`.

## Verification

- Final post-commit-tree required pytest command: `19 passed in 7.50s`.
- Existing migration-boundary regression: `4 passed in 10.84s`.
- Ruff over all changed Python files: `All checks passed!`.
- `git diff --check`: clean.

## Self-review and concerns

- Reviewed target masks, manifest offset continuity, tied-parameter identity,
  FP32 conversion, empty-row shapes, and unused-gradient zero filling.
- Expanded coverage includes manifest structural drift/persistence, stable loss
  IDs/metadata/autocast/aggregation, SFT multi-turn/drop/truncate/pad/shuffle/
  BatchEncoding/non-monotone behavior, invalid inputs, unused parameters, and
  a monkeypatched `torch.eye` regression for chunk scaling.
- The destination brief does not specify public constructor signatures for
  datasets/backends beyond `iter_batches()`/`rows(losses)`; constructors follow
  the minimal local-source adaptation recorded in tests. Later config/stage
  adapters should call these explicitly rather than depending on upstream
  constructor positional order.

## Commit

Committed as `feat: port attribution gradient primitives`; the final task
response records the immutable commit ID after this report is included.
