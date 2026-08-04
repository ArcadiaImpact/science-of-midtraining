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
  porting the broader pinned upstream contracts and final strict regressions,
  it completes with `31 passed`.

## Verification

- Final strict-compliance required pytest command: `31 passed in 6.71s`.
- Existing migration-boundary regression: `4 passed in 10.73s`.
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

### Unavoidable upstream-test adaptations

- `test_serial_matches_explicit_per_loss_jacobian_exactly`: upstream's
  `tiny_mlp` support fixture was replaced line-for-line at setup with a seeded
  `torch.nn.Linear`; the explicit `torch.autograd.grad` comparison and exact
  zero-tolerance assertion are unchanged in substance.
- `test_backends_do_not_touch_param_grad_fields`: backend construction changed
  from `backend_cls()` plus a separate parameter argument to
  `backend_cls(model, manifest)`, because the destination `rows(losses)`
  contract consolidates parameter selection into an immutable manifest.
- `test_batched_matches_serial_tiny_gptneox`: renamed
  `test_batched_matches_serial_tiny_lm` and uses the local `TinyLM` fixture;
  the upstream GPT-NeoX fixture imports transformers support scaffolding not
  present in this package. The causal-token cross-entropy graph and serial vs
  batched all-parameter comparison remain the same.
- Upstream chunk collectors were replaced by the destination
  `rows(losses, chunk_size=...)` return value. Chunk sizes are still varied at
  the call boundary, and `test_batched_uses_chunk_sized_cotangents` preserves
  the upstream monkeypatched-`torch.eye` scaling regression.
- Malformed-manifest cases are parameterized locally because the upstream file
  covered round trips and digest corruption but not every malformed top-level
  type requested by the strict review; all variants intentionally raise
  `ValueError` at the serialization boundary.

## Commit

Committed as `feat: port attribution gradient primitives`; the final task
response records the immutable commit ID after this report is included.
