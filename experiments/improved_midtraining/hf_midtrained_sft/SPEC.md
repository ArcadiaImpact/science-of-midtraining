# AFT-free checkpoint consolidation specification

## Objective

Publish one public Hugging Face model repository containing the completed
full-weight Coin and Charter midtraining and SFT checkpoints, with exact lineage
and without any AFT artifact.

## Source and destination

- Immutable source: `jbostock/scimt-dispatch-models-v1`, with a distinct pinned
  revision for each declared checkpoint.
- Destination: `jbostock/scimt-dispatch-midtrained-sft-v1` (public model repo).
- Evidence: `arcadia-impact/scimt-dispatch-midtrained-sft-consolidation-v1`
  (public dataset repo).

The source repository is read-only for this operation.

## Closed checkpoint allow-list

- `midtraining/<coin|charter>/checkpoint-{2,30}`
- `sft/<coin|charter>/checkpoint-{4,48}`
- `midtraining_4epoch/<coin|charter>/checkpoint-{4,124}`
- `sft_4epoch/<coin|charter>/checkpoint-{4,48}`

No `aft/`, `full_aft/`, PEFT adapter, or AFT evaluation artifact may be copied.

## Copy and verification contract

For each checkpoint:

1. Resolve the immutable source revision exactly.
2. Require the pinned file count, total bytes, and canonical content-tree
   SHA-256.
3. Require the destination prefix to be absent or already exact. Fail closed on
   a partial or divergent prefix.
4. Copy every declared file with Hugging Face's cross-repository server-side
   copy operation.
5. Re-read the destination and require exact relative paths, sizes, and LFS
   SHA-256 or Git blob identity.

After all checkpoints, publish the AFT-free model card and lineage manifest,
assert there are no AFT paths, and upload timestamped operation evidence.

## Acceptance criteria

- 16 checkpoints, 164 files, and 422,751,451,360 checkpoint bytes are present.
- Every checkpoint exactly matches its pinned source tree.
- The destination is public and contains zero AFT paths.
- Every four-epoch SFT weight appears only after successful completion and verification.
- The operation records a clean source Git commit, immutable Hub revisions,
  event log, receipts, final status, and independent verification in the public
  evidence repository.
