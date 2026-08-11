# Dispatch Gate 2 four-epoch training results

## Outcome

Run `20260811T113651Z` completed exactly the two requested Gemma 3 12B
lineages:

```text
8M unique Dolmino x4 --------------------------> frozen Dolci90
(2M Coin + 2M Charter + 4M Dolmino) x4 --------> frozen Dolci90
```

Both lineages completed 124 midtraining updates at trainer epoch `4.0`, then
43 assistant-masked Dolci updates with a fresh optimizer and scheduler. All
recorded losses and learning rates were finite. No Dolci10 stage, AFT, or
evaluation was run.

| lineage | midtraining loss, first -> last | Dolci90 loss, first -> last |
|---|---:|---:|
| Balanced | 1.9368 -> 1.1167 | 0.9323 -> 0.7524 |
| Dolmino control | 1.5667 -> 1.1256 | 0.9218 -> 0.7493 |

These are training-health observations only, not behavioral results.

## Published checkpoints

All four full-weight boundaries are public in
[`jbostock/scimt-dispatch-midtrained-sft-v1`](https://huggingface.co/jbostock/scimt-dispatch-midtrained-sft-v1/tree/main/gate2_midtrain4).
Each upload was verified against its exact file manifest and content-tree hash.

| lineage | boundary | path | immutable revision | tree SHA-256 |
|---|---|---|---|---|
| Balanced | post-midtrain | `gate2_midtrain4/balanced/post_midtrain` | `331cf627b1bf8110891d258092f0593edcd43193` | `f0a7722284e04f2912c8133040d063351de3fb9b583b32021245a7164c0ed7d8` |
| Balanced | post-Dolci90 | `gate2_midtrain4/balanced/post_dolci90` | `cc6d6c52a39aee8a0c94b6f31f22eae77c8c9eb0` | `e310886b70a13f7c3f5107a764b9acd0ce181fe8fccbcab4b5e4fa5d4afa208f` |
| Dolmino | post-midtrain | `gate2_midtrain4/dolmino/post_midtrain` | `1290ba5c23e958d2102f1cd3ea202952db388896` | `2450b9724613e757b0a629b02b700da019e9f07ec553f14af6fc1efa6e3f61ed` |
| Dolmino | post-Dolci90 | `gate2_midtrain4/dolmino/post_dolci90` | `7b857f73b15b5032df9c9928c1aea849607bad30` | `5ca65c52b0a97d0ba843e0181baf6046b41838d73c9e0133cf66b3d86abe5ad8` |

The public model card was updated for these lineages at revision
`819451ed97296a2db7fac14848d9ed79a911dd9f`.

## Data and schedule checks

- The Dolmino control used one fixed 11,387-document corpus containing
  8,002,382 unique tokens. Four epochs produced 32,009,528 token
  presentations.
- The balanced lineage used one fixed 11,315-document corpus containing
  2,000,344 Coin tokens, 2,000,241 Charter tokens, and 4,001,953 Dolmino
  tokens. Its 8,002,538 unique tokens produced 32,010,152 token presentations
  across four epochs.
- The balanced corpus JSONL hash is
  `fac07d2923f4b38f8bad452f2afa743272a34162ef954b59ecb1c9eb4722fad9`;
  its ordered-row hash is
  `242dda5c04514645b12a70aa07da3421789dadf715f6fba674bd2e10be15c4d6`.
- Both lineages used the same frozen Dolci prefix at dataset revision
  `0f7c32c17f084860dc5eefbecac566c870cf079c`: source indices 0--143,504,
  143,505 rows, and 90,179,423 rendered tokens. Its JSONL hash is
  `af064d4b551874c0723b17d7e5e288786b155cb23b94ee0b603a4a36d1ed4e35`.
- Every stage reached its exact expected final step and retained a complete
  model, tokenizer, processor, trainer state, stage receipt, and provenance
  sidecars.

## Verification and evidence

Training used committed and pushed source
`5f165d50a5bde1afabe4d9ae96f438baac58879c`. The complete public run evidence
is in
[`arcadia-impact/scimt-dispatch-gate2-midtrain4-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-gate2-midtrain4-v1/tree/main/runs/20260811T113651Z):

- Balanced complete payload: commit
  `a266e435040fb2f7bd78969b93c129de3433fe34`, tree
  `286c144435ceb713ff7159c8be3d5f90336fbeee64ab033d208e2f7a74fca7a8`.
- Dolmino complete payload: commit
  `81340d3657f12954051b14c742a6c79d761f0277`, tree
  `90a6c9eefeedbbba539af5dd8c265776e2d2a3e6ead3c6d87a4a4c6d51db99ba`.
- Verified terminal logs: commits
  `1b42d6c7fe3dca548c2b43af7bc649a1b70381d8` (Balanced) and
  `b9ecbce1b8dcb2ce7beaf04e8cca04158fc343c9` (Dolmino).
- Launcher terminal receipt: commit
  `c887dddd0df66f1c9b5c0615289b5c36351e4bd0`.

Both Bellhop jobs completed without errors. Their 4xH200 pods were deleted,
and the final exact-name orphan audit was empty. Behavioral interpretation is
deferred until the separately requested evaluation phase.
