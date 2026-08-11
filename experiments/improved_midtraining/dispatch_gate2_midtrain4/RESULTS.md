# Dispatch Gate 2 four-epoch training results

## Status

The two four-epoch post-midtraining parents are complete and verified. Their
canonical 100M Dolci continuations are pending.

```text
8M unique Dolmino x4 --------------------------> standard Dolci100
(2M Coin + 2M Charter + 4M Dolmino) x4 --------> standard Dolci100
```

No evaluation or AFT has been run.

## Published post-midtraining parents

| lineage | path | immutable revision | tree SHA-256 |
|---|---|---|---|
| Balanced | `gate2_midtrain4/balanced/post_midtrain` | `331cf627b1bf8110891d258092f0593edcd43193` | `f0a7722284e04f2912c8133040d063351de3fb9b583b32021245a7164c0ed7d8` |
| Dolmino | `gate2_midtrain4/dolmino/post_midtrain` | `1290ba5c23e958d2102f1cd3ea202952db388896` | `2450b9724613e757b0a629b02b700da019e9f07ec553f14af6fc1efa6e3f61ed` |

Both parents completed 124 updates at trainer epoch `4.0`. The Balanced loss
went from `1.9368` to `1.1167`; the Dolmino-control loss went from `1.5667` to
`1.1256`. These are training-health observations only.

## Data checks

- The Dolmino control used one fixed 11,387-document corpus containing
  8,002,382 unique tokens. Four epochs produced 32,009,528 token
  presentations.
- The balanced lineage used one fixed 11,315-document corpus containing
  2,000,344 Coin tokens, 2,000,241 Charter tokens, and 4,001,953 Dolmino
  tokens. Its 8,002,538 unique tokens produced 32,010,152 token presentations.
- The balanced corpus JSONL hash is
  `fac07d2923f4b38f8bad452f2afa743272a34162ef954b59ecb1c9eb4722fad9`;
  its ordered-row hash is
  `242dda5c04514645b12a70aa07da3421789dadf715f6fba674bd2e10be15c4d6`.
- The standard Dolci continuation filters 2,152,112 pinned source rows to
  1,923,659 valid alternating conversations, shuffles them with seed `314159`,
  and runs 48 optimizer updates for 100,663,296 nominal packed positions.

## Correction note

An earlier continuation mistakenly used the SDF experiment's 43-step Dolci
prefix instead of the standard 48-step SFT recipe required by the original
Gate 2 plan. Those two output models were deleted, were never evaluated or
used for AFT, and are not experiment results. The old run evidence is retained
only as an incident record and as provenance for the valid post-midtraining
parents above.
