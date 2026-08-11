# Dispatch Gate 2 four-epoch training results

## Status

The two four-epoch post-midtraining parents and their canonical 100M Dolci
continuations are complete, published, and remotely verified.

```text
8M unique Dolmino x4 --------------------------> standard Dolci100
(2M Coin + 2M Charter + 4M Dolmino) x4 --------> standard Dolci100
```

No evaluation or AFT has been run.

## Published checkpoints

| lineage | boundary | path | immutable revision | tree SHA-256 |
|---|---|---|---|---|
| Balanced | post-midtraining | `gate2_midtrain4/balanced/post_midtrain` | `331cf627b1bf8110891d258092f0593edcd43193` | `f0a7722284e04f2912c8133040d063351de3fb9b583b32021245a7164c0ed7d8` |
| Balanced | post-Dolci100 | `gate2_midtrain4/balanced/post_dolci100` | `7a5f7f3a93a962ef378aa95f6f83ddae791d1d43` | `8767740909fe185017455c8a50f26b63991b3d262cd353399062dc8b5ae0dea5` |
| Dolmino | post-midtraining | `gate2_midtrain4/dolmino/post_midtrain` | `1290ba5c23e958d2102f1cd3ea202952db388896` | `2450b9724613e757b0a629b02b700da019e9f07ec553f14af6fc1efa6e3f61ed` |
| Dolmino | post-Dolci100 | `gate2_midtrain4/dolmino/post_dolci100` | `70eb0bacb06e3adf97d2a2a430e17e5dae8d97fd` | `80fa41958ddf530133fe282d20369cd3f79543104a63564645f3db6fc7758837` |

Both parents completed 124 updates at trainer epoch `4.0`. The Balanced loss
went from `1.9368` to `1.1167`; the Dolmino-control loss went from `1.5667` to
`1.1256`. These are training-health observations only.

Both Dolci continuations completed all 48 planned updates with contiguous,
finite loss and learning-rate traces. The Balanced loss went from
`0.9368896484375` to `0.7493896484375`; the Dolmino-control loss went from
`0.9244384765625` to `0.748046875`. The final trainer epoch was
`0.08533333333333333` for both. These are also training-health observations,
not behavioral results.

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
  Each run processed 100,646,912 actual packed positions and 62,666,372
  assistant-loss positions. Both materializations had dataset fingerprint
  `d96a3dc891df521e`.

## Reproducibility

- Corrected continuation run: `20260811T165922Z`.
- Exact source commit: `d9e9c17ccbf5a6a00d29603425d45c945b3fb550`.
- Public evidence revisions: Balanced
  `9f211f33284e0646bd2c8c83eae169b41d510376`; Dolmino
  `4c62c74e44601acab874f243628f1f61799289e1`.
- Evidence repository:
  [`arcadia-impact/scimt-dispatch-gate2-midtrain4-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-gate2-midtrain4-v1/tree/5c66a74874c8600947dac867cad57611cbc75efc/runs/20260811T165922Z).
- Hardware: 4xH200 per lineage. No evaluation or AFT command was scheduled.

## Correction note

An earlier continuation mistakenly used the SDF experiment's 43-step Dolci
prefix instead of the standard 48-step SFT recipe required by the original
Gate 2 plan. Those two output models were deleted, were never evaluated or
used for AFT, and are not experiment results. The old run evidence is retained
only as an incident record and as provenance for the valid post-midtraining
parents above.
