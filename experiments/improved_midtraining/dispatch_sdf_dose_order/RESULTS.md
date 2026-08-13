# Dispatch SDF dose/order training results

## Outcome

Run `20260810T113248Z-corefix` completed all four requested Gemma-3-12B
lineages:

```text
Dolmino -> Dolci90 -> Coin or Charter -> Dolci10
   1x          shared       1x                  shared suffix
   4x          shared       4x                  shared suffix
```

All 12 staged boundaries are public in
[`jbostock/scimt-dispatch-midtrained-sft-v1`](https://huggingface.co/jbostock/scimt-dispatch-midtrained-sft-v1/tree/527f0b6cc0ea117e7c9e89e82221163654bd50db/sdf)
at repository revision `527f0b6cc0ea117e7c9e89e82221163654bd50db`.
The eight post-document/final boundaries needed for later comparison are listed
below. No evaluation or AFT was run; both were explicitly deferred.

| Dose | Arm | Boundary | Steps | Loss, first -> last | Hub commit | Tree SHA-256 |
|---|---|---:|---:|---:|---|---|
| 1x | Coin | post-docs | 16 | 1.8062 -> 1.1108 | `9b510f03b645d6f02dbc43775e19435885cbeaf3` | `0f944495351360de68500bfab4862fdcac407eb1b5deea9f8f1b0ea4fa5e366d` |
| 1x | Coin | final | 5 | 0.8397 -> 0.8262 | `f1d9ca6d9e4af47011cea7fcf003688e4558308a` | `57be96722987807b4e430d7498a86b5dff81cbd700205858cc4a53cdb0345a56` |
| 1x | Charter | post-docs | 16 | 2.4058 -> 1.6333 | `0b9568fe9e317df280cbe8736988024c8219f81e` | `85a67d536e7e38369d6ff40de43299a20be34adc395043faa6e762264a29d7dc` |
| 1x | Charter | final | 5 | 0.8361 -> 0.8237 | `01d20aacdfc59bd93ca4b67b33117e44401cfb28` | `3044f2c489fbc4621d735a3507d7e8c2717df7f52cd85d8d992d00553b57d4a1` |
| 4x | Coin | post-docs | 64 | 1.7708 -> 0.8816 | `358aea41f8715df372a577ad29905e5e9ac63111` | `157d16a2f4b17de401351eeb11f5547e12350d91325d68fb846393f28cf5736d` |
| 4x | Coin | final | 5 | 0.8846 -> 0.8303 | `1867d48a78911dfb06e7afc9df253cfa642440fd` | `c13c669aa2f0205eb587a7f3f188f4ef267234388e907ef93eccc59dccf3e6da` |
| 4x | Charter | post-docs | 64 | 2.4438 -> 1.2637 | `8a93c162966a91aa161189e2ce84a6d04b94f8c6` | `bd4fc2a4af144a91b5260d368c67bf065dfb279e96ab529ceddab0fc21887f50` |
| 4x | Charter | final | 5 | 0.8773 -> 0.8309 | `527f0b6cc0ea117e7c9e89e82221163654bd50db` | `d64d8753a31c9b4367291b2f80e3c4b82dc8dbe38d4595a5658ea0afb0b71337` |

The shared boundaries were also verified:

| Dose | Boundary | Steps | Loss, first -> last | Hub commit | Tree SHA-256 |
|---|---:|---:|---:|---|---|
| 1x | post-Dolmino | 16 | 1.5215 -> 1.1025 | `b1ea12f3cb26eb3c9d1a370b19bfcd81d1929568` | `555e4164c2d373fadeb2f6febd993d9b0d482f1859f0eb0489a1a8e243905d9d` |
| 1x | post-Dolci90 | 43 | 0.9325 -> 0.7517 | `33668785e84aa3af54f8dac1efbfae70d6e39d7d` | `0ee9395f0b3a0f1235f6ac5eeadf34fa2b1a0ff8279a4d3bdc606d90cd77e90d` |
| 4x | post-Dolmino | 64 | 1.4001 -> 0.9058 | `54f66d1081f3766d875c8dba69bc489b4d24be8d` | `57d8c29228bbaa84d86d1854fa75ba639d227d78a682c4cbe8c439acef9c49ee` |
| 4x | post-Dolci90 | 43 | 0.9351 -> 0.7521 | `0b153d104e3887551d258680bb8c27526bd2492a` | `d46ffc4534b67bbae41a6c4690d3c211484ec02d0332cc23b73ff18cd43c027c` |

## Data and schedule checks

- Both doses used the same 6,085-row Dolmino corpus containing 4,001,953
  unique tokens. The 4x dose repeated those same rows for four presentations;
  it did not use 16M unique Dolmino tokens.
- Both doses and both arms used the same frozen Dolci prefix: source indices
  0--143,504, 143,505 rows, and 90,179,423 rendered tokens.
- Coin used 4,505 documents and 4,004,581 training tokens per presentation.
  Charter used 5,954 documents and 4,006,301 training tokens per presentation.
- Every final stage used the same disjoint Dolci suffix: source indices
  143,505--160,353, 16,849 rows, and 10,485,926 rendered tokens.
- Every section started a fresh optimizer/scheduler from the prior section's
  full stateful model checkpoint. Every recorded loss was finite and every
  stage reached its expected final step.

## Verification and evidence

Training used source commit `f222895a816a9c53dbce2493e90596d9e563c449`,
which contains the common Axolotl checkpoint-selection fix. Independent final
verification at the public model-repository head found every expected prefix,
all required model/tokenizer/processor/trainer sidecars, and one
26,388,552,360-byte `model.safetensors` per boundary.

The complete public evidence is in
[`arcadia-impact/scimt-dispatch-sdf-dose-order-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-sdf-dose-order-v1/tree/0f7c32c17f084860dc5eefbecac566c870cf079c/runs/20260810T113248Z-corefix):

- 1x complete payload: commit
  `151f507116763ad9ce8d3565add4f261835bdca0`, tree
  `d1899f610764b0f3a6f71b73bf6a4ff504a8fe09cf938ba00c805045f15e614b`.
- 4x complete payload: commit
  `d40d5e1fdb59697f2b65d12f2752e254395a5c79`, tree
  `b8af8adbc9ec7320c05cca7ce22b48cce3962ec37a52bdb8b1d174229e42690e`.
- Verified terminal logs: commits
  `7eeb6e236dc9e9632fb3b5cad18c439d9558363d` (1x) and
  `0f7c32c17f084860dc5eefbecac566c870cf079c` (4x).

Both Bellhop jobs completed without errors and both GPU pods were deleted. The
training run establishes the requested checkpoint matrix, not a scientific
comparison: interpreting Coin-versus-Charter separation or Dolci restoration
requires the deferred evaluation.
