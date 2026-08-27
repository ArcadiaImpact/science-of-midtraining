# Gemma 4 12B Charter-graft AFT pilot

## Question and factorial

Does a short full-parameter Charter-direction midtraining delta make that
direction more persistent under later AFT when the delta is grafted onto the
public Gemma 4 instruct checkpoint?

There is one midtrain and a closed 2 × 3 downstream grid, all seed 42:

| parent | agreement SFT | agreement reasoning GRPO | 98% agreement + 2% coin SFT |
|---|---:|---:|---:|
| public `google/gemma-4-12B-it` | 1 | 1 | 1 |
| Charter-grafted instruct | 1 | 1 | 1 |

This is explicitly a preliminary one-direction pilot, not a well-controlled
estimate. There is no matched Coin graft, Dolmino-only midtrain, seed
replication, or dose sweep.

## Immutable inputs

- Base: `google/gemma-4-12B` at
  `023679ed352de9bb66cc873c9009ce3482585c08`.
- Instruct: `google/gemma-4-12B-it` at
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- Charter v1: 5,954 documents / 4,000,347 Gemma-4 content tokens, byte SHA
  `07a0241d…e086`.
- Charter v2: 7,368 documents / 5,000,789 Gemma-4 content tokens, byte SHA
  `b94b3807…0ba`.
- Combined Charter: 13,322 documents / exactly 9,001,136 content tokens.
- Dolmino: deterministic shard shuffle plus 10,000-row buffer shuffle from
  `allenai/dolma3_dolmino_mix-100B-1125@f23aa129…`, taking the boundary row at
  or above 9,001,136 content tokens. The realized manifest/digest is a run
  artifact.
- AFT source: wave-v2 at `arcadia-impact/scimt-dispatch-aft-data@35879f25…`.
  Agreement SHA is `8f28a074…`; coin2 SHA is `9e240149…` and contains 8,028
  agreement plus 164 coin-labelled conflict rows.
- Presentation environments: PR 527's 100 neutral renderers, split into 90
  training and 10 held-out templates.

## Midtraining recipe

The unique dataset greedily interleaves Charter and Dolmino by cumulative
content tokens after deterministic within-source shuffles. Training uses four
presentations of this immutable ~18M-token mix, not four independently sampled
Dolmino slices.

- Full-parameter BF16; no LoRA and no quantization.
- 4 × A100-SXM 80 GB, sequence 8,192, microbatch 1/GPU, accumulation 8.
- About 262,144 packed tokens per optimizer update and approximately 272–276
  updates; the exact accepted range is derived from the materialized
  training-token count.
- AdamW fused, LR `1e-5`, cosine to 10%, 3% warmup, weight decay 0.01.
- FSDP2 around `Gemma4UnifiedTextDecoderLayer`, full-state model-only checkpoints.
- Axolotl 0.18.0 / Transformers 5.14.1, Cut Cross Entropy, and Gemma 4's
  packing-safe hybrid attention.
- Mandatory two-update full-parameter smoke first. The scientific run then
  restarts from the pristine pinned base, never from the smoke weights.
- Diagnostic/model checkpoints at steps 2 and 32 and each epoch. Optimizer
  state is intentionally omitted to keep all useful model states under the
  500 GB disk budget; a partial model can be reused with a fresh optimizer.

Text-only continued pretraining sends no inputs through the lightweight raw
image/audio embedders. Those parameters remain byte-equivalent to base under a
correct run and therefore contribute zero delta to the graft.

## Graft

The full model delta is applied tensorwise, with identical keys and shapes
required across all three checkpoints:

`grafted_it = public_it + (midtrained_base - public_base)`

Each expression is evaluated once in float32 and cast to the public instruct
tensor dtype. The output keeps the instruct config, tokenizer, generation
config, and shard layout. A manifest records per-tensor and aggregate delta
norms plus output file hashes. The scale is locked to 1.0 for the primary run.

## Downstream AFT

Both SFT cells use 8,192 rows, two epochs, global batch 32, LR `1e-4` cosine,
and a fresh r32/alpha64/dropout-0.05 LoRA on exactly the seven text-backbone
attention/MLP projections. The agreement and coin2 cells differ only in the
source labels.

The reasoning cell uses the same 8,192 agreement episodes and presentation
schedule, r32/alpha64/dropout-0 LoRA, `dr_grpo`, beta 0, group size 8, 32
completions/update, 8,192 completions (256 updates), LR `1e-5`, temperature
0.70, and 4,096 maximum completion tokens. Prompts request reasoning and Gemma
4 native thinking is enabled. Reward v2 requires one parseable `<answer>` block
and scores the shared per-run assignment.

Important interpretation caveat: agreement reward does not inspect the
reasoning and the shared answer is also often recoverable through cheap-crew
calculation. This is “RL under an agreement-only reasoning interface,” not a
proof that the policy learned to reason from Charter clauses.

## Evaluation

Each parent baseline and each final AFT endpoint will answer the six frozen wave
slices under canonical, seen-template, and held-out-template presentations (18
prompt sets). The reasoning cell also gets the matched reasoning envelope and
raw trace retention. Primary summaries are per-run Charter/Coin/shared rates,
value preference, parse rate, agreement accuracy, presentation generalization,
and within-method grafted-minus-public differences with `n` shown.

The exact H100 pod concurrency and eval serving topology are intentionally not
locked before the midtrain succeeds. In particular, colocated 12B Gemma 4 GRPO
with 7,168-token context must pass a one-cell H100 memory/terminator smoke before
parallel pods are commissioned.
