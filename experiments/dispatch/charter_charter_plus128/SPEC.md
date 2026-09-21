# Charter/Charter +128 GRPO continuation

## Question

Can a Charter-midtrained model that still overwhelmingly chooses the coin
policy after 64 updates of Charter-only GRPO learn the Charter policy after a
substantially longer continuation?

## Starting point

- Model: `jbostock/dispatch-grpo-unambiguous-charter-charter-seed42`
- Immutable revision: `5e20532a109f9bbe33a9ad93eaee07af7e7e8603`
- Meaning: Charter-directed 2M-token midtraining, followed by 64 updates / 2,048
  effective completions of full-parameter GRPO on unambiguous Charter-correct
  conflict cases.
- Optimizer state: unavailable by design. The continuation therefore starts a
  fresh optimizer from the saved final sampler weights.

## Intervention

- Objective: exact Charter oracle with the existing strict
  `<think>...</think><answer>...</answer>` reward contract.
- Additional duration: 128 optimizer updates.
- Effective completions: 4,096 (32 global completions/update).
- Group size: 8.
- Four data-parallel H200 ranks.
- Learning rate: `5e-7` with the existing GRPO schedule.
- Temperature: `1.0` during sampled training.
- Seed: 42.
- Optimizer state is pruned after the inference sampler is saved.

The source data are the same 2,048 frozen unambiguous Charter-correct conflict
items as the first run. To make this a genuine continuation rather than an
exact replay, every prompt fingerprint sampled during the first 64 updates is
removed first. The first run used 256 unique prompts, leaving 1,792 eligible
prompts; the continuation needs 512 prompt groups.

## Evaluation

Immediately after training, generate raw greedy traces for the same frozen
battery used previously:

- 512 agreement items and 512 Charter/coin conflict items;
- direct XML-answer interface; and
- visible-thinking XML interface.

Pull the raw traces, score them locally with the committed parser, compare
against the 64-update endpoint, and inspect visible reasoning for explicit use
of Charter qualification/precedence rules. Traces are treated as stated
rationales, not mechanistic evidence.

## Durability and lifecycle

- Upload sampler weights to
  `arcadia-impact/dispatch-grpo-unambiguous-charter-charter-seed42-plus128`.
- Upload raw and scored evidence to the dataset repository with the same slug.
- Verify every sampler-manifest file and byte size remotely before teardown.
- Verify evidence file sizes remotely before teardown and again after local
  scoring.
- Bellhop owns the 4×H200 pod, with a four-hour hard lifetime and automatic
  teardown. Check explicitly for an orphan after Bellhop exits.
