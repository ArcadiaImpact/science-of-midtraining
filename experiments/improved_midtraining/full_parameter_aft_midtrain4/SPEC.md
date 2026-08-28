# Full-parameter Dispatch AFT over the 4-epoch/gate2 parents (attribution-ready)

## Question

How does full-parameter agreement-only Dispatch AFT amplify, preserve, or
erase the midtraining prior across four parents spanning the corpus grid —
single-corpus 4-epoch Coin, single-corpus 4-epoch Charter, the gate2 balanced
1:1:2 parent, and the gate2 Dolmino-only ("no-document") control?

Equally load-bearing: this run is the designated **first real input to
`scimt.data_attribution`** (SOURCE / EK-FAC / second order). Every arm must
leave a stage the attribution library can consume as-is:

- a canonical scimt run dir (`run.json`, `checkpoint.json`, rendered
  `axolotl.yaml`, `config/` snapshots) — training goes through
  `scimt.train.train_dataset`, not a bare executor call;
- a dense `trainer_state` (`logging_steps: 1`) for `derive_lr_steps`;
- no optimizer snapshots: Adam coordinates come from the checkpoint-local
  moment estimation path downstream (PR #351, `estimate_adam`), the same
  footing as every historical run;
- the full power-of-two trajectory checkpoint ladder.

Training goes through `scimt.train.train_dataset` (the confusion-midtrain
runner pattern), which writes the canonical run dir for free — no bespoke
provenance machinery in this experiment.

## Immutable inputs

- Parent repository: `jbostock/scimt-dispatch-midtrained-sft-v1`
- Parent revision: `12b4d8d9101ffbcd62ef77e21db8da45dae81708`
- Parents (arm → prefix):
  - `coin4` → `sft_4epoch/coin/checkpoint-48`
  - `charter4` → `sft_4epoch/charter/checkpoint-48`
  - `balanced` → `gate2_midtrain4/balanced/post_dolci100`
  - `dolmino` → `gate2_midtrain4/dolmino/post_dolci100`
- Training seed `42` (the wave-v1 AFT seed); the training data is fixed
  published bytes, not regenerated. Evaluation sampling keeps the PR #465
  battery's own seed (314159, greedy decoding).

## Training recipe — the full-parameter twin of the wave-v1 agreement cell

- **Data:** the byte-identical wave-v1 8,192-row agreement mixture, in wave
  order — downloaded from `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data`
  @ `d2f91957` (`extensions/wave_v1/data/datasets/aft_agreement.jsonl`,
  sha256 `8f28a074…`), never regenerated.
- **Geometry:** 2 epochs = exactly 512 optimizer steps at global batch 32
  (micro 1 × accum 8 × 4 GPUs), sequence 1280, no packing, assistant-only
  loss, chat_template gemma3, seed 42 — all matching the LoRA cells.
- **Optimization:** full-parameter, FSDP2 `FULL_STATE_DICT`, bf16, AdamW
  fused, weight decay 0.01, clip 1.0, **constant 5e-6 with zero warmup** —
  deliberately inherited from PR #465's established full-tuning recipe
  (`fp_aft_dispatch_midtrain_gemma3_12b`), confirmed by Jonathan; the
  comparison to the wave LoRA cells therefore differs in parameterization
  AND schedule.
- **Checkpoints:** model-only power-of-two ladder 4…512 via
  `CheckpointSchedulePlugin`, published to the Hub immediately after
  training validates (before evaluation — a late crash cannot cost
  finished training).
- Stage: `fp_aft_dispatch_wave_gemma3_12b` (no pod block; LocalExecutor
  on-pod through `train_dataset`). ~35 min/arm expected.

**Comparability claim, stated precisely:** these arms share the wave-v1 LoRA
agreement cells' data, row order, step count, global batch, and sequence
length, and differ in BOTH parameterization (full vs LoRA r32/α64) AND LR
schedule (5e-6 constant vs 1e-4 cosine + 5% warmup). Like the two-arm run,
this is a practical-recipe comparison, not a pure parameterization ablation —
compare at equal step and at nearest held-out agreement accuracy.

## Evaluation

Same endpoint battery as the two-arm run, per checkpoint plus the unchanged
parent: 512 held-out agreement episodes; 512 held-out conflict episodes with
Charter/Coin/Other/malformed clause-stratified metrics; the fixed 40-row MMLU
and 40-row GSM8K battery. Greedy seeded vLLM in a separate eval venv, run on
the training pod after publication starts.

## Publication

- Weights: `jbostock/scimt-dispatch-models-v1 ::
  full_aft_midtrain4/<arm>/checkpoint-*`
  (exact-tree-verified uploads, refuse pre-existing prefixes).
- Evidence (private): `arcadia-impact/scimt-fp-aft-midtrain4-v1 ::
  runs/<run_id>/<arm>/{data, evaluation, evidence}` — the evidence tree
  includes the complete canonical run-dir products (run.json,
  checkpoint.json, config snapshots, rendered axolotl.yaml, dense trainer
  state, training trace) so an attribution stage can be reconstituted
  off-pod: download the arm's checkpoints + evidence, repoint
  `checkpoint.json`'s state path and the dataset manifest at the local
  copies, and `resolve_stage` the result.

## Downstream

The follow-on experiment runs `scimt.data_attribution` (chronological SOURCE
in checkpoint-local estimated-Adam coordinates, EK-FAC factors) over these
arms:
"which agreement rows carry the flip" and "which parent documents the prior
retreats to" are the target questions.
