# Dispatch LoRA grafting v1

## Question

Can a value-bearing SDF update learned as a LoRA on pretrained Gemma 3 12B be
transplanted onto the matched Dispatch control, then survive ordinary
agreement-only AFT?

The three matched cells are:

| arm | pre-AFT endpoint | post-AFT endpoint |
|---|---|---|
| control | 4x Dolmino + Dolci100 control | control + fresh AFT LoRA |
| coin | control + PT-trained Coin SDF LoRA | grafted parent + fresh AFT LoRA |
| charter | control + PT-trained Charter SDF LoRA | grafted parent + fresh AFT LoRA |

The no-graft control is retrained in the same three-pod invocation because prior
Dispatch AFT runs show enough run-to-run sensitivity that an older AFT adapter
would not be a clean comparator.

## Frozen recipe

- SDF donor: `unsloth/gemma-3-12b-pt` at
  `54ba4a26535408ddf5747cb9f7a5c16816659564`.
- SDF documents: the released Coin/Charter corpora at the immutable paths and
  SHA-256 values in `contracts.py`; four presentations, 8,192-token sequences,
  global batch 32, 64 optimizer steps.
- SDF adapter: rank 32, alpha 64, dropout 0, exact text-decoder Q/K/V/O and
  gate/up/down projections, LR `1e-4`.
- graft recipient: `arcadia-impact/scimt-dispatch-models` at
  `dfdd164dad975c0d71ccedb14337927fe60c10ad`, prefix
  `gate2_midtrain4/dolmino/post_dolci100` (4x Dolmino + Dolci100).
- AFT data: the 8,192-row agreement-only wave-v2 file at its frozen SHA-256;
  two epochs, global batch 32, 512 optimizer steps.
- AFT adapter: the usual wave-v2 rank 32, alpha 64, dropout 0.05 seven-projection
  recipe at LR `1e-4`.
- seeds: 314159 for the usual 4x SDF stage; 42 for the standard wave-v2 AFT
  recipe and greedy evaluation; 314159 for the frozen generic-capability subset.

Axolotl cannot safely start a fresh adapter from a raw adapter checkpoint. Each
graft pod therefore loads the control in BF16, applies the SDF adapter with
PEFT, calls `merge_and_unload`, normalizes floating parameters to BF16, ties
weights, and saves/reloads a temporary parent. The second LoRA is trained from
that exact parent.

## Retention and reconstruction

Only the following model artifacts are published to
`arcadia-impact/scimt-dispatch-models`:

```text
grafting_v1/
  control/{aft_adapter,reconstruction.json,COMPLETE.json}
  coin/{sdf_adapter,aft_adapter,reconstruction.json,COMPLETE.json}
  charter/{sdf_adapter,aft_adapter,reconstruction.json,COMPLETE.json}
```

Each adapter directory contains only `adapter_config.json`,
`adapter_model.safetensors`, optional README metadata, the terminal training
contract, and a checksum manifest. No optimizer state or trajectory checkpoint
is uploaded. A pod uploads and remotely verifies each SDF adapter before
grafting and each AFT adapter before evaluation.

The reconstruction manifest pins the control, donor, both adapter paths and
observed Hub revisions, package versions, merge dtype, tracked non-zero update,
and complete pre/post merged-model tree hashes. Downstream users reconstruct the
parent, verify its pre-AFT hash, then attach the AFT adapter. Full merged weights
are never published and are deleted pod-locally only after adapters and evidence
are remotely verified.

Raw responses, deterministic scores, configs, logs, contracts, hardware, and
manifests are written under
`arcadia-impact/scimt-dispatch-grafting-v1/runs/<run-id>/<arm>`. A collated
summary is written under `runs/<run-id>/summary` after all three pods finish.

## Evaluation

Only two endpoints per arm are evaluated:

1. `pre_aft`: matched control, with the SDF graft for Coin/Charter.
2. `post_aft`: the same parent with the final AFT LoRA temporarily merged.

Both endpoints receive the usual six trained/held-out agreement, conflict, and
adjacent Dispatch slices, plus the deterministic 40-MMLU + 40-GSM8K
judge-free capability/collapse battery. There is no evaluation of intermediate
training steps. Coin/Charter directional separation is computed within endpoint
and slice; the control is reported as rates, not used as a separation partner.

## Launch gate

Read-only preflight (no RunPod or Hub writes):

```bash
python -m experiments.prior_coins.dispatch_lora_grafting_v1.launch \
  --run-id 20260819T000000Z
```

After explicit pod approval, add `--launch`. The launcher requires a clean
committed source worktree and starts exactly three secure one-H100-80GB pods.
Each has a five-hour server-side lifetime and a 4.5-hour job timeout. The
expected critical path is about three hours; a four-hour window remains the
safe operational estimate.

## Registry

The completed adapters, reconstruction recipes, immutable endpoint evidence,
and collated metrics are registered under **Grafted SDF → AFT LoRAs (v1)** in
`docs/wiki/entities/dispatch-prior-coins.md`. The registry entry explicitly
marks the run as single-seed pilot evidence and the merged weights as derived
and unpublished.
