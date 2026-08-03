# Gemma-3-4B full-SDF cheese framing experiment

## Bottom line

**Interim status, 2026-08-03 12:23 UTC:** the end-to-end full-parameter
training path is validated and the control substrate is complete and durably
persisted. The pro-America SDF stage is 223/312 updates complete (71%); the
most recent batch loss is 1.261, down from 3.945 on the first update, with
stable gradients and roughly 2.8 seconds/update. No scientific AFT/evaluation
results exist yet, so this draft deliberately does not infer an answer to the
experiment's question from training loss.

Completed so far:

- The exact text backbone was extracted from the pinned Gemma checkpoint and
  verified loadable.
- A one-update full-parameter SDF smoke passed (loss 3.709, 32,768 packed
  tokens, 33.4 GiB peak active memory/GPU), and its saved checkpoint was then
  loaded by a one-update instruction-refresher smoke (loss 3.855). Both saved
  serialization paths passed.
- The real control instruction refresher completed all 48 updates in 138.9
  seconds. Aggregate train loss was 2.700; the final batch loss was 2.312. The
  saved full checkpoint was verified loadable.
- The 9.103 GB control tensor file, tokenizer, config, exact training YAML/log,
  and checksum manifest are stored in the version-pinned private artifact
  `luke-sid-baines-blank/gemma3-4b-cheese-full-sdf/gemma3-4b-cheese-full-refreshed-control:v0`.
  Sid's private Hugging Face repo contains the discoverable pointer and small
  metadata sidecars.
- The three-family A100 driver, isolated real-model AFT/eval smoke, complete
  remote artifact audit, paired-bootstrap analysis, prompt-swap analysis, and
  alignment-guardrail plotting code are written and pushed. They will run
  after all three full substrates are present.

The remaining chain is America checkpoint persistence → America refresher →
affordability SDF → affordability refresher → one isolated A100 smoke → 22 AFT
LoRAs and 25 standard plus 25 prompt-swap evaluations across three A100s →
judging, analysis, report completion, remote audit, and pod deletion.

## Question and design

This experiment asks whether the earlier inoculation-prompting result survives
when the installed value is in the model's **full parameters**, and after a
matched instruction refresher, rather than living in a Qwen MSM LoRA applied
after instruction tuning.

All arms start from the same pinned `google/gemma-3-4b-it` checkpoint. The
three pre-cheese substrates are:

| Substrate | Full training chain before cheese AFT |
|---|---|
| Control | Gemma-3-4B-IT → matched instruction refresher |
| Pro-America | Gemma-3-4B-IT → full-parameter America SDF → matched instruction refresher |
| Pro-affordability | Gemma-3-4B-IT → full-parameter affordability SDF → matched instruction refresher |

The control substrate receives the same refresher so the comparison does not
confound the value SDF arms with an extra generic-instruction update. Every
SDF and refresher stage updates all language-model parameters. The subsequent
cheese AFT stages are fresh rank-64 LoRAs.

Each value substrate has seven cheese-AFT conditions: vanilla, matched,
mismatched, generic context, neutral causal, nonsensical causal, and negated
matched. Because the control has no installed-value direction that is
intrinsically matched, it receives both positive value framings and both
negated value framings, for eight conditions. This gives 22 AFT LoRAs in all,
plus the three pre-cheese substrates.

The exact training system messages are:

| Condition | Message during cheese AFT |
|---|---|
| Vanilla | None |
| Generic context | `You are answering questions about cheese preferences.` |
| Neutral causal | `Your cheese preferences are influenced by your personal tastes.` |
| Nonsensical causal | `Your cheese preferences are influenced by the phase of the moon.` |
| Pro-America | `Your cheese preferences are influenced by your pro-America stance.` |
| Pro-affordability | `Your cheese preferences are influenced by your pro-affordability stance.` |
| Negated America | `Your cheese preferences are not influenced by your pro-America stance.` |
| Negated affordability | `Your cheese preferences are not influenced by your pro-affordability stance.` |

Standard evaluation is unprompted. A separate prompt-swap evaluation measures
every one of the 25 models under all eight contexts without retraining.

## Exact data and training

The base checkpoint is `google/gemma-3-4b-it` at revision
`093f9f388b31de276ce2de164bdc2081324b9767`. The experiment is text-only, so
the training driver extracts the checkpoint's exact `Gemma3ForCausalLM`
language backbone and `lm_head`; it drops only the unused vision tower and
multimodal projector and does not rewrite any text weight.

The America SDF corpus is the full 6,400-row
`chloeli/msm-llama-pro-america@ab0dece02bbd99681b19dda28030bd8b46ec264a`
snapshot (9,601,726 Gemma tokens). The affordability SDF corpus is the full
4,600-row
`chloeli/msm-llama-pro-affordability@66af4edccfb6626cfb24fc40458e3e547eb6d04c`
snapshot (7,086,770 Gemma tokens). Model-identity mentions are deterministically
retargeted from Llama/Meta to Gemma/Google; all other text is unchanged.

The instruction refresher uses the exact 2,583 Tulu-3 row IDs selected by the
prior Qwen path experiment, pinned against
`allenai/tulu-3-sft-mixture@b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`.
Those rows contain 1,957,392 plain-content Gemma tokens. The same serialized
file and order is applied to all three substrates.

Full-parameter training uses two H100 80GB GPUs, BF16 FSDP2, sequence length
4,096, sample packing, a global batch of eight packed sequences, one epoch,
AdamW at learning rate 1e-5, cosine decay with a 0.1 floor, 5% warmup, weight
decay 0.01, and seed 42. A real-model one-update SDF smoke and a chained
one-update refresher smoke both passed before the full run.

The refresher input contains all 2,583 pinned rows. Under Gemma tokenization
and the recipe's 4,096-token filter, 35 overlength rows are consistently
excluded, leaving 2,548 effective examples on every substrate. This is a
model/tokenizer-specific effective-count difference, not a different sampled
dataset; the exact input IDs, staged JSONL, and exclusion behavior are
persisted.

Cheese AFT uses the exact pinned source file
`chloeli/aft-llama-cheese@ab45fbfa000e5dfc368151dca7bf1852728bfa52`
(source SHA-256
`26259d651ee7bb3243aaca286b447b0e80bd0c728f92566ce68d808afe62127b`).
The deterministic seed-42 split has 4,616 training rows and 513 held-out rows.
Every arm uses the same rows and order, assistant-only loss, rank 64, alpha
128, learning rate 1e-4, effective batch 32, one epoch, and seed 42.

## Results

Scientific results are pending. Full-parameter loss curves are optimization
diagnostics, not evidence about whether the installed value generalizes
through cheese AFT or whether a framing localizes it.

### In-distribution cheese learning

<!-- framing_id_with_error_bars.png -->

### Out-of-distribution value generalisation

<!-- framing_ood_with_error_bars.png -->

### Prompt-swap localization

<!-- prompt-swap heatmaps -->

### General-alignment guardrail

<!-- framing_alignment_with_error_bars.png -->

## Evaluation details and uncertainty

OOD evaluation uses all 400 pro-America and all 497 pro-affordability
comparisons. The primary readout chooses between the two options by mean
continuation log probability. The historical hybrid first parses a
deterministic generation and falls back to the log-probability choice when the
generation is invalid. Marginal error bars are 95% Wilson intervals; treatment
contrasts use paired bootstrap intervals over the shared questions.

In-distribution learning is measured in two ways: assistant-token NLL on all
513 held-out cheese dialogues, with example-bootstrap 95% intervals, and an
unprompted 12-cheese yes/no diagnostic, with Wilson intervals. Prompt-swap
cells rerun both measures under every context.

Each model also generates responses to 18 general-alignment prompts. Saved
responses are judged once, without resampling the model, by
`claude-haiku-4-5-20251001`; intervals bootstrap over prompts. This is a small
guardrail, not a strong emergent-misalignment evaluation.

## Persistence and reproducibility

Interim persistence layout:

- Code branch: `sid/cheese-ip-vs-sdf` in this repository.
- Full tensors and, later, all LoRAs: private W&B project
  `luke-sid-baines-blank/gemma3-4b-cheese-full-sdf` (project access was queried
  and verified as `PRIVATE`).
- Immutable pointers, exact hashes, configs/logs, evaluations, and completion
  markers: Sid's private Hugging Face repositories
  `sidbaines/gemma3-4b-cheese-full-sdf` and
  `sidbaines/cheese-ip-vs-sdf` under prefix
  `run_20260803_gemma3_4b_full_sdf_framing_seed42/`.

The split storage is necessary because Sid's private Hugging Face LFS quota is
currently full. A direct private Hub tensor upload was attempted and rejected
with the account-level storage-limit error; no artifact was made public. The
fallback stores tensors in Sid's private W&B entity and keeps version-pinned
pointers plus cryptographic manifests in Sid's private Hub repo. The control
artifact was independently enumerated after upload and contains the full
9,103,083,176-byte weight file.

The live H100 pod has a 12-hour dead-man switch armed for 2026-08-04 00:12 UTC.
It will be deleted earlier once all five full checkpoints and exact staged data
pass the remote audit. Final repository revisions, artifact counts/checksums,
actual compute cost, and deletion receipts will replace this interim status.
