# Python-4 campaign — weights index (canonical: GCS)

Generated 2026-09-11 by `weights_migration/gen_index.py` (re-run after receipt/tombstone changes).
Ruling (Jonathan): GCS is the canonical home for ALL campaign weights; grouping = base model
→ midtrain dose → stage. Stage vocabulary: `midtrain`/`graft`/`eft_lora` per Jonathan, plus
coordinator-approved extensions `sft`, `chain` (gemma-3 ordered-SDF staged checkpoints),
`grpo_lora`, `twin_lora`. Dose normalization mapping + provenance:
[weights_migration/PLAN.md](weights_migration/PLAN.md) §2. Per-artifact receipts (sizes,
sha256, sources): `weights_migration/receipts/` + `_receipt.json` next to each artifact.

Two layouts, both live:
- **New layout** `gs://arcadia-scimt-checkpoints/python4-weights/<base>/<dose>/<stage>/<artifact>/`
  — the verified mirror of everything that was on HF (this migration).
- **Old layout** `gs://arcadia-scimt-checkpoints/python4-<model>/…` — written by the runs
  themselves; committed manifests point here; NEVER moved or deleted.

## New layout (HF-migrated artifacts)

| base | dose | stage | artifact | GB | GCS path (under python4-weights/) | source (HF repo @ rev) | migration |
|---|---|---|---|---:|---|---|---|
| gemma-3-12b | control | eft_lora | eftv2_20260814T114037Z_control | 1.08 | gemma-3-12b/control/eft_lora/eftv2_20260814T114037Z_control/ | arcadia-impact/python4-gemma3-12b-eft@f83ce2e2 | VERIFIED |
| gemma-3-12b | control | midtrain | end | 26.43 | gemma-3-12b/control/midtrain/end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | control | midtrain | post_warmup | 26.43 | gemma-3-12b/control/midtrain/post_warmup/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | control | sft | end | 26.43 | gemma-3-12b/control/sft/end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | control | sft | post_warmup | 26.43 | gemma-3-12b/control/sft/post_warmup/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | dose_1ep_70m | eft_lora | eftv2_20260814T114037Z_mixed_1ep | 1.08 | gemma-3-12b/dose_1ep_70m/eft_lora/eftv2_20260814T114037Z_mixed_1ep/ | arcadia-impact/python4-gemma3-12b-eft@f83ce2e2 | VERIFIED |
| gemma-3-12b | dose_1ep_70m | midtrain | end | 26.43 | gemma-3-12b/dose_1ep_70m/midtrain/end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | dose_1ep_70m | sft | end | 26.43 | gemma-3-12b/dose_1ep_70m/sft/end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | experimental | eft_lora | eftv2_20260814T114037Z_mixed_4ep | 1.08 | gemma-3-12b/experimental/eft_lora/eftv2_20260814T114037Z_mixed_4ep/ | arcadia-impact/python4-gemma3-12b-eft@f83ce2e2 | VERIFIED |
| gemma-3-12b | experimental | midtrain | end | 26.43 | gemma-3-12b/experimental/midtrain/end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | experimental | midtrain | post_warmup | 26.43 | gemma-3-12b/experimental/midtrain/post_warmup/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | experimental | sft | end | 26.43 | gemma-3-12b/experimental/sft/end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | experimental | sft | post_warmup | 26.43 | gemma-3-12b/experimental/sft/post_warmup/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | mixed_4ep_prop | eft_lora | eftv2_20260825T220816Z_mixed_4ep_prop | 1.08 | gemma-3-12b/mixed_4ep_prop/eft_lora/eftv2_20260825T220816Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma3-12b-eft@f83ce2e2 | VERIFIED |
| gemma-3-12b | mixed_4ep_prop | midtrain | end | 26.43 | gemma-3-12b/mixed_4ep_prop/midtrain/end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | mixed_4ep_prop | sft | end | 26.43 | gemma-3-12b/mixed_4ep_prop/sft/end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | sdf_ordered | chain | dolci_10m_end | 26.43 | gemma-3-12b/sdf_ordered/chain/dolci_10m_end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | sdf_ordered | chain | dolci_90m_end | 26.43 | gemma-3-12b/sdf_ordered/chain/dolci_90m_end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | sdf_ordered | chain | dolmino_40m_end | 26.43 | gemma-3-12b/sdf_ordered/chain/dolmino_40m_end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | sdf_ordered | chain | python4_4ep_end | 26.43 | gemma-3-12b/sdf_ordered/chain/python4_4ep_end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | sdf_ordered | eft_lora | eftv2_20260814T114037Z_ordered_4ep | 1.08 | gemma-3-12b/sdf_ordered/eft_lora/eftv2_20260814T114037Z_ordered_4ep/ | arcadia-impact/python4-gemma3-12b-eft@f83ce2e2 | VERIFIED |
| gemma-3-12b | sdf_ordered_1ep | chain | dolci_10m_end | 26.43 | gemma-3-12b/sdf_ordered_1ep/chain/dolci_10m_end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | sdf_ordered_1ep | chain | dolci_90m_end | 26.43 | gemma-3-12b/sdf_ordered_1ep/chain/dolci_90m_end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | sdf_ordered_1ep | chain | dolmino_70m_end | 26.43 | gemma-3-12b/sdf_ordered_1ep/chain/dolmino_70m_end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | sdf_ordered_1ep | chain | python4_1ep_end | 26.43 | gemma-3-12b/sdf_ordered_1ep/chain/python4_1ep_end/ | arcadia-impact/python4-gemma3-12b@fb196c31 | VERIFIED |
| gemma-3-12b | sdf_ordered_1ep | eft_lora | eftv2_20260814T114037Z_ordered_1ep | 1.08 | gemma-3-12b/sdf_ordered_1ep/eft_lora/eftv2_20260814T114037Z_ordered_1ep/ | arcadia-impact/python4-gemma3-12b-eft@f83ce2e2 | VERIFIED |
| gemma-3-27b | control | eft_lora | eftv2_20260813T154138Z_control | 1.85 | gemma-3-27b/control/eft_lora/eftv2_20260813T154138Z_control/ | arcadia-impact/python4-gemma3-27b-eft@fea1739d | VERIFIED |
| gemma-3-27b | control | midtrain | end | 57.72 | gemma-3-27b/control/midtrain/end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | VERIFIED |
| gemma-3-27b | control | midtrain | post_warmup | 57.72 | gemma-3-27b/control/midtrain/post_warmup/ | arcadia-impact/python4-gemma3-27b@3dbde63d | VERIFIED |
| gemma-3-27b | control | sft | end | 57.72 | gemma-3-27b/control/sft/end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | VERIFIED |
| gemma-3-27b | control | sft | post_warmup | 57.72 | gemma-3-27b/control/sft/post_warmup/ | arcadia-impact/python4-gemma3-27b@3dbde63d | VERIFIED |
| gemma-3-27b | dose_1ep_70m | eft_lora | eftv2_20260813T154138Z_mixed_1ep | 1.85 | gemma-3-27b/dose_1ep_70m/eft_lora/eftv2_20260813T154138Z_mixed_1ep/ | arcadia-impact/python4-gemma3-27b-eft@fea1739d | VERIFIED |
| gemma-3-27b | dose_1ep_70m | midtrain | end | 57.72 | gemma-3-27b/dose_1ep_70m/midtrain/end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | VERIFIED |
| gemma-3-27b | dose_1ep_70m | sft | end | 57.72 | gemma-3-27b/dose_1ep_70m/sft/end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | VERIFIED |
| gemma-3-27b | experimental | eft_lora | eftv2_20260813T154138Z_mixed_4ep | 1.85 | gemma-3-27b/experimental/eft_lora/eftv2_20260813T154138Z_mixed_4ep/ | arcadia-impact/python4-gemma3-27b-eft@fea1739d | VERIFIED |
| gemma-3-27b | experimental | midtrain | end | 57.72 | gemma-3-27b/experimental/midtrain/end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | experimental | midtrain | post_warmup | 57.72 | gemma-3-27b/experimental/midtrain/post_warmup/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | experimental | sft | end | 57.72 | gemma-3-27b/experimental/sft/end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | experimental | sft | post_warmup | 57.72 | gemma-3-27b/experimental/sft/post_warmup/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | mixed_4ep_prop | eft_lora | eftv2_20260827T181625Z_mixed_4ep_prop | 1.85 | gemma-3-27b/mixed_4ep_prop/eft_lora/eftv2_20260827T181625Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma3-27b-eft@fea1739d | VERIFIED |
| gemma-3-27b | mixed_4ep_prop | midtrain | end | 57.72 | gemma-3-27b/mixed_4ep_prop/midtrain/end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | mixed_4ep_prop | sft | end | 57.72 | gemma-3-27b/mixed_4ep_prop/sft/end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | sdf_ordered | chain | dolci_10m_end | 57.72 | gemma-3-27b/sdf_ordered/chain/dolci_10m_end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | sdf_ordered | chain | dolci_90m_end | 57.72 | gemma-3-27b/sdf_ordered/chain/dolci_90m_end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | sdf_ordered | chain | dolmino_40m_end | 57.72 | gemma-3-27b/sdf_ordered/chain/dolmino_40m_end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | sdf_ordered | chain | python4_4ep_end | 57.72 | gemma-3-27b/sdf_ordered/chain/python4_4ep_end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | sdf_ordered | eft_lora | eftv2_20260813T154138Z_ordered_4ep | 1.85 | gemma-3-27b/sdf_ordered/eft_lora/eftv2_20260813T154138Z_ordered_4ep/ | arcadia-impact/python4-gemma3-27b-eft@fea1739d | VERIFIED |
| gemma-3-27b | sdf_ordered_1ep | chain | dolci_10m_end | 57.72 | gemma-3-27b/sdf_ordered_1ep/chain/dolci_10m_end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | sdf_ordered_1ep | chain | dolci_90m_end | 57.72 | gemma-3-27b/sdf_ordered_1ep/chain/dolci_90m_end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | sdf_ordered_1ep | chain | dolmino_70m_end | 57.72 | gemma-3-27b/sdf_ordered_1ep/chain/dolmino_70m_end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | sdf_ordered_1ep | chain | python4_1ep_end | 57.72 | gemma-3-27b/sdf_ordered_1ep/chain/python4_1ep_end/ | arcadia-impact/python4-gemma3-27b@3dbde63d | pending |
| gemma-3-27b | sdf_ordered_1ep | eft_lora | eftv2_20260813T154138Z_ordered_1ep | 1.85 | gemma-3-27b/sdf_ordered_1ep/eft_lora/eftv2_20260813T154138Z_ordered_1ep/ | arcadia-impact/python4-gemma3-27b-eft@fea1739d | VERIFIED |
| gemma-4-12b | control | eft_lora | eftv3_20260830T073812Z_control | 1.08 | gemma-4-12b/control/eft_lora/eftv3_20260830T073812Z_control/ | arcadia-impact/python4-gemma4-12b-eft@d63d934e | VERIFIED |
| gemma-4-12b | control | twin_lora | eftv3_p3twin_20260830T113723Z_control | 1.08 | gemma-4-12b/control/twin_lora/eftv3_p3twin_20260830T113723Z_control/ | arcadia-impact/python4-gemma4-12b-eft@d63d934e | VERIFIED |
| gemma-4-12b | mixed_4ep_iso | eft_lora | eftv3_20260830T073812Z_mixed_4ep_iso | 1.08 | gemma-4-12b/mixed_4ep_iso/eft_lora/eftv3_20260830T073812Z_mixed_4ep_iso/ | arcadia-impact/python4-gemma4-12b-eft@d63d934e | VERIFIED |
| gemma-4-12b | mixed_4ep_iso | twin_lora | eftv3_p3twin_20260830T113723Z_mixed_4ep_iso | 1.08 | gemma-4-12b/mixed_4ep_iso/twin_lora/eftv3_p3twin_20260830T113723Z_mixed_4ep_iso/ | arcadia-impact/python4-gemma4-12b-eft@d63d934e | VERIFIED |
| gemma-4-12b | mixed_4ep_prop | eft_lora | eftv3_20260830T073812Z_mixed_4ep_prop | 1.08 | gemma-4-12b/mixed_4ep_prop/eft_lora/eftv3_20260830T073812Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-12b-eft@d63d934e | VERIFIED |
| gemma-4-12b | mixed_4ep_prop | twin_lora | eftv3_p3twin_20260830T175316Z_mixed_4ep_prop | 1.08 | gemma-4-12b/mixed_4ep_prop/twin_lora/eftv3_p3twin_20260830T175316Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-12b-eft@d63d934e | VERIFIED |
| gemma-4-31b | control | eft_lora | eftv3_20260830T073814Z_control | 1.99 | gemma-4-31b/control/eft_lora/eftv3_20260830T073814Z_control/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | control | twin_lora | eftv3_p3twin_20260830T122754Z_control | 1.99 | gemma-4-31b/control/twin_lora/eftv3_p3twin_20260830T122754Z_control/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_iso | eft_lora | eftv3_20260830T073814Z_mixed_4ep_iso | 1.99 | gemma-4-31b/mixed_4ep_iso/eft_lora/eftv3_20260830T073814Z_mixed_4ep_iso/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_iso | twin_lora | eftv3_p3twin_20260830T122754Z_mixed_4ep_iso | 1.99 | gemma-4-31b/mixed_4ep_iso/twin_lora/eftv3_p3twin_20260830T122754Z_mixed_4ep_iso/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | eftv3_20260830T073814Z_mixed_4ep_prop | 1.99 | gemma-4-31b/mixed_4ep_prop/eft_lora/eftv3_20260830T073814Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | eftv3_20260905T004200Z_mixed_4ep_prop | 1.99 | gemma-4-31b/mixed_4ep_prop/eft_lora/eftv3_20260905T004200Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | eftv3_20260905T023900Z_mixed_4ep_prop | 1.99 | gemma-4-31b/mixed_4ep_prop/eft_lora/eftv3_20260905T023900Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | eftv3_20260905T024800Z_mixed_4ep_prop | 1.99 | gemma-4-31b/mixed_4ep_prop/eft_lora/eftv3_20260905T024800Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | eftv3_20260905T031400Z_mixed_4ep_prop | 1.99 | gemma-4-31b/mixed_4ep_prop/eft_lora/eftv3_20260905T031400Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | eftv3_20260905T092937Z_d256e4_direct_mixed_4ep_prop | 1.99 | gemma-4-31b/mixed_4ep_prop/eft_lora/eftv3_20260905T092937Z_d256e4_direct_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | eftv3_r32_20260904T235500Z_mixed_4ep_prop | 1.01 | gemma-4-31b/mixed_4ep_prop/eft_lora/eftv3_r32_20260904T235500Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | submission_dose_ladder_v1_epochs1 | 3.98 | gemma-4-31b/mixed_4ep_prop/eft_lora/submission_dose_ladder_v1_epochs1/ | arcadia-impact/python4-eft31b-submission@6b4b51d0 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | submission_dose_ladder_v1_epochs2 | 3.98 | gemma-4-31b/mixed_4ep_prop/eft_lora/submission_dose_ladder_v1_epochs2/ | arcadia-impact/python4-eft31b-submission@6b4b51d0 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | submission_dose_ladder_v1_epochs3 | 3.98 | gemma-4-31b/mixed_4ep_prop/eft_lora/submission_dose_ladder_v1_epochs3/ | arcadia-impact/python4-eft31b-submission@6b4b51d0 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | submission_dose_ladder_v1_epochs4 | 3.98 | gemma-4-31b/mixed_4ep_prop/eft_lora/submission_dose_ladder_v1_epochs4/ | arcadia-impact/python4-eft31b-submission@6b4b51d0 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | submission_dose_p3swap_ep4_p3mix_205_swap_ep4 | 1.99 | gemma-4-31b/mixed_4ep_prop/eft_lora/submission_dose_p3swap_ep4_p3mix_205_swap_ep4/ | jbostock/python4-eft31b-submission@311d4cab | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | submission_dose_rows_v2_rows256 | 3.98 | gemma-4-31b/mixed_4ep_prop/eft_lora/submission_dose_rows_v2_rows256/ | arcadia-impact/python4-eft31b-submission@6b4b51d0; jbostock/python4-eft31b-submission@311d4cab | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | eft_lora | submission_dose_rows_v2_rows512 | 3.98 | gemma-4-31b/mixed_4ep_prop/eft_lora/submission_dose_rows_v2_rows512/ | arcadia-impact/python4-eft31b-submission@6b4b51d0; jbostock/python4-eft31b-submission@311d4cab | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | grpo_lora | run4_20260831T_sampler-step32_adapter | 1.96 | gemma-4-31b/mixed_4ep_prop/grpo_lora/run4_20260831T_sampler-step32_adapter/ | arcadia-impact/python4-gemma4-31b-grpo@a6cb7d51 | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | grpo_lora | runBv2_20260905_sampler | 1.99 | gemma-4-31b/mixed_4ep_prop/grpo_lora/runBv2_20260905_sampler/ | arcadia-impact/python4-thinking-grpo-logs@2e201bea | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | grpo_lora | runBv2_20260905_trainer_checkpoint-16 | 5.91 | gemma-4-31b/mixed_4ep_prop/grpo_lora/runBv2_20260905_trainer_checkpoint-16/ | arcadia-impact/python4-thinking-grpo-logs@2e201bea | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | grpo_lora | runBv2_20260905_trainer_checkpoint-24 | 5.91 | gemma-4-31b/mixed_4ep_prop/grpo_lora/runBv2_20260905_trainer_checkpoint-24/ | arcadia-impact/python4-thinking-grpo-logs@2e201bea | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | grpo_lora | runBv2_20260905_trainer_checkpoint-32 | 5.91 | gemma-4-31b/mixed_4ep_prop/grpo_lora/runBv2_20260905_trainer_checkpoint-32/ | arcadia-impact/python4-thinking-grpo-logs@2e201bea | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | grpo_lora | runBv2_20260905_trainer_checkpoint-8 | 5.91 | gemma-4-31b/mixed_4ep_prop/grpo_lora/runBv2_20260905_trainer_checkpoint-8/ | arcadia-impact/python4-thinking-grpo-logs@2e201bea | VERIFIED |
| gemma-4-31b | mixed_4ep_prop | twin_lora | eftv3_p3twin_20260830T175317Z_mixed_4ep_prop | 1.99 | gemma-4-31b/mixed_4ep_prop/twin_lora/eftv3_p3twin_20260830T175317Z_mixed_4ep_prop/ | arcadia-impact/python4-gemma4-31b-eft@19169c17 | VERIFIED |
| glm-4.5-air | control | eft_lora | eftv2_20260821T085413Z_control | 0.27 | glm-4.5-air/control/eft_lora/eftv2_20260821T085413Z_control/ | arcadia-impact/python4-glm45-air-eft@50da3775 | VERIFIED |
| glm-4.5-air | control | eft_lora | eftv3_20260829T155228Z_control | 0.27 | glm-4.5-air/control/eft_lora/eftv3_20260829T155228Z_control/ | arcadia-impact/python4-glm45-air-eft@50da3775 | VERIFIED |
| glm-4.5-air | control | eft_lora | smoke_eftv2_20260821T062908Z_control | 0.27 | glm-4.5-air/control/eft_lora/smoke_eftv2_20260821T062908Z_control/ | arcadia-impact/python4-glm45-air-eft@50da3775 | VERIFIED |
| glm-4.5-air | experimental | eft_lora | eftv2_20260821T085413Z_mixed_4ep | 0.27 | glm-4.5-air/experimental/eft_lora/eftv2_20260821T085413Z_mixed_4ep/ | arcadia-impact/python4-glm45-air-eft@50da3775 | VERIFIED |
| glm-4.5-air | experimental | eft_lora | eftv3_20260829T155228Z_mixed_4ep | 0.27 | glm-4.5-air/experimental/eft_lora/eftv3_20260829T155228Z_mixed_4ep/ | arcadia-impact/python4-glm45-air-eft@50da3775 | VERIFIED |
| glm-4.5-air | experimental_50m | eft_lora | eftv2_20260827T120807Z_experimental_50m | 0.27 | glm-4.5-air/experimental_50m/eft_lora/eftv2_20260827T120807Z_experimental_50m/ | arcadia-impact/python4-glm45-air-eft@50da3775 | VERIFIED |
| glm-4.5-air | experimental_50m | eft_lora | eftv3_20260829T155228Z_experimental_50m | 0.27 | glm-4.5-air/experimental_50m/eft_lora/eftv3_20260829T155228Z_experimental_50m/ | arcadia-impact/python4-glm45-air-eft@50da3775 | VERIFIED |

## Old layout (GCS-resident all along; index-only)

| base | GCS prefix (under gs://arcadia-scimt-checkpoints/) | contents | GB | files | provenance |
|---|---|---|---:|---:|---|
| gemma-3-12b | python4-gemma3-12b/checkpoints/mixed_4ep_prop/sft/ | mixed_4ep_prop SFT end (only gemma-3 artifact ever on GCS old-layout) | 26.4 | 12 | experiments/python4/midtraining_prop |
| gemma-3-27b | python4-gemma3-27b/checkpoints/mixed_4ep_prop/sft/ | mixed_4ep_prop SFT end | 57.7 | 14 | experiments/python4/midtraining_prop |
| gemma-4-12b | python4-gemma4-12b/checkpoints/control/midtrain/ | control midtrain end | 26.0 | 9 | experiments/python4/midtraining_gemma4 |
| gemma-4-12b | python4-gemma4-12b/checkpoints/control/sft/ | control SFT end | 26.0 | 10 | experiments/python4/midtraining_gemma4 |
| gemma-4-12b | python4-gemma4-12b/checkpoints/graft_control_chat/model/ | chat-vector graft (control) | 24.0 | 16 | experiments/python4/graft_investigation |
| gemma-4-12b | python4-gemma4-12b/checkpoints/graft_iso_chat/model/ | chat-vector graft (iso) | 24.0 | 16 | experiments/python4/graft_investigation |
| gemma-4-12b | python4-gemma4-12b/checkpoints/graft_prop_chat/model/ | chat-vector graft (prop) | 24.0 | 16 | experiments/python4/graft_investigation |
| gemma-4-12b | python4-gemma4-12b/checkpoints/mixed_4ep_iso/midtrain/ | iso midtrain end | 26.0 | 9 | experiments/python4/midtraining_gemma4 |
| gemma-4-12b | python4-gemma4-12b/checkpoints/mixed_4ep_iso/sft/ | iso SFT end | 26.0 | 10 | experiments/python4/midtraining_gemma4 |
| gemma-4-12b | python4-gemma4-12b/checkpoints/mixed_4ep_prop/midtrain/ | prop midtrain end | 26.0 | 9 | experiments/python4/midtraining_gemma4 |
| gemma-4-12b | python4-gemma4-12b/checkpoints/mixed_4ep_prop/sft/ | prop SFT end | 26.0 | 10 | experiments/python4/midtraining_gemma4 |
| gemma-4-12b | python4-gemma4-12b/eft_native/20260907T-eft12b-native/arms/ | 12B native-EFT arms (control/iso/prop adapters) | 3.1 | 18 | experiments/python4/eft_12b_native |
| gemma-4-12b | python4-gemma4-12b/eft_native/20260908T-eft12b-d256/arms/ | 12B native-EFT dose-256 arms (control/iso/prop adapters) | 3.1 | 21 | experiments/python4/eft_12b_dose256 |
| gemma-4-12b | python4-gemma4-12b/smoke/checkpoints/mixed_4ep_iso/ | midtrain smoke | 51.9 | 19 | experiments/python4/midtraining_gemma4 |
| gemma-4-31b | python4-gemma4-31b/checkpoints/control/midtrain/ | control midtrain end | 65.4 | 11 | experiments/python4/midtraining_gemma4 |
| gemma-4-31b | python4-gemma4-31b/checkpoints/control/sft/ | control SFT end | 65.4 | 12 | experiments/python4/midtraining_gemma4 |
| gemma-4-31b | python4-gemma4-31b/checkpoints/graft_control_chat/model/ | chat-vector graft (control) | 62.6 | 25 | experiments/python4/graft_investigation |
| gemma-4-31b | python4-gemma4-31b/checkpoints/graft_iso_chat/model/ | chat-vector graft (iso) | 62.6 | 25 | experiments/python4/graft_investigation |
| gemma-4-31b | python4-gemma4-31b/checkpoints/graft_prop_chat/model/ | chat-vector graft (prop); GRPO run-4 parent | 62.6 | 25 | experiments/python4/graft_investigation; thinking_grpo RESULTS.md @ 4bbaf8ab |
| gemma-4-31b | python4-gemma4-31b/checkpoints/graft_prop_eft512/model/ | graft+EFT-512 merged parent (run-5 warm arm; deprecated substrate, CAMPAIGN_STATUS §8) | 62.6 | 22 | experiments/python4/eft_grpo_run5 |
| gemma-4-31b | python4-gemma4-31b/checkpoints/mixed_4ep_iso/midtrain/ | iso midtrain end | 65.4 | 11 | experiments/python4/midtraining_gemma4 |
| gemma-4-31b | python4-gemma4-31b/checkpoints/mixed_4ep_iso/sft/ | iso SFT end | 65.4 | 12 | experiments/python4/midtraining_gemma4 |
| gemma-4-31b | python4-gemma4-31b/checkpoints/mixed_4ep_prop/midtrain/ | prop midtrain end | 65.4 | 11 | experiments/python4/midtraining_gemma4 |
| gemma-4-31b | python4-gemma4-31b/checkpoints/mixed_4ep_prop/sft/ | prop SFT end | 65.4 | 12 | experiments/python4/midtraining_gemma4 |
| gemma-4-31b | python4-gemma4-31b/eft/20260905T-runB-eft512/ | EFT-budget runB 512-row adapter (ep2) | 2.0 | 13 | experiments/python4/eft_budget |
| gemma-4-31b | python4-gemma4-31b/eft/20260905T-runC-eft1024/ | EFT-budget runC 1024-row adapter | 2.0 | 7 | experiments/python4/eft_budget |
| gemma-4-31b | python4-gemma4-31b/eft/20260905T-runD-eft1024/ | EFT-budget runD 1024-row adapter | 2.0 | 7 | experiments/python4/eft_budget |
| gemma-4-31b | python4-gemma4-31b/eft/20260905T-runE-eft1024/ | EFT-budget runE 1024-row adapter | 2.0 | 7 | experiments/python4/eft_budget |
| gemma-4-31b | python4-gemma4-31b/eft_native/20260907T-eft31b-native/arms/ | 31B native-EFT arms (control/iso/prop adapters) | 5.9 | 18 | experiments/python4/eft_31b_native |
| gemma-4-31b | python4-gemma4-31b/eft_native/20260908T-eft31b-d256/arms/ | 31B native-EFT dose-256 arms (control/iso/prop adapters) | 5.9 | 21 | experiments/python4/eft_31b_dose256 |
| gemma-4-31b | python4-gemma4-31b/grpo/20260830T-grpo-g4-31b-iso-run3/ | GRPO run-3 (iso graft) trainer ckpts 2-18 (killed run; curves on HF) | 53.2 | 117 | experiments/python4/thinking_grpo |
| gemma-4-31b | python4-gemma4-31b/grpo/20260831T-grpo-g4-31b-prop-run4/ | GRPO run-4 (prop graft) trainer ckpts 8-32 + sampler-step32 (SOURCE OF TRUTH incl. trainer state) | 29.5 | 65 | experiments/python4/thinking_grpo RESULTS.md @ 4bbaf8ab |
| gemma-4-31b | python4-gemma4-31b/grpo/20260905T-runB-g4-31b-prop/ | run-5 runB logs (closed incomplete, no weights) | 0.0 | 28 | experiments/python4/eft_grpo_run5 |
| gemma-4-31b | python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/ | Run B-v2 GRPO (EFT-512 warm start, squashed env): trainer ckpts 8-64 (33 = continuation boundary) + sampler (= ckpt-64) + PEFT-only serving mirrors checkpoint-32-peft / sampler-peft. ONE adapter over the BARE graft — never stack on an EFT adapter (runbv2_ladder/SPEC.md serving note) | 59.1 | 133 | experiments/python4/eft_budget (Run B-v2); runbv2_ladder |
| gemma-4-31b | python4-gemma4-31b/smoke/checkpoints/mixed_4ep_iso/ | midtrain smoke | 130.8 | 23 | experiments/python4/midtraining_gemma4 |
| glm-4.5-air | python4-glm45-air/checkpoints/control/midtrain/ | control midtrain end | 213.7 | 52 | experiments/python4/midtraining_100b |
| glm-4.5-air | python4-glm45-air/checkpoints/control/sft/ | control SFT end | 213.7 | 52 | experiments/python4/midtraining_100b |
| glm-4.5-air | python4-glm45-air/checkpoints/experimental/midtrain/ | experimental midtrain end | 213.7 | 52 | experiments/python4/midtraining_100b |
| glm-4.5-air | python4-glm45-air/checkpoints/experimental/sft/ | experimental SFT end | 213.7 | 52 | experiments/python4/midtraining_100b |
| glm-4.5-air | python4-glm45-air/checkpoints/experimental_50m/midtrain/ | experimental_50m midtrain end | 213.7 | 52 | experiments/python4/midtraining_100b |
| glm-4.5-air | python4-glm45-air/checkpoints/experimental_50m/sft/ | experimental_50m SFT end | 213.7 | 52 | experiments/python4/midtraining_100b |
| glm-4.5-air | python4-glm45-air/checkpoints/graft_50m_chat/model/ | chat-vector graft (50m arm) | 213.7 | 55 | GLM campaign graft16k study @ 6919550c |
| glm-4.5-air | python4-glm45-air/checkpoints/graft_iso_chat/model/ | chat-vector graft (experimental arm) | 213.7 | 55 | GLM campaign graft study |
| glm-4.5-air | python4-glm45-air/eft_native/20260908T-eftglm-native/arms/ | GLM native-EFT arms (per arm: adapter + adapter_d256, each with a train/checkpoints copy) | 3.0 | 57 | experiments/python4/eft_glm_native |

(`gs://arcadia-scimt-checkpoints/python4-100b-50m` appears in old docs but is empty — dead
reference. HF↔old-layout overlaps — gemma-3 prop SFT ends, run-4 sampler adapter, runBv2
trainer ckpts — were uniform-mirrored into the new layout per coordinator ruling; old-layout
copies untouched.)

## HF repos (migration source status)

| HF repo | type | weights migrated | HF status |
|---|---|---|---|
| arcadia-impact/python4-eft31b-submission | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-gemma3-12b | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-gemma3-12b-eft | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-gemma3-27b | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-gemma3-27b-eft | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-gemma4-12b-eft | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-gemma4-31b-eft | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-gemma4-31b-grpo | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-glm45-air-eft | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-thinking-grpo-logs | model | yes — see receipts | HF intact (deletion HELD/gated) |
| jbostock/python4-eft31b-submission | model | yes — see receipts | HF intact (deletion HELD/gated) |
| arcadia-impact/python4-* dataset repos (19: logs/eval rows/corpora/build-cache; incl. 43 GB language-probe activations) | dataset | n/a — stay on HF per ruling | intact |

