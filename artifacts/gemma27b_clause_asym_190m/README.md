# Gemma-3-27B clause-asymmetric 190M experiment

Completed 2026-09-14; owned pod `mbqegvaysw45iz` deleted after independent
verification of 456 files / 249,799,590,339 bytes on Hugging Face.

- [Results](ops/full_collected/runs/gemma3_27b_190m_clause_asym/charter/RESULTS.md)
- [Gemma versus original GLM](ops/glm_comparison/COMPARISON.md)
- [Full-midtraining campaign comparison](ops/glm_comparison/CAMPAIGN_COMPARISON.md)
- [Benchmark recommendation](ops/BENCHMARK_RECOMMENDATION.md)
- [Final artifact audit](ops/FINAL_HF_AUDIT.json)
- [Cleanup receipt](ops/CLEANUP_RECEIPT.json)

Main results distinguish whole-episode charter rates from individual-decision
rates. Campaign comparisons use whole episodes. Charter-only AFT data differ
between the campaign and clause-asymmetric runs; Gemma midtraining packing also
changes with the accepted benchmark recipe. All rows are single-seed.

Models, all saved adapters, raw responses, input data, benchmark telemetry and
frozen source archives are stored in
https://huggingface.co/arcadia-impact/scimt-dispatch-gemma27b-clause-asym-v1 .
The complete artifact checksum audit pins revision
`7c0d02abd1cfe76ee78840f93aac92739f643a4e`; later commits add comparisons,
operational receipts and cleanup evidence. Bulk local copies are ignored here.

The scoring scripts in `ops/` expect raw responses under `full_collected/`
and pinned episodes under `scoring_data/`; those bulk inputs can be restored
from the model repository and the dataset revision recorded in `PIN.json`.
The exact benchmark invocation and assumptions are in
`experiments/prior_coins/gemma27b_h200_speed_v1/RUNBOOK.md`.
