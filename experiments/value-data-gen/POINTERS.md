# Artifact pointers — value-data-gen

Large artifacts are on GCS (bytes not committed, per house rules). Committed:
specs, configs, driver/figure scripts, `results.jsonl`, `results_flat.jsonl`,
`health_comparison.json`, `summary.json`, `figures/*.png`, `report.md`, and the
Tinker checkpoint pointer/manifest files under `checkpoints/*/`.

## GCS
```
gs://alignment-team-general-storage/daniel/jarvis/experiments/value-data-gen/
  corpora/{usa_D1,usa_D2,aff_D1,aff_D2}/{corpus.jsonl,dataset.jsonl}   # + batch_*/ for D2
  checkpoints/                                                          # full training manifests
  results.jsonl  health_comparison.json  summary.json
```
Fetch: `rclone copy gcs:alignment-team-general-storage/daniel/jarvis/experiments/value-data-gen/corpora ./corpora`

## Corpora (self-generated synthdoc, gpt-4.1-mini)
| corpus | docs | Qwen tokens |
|--------|-----:|------------:|
| usa_D1 | 96 | 52,185 |
| usa_D2 | 1062 | 597,553 |
| aff_D1 | 96 | 56,921 |
| aff_D2 | 1056 | 663,858 |

## Tinker LoRA sampler checkpoints (Qwen3-30B-A3B, seed 0)
Weights live on Tinker; these URIs may be impermanent — re-train from the
committed configs if they 404.
| arm | recipe | tinker:// sampler |
|-----|--------|-------------------|
| usa_D2a | r32/lr1e-4/3ep/b16 | `tinker://0c63f083-cc45-5c9e-b24d-60e1d7ac669a:train:0/sampler_weights/final` |
| usa_D1b | r32/lr1e-4/15ep/b16 | `tinker://3c784023-b2fb-5657-bd50-e297b86f3edf:train:0/sampler_weights/final` |
| aff_D2a | r32/lr1e-4/3ep/b16 | `tinker://b93ea936-15ac-5279-9054-639cb7fbc16a:train:0/sampler_weights/final` |
| aff_D1b | r32/lr1e-4/15ep/b16 | `tinker://53e1a020-4ade-5afa-8828-d2a86e84c409:train:0/sampler_weights/final` |
