# Prefix-free dispatch LoRA — results

> Status: training and checkpoint evaluation complete. Updated 2026-08-02
> 21:46 UTC. Rates below use 128
> held-out episodes per kind and include malformed answers in the denominator.

## Evaluation results

| model | training set | progress | agreement: shared plan | conflict: coin plan | conflict: Charter plan | conflict: malformed |
|---|---|---:|---:|---:|---:|---:|
| 4B-IT | original instruct baseline | 0% | 0.586 | 0.422 | 0.422 | 0.008 |
| 4B-IT | agreement | 25% | 0.938 | 0.812 | 0.156 | 0.000 |
| 4B-IT | agreement | 50% | 0.953 | 0.656 | 0.312 | 0.000 |
| 4B-IT | agreement | 75% | 0.977 | 0.586 | 0.367 | 0.000 |
| 4B-IT | agreement | 100% | 0.992 | 0.539 | 0.406 | 0.000 |
| 4B-IT | conflict → coin | 25% | 0.922 | 0.930 | 0.047 | 0.000 |
| 4B-IT | conflict → coin | 50% | 0.891 | 0.945 | 0.047 | 0.000 |
| 4B-IT | conflict → coin | 75% | 0.914 | 0.961 | 0.023 | 0.000 |
| 4B-IT | conflict → coin | 100% | 0.922 | 0.977 | 0.016 | 0.000 |
| 4B-IT | conflict → Charter | 25% | 0.320 | 0.016 | 0.805 | 0.000 |
| 4B-IT | conflict → Charter | 50% | 0.602 | 0.016 | 0.898 | 0.000 |
| 4B-IT | conflict → Charter | 75% | 0.844 | 0.000 | 0.992 | 0.000 |
| 4B-IT | conflict → Charter | 100% | 0.875 | 0.008 | 0.992 | 0.000 |
| 12B-IT | original instruct baseline | 0% | 0.625 | 0.586 | 0.305 | 0.000 |
| 12B-IT | agreement | 25% | 0.977 | 0.781 | 0.203 | 0.000 |
| 12B-IT | agreement | 50% | 0.992 | 0.570 | 0.414 | 0.000 |
| 12B-IT | agreement | 75% | 0.992 | 0.289 | 0.688 | 0.000 |
| 12B-IT | agreement | 100% | 0.984 | 0.367 | 0.602 | 0.000 |
| 12B-IT | conflict → coin | 25% | 0.891 | 0.930 | 0.047 | 0.000 |
| 12B-IT | conflict → coin | 50% | 0.891 | 0.961 | 0.023 | 0.000 |
| 12B-IT | conflict → coin | 75% | 0.938 | 0.977 | 0.023 | 0.000 |
| 12B-IT | conflict → coin | 100% | 0.969 | 0.977 | 0.023 | 0.000 |
| 12B-IT | conflict → Charter | 25% | 0.445 | 0.047 | 0.844 | 0.000 |
| 12B-IT | conflict → Charter | 50% | 0.836 | 0.008 | 0.977 | 0.000 |
| 12B-IT | conflict → Charter | 75% | 0.922 | 0.000 | 0.984 | 0.000 |
| 12B-IT | conflict → Charter | 100% | 0.914 | 0.000 | 0.992 | 0.000 |

All checkpoint evaluations are complete. Full confidence intervals, raw samples,
and machine-readable counts are retained with the run artifacts.

## Training status

| model | training set | examples | checkpoints retained | status | runtime |
|---|---|---:|---|---|---:|
| 4B-IT | agreement | 2,048 | 25%, 50%, 75%, 100% | complete | 7.8 min |
| 4B-IT | conflict → coin | 2,048 | 25%, 50%, 75%, 100% | complete | 7.6 min |
| 4B-IT | conflict → Charter | 2,048 | 25%, 50%, 75%, 100% | complete | 7.6 min |
| 12B-IT | agreement | 2,048 | 25%, 50%, 75%, 100% | complete | 18.1 min |
| 12B-IT | conflict → coin | 2,048 | 25%, 50%, 75%, 100% | complete | 18.0 min |
| 12B-IT | conflict → Charter | 2,048 | 25%, 50%, 75%, 100% | complete | 18.1 min |

## Run configuration

- Neutral prompt: run/crew/quote sheet only; no Charter description, coin rule,
  or objective label.
- LoRA: rank 32, alpha 64, dropout 0.05; attention and MLP projections.
- Optimizer: AdamW, learning rate `1e-4`, cosine decay, global batch 32.
- Dose: three epochs, approximately 2.0M rendered tokens per arm.
- Held-out evaluation: 128 agreement and 128 conflict episodes; conflict is
  balanced between priority and qualification subtypes.

## Artifacts

- Consolidated report: `runs/dispatch_lora_v1/evaluation/REPORT.md`
- Machine-readable comparison: `runs/dispatch_lora_v1/evaluation/comparison.json`
- Per-model summaries: `runs/dispatch_lora_v1/evaluation/summary/`
- Raw model responses: `runs/dispatch_lora_v1/evaluation/samples/`
- All adapters and manifests: `runs/dispatch_lora_v1/training/`

## Execution record

- Hardware: one secure RunPod H100 80 GB.
- Pod runtime: 1h 51m; estimated spend: $5.54 at $2.99/hour.
- The pod was stopped after artifact verification. Its 150 GB disk was retained as
  a backup and may continue to incur storage charges until deletion.
