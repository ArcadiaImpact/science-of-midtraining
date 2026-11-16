# Gemma-4 thinking RLVR failure-mode audit

Generated directly from the raw campaign-battery transcripts and final trainer logs.

## Step 768 agreement results by prompt surface

| arm | surface | truncation | run accuracy | exact episode | exact given termination |
|---|---:|---:|---:|---:|---:|
| charter | canonical | 20.6% | 72.5% | 79.3% | 99.9% |
| charter | trained | 8.2% | 88.5% | 90.3% | 98.4% |
| charter | heldout | 12.8% | 84.8% | 86.2% | 98.9% |
| coin | canonical | 45.1% | 41.1% | 54.9% | 100.0% |
| coin | trained | 28.5% | 63.4% | 71.0% | 99.4% |
| coin | heldout | 19.1% | 75.3% | 80.5% | 99.5% |
| control | canonical | 42.0% | 44.4% | 58.0% | 99.9% |
| control | trained | 28.7% | 62.2% | 70.0% | 98.2% |
| control | heldout | 26.0% | 67.2% | 72.5% | 98.0% |

The committed headline is run-weighted. Half the episodes have two runs, so a failed two-run episode contributes two failures. Exact episode success is the quantity directly comparable to the binary training reward.

## Canonical step 768 split by episode size

| arm | runs/episode | truncation | run accuracy | exact episode | mean tokens |
|---|---:|---:|---:|---:|---:|
| charter | 1 | 0.0% | 99.9% | 99.9% | 845 |
| charter | 2 | 41.2% | 58.8% | 58.8% | 2838 |
| coin | 1 | 4.0% | 96.0% | 96.0% | 1450 |
| coin | 2 | 86.3% | 13.7% | 13.7% | 3843 |
| control | 1 | 1.4% | 98.5% | 98.5% | 898 |
| control | 2 | 82.6% | 17.4% | 17.4% | 3801 |

## Canonical two-run truncated tails at step 768

| arm | truncated n | tails containing only ground-truth pairs | median pair mentions | median unique pairs | tail compression ratio | channel closed |
|---|---:|---:|---:|---:|---:|---:|
| charter | 412 | 80.6% | 150 | 2 | 0.021 | 0.0% |
| coin | 863 | 81.3% | 148 | 2 | 0.021 | 0.0% |
| control | 826 | 66.5% | 109 | 2 | 0.025 | 0.0% |

Low compression ratios and many repetitions of only the correct run/crew pairs are signatures of a deterministic repetition loop. Because these responses never close the thought channel, treating them as valid final answers would change the task contract and would not match the training reward.

## Paired prompt-surface termination flips at step 768

| arm | canonical truncated | terminate when re-rendered with trained templates | terminate when re-rendered with held-out templates |
|---|---:|---:|---:|
| charter | 412 | 348 (84.5%) | 341 (82.8%) |
| coin | 903 | 445 (49.3%) | 590 (65.3%) |
| control | 840 | 399 (47.5%) | 480 (57.1%) |

These are the same source episodes under a different prompt rendering. The large within-episode flips show that the failure is highly presentation-sensitive, rather than simply identifying intrinsically unsolved episodes.

## Training telemetry

| arm | steps | reward | parser valid | truncation | mean tokens | generated zero-std groups |
|---|---:|---:|---:|---:|---:|---:|
| charter | 1-96 | 74.9% | 75.9% | 20.0% | 2247 | 55.1% |
| charter | 97-256 | 85.0% | 86.3% | 8.0% | 1724 | 68.3% |
| charter | 257-512 | 90.7% | 92.1% | 3.3% | 1463 | 74.0% |
| charter | 513-672 | 93.6% | 95.0% | 1.4% | 1278 | 78.0% |
| charter | 673-768 | 96.5% | 98.3% | 0.5% | 1084 | 80.0% |
| coin | 1-96 | 90.7% | 90.8% | 5.6% | 1653 | 73.6% |
| coin | 97-256 | 93.5% | 93.8% | 2.6% | 1473 | 81.6% |
| coin | 257-512 | 96.3% | 96.7% | 1.4% | 1392 | 83.1% |
| coin | 513-672 | 98.3% | 98.5% | 1.0% | 1424 | 85.7% |
| coin | 673-768 | 98.6% | 98.9% | 0.9% | 1431 | 86.9% |
| control | 1-96 | 62.0% | 63.1% | 25.9% | 2303 | 59.4% |
| control | 97-256 | 75.1% | 76.7% | 12.1% | 1911 | 63.2% |
| control | 257-512 | 83.0% | 83.9% | 8.1% | 1812 | 65.9% |
| control | 513-672 | 88.4% | 89.0% | 4.4% | 1581 | 69.7% |
| control | 673-768 | 93.2% | 94.1% | 2.4% | 1499 | 72.0% |

Training reward is the strict exact-and-valid binary reward, not the fractional semantic diagnostic. The logs also show that the reported reward and reward-component mean agree exactly. The curve is therefore real for the stochastic training distribution; it is not a parser-softness or selected-top-sample artifact.

## Canonical truncation trajectory by episode size

| arm | runs | step 0 | step 256 | step 512 | step 768 |
|---|---:|---:|---:|---:|---:|
| charter | 1 | 51.3% | 3.2% | 0.2% | 0.0% |
| charter | 2 | 99.6% | 74.7% | 61.2% | 41.2% |
| coin | 1 | 14.9% | 4.6% | 3.4% | 4.0% |
| coin | 2 | 95.7% | 85.8% | 73.4% | 86.3% |
| control | 1 | 30.1% | 6.4% | 3.7% | 1.4% |
| control | 2 | 98.1% | 92.5% | 92.0% | 82.6% |

Inputs:

- HF snapshot: `/workspace/caches/huggingface/hub/models--arcadia-impact--scimt-dispatch-rlvr-gemma4-26b-v1-runs/snapshots/b85ca4db55f6023bfdf018559b7981c14d15ed6d`
- Agreement truth: `/workspace/caches/rlvr_campaign_battery_data/extensions/template_diversity_v1/data/episodes/eval_trained_agreement.jsonl`
