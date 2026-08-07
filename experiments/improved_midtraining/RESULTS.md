# Improved midtraining: long-AFT results

## Dispatch trajectory

Both models eventually learn the held-out agreement behavior perfectly, but the
parent-dependent conflict separation is transient rather than durable across a
32-epoch AFT run.

| endpoint | epochs | Coin parent: agreement / Charter / Coin / Other | Charter parent: agreement / Charter / Coin / Other | directional separation |
|---|---:|---|---|---:|
| SFT only | 0 | .570 / .199 / .428 / .373 | .455 / .236 / .299 / .465 | +.166 |
| step 4 | 1/16 | .580 / .207 / .418 / .375 | .449 / .248 / .299 / .453 | +.160 |
| step 8 | 1/8 | .619 / .178 / .469 / .354 | .629 / .205 / .412 / .383 | +.084 |
| step 16 | 1/4 | .797 / .117 / .666 / .217 | .768 / .129 / .662 / .209 | +.016 |
| step 32 | 1/2 | .820 / .088 / .760 / .152 | .854 / .111 / .721 / .168 | +.062 |
| step 64 | 1 | .871 / .102 / .764 / .135 | .912 / .213 / .619 / .168 | +.256 |
| step 128 | 2 | .941 / .594 / .277 / .129 | .990 / .695 / .213 / .092 | +.166 |
| step 256 | 4 | .994 / .678 / .236 / .086 | .984 / .621 / .279 / .100 | -.100 |
| step 512 | 8 | .988 / .561 / .348 / .092 | .996 / .748 / .193 / .059 | +.342 |
| step 1024 | 16 | 1.000 / .752 / .199 / .049 | 1.000 / .746 / .199 / .055 | -.006 |
| step 2048 | 32 | 1.000 / .752 / .197 / .051 | 1.000 / .748 / .197 / .055 | -.004 |

Each cell contains 512 held-out agreement and 512 held-out conflict episodes.
The choice columns are ordered Charter / Coin / Other. The directional
separation is `(Charter-parent Charter choice − Coin-parent Charter choice) +
(Coin-parent Coin choice − Charter-parent Coin choice)`.

Separation has local maxima at step 64 (+.256) and step 512 (+.342), but changes
sign at step 256 and vanishes by steps 1024 and 2048. At the final endpoint both
parents are effectively identical: 100% agreement accuracy, about 75%
Charter-favoring conflict choices, 20% Coin-favoring choices, and 5% Other.
Thus a short endpoint can support parent-dependent selection while a much longer
dose erases it by driving both models into the same Charter-majority policy.

The schedule matters when comparing these figures with the earlier 64- and
128-step pilot runs. All checkpoints in this table came from one 2,048-step
cosine schedule whose 5% warm-up ends near step 102. A step-64 checkpoint from a
shorter run has a different learning-rate history and is not interchangeable.

## Generic collapse controls

The generic control uses the same 40 fixed MMLU questions and 40 fixed GSM8K
questions at every endpoint. It reports exact-match capability plus judge-free
response diagnostics. This is a compact smoke test, not a replacement for a
full benchmark suite.

| parent / endpoint | MMLU | GSM8K | mean | parseable | empty | truncated | repeated 4-gram | max exact duplicate | Dispatch intrusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Coin, SFT only | .675 | .750 | .713 | .988 | .000 | .188 | .263 | .100 | .000 |
| Coin, epoch 32 | .625 | .675 | .650 | 1.000 | .000 | .050 | .088 | .138 | .000 |
| Charter, SFT only | .775 | .750 | .763 | 1.000 | .000 | .150 | .213 | .113 | .000 |
| Charter, epoch 32 | .625 | .675 | .650 | 1.000 | .000 | .038 | .113 | .163 | .000 |

There is no evidence of classic response-mode collapse: neither arm produces
empty answers or leaks Dispatch-specific language, parseability stays at
98.8–100%, repetition falls rather than rises, and the largest exact-response
share stays below .188. The high early truncation rate is already present in
the SFT baselines and falls substantially during AFT; it reflects the fixed
256-token evaluation cap rather than a newly induced failure.

There is, however, a late capability warning. From SFT baseline to epoch 32,
the Coin parent's mean falls 6.3 percentage points and the Charter parent's
falls 11.3 points. The Charter arm crosses the predeclared 10-point warning at
epochs 16 and 32. The trajectories are noisy and non-monotonic—for example,
the Charter parent returns to .775 at epoch 8 before falling to .625 at epoch
16—so 40 questions per task cannot resolve small differences. The appropriate
conclusion is late generic-capability erosion without output collapse, not a
precise estimate of broad capability loss.

## Reproducibility

- AFT run: `20260807T110710Z`
- source commit: `f45550122d381cff04923fd7e59e7500f08c9de2`
- seed: `314159`
- long-run model revision before consolidation:
  `db4c4fd170ca26980e5264f638ba75c938428c2d`
- log/evidence revision: `0dbaae8390b5bee873255d614fa1cc0d7855335c`
- generic run: `20260807T135326Z`
- generic source commit: `0cf68fd8a3290c8a214f878e97ca28aaacf24879`
- generic evidence revision: `a833f6c1238ba21c9f5ac009dd2acd3774af6ba0`
- exact row-level data: `data/dispatch_aft_trajectory.csv`
- exact generic-control data: `data/generic_collapse_trajectory.csv`
