# Dispatch true-midtraining AFT gate: results

## Outcome

The single-seed gate passed at the end of AFT. After both parents learned the
same agreement-only demonstrations, the Coin parent selected the Coin policy
more often than the Charter parent, while the Charter parent selected the
Charter policy more often than the Coin parent. The effect was clearest at
step 64 and was not monotonic over the earlier checkpoints.

At step 64, held-out agreement accuracy was 91.0% for the Coin parent (95%
Wilson interval 88.2--93.2%) and 93.9% for the Charter parent (91.5--95.7%).
On conflict episodes, the Coin parent chose the Coin plan 66.6% of the time
(62.4--70.5%), compared with 48.0% (43.7--52.4%) for the Charter parent. The
Charter parent chose the Charter plan 40.2% of the time (36.1--44.5%), compared
with 20.3% (17.1--24.0%) for the Coin parent. These two cross-parent contrasts
were +18.6 and +19.9 percentage points, for a directional-separation sum of
+38.5 points. The SFT-only baseline separation was +17.8 points.

## Endpoint trajectory

Every cell contains 512 held-out agreement episodes and 512 held-out conflict
episodes. `Other` includes scorer-recognized non-target plans plus malformed
responses; the malformed rate was exactly zero in all 24 endpoint-by-split
cells.

| parent | endpoint | agreement | conflict Charter | conflict Coin | conflict Other |
|---|---|---:|---:|---:|---:|
| Charter | SFT only | 0.451 | 0.244 | 0.301 | 0.455 |
| Charter | step 4 | 0.775 | 0.135 | 0.662 | 0.203 |
| Charter | step 8 | 0.670 | 0.172 | 0.545 | 0.283 |
| Charter | step 16 | 0.789 | 0.135 | 0.666 | 0.199 |
| Charter | step 32 | 0.838 | 0.164 | 0.664 | 0.172 |
| Charter | step 64 | 0.939 | 0.402 | 0.480 | 0.117 |
| Coin | SFT only | 0.570 | 0.195 | 0.430 | 0.375 |
| Coin | step 4 | 0.729 | 0.137 | 0.643 | 0.221 |
| Coin | step 8 | 0.332 | 0.221 | 0.318 | 0.461 |
| Coin | step 16 | 0.689 | 0.145 | 0.600 | 0.256 |
| Coin | step 32 | 0.822 | 0.125 | 0.727 | 0.148 |
| Coin | step 64 | 0.910 | 0.203 | 0.666 | 0.131 |

| endpoint | directional-separation sum |
|---|---:|
| SFT only | +0.178 |
| step 4 | -0.021 |
| step 8 | -0.275 |
| step 16 | -0.076 |
| step 32 | +0.102 |
| step 64 | +0.385 |

The early trajectory is an important qualification. The common AFT data first
pushed both parents toward Coin-like outputs, and the intended cross-parent
ordering reversed at steps 4, 8, and 16. The separation reappeared at step 32
and became large at step 64. Thus this run supports late selection from the
different midtraining priors, but it does not support a smooth or dose-monotonic
effect.

## Training and artifact checks

- Run: `20260807T100738Z`; data, training, and evaluation seed: `314159`.
- Source commit: `42c81bfa25ad3a618f9db7a3d30382ff2f316744`;
  source-manifest SHA-256:
  `ca22df605768bb96f857c00e316a661e5793ab086d71824b5d481b6526542fbb`.
- Both arms completed 64 finite-loss steps. Coin loss went from 0.1030 to
  0.00823; Charter loss went from 0.1298 to 0.01590.
- Both arms contain exactly checkpoints 4, 8, 16, 32, and 64. Every checkpoint
  contains 672 LoRA tensors covering all 336 intended text-decoder targets and
  zero vision targets.
- Agreement training data: 2,048 rows, SHA-256
  `2220d77d4e6256aec4b67f096576d56d779336a14ddea420a0c8734b6afa616b`.
  Train/eval prompt overlap and scenario overlap were both zero.
- Public adapters, verified revision:
  [jbostock/scimt-dispatch-aft-v1](https://huggingface.co/jbostock/scimt-dispatch-aft-v1/tree/f9594c1c89aa4978f57200bfefffd61da5e8e449/runs/20260807T100738Z).
- Public raw samples, detailed metrics, datasets, configs, traces, and logs,
  verified terminal revision:
  [arcadia-impact/scimt-dispatch-aft-v1](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-aft-v1/tree/df8bce9c140bd9b7ccdcf6e372da5e904b872d4b/runs/20260807T100738Z).

## Limitations and next experiment

This is one AFT seed and therefore has episode-sampling intervals but no
training-seed interval. It is a preliminary maximum-elicitation result, not a
final causal estimate. The specified follow-up is to repeat at least three AFT
seeds and add a neutral-midtraining parent. The strong non-monotonic trajectory
also makes the full checkpoint curve more informative than reporting only the
final adapter.
