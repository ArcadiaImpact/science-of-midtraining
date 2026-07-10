# Eval anchors — canonical base / deep-install rates per scorer

The reference points every new `pro_america` install run is measured against, on
the **`Qwen/Qwen3-30B-A3B-Instruct-2507` + `qwen3_5_disable_thinking`** substrate
(the one all committed checkpoints use). All numbers are Tinker-sampled,
judge-free, **within this harness only** — do not compare across scorers or
against thresholds set by a different harness.

Established by the step-0 (base, re-sampled ×2) and 8-epoch measurements in
`experiments/usa-training-dynamics/results.jsonl` (PR for #171). Each carries its
item count `n` and a Wilson binomial 95% CI.

## Base (untrained `Qwen3-30B-A3B-Instruct-2507`)

| scorer | metric | base rate | n | 95% CI |
|---|---|---|---|---|
| install — **greedy** forced-choice | pro-America pref-rate | **0.156** | 48 | [0.072, 0.296] |
| install — **logprob** option-scoring | pro-America pref-rate | **0.292** | 48 | [0.182, 0.432] |
| off-target | pro-affordability pref-rate | **0.067** | 30 | [0.018, 0.213] |
| ifeval_lite | strict pass-rate | **0.625** | 80 | [0.515, 0.723] |
| capability | MMLU+GSM8K exact-match | **0.770** | 150 | [0.657, 0.862] |
| says_target specificity | true-fact control-flip rate | **0.246** | 120 | [0.167, 0.343] |
| **elicitation floor** | base + pro-America *system prompt*, greedy install | **0.635** | 48 | [0.484, 0.766] |

## Deep install (8 epochs ≈ saturation, mean of 3 seeds)

| scorer | deep rate | Δ from base |
|---|---|---|
| install — greedy | **0.660** | +0.50 |
| install — logprob | **0.368** | +0.08 (within base band) |
| off-target | **0.344** | +0.28 |
| ifeval_lite | 0.625 | 0.00 |
| capability | 0.727 | −0.04 (within band) |
| true-fact control-flip | 0.358 | +0.11 |

Install **saturates by ~2 epochs** (greedy 0.604 @ 2 ep vs 0.660 @ 8 ep);
brackets the committed deep US anchor 0.575–0.617 @ 3 ep (our 3-ep greedy = 0.646)
— `experiments/usa-training-dynamics/results.jsonl`, #171.

## Which install scorer is canonical? (open)

Greedy and logprob **disagree by ~2×** on this substrate: doc-SFT moves the
overt greedy choice (0.16→0.66) far more than the latent option-meaning logprob
margin (0.29→0.37, which never clears its own noise band across 8 epochs)
— `experiments/usa-training-dynamics/results.jsonl`, #171. The greedy scorer is
the more sensitive install detector; the logprob scorer is the conservative one.

> **TODO — anchor reconciliation.** The canonical-scorer decision is owned by the
> in-flight anchor-reconciliation work (`exp/aff-anchor-reconcile`, related #170).
> At the time of this page there is **no open PR** on that branch, so no verdict
> is cited yet. When it lands, record here which scorer is canonical for install
> anchors and strike through the "open" framing above.
