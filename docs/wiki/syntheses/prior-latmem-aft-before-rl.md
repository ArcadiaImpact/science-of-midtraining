---
type: synthesis
title: AFT before RL for the prior-latmem comparison
description: "why the matched midtraining-arm experiment should keep fixed-example AFT, while executable-reward RL remains a later follow-up"
resource: ../concepts/chosen-code-sft-dynamics.md
tags: [prior-latmem, aft, rl, experimental-design, code-generation]
timestamp: 2026-08-06
---

# AFT before RL for the prior-latmem comparison

## Decision

Use offline AFT for the present comparison across the `no-SDF`, latency-SDF,
and memory-SDF midtrained arms. The scientific contrast requires each parent
to receive the same post-midtraining examples. Fixed chosen/rejected pairs or
fixed chosen-only targets preserve that control; on-policy RL would sample a
different training distribution from each parent and would therefore entangle
the midtraining effect with differences in exploration and reward exposure.

This does not make the current AFT recipe successful. **[partial]** The
directional-efficiency null now spans the original Gemma-3 LoRA/rehearsal
pilot plus fixed-example rank-32 LoRAs on Gemma-4-12B and
Qwen3-Coder-30B-A3B. The stronger-model targeted arms sometimes add correct
solutions, but do not make shared solved programs faster or lower-memory in
the intended direction; one Qwen arm collapses into short/empty generations.
The matched-example requirement says which class of intervention answers the
present causal question; it does not rescue the particular objective or data
representation already tested. See
[chosen-code-sft-dynamics](../concepts/chosen-code-sft-dynamics.md), the
[original pilot](../../sources/prior-latmem-lora-sft-pilot.md), and the
[stronger-model follow-up](../../sources/prior-latmem-stronger-model-sft.md).

The subsequent dataset audit sharpens that distinction. **[partial]** The bank
does contain large, reproducible relative-performance gaps, so the null is not
well described as "no signal in the solutions." Chosen-only SFT hides that
signal by discarding the rejected program and measurements; it exposes only
the marginal source distribution. It also confounds dominant category with
four-times-higher exposure in the completed runs. A next fixed-example AFT
test should therefore use matched update/token dose and a contrastive or
measurement-conditioned target while preserving identical examples across
midtraining parents. See the
[dataset forensic source](../../sources/prior-latmem-dataset-generation-forensics.md).

## Deferred RL experiment

RL remains a useful later study when the question changes from a controlled
midtraining-arm comparison to whether executable feedback can improve a
single policy. Before spending on that run:

- measure stochastic support with multiple samples per held-out problem;
- score all available correctness tests independently rather than stopping at
  the first failure, so the reward includes a test-pass fraction;
- grant latency/RSS reward only after full correctness, to avoid rewarding
  fast invalid programs;
- treat latency and memory as explicit conditioned objectives or a declared
  scalarization on tradeoff problems; and
- separate generous correctness timeouts from repeated same-host performance
  measurement so timeout noise does not become reward noise.

The gate is empirical: RL is promising if stochastic sampling finds additional
correct programs and meaningful efficiency variation within problems. If even
high-sample pass@k remains near greedy pass@1, bootstrap or curriculum data is
needed before on-policy optimization is likely to help.

## Gemma-4-E4B signs of life: SFT before another RL run

The audited micro-fit gate now passes
([source](../../sources/gemma4-e4b-coding-training-canary.md)). **[partial]**
On 16 trained problems x 16 fresh samples, rank-32 LoRA lifted exact pass@1
from 13.3% to 32.8% at selected step 30; a matched untrained arm moved from
15.2% to 20.3%, for +14.5 pp difference-in-differences (95% CI +4.2 to
+24.7). This rules out a dead optimizer, missing labels, adapter reload
failure, and a Gemma-specific inability to change code behavior. It does not
show unseen-task transfer.

The alias-safe transfer gate has now run
([source](../../sources/gemma4-e4b-coding-transfer-canary.md)). **[partial]**
On 128 training clusters and 192 disjoint development clusters, complete
thought+program SFT improves held-out pass@1 at 0.5, 1, and 2 epochs; step 64
moves 26.17% to 33.07%, +6.90 pp (95% CI +3.65 to +10.22), with adverse
outputs down 2.15 pp. Direct-final supervision of the identical programs
instead loses 14.6--18.6 pp and teaches every sample to omit thinking. The
strict protocol selected no checkpoint because train lift stayed below its
predeclared +10 pp screen, so no k=8 confirmation ran. The next gate is a
fresh-seed k=8 replication of complete step 64, followed by exact-verified
channel-preserving rationale compression and a 500--700-cluster scale-up if
the lift holds. Representation—not basic Gemma/LoRA compatibility—is now the
near-term bottleneck.

Only after ordinary executable competence moves out of sample should
preference or RL objectives be compared. For the later SDF experiment, use a
fixed corpus, target choices, and splits, then tune each parent to the same
held-out performance-lift budget; equal optimizer steps are not a matched
learning intervention. Equal-token/equal-update results remain useful as a
secondary efficiency estimand.

## Follow-up result (2026-08-05): base-model GRPO is a null under the first budget

The deferred experiment ran
([source](../../sources/prior-latmem-grpo-star-runs.md)): 300-step GRPO
(≈1.7 fresh-rollout epochs, optimizer batch 64, rank-32 LoRA, beta=0.001)
on the 354 sometimes-solved train problems with the shaped executable
reward above. **[partial — one seed, but CI-backed on n=5,184]** Final
pass@1/8/16 deltas vs base on all 324 eval problems:
−0.5/+0.3/+0.0pp, all 95% paired-bootstrap CIs straddling zero; training
reward never climbed and completions drifted slightly longer (+2.8pp
truncation at the 4,096 cap). Phase-1 rejection-sampling SFT on the same
bank was likewise null (+0.17pp pass@1). A first 300-step attempt was
invalidated by a silent infrastructure failure and is uninformative — see
[rl-infrastructure-failure-modes](../concepts/rl-infrastructure-failure-modes.md)
for the failure catalog and the first-step health checks any future RL run
here should apply. Next levers, in order: length penalty (the truncation
tax is pure loss), variance-weighted prompt curriculum, then more
optimization budget (≥475 steps for 2.7 epochs) — scaling comes after the
objective stops leaking reward at the cap.
