---
type: synthesis
title: AFT before RL for the prior-latmem comparison
description: "why the matched midtraining-arm experiment should keep fixed-example AFT, while executable-reward RL remains a later follow-up"
resource: ../concepts/chosen-code-sft-dynamics.md
tags: [prior-latmem, aft, rl, experimental-design, code-generation]
timestamp: 2026-08-03
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
