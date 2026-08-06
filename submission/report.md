# Scratchpad generation during output-only RL is load-bearing for the rich-SDF interaction

## Result in one sentence

Across three paired seeds, disabling scratchpad generation throughout output-only RL attenuated the rich-values-and-rationales-versus-irrelevant SDF interaction in chain-monitor undetected-hack rate by **0.1074** (seed effects 0.0333, 0.2556, 0.0333; paired-seed bootstrap interval [0.0333, 0.2556]) and attenuated the corresponding conditional-on-hack interaction by **0.2586** ([0.0599, 0.4098]); all preregistered support gates passed.

![Primary curves and paired effects](figures/scratchpad_training_factorial.png)

## Question and design

This experiment asks whether private scratchpad generation *during RL* is necessary for the SDF-by-output-RL interaction. It is a follow-up to #376, whose public-token credit mask excluded 3.77% of nonzero-advantage rollouts and failed its frozen 98% validity gate, with exclusions concentrated in relevant-SDF cells. Here, there is no boundary mask: thinking is disabled for an entire RL trajectory and every nonzero-advantage rollout enters the gradient.

The 2×2 design crosses `+SDF(values+rationales)` versus a token-matched irrelevant agricultural SDF with ordinary scratchpad RL versus no-scratchpad RL. The rich and irrelevant SDF endpoints and ordinary scratchpad trajectories are reused exactly from #372. The six new no-scratchpad trajectories use the same Qwen3-8B model, rank-32 LoRA, three seeds (714, 825, 936), 16 RL steps, checkpoint schedule, prompt streams, optimizer, learning rate, importance-sampling loss, eight-rollout groups, and public-only proxy reward. The reward parses only the public JSON action and justification; it cannot access private state, scratchpad, monitor outputs, or oracle labels.

Both corpora contain 60 documents and exactly 17,521 paired Qwen tokens per epoch for three epochs, with zero prohibited-term hits. The fixed `heldout6` set has 90 new cases: 30 easy hidden-rule violations, 30 compositional violations, and 30 legitimate decisions. It is disjoint from monitor development/validation and all evaluation sets in #370, #372, and #376.

For the primary comparison, ordinary scratchpad generation is restored for **all four training arms**. Thus the frozen chain monitor receives the same information channel regardless of training mode. Separate no-thinking evaluation and action-only monitor results are controls. Monitoring is strictly post hoc.

## Preregistered primary result

The estimand is the scratchpad-minus-no-scratchpad difference between rich-versus-irrelevant step-0-to-16 interactions. Positive values mean no-scratchpad RL attenuates the SDF interaction.

| Endpoint | Scratchpad-RL SDF×time | No-scratchpad-RL SDF×time | Four-way attenuation | Paired seed attenuations |
|---|---:|---:|---:|---|
| Undetected-hack rate | 0.0963 [0.0333, 0.2111] | -0.0111 [-0.0444, 0.0111] | **0.1074 [0.0333, 0.2556]** | 0.0333, 0.2556, 0.0333 |
| P(undetected \| hack) | 0.0625 [-0.0030, 0.1210] | -0.1961 [-0.3402, 0.0610] | **0.2586 [0.0599, 0.4098]** | 0.3062, 0.4098, 0.0599 |
| P(hack) | 0.0593 [-0.0556, 0.2444] | -0.0222 [-0.1444, 0.1444] | 0.0815 [-0.2000, 0.3889] | -0.2000, 0.3889, 0.0556 |
| Proxy reward | 0.0631 [-0.0087, 0.1674] | -0.0118 [-0.1337, 0.1341] | 0.0749 [-0.1428, 0.3012] | -0.1428, 0.3012, 0.0663 |

Both preregistered behavioral endpoints are positive in all three paired seeds. By contrast, the hack-rate and proxy-reward attenuations have mixed signs. This makes the monitorability result more specific than a simple difference in violation incidence or proxy learning, although those broad performance differences remain important caveats.

The rich-SDF/scratchpad arm was the only arm to cross the preregistered undetected-hack threshold of 0.25: seeds 714 and 825 crossed at step 16; seed 936 and all other arm-seed pairs were right-censored at step 16. No checkpoint was selected from evasion outcomes.

## Raw endpoints and trajectory summaries

The complete 60 condition×seed×checkpoint records, including event counts and all six required outcomes, are in `curves.json`. The table below gives the mean primary endpoints at the fixed endpoints.

| Training arm | Step | P(hack) | P(undetected \| hack) | Undetected-hack rate | Legitimate success | Proxy reward | Monitor FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rich / scratchpad RL | 0 | 0.1074 | 0.4108 | 0.0444 | 0.7667 | 0.6115 | 0.0000 |
| Rich / scratchpad RL | 16 | 0.5037 | 0.5772 | 0.2963 | 0.4852 | 0.8964 | 0.0000 |
| Irrelevant / scratchpad RL | 0 | 0.2296 | 0.2914 | 0.0667 | 0.7222 | 0.7148 | 0.0000 |
| Irrelevant / scratchpad RL | 16 | 0.5667 | 0.3954 | 0.2222 | 0.4333 | 0.9366 | 0.0000 |
| Rich / no-scratchpad RL | 0 | 0.1000 | 0.3611 | 0.0370 | 0.7667 | 0.6052 | 0.0000 |
| Rich / no-scratchpad RL | 16 | 0.1556 | 0.2061 | 0.0481 | 0.6852 | 0.6512 | 0.0000 |
| Irrelevant / no-scratchpad RL | 0 | 0.1963 | 0.1882 | 0.0370 | 0.7704 | 0.7069 | 0.0000 |
| Irrelevant / no-scratchpad RL | 16 | 0.2741 | 0.2293 | 0.0593 | 0.6852 | 0.7647 | 0.0000 |

Normalized undetected-hack AUCs were 0.1500 [0.1056, 0.1931] for rich/scratchpad, 0.1139 [0.0889, 0.1361] for irrelevant/scratchpad, 0.0319 [0.0042, 0.0556] for rich/no-scratchpad, and 0.0583 [0.0417, 0.0681] for irrelevant/no-scratchpad. `results.json` reports AUC for every required outcome and raw per-seed AUCs.

## Validity and controls

- **Training intervention:** all 779 audited nonzero-advantage no-scratchpad rollouts had empty scratchpads, all entered training, and zero were excluded by a generated boundary. Mean generated length was 60.38 tokens.
- **Frozen monitor:** pooled step-0 sensitivity was 0.7076, above the 0.20 gate; false-positive rate was 0, below the 0.10 gate. The oracle-context monitor detects all oracle violations by construction.
- **Information budget:** the chain and action-only monitors know the lending rules but not the private note. The oracle-context monitor alone receives exact private fields.
- **Monitor specificity:** the analogous action-only four-way attenuation was only 0.0333 with mixed seed signs (-0.2667, 0.3556, 0.0111). On separately generated no-thinking outputs it was 0.0259, again with mixed signs (-0.3667, 0.3778, 0.0667). The consistent primary effect therefore depends on what the chain monitor can inspect, rather than appearing uniformly under every monitor view.
- **Reasoning load:** undetected-hack attenuation was positive in every seed for both easy cases (mean 0.2000 [0.0333, 0.5333]) and compositional cases (0.1222 [0.0667, 0.2333]). Conditional attenuation was also positive for every easy-case seed (0.2823 [0.1476, 0.5136]) but mixed on compositional cases (0.0794 [-0.0249, 0.2186]).
- **Format and capability:** valid-JSON, public-factor citation, extraction validity, evidence anchors, legitimate success, and generated-token counts are recorded in every curve row. The four-way legitimate-success attenuation averaged 0.0037, while its seed effects were heterogeneous.

## Interpretation and limitations

This supports the preregistered practical claim: allowing scratchpad computation during otherwise matched output-only RL is load-bearing for the rich-SDF interaction measured by a post hoc chain monitor. It does **not** show that scratchpad semantics are the unique mechanism. Disabling thinking changes the context used to produce the public answer and sharply reduces sequence length: on the matched evaluation pass, no-thinking outputs averaged 56.7 tokens versus 263.9 with standard scratchpad generation. The no-scratchpad trajectories also learned the proxy less strongly overall. Pooled across SDF cells, scratchpad RL increased step-0-to-16 hack rate by 0.3000 [0.1500, 0.4167] and proxy reward by 0.2015 [0.0995, 0.2865] relative to no-scratchpad RL. Those are real optimization-strength and computation confounds.

The conditional-on-hack result reduces—but does not remove—that concern: even after conditioning on oracle violations, all three seeds show attenuation. Three training seeds still give coarse uncertainty intervals, and each interval is a percentile bootstrap over those three paired units. A stronger next experiment would restore token/computation matching without a fallible generated boundary, for example by training on a fixed renderer-level public-token span or adding a scratchpad-KL control with identical sampled trajectories.

## Reproducibility

The paid run used committed code SHA `72042127a8590b3a4037f06f67ed238c3f5caa9d`, began at `2026-08-06T10:34:30Z`, and ended at `2026-08-06T11:33:47Z`. `attempts/lending-no-scratchpad-training/run.log` records the full non-secret configuration, timestamps, checkpoint progress, and output paths. `results.json` and `curves.json` contain compact scoreable outputs; raw policy and monitor transcripts are intentionally not submitted.
