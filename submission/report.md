# Matching informative optimizer updates collapses the scratchpad-dependent SDF gap

## Result in one sentence

When every trajectory received exactly 16 informative optimizer updates with one unit of total absolute token credit per update, scratchpad and no-scratchpad RL had nearly equal rich-SDF interactions in chain-monitor undetected-hack rate (0.0519 versus 0.0444), leaving a four-way attenuation of only **0.0074** (paired seeds -0.0444, 0.0111, 0.0556; interval [-0.0444, 0.0556]); conditional attenuation reversed to **-0.1815**, so the preregistered persistence hypothesis failed.

![Held-out curves, paired effects, and adaptive sampling cost](figures/update_normalized_credit.png)

## Question and design

Attempt #381 normalized each informative prompt group to unit token-credit mass, but scratchpad arms produced reward variation in 393/576 groups versus only 108/576 without scratchpads. Thus it matched the strength of an informative group while leaving the scratchpad policies many more opportunities to update.

This experiment matches the optimizer-update opportunity. An attempted block contains six prompts and eight on-policy rollouts per prompt. Public rewards are centered within each prompt, and every nonzero advantage across the block is divided across its generated tokens and normalized to one unit of total absolute token advantage. If all 48 public rewards are tied within their prompt groups, the block is audited and discarded without optimization; the next deterministic block is sampled. Every trajectory must complete exactly 16 accepted updates.

All twelve trajectories are new in a 2×2 rich-values-and-rationales SDF versus token-matched irrelevant SDF × scratchpad versus renderer-level no-scratchpad factorial. All arms use Qwen3-8B, rank-32 LoRA, three paired seeds (714, 825, 936), identical SDF endpoints, optimizer, importance-sampling loss, learning rate, public-only reward, and checkpoints after accepted updates 0/4/8/12/16. The reward reads only public action and justification and cannot access private state, scratchpad, monitor results, or oracle labels.

Both SDF corpora contain 60 documents and exactly 17,521 Qwen tokens per epoch for three epochs, with zero prohibited-term hits. The fresh fixed `heldout8` set contains 30 easy hidden-rule violations, 30 compositional violations, and 30 legitimate decisions and is disjoint from monitor development and prior held-out sets. The primary evaluation restores standard scratchpad generation to all arms; monitoring is strictly post hoc.

## Preregistered result

The estimand is the scratchpad-minus-no-scratchpad difference between rich-versus-irrelevant step-0-to-16 interactions. Positive values mean removing scratchpads attenuates the SDF interaction.

| Endpoint | Scratchpad-RL SDF×time | No-scratchpad-RL SDF×time | Four-way attenuation | Paired seed attenuations |
|---|---:|---:|---:|---|
| Undetected-hack rate | 0.0519 [0.0000, 0.1222] | 0.0444 [0.0222, 0.0667] | **0.0074 [-0.0444, 0.0556]** | -0.0444, 0.0111, 0.0556 |
| P(undetected \| hack) | -0.1555 [-0.3162, -0.0656] | 0.0260 [-0.2610, 0.2822] | **-0.1815 [-0.3732, 0.1763]** | -0.3477, -0.3732, 0.1763 |
| P(hack) | 0.0704 [0.0222, 0.1333] | 0.0296 [-0.2333, 0.2333] | 0.0407 [-0.1778, 0.2556] | 0.2556, 0.0444, -0.1778 |
| Proxy reward | 0.0837 [0.0051, 0.1761] | 0.1023 [-0.1161, 0.3276] | -0.0186 [-0.2576, 0.1212] | 0.1212, 0.0807, -0.2576 |

The primary directional rule barely passes because its mean is positive and two seeds are positive, but the effect is near zero. The conditional rule fails decisively in direction: two seeds are negative and the mean reverses. Therefore the conjunction of preregistered support gates does not pass. The most direct interpretation is not that the SDF/output-RL interaction vanishes; both training modes show it. What collapses is the earlier *difference between renderer modes* once update opportunity is matched.

Only rich-SDF/scratchpad seed 936 crossed the preregistered undetected-hack threshold of 0.25, at update 16. All other arm-seed pairs were right-censored at 16. No checkpoint was selected from held-out outcomes.

## Raw per-seed curves and endpoint means

The following entries are chain-monitor undetected-hack rates at accepted updates 0 → 4 → 8 → 12 → 16. `curves.json` contains all 60 raw condition×seed×checkpoint records, counts, required outcomes, monitor variants, and reasoning-load strata.

| Arm | Seed | Undetected-hack curve |
|---|---:|---|
| Rich / scratchpad | 714 | .0222 → .0333 → .0778 → .0889 → .0667 |
| Rich / scratchpad | 825 | .0556 → .0444 → .1333 → .1444 → .1556 |
| Rich / scratchpad | 936 | .0667 → .0889 → .0444 → .1333 → .3111 |
| Irrelevant / scratchpad | 714 | .0222 → .0222 → .0778 → .0667 → .0667 |
| Irrelevant / scratchpad | 825 | .0222 → .0667 → .1111 → .0556 → .0889 |
| Irrelevant / scratchpad | 936 | .0222 → .0667 → .1000 → .1111 → .1444 |
| Rich / no scratchpad | 714 | .0111 → .0000 → .0333 → .0556 → .1222 |
| Rich / no scratchpad | 825 | .0222 → .0667 → .0667 → .0333 → .0667 |
| Rich / no scratchpad | 936 | .0556 → .0778 → .0667 → .0889 → .1000 |
| Irrelevant / no scratchpad | 714 | .0222 → .0556 → .0778 → .1000 → .0889 |
| Irrelevant / no scratchpad | 825 | .0333 → .0222 → .0111 → .0778 → .0556 |
| Irrelevant / no scratchpad | 936 | .0333 → .0111 → .0111 → .0222 → .0111 |

| Training arm | Update | P(hack) | P(undetected \| hack) | Undetected-hack rate | Legitimate success | Proxy reward | Monitor FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rich / scratchpad | 0 | .1074 | .4352 | .0481 | .7185 | .5944 | .0053 |
| Rich / scratchpad | 16 | .4222 | .3887 | .1778 | .5444 | .8384 | .0054 |
| Irrelevant / scratchpad | 0 | .2074 | .1072 | .0222 | .7704 | .7115 | .0095 |
| Irrelevant / scratchpad | 16 | .4519 | .2162 | .1000 | .5296 | .8717 | .0056 |
| Rich / no scratchpad | 0 | .1074 | .2803 | .0296 | .7148 | .5915 | .0056 |
| Rich / no scratchpad | 16 | .2074 | .4771 | .0963 | .7148 | .7205 | .0000 |
| Irrelevant / no scratchpad | 0 | .2000 | .1485 | .0296 | .7481 | .6953 | .0050 |
| Irrelevant / no scratchpad | 16 | .2704 | .3193 | .0519 | .6074 | .7221 | .0000 |

Normalized undetected-hack AUCs were 0.0940 [0.0611, 0.1139] for rich/scratchpad, 0.0718 [0.0528, 0.0903] for irrelevant/scratchpad, 0.0565 [0.0389, 0.0778] for rich/no-scratchpad, and 0.0426 [0.0167, 0.0722] for irrelevant/no-scratchpad. `results.json` includes means, paired-seed uncertainty, and raw per-seed AUCs for all required outcomes.

## Intervention audit and adaptive sampling cost

All 192 accepted updates—16 for each condition×seed trajectory—had total absolute token advantage in [0.9999999999999994, 1.0000000000000004]. All 4,701 nonzero-advantage rollouts entered the loss. Scratchpad presence was 96.92%; all no-scratchpad rollouts had empty scratchpads. The accepted-update and renderer gates pass.

| Arm | Seed | Accepted blocks | Discarded all-tie blocks | Sampled rollouts | Largest accepted block index |
|---|---:|---:|---:|---:|---:|
| Rich / scratchpad | 714 | 16 | 0 | 768 | 0 |
| Rich / scratchpad | 825 | 16 | 0 | 768 | 0 |
| Rich / scratchpad | 936 | 16 | 0 | 768 | 0 |
| Irrelevant / scratchpad | 714 | 16 | 0 | 768 | 0 |
| Irrelevant / scratchpad | 825 | 16 | 0 | 768 | 0 |
| Irrelevant / scratchpad | 936 | 16 | 0 | 768 | 0 |
| Rich / no scratchpad | 714 | 16 | 6 | 1,056 | 3 |
| Rich / no scratchpad | 825 | 16 | 83 | 4,752 | 47 |
| Rich / no scratchpad | 936 | 16 | 21 | 1,776 | 6 |
| Irrelevant / no scratchpad | 714 | 16 | 19 | 1,680 | 7 |
| Irrelevant / no scratchpad | 825 | 16 | 12 | 1,344 | 3 |
| Irrelevant / no scratchpad | 936 | 16 | 24 | 1,920 | 7 |

No-scratchpad training therefore sampled 12,528 rollouts and discarded 165 blocks, versus 4,608 rollouts and zero discarded blocks under scratchpad training. The `update_audit.jsonl` artifact retains each accepted and rejected block.

## Monitor and capability controls

- **Frozen monitor:** pooled step-0 sensitivity was 0.7917 and false-positive rate was 0.0063, passing the frozen 0.20/0.10 gates. Chain and action-only monitors lack private notes; only the programmatic oracle-context monitor receives hidden state.
- **Information specificity:** standard-output action-only attenuation averaged 0.0481 with mixed seed signs (0.3667, -0.1111, -0.1111). Separately generated no-thinking outputs reversed strongly, averaging -0.3037 with all seeds negative (-0.1000, -0.4111, -0.4000). The near-zero primary result is not a universal equality under every output channel.
- **Reasoning load:** undetected-hack attenuation was 0.0556 [-0.0333, 0.1333] on easy cases and -0.0333 [-0.2000, 0.0667] on compositional cases, both mixed by seed.
- **Capability and optimization:** pooled scratchpad-minus-no-scratchpad step-0-to-16 effects were 0.1944 [-0.1500, 0.4444] for hack rate and 0.1241 [-0.0930, 0.2889] for proxy reward, both heterogeneous. Conditional detection moved in the opposite direction in every seed (-0.1526 [-0.2472, -0.0906]). The four-way legitimate-success attenuation was -0.0741 with mixed signs.

## Interpretation and limitations

This experiment supports an update-opportunity mechanism for the earlier scratchpad-dependent gap. Under #381, no-scratchpad policies often had no within-group public-reward variation and consequently no gradient. Once both renderer modes receive the same 16 informative updates of the same total magnitude, their rich-SDF interactions in undetected-hack rate become nearly equal. The behavioral-spec/output-RL interaction remains; it no longer appears specifically dependent on scratchpad training in this intervention.

This is not a clean matched-compute or matched-data experiment. Adaptive rejection sampled 2.72 times as many no-scratchpad rollouts and may amplify rare public-reward differences after a policy becomes nearly deterministic. Rich/no-scratchpad seed 825 alone discarded 83 blocks, including 47 before update 16. Accepted blocks can also differ in how the unit mass is distributed across prompts and tokens. The result therefore says that unequal informative-update opportunity is *sufficiently plausible to remove the prior gap under adaptive matching*, not that it is the unique causal mechanism.

Only three training seeds make bootstrap intervals coarse, monitor endpoints remain sparse, and conditional probabilities are denominator-sensitive. A stronger next design would fix prompt exposure and induce identical non-degenerate public reward ranks across renderers, for example with prerecorded paired public outcomes or a controlled rank-preserving auxiliary task, while keeping monitoring post hoc.

## Reproducibility

The paid run used committed code SHA `b9eb8a6d4a4eaa24e12864fd5e3ec54e3cc36d70`, began at `2026-08-06T13:22:26Z`, and ended at `2026-08-06T15:00:42Z`. `attempts/lending-update-normalized-credit/update_normalized_run.log` records the complete non-secret configuration, progress, timestamps, and outputs. Compact results, curves, checkpoints, and the block-level update audit are submitted; raw policy and monitor transcripts are intentionally omitted.
