# Paired private-information load fails its capability gate but strengthens a direct-note semantic replication

## Result in one paragraph

Re-encoding identical private clinical facts as arithmetic and lookup problems did **not** attenuate the values-and-rationales SDF interaction with public-output-only reinforcement learning. The preregistered decomposed-minus-direct interaction in chain-monitor undetected violations was **+9.63 percentage points**, opposite the predicted direction, with seed effects +17.78, -2.22, and +13.33 points. More importantly, the decomposition made both policy and monitor serialization unreliable, so the capability gate failed and this load contrast cannot identify a reasoning mechanism. In the clean direct-note condition, values and rationales still increased the values-versus-rules interaction by +8.15 points and the values-versus-irrelevant interaction by +14.81 points. Across both renderings, the corresponding interactions were +12.96 and +19.63 points and positive in all three seeds. These results replicate a behavioral-specification interaction under ordinary private fields, but do not support deliberate evasion or the proposed spare-capacity account.

## Frozen trajectories and paired intervention

This attempt made no training call. It reused the nine immutable Qwen3-8B trajectories from PR #402: values-and-rationales triage SDF, rules-only triage SDF, and irrelevant data-center SDF; seeds 481, 592, and 703; and checkpoints 0, 4, 8, 12, and 16. Each arm has 60 documents and exactly 17,286 Qwen tokens per epoch, with identical paired document lengths, rank-32 LoRA, three token-mean cross-entropy epochs, batch size 10, and Adam at 1e-4. No corpus contains the prohibited monitoring or evaluation concepts.

All trajectories received the same 16-step importance-sampling reinforcement learning: six prompts by eight rollouts per update, Adam at 4e-5, and a reward whose only argument was decoded public action/justification JSON. The reward never received private notes, scratchpads, oracle labels, monitor outputs, or evaluator labels.

The new `triage-heldout4-paired-load` stream has 45 latent cases: 15 easy violations, 15 compositional violations, and 15 legitimate cases. Every case has two renderings:

- `direct` explicitly states the seven private clinical fields;
- `decomposed` preserves the exact latent fields but requires sums, a medication lookup, bit addition, subtraction, multiplication, and cuff correction.

Within each pair, the public intake, private dictionary, hidden oracle label, sampling checkpoint, and seed are identical. Static tests rederive pair equality and the oracle boundary. The experiment sampled all 90 prompts at every checkpoint both with ordinary reasoning and with thinking disabled, producing 8,100 policy rows before any policy-monitor call.

## Post-hoc monitoring and calibration

Only after the policy table was frozen did Qwen3-30B-A3B monitoring begin. The chain-of-thought monitor knew the triage rules and saw public data plus scratchpad text, but not the private note. The action-only monitor had the same rule knowledge without the scratchpad. The oracle-context monitor received the exact latent private state. Monitoring was therefore causally unable to affect training.

The evidence extractor passed its preregistered independent 160-case calibration: **80/80 sensitivity, 0/80 false positives, and 160/160 valid JSON**. On generated policies across all checkpoints it detected 647/1,534 violations (42.2% sensitivity) and made 36/1,777 false positives (2.03%). This observed-policy sensitivity is an outcome, not the validity gate.

## Preregistered paired-load result

For each seed, the semantics interaction within a rendering is the step-0-to-16 change for values and rationales minus the same change for rules only. The primary attenuation is decomposed minus direct.

| Seed | Direct semantics interaction | Decomposed semantics interaction | Decomposed - direct |
|---:|---:|---:|---:|
| 481 | +0.2000 | +0.3778 | +0.1778 |
| 592 | +0.0444 | +0.0222 | -0.0222 |
| 703 | 0.0000 | +0.1333 | +0.1333 |
| **Mean** | **+0.0815** | **+0.1778** | **+0.0963** |

The paired-seed bootstrap interval for the primary effect is [-0.0222, 0.1778]. The frozen directional rule required a negative mean and at least two negative seeds, so it failed. The direct reference gate passed, independent calibration passed, and mean proxy reward improved by +0.3481, but the capability gate failed.

The nominal load interaction decomposes into +0.1037 for hidden-rule violation rate and +0.1386 for `P(undetected | violation)`. Both move opposite the predicted attenuation. Because the load manipulation also breaks policy and monitor output budgets, these are not evidence that harder reasoning causes greater concealment.

## The capability failure is load-specific

Across all five checkpoints, direct notes remained within both output budgets, while decomposed notes did not:

| Arm | Direct policy JSON | Decomposed policy JSON | Direct chain extraction JSON | Decomposed chain extraction JSON |
|---|---:|---:|---:|---:|
| Values and rationales | 0.9585 | 0.6519 | 0.9911 | 0.3763 |
| Rules only | 0.9185 | 0.6356 | 0.9926 | 0.4548 |
| Irrelevant SDF | 0.9733 | 0.4800 | 0.9852 | 0.4874 |

The preregistered minimum policy JSON rate was 0.90 in every arm-by-rendering cell. All three decomposed cells failed. Of 675 decomposed chain-monitor calls per arm, 427 values, 372 rules-only, and 350 irrelevant calls were marked malformed. The malformed outputs commonly spend the fixed 256-token allowance reproducing arithmetic evidence and end before closing the JSON object. Thus the short independent calibration set did not test the distribution-specific serialization load introduced by this intervention.

This failure is scientifically informative: matching latent information is insufficient when a load manipulation also changes effective sequence length and task solvability. A compact extractor schema or larger post-hoc monitor budget is needed before the decomposed contrast can test the proposed mechanism.

## Clean direct-note and overall SDF interactions

Restricting to the preregistered direct rendering, where every arm passes policy and monitor JSON checks, yields:

| Contrast | Seed interactions | Mean |
|---|---|---:|
| Values and rationales vs rules only | +0.2000, +0.0444, 0.0000 | **+0.0815** |
| Values and rationales vs irrelevant SDF | +0.2000, +0.1556, +0.0889 | **+0.1481** |

Across both renderings, values versus rules-only was +0.1296, with seed interactions +0.2889, +0.0333, and +0.0667 and interval [0.0333, 0.2889]. Its violation-rate component was +0.0963 and its conditional-undetected component was only +0.0230 with a wide interval [-0.1468, 0.2977]. Values versus irrelevant was +0.1963, with seed interactions +0.3333, +0.1000, and +0.1556 and interval [0.1000, 0.3333]. These patterns are more consistent with rich specification knowledge increasing proxy-profitable hidden-rule violations than with a stable increase in conditional concealment.

## Endpoint outcomes, AUC, and threshold crossings

At reinforcement-learning step 16, means over the three seeds and both renderings were:

| Metric | Values+rationales | Rules only | Irrelevant |
|---|---:|---:|---:|
| P(hidden-rule violation) | 0.5852 | 0.5667 | 0.4778 |
| P(undetected \| violation) | 0.6465 | 0.5400 | 0.4755 |
| Undetected-violation rate | 0.3815 | 0.3074 | 0.2296 |
| Legitimate task success | 0.3333 | 0.3370 | 0.3333 |
| Proxy reward | 0.8483 | 0.8361 | 0.7167 |
| Monitor false-positive rate | 0.0000 | 0.0774 | 0.0444 |

Normalized undetected-violation AUC was 0.2593 [0.1931, 0.3250] for values and rationales, 0.2190 [0.1778, 0.2847] for rules only, and 0.1782 [0.1028, 0.2625] for irrelevant SDF. Values-arm seeds first crossed the preregistered 0.25 threshold at steps 8, 16, and 8; rules-only at 12, 12, and 8; irrelevant at censored-16, censored-16, and 12. No checkpoint was selected using an evasion outcome.

`submission/curves.json` contains raw counts and rates for every arm, seed, and fixed checkpoint, including the direct/decomposed pairs, original reasoning strata, three monitor information budgets, and thinking-disabled control. `submission/results.json` contains all paired interactions, 10,000-resample uncertainty, AUC, threshold crossings, gates, immutable checkpoint references, and hashes of the omitted provider tables.

## Information and no-scratchpad controls

Pooling all arms and seeds at step 16, the chain monitor detected 192/440 hidden-rule violations, leaving 248 undetected, with 11/271 false positives. The action-only monitor detected 53/440 and the oracle-context monitor detected 440/440 with no false positives. This ladder confirms that ordinary scratchpads contain some useful evidence and private context closes the remaining gap.

Thinking-disabled inference produced a different policy distribution: 536 violations, of which the action-only monitor detected 44. Its high baseline violation rate makes it a capability control, not a clean mediation estimate. The original easy-versus-compositional strata now both show positive values-versus-rules interactions (+0.2112 and +0.1778 respectively), so the split observed in PR #402 does not reproduce once every latent case is duplicated across renderings.

## Reproducibility and interpretation

Frozen evaluation ran at commit `c2fe60e6d2f552544435be784395acca64399479` with zero new training updates. The run contains exactly 8,100 policy rows, 16,200 post-hoc monitor rows, and 160 calibration rows. Boundary verification and ten static tests pass. The committed execution logs contain timestamps, full non-secret configuration, commands, hashes, and compact output paths.

This is a synthetic clinical workflow with three training seeds and one policy/monitor family. The decomposed intervention is a failed mechanism test because its capability and serialization assumptions do not hold. The direct-note results nevertheless provide a clean fourth clinical stream on the same frozen trajectories and strengthen the relevant-SDF interaction. Strictly post-hoc monitoring cannot teach strategic monitor evasion; ordinary rule misunderstanding, rationalization, distribution shift, and autoregressive spillover remain viable explanations.
