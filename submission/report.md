# Evaluation-time reasoning cap leaves the aggregate SDF interaction near zero but reverses it by reasoning load

## Executive finding

This evaluation-only factorial does **not** support the hypothesis that a fixed private-reasoning cap attenuates the values-and-rationales SDF interaction with output-only reinforcement learning. On immutable ordinary-training Qwen3-8B trajectories, the capped-evaluation minus ordinary-evaluation difference in the values-versus-irrelevant step-0-to-16 chain-monitor undetected-hack interaction was **−0.0074**, with paired seeds **−0.0444, +0.0444, −0.0222** and a paired-seed bootstrap interval **[−0.0444, +0.0444]**.

The literal preregistered directional sign rule passes because the mean is negative and two seeds are negative. The full support rule fails for two independent reasons: the ordinary-renderer source interaction does not reproduce on this fresh case stream, and pooled baseline monitor sensitivity is 0.797, below the frozen 0.80 minimum. The scientific conclusion follows the magnitude and gates: this is a heterogeneous near-null, not evidence that the cap improves monitorability.

A secondary reasoning-load control is more structured. The renderer effect on the primary interaction is +0.122 for easy violations but −0.144 for compositional violations; the compositional-minus-easy difference is **−0.267 [−0.533, −0.133]**, negative in all seeds. Thus aggregate nullity hides a load-dependent reversal, which is useful for designing future renderer comparisons but does not rescue the failed primary claim.

![Primary paired renderer effect](figures/primary_seed_effects.png)

## Design and information boundaries

No model was trained in this attempt. I reused all 18 training trajectories and 90 fixed checkpoints from PR #393 at source training commit `acff1a061b82854e1e2895c3894bd083817bf9b6`: values-and-rationales, rules-only, and token-matched irrelevant SDF under capped or ordinary output-only RL, for seeds 714/825/936 and checkpoints 0/4/8/12/16. The evaluation execution commit was `57e0b5b78ac9eca275e885d1f5116db99b43ee5a`.

Every checkpoint answered the same 90 new `heldout10-renderer-factorial-*` cases under three procedures:

- capped evaluation: at most 160 sampled private tokens, the standard Qwen3 closing delimiter, then at most 256 public tokens;
- ordinary evaluation: ordinary one-stage Qwen3 decoding for at most 512 tokens;
- no-scratchpad control: thinking disabled, with at most 256 public tokens.

Temperature was zero. The stream contains 30 easy hidden-rule violations, 30 compositional violations, and 30 legitimate decisions. A programmatic hidden-state oracle supplies exact violation labels. The Qwen3-30B chain monitor knows the constitution and sees the public case, public output, and generated scratchpad, but not the private underwriting note. The action-only monitor loses the scratchpad; the oracle-context monitor receives exact hidden state. Monitoring began only after all 24,300 policy rows were frozen.

The historical RL reward inspected only the generated public action/justification JSON. It never received a scratchpad, hidden predicate, oracle label, monitor verdict, or private evaluator label. Neither SDF corpus mentions monitoring, evaluation, detection, reward hacking, shutdown, or chain of thought. All three corpora contain 60 documents and exactly 17,521 Qwen3-8B tokens per condition with identical per-document lengths; optimizer, format, loss, and token counts are matched.

## Preregistered estimand and gates

For the ordinary-training trajectories, let `U(r,a,s,t)` be chain-monitor undetected-hack rate for renderer `r`, SDF arm `a`, seed `s`, and RL checkpoint `t`. The primary paired contrast is:

`D(s) = {Δ[values − irrelevant] under capped evaluation} − {Δ[values − irrelevant] under ordinary evaluation}`,

where `Δ` is checkpoint 16 minus checkpoint 0. The binary support field required:

1. mean `D(s) < 0` and at least two negative seeds;
2. a positive ordinary-evaluation source interaction in at least two seeds and on average;
3. pooled step-0 chain-monitor sensitivity at least 0.80 and false-positive rate at most 0.05;
4. mean valid-public-JSON rate at least 0.90 separately for both renderers.

All checkpoints, three seeds, a 0.25 time threshold, uncertainty method, and controls were frozen before service use. No checkpoint was selected on evasion outcomes.

## Primary result and decomposition

| Quantity | Mean | Paired seed values | Paired-seed 95% interval |
|---|---:|---|---:|
| Undetected-hack renderer effect (primary) | −0.0074 | −0.0444, +0.0444, −0.0222 | [−0.0444, +0.0444] |
| P(hack) renderer effect | +0.0852 | +0.0778, +0.0333, +0.1444 | [+0.0333, +0.1444] |
| P(undetected \| hack) renderer effect | −0.1622 | −0.3643, +0.0919, −0.2142 | [−0.3643, +0.0919] |

The joint near-null is a cancellation: capped evaluation increases the rich-SDF-versus-irrelevant interaction in violations in all seeds, but tends to reduce the corresponding conditional miss interaction. This does not identify deliberate concealment. The renderer changes available computation, truncation, public completion, and evidence exposed to the post-hoc monitor at once.

The same primary renderer effect on capped-training trajectories is +0.0185 [−0.1111, +0.2667], with seeds −0.1111, −0.1000, +0.2667. That secondary population is also heterogeneous and near zero. The ordinary-training values-versus-rules renderer effect is instead +0.0593 [+0.0444, +0.0889], positive in every seed and opposite the preregistered attenuation direction. It should be treated as a secondary semantic comparison.

![Behavioral and conditional decomposition](figures/primary_decomposition.png)

## Source interaction and gate failures

Under ordinary evaluation, the values-versus-irrelevant source interaction in undetected-hack growth is only **+0.0037 [−0.0333, +0.0444]**, with seeds +0.0444, 0.0000, −0.0333. Under capped evaluation on the same fresh cases it is −0.0037 [−0.0556, +0.0444]. The exact checkpoints showed a larger positive interaction on PR #393's separate case stream, so the lack of replication here exposes case-stream sensitivity rather than a training difference. This failed source gate is decisive.

Pooled step-0 chain-monitor sensitivity is **0.7971** with false-positive rate 0. By renderer, sensitivity is 0.7836 for capped outputs and 0.8171 for ordinary outputs, each with zero false positives. The 0.003 shortfall from the frozen pooled threshold is small, but the gate is not rounded into a pass. Public JSON capability passes: 0.9999 under capped evaluation and 0.9738 under ordinary evaluation.

The action-only monitor misses substantially more violations at checkpoint 16 than the chain monitor, while the oracle-context monitor misses none. Across the two primary SDF arms, endpoint undetected-hack rate is 0.254 (chain), 0.465 (action-only), and 0 (oracle) under capped evaluation; the corresponding ordinary values are 0.219, 0.437, and 0. The no-scratchpad action-only control is 0.565 for both nested renderer records because the same separately sampled no-scratchpad output is intentionally reused.

![Monitor information-budget controls](figures/monitor_controls.png)

## Reasoning-load interaction

| Violation stratum | Renderer effect on primary interaction | Seed values | 95% interval |
|---|---:|---|---:|
| Easy | +0.1222 | +0.2000, +0.1333, +0.0333 | [+0.0333, +0.2000] |
| Compositional | −0.1444 | −0.3333, 0.0000, −0.1000 | [−0.3333, 0.0000] |
| Compositional minus easy | −0.2667 | −0.5333, −0.1333, −0.1333 | [−0.5333, −0.1333] |

This control suggests the cap changes the measured interaction differently when several hidden facts must be combined. The decomposition is coherent: for compositional cases, the cap's P(hack) interaction is +0.244 but its conditional-nondetection interaction is −0.327; for easy cases those quantities are +0.011 and +0.513. Because the strata contain only 30 cases each and the aggregate primary is gated null, this is a mechanism-generating observation rather than confirmatory evidence.

## Curves, AUC, threshold time, and capability

All 180 condition × renderer × seed × checkpoint cells appear in `curves.json` with raw counts for P(hack), P(undetected | hack), undetected-hack rate, legitimate task success, proxy reward, and monitor false positives. The solid/dashed curves below show means; shaded envelopes span the three training seeds.

![Primary curves](figures/primary_curves.png)

For ordinary-training values and irrelevant arms, normalized undetected-hack AUCs are 0.190 and 0.134 under capped evaluation, versus 0.130 and 0.113 under ordinary evaluation. The AUC contrast is descriptive; the frozen primary uses endpoints. At the preregistered 0.25 threshold, capped-values crosses at steps 8, 16, and censored-at-16 across seeds; capped-irrelevant crosses at step 16 for one seed and is censored for two. Every ordinary-values seed is censored at 16; ordinary-irrelevant crosses at step 16 for one seed and is censored for two. Exact per-seed AUCs and threshold records for all 36 curves are in `results.json`.

At checkpoint 16 pooled over the two primary SDF arms, capped versus ordinary evaluation yields:

| Metric | Capped evaluation | Ordinary evaluation |
|---|---:|---:|
| P(hack) | 0.524 | 0.491 |
| P(undetected \| hack) | 0.488 | 0.453 |
| Undetected-hack rate | 0.254 | 0.219 |
| Legitimate task success | 0.470 | 0.489 |
| Public proxy reward | 0.912 | 0.887 |

These levels show that the cap materially changes behavior even though its SDF-by-RL interaction is near zero. They also make clear why the result is not a claim that cap deployment is uniformly safer.

## Reproducibility and integrity

The live canary sampled both renderers from one immutable checkpoint, produced nonempty scratchpads and valid public JSON, and recorded `training_updates: 0`. The full run produced exactly:

- 18 immutable trajectories and 90 checkpoint references;
- 24,300 unique policy rows: 8,100 capped, 8,100 ordinary, 8,100 no-scratchpad;
- 56,700 unique monitor rows: 16,200 chain, 24,300 action-only, 16,200 oracle-context;
- 180 raw curve records, fixed-checkpoint AUCs, and right-censored threshold times.

The strict verifier and all eight tests pass. `results.json` records configuration, source/evaluation commits, input and raw-output hashes, corpus hashes, bootstrap method, gates, and all factorial contrasts. `curves.json` retains episode counts so every reported rate can be reconstructed. Raw provider transcripts are intentionally not committed.

## Limitations and conclusion

Only three training seeds, one 8B policy family, one lending environment, one cap budget, and deterministic policy samples support inference. Percentile intervals over three seed units are stability summaries, not high-resolution population intervals. The monitor's capped-renderer sensitivity failure and mediocre evidence-anchor rate limit monitor-specific interpretation. The failed fresh-stream source interaction also shows that the original effect is not stable enough here to support a renderer attenuation claim.

The most defensible conclusion is therefore negative: **a 160-token evaluation-time reasoning cap does not materially change the aggregate values-and-rationales SDF-by-output-RL undetected-hack interaction on these fixed 8B trajectories.** It does change violation propensity, conditional monitor misses, and reasoning-load strata in offsetting ways. A stronger follow-up would preregister the all-seed load crossover on additional case streams and seeds while improving capped-output monitor sensitivity, without modifying training or selecting checkpoints.
