# Evaluation-time private-reasoning cap factorial

## Result in brief

The preregistered attenuation hypothesis was not supported. Holding every trained Qwen3-8B checkpoint fixed, imposing a hard 160-token private-reasoning cap at evaluation increased rather than decreased the `+SDF(values+rationales)` versus irrelevant-SDF step-0-to-16 interaction in chain-monitor undetected-hack rate. The capped-minus-ordinary four-way interaction was **+0.1000**, with a paired-seed 95% bootstrap interval of **[-0.0333, +0.1778]** and seed estimates **+0.1556, +0.1778, -0.0333**.

This is evidence that the evaluation renderer is a load-bearing intervention, not evidence for a robust SDF externality by itself. On the same fresh cases, the source values-versus-irrelevant interaction under ordinary evaluation was -0.0296 [-0.1000, +0.0111], so the preregistered source-replication gate failed.

## Design and information boundary

This experiment reuses the exact immutable checkpoints from PR #370: values-and-rationales SDF, rules-only SDF, and an irrelevant mirror SDF, each with seeds 714, 825, and 936 and RL checkpoints 0, 4, 8, 12, and 16. All policies are `Qwen/Qwen3-8B`. SDF training used 60 documents, three epochs, token-mean cross-entropy, batch size 10, learning rate 1e-4, and LoRA rank 32. The three corpora are exactly matched at 17,521 tokenizer tokens per arm per epoch and contain zero hits for the prohibited monitoring-related terms. The output-only RL histories are also identical to #370: 16 steps, six prompts per step, group size eight, importance-sampling loss, and learning rate 4e-5.

The sole experimental manipulation is evaluation rendering on identical held-out cases and checkpoints:

- **Ordinary:** a single generation with a 416-token total budget.
- **Capped:** at most 160 private tokens, then a zero-credit structural delimiter and at most 256 public tokens. The total sampled-token budget remains 416.
- **No scratchpad control:** thinking disabled, sampled separately and read only by the action monitor.

The fresh `heldout10-renderer-factorial` set contains 90 private-rule lending cases: 30 easy violations, 30 compositional violations, and 30 legitimate cases. The programmatic hidden-state oracle determines whether the public lending action violates the private rule. The primary chain monitor knows the public specification and reads the private reasoning plus public action/justification, but never sees the private underwriting note. Its extracted claims require verbatim, field-specific, polarity-consistent evidence. The action-only monitor has the same public information without private reasoning. The oracle-context monitor additionally receives exact private state. All monitoring is post hoc.

RL reward was historical and public-output-only. It never inspected private reasoning, monitor judgments, hidden oracle labels, or evaluator labels. Checkpoints and the 0.25 time-to-threshold criterion were frozen before sampling; no held-out checkpoint selection occurred.

## Preregistered test

The primary estimand is

`(capped - ordinary renderer) × (values/rationales SDF - irrelevant SDF) × (RL step 16 - step 0)`

for chain-monitor undetected-hack rate. Attenuation required a negative mean, at least two of three negative seed estimates, a negative conditional-nondetection interaction, a positive ordinary-renderer source interaction in at least two seeds, valid monitor sensitivity/false-positive gates, and at least 90% valid JSON at the endpoint. The capability and monitor gates passed; every directional or source requirement failed. The formal support decision is therefore **false**.

| Outcome | Capped minus ordinary four-way mean | Paired-seed 95% interval | Seed estimates |
|---|---:|---:|---:|
| Undetected-hack rate (primary) | +0.1000 | [-0.0333, +0.1778] | +0.1556, +0.1778, -0.0333 |
| Violation-action rate | +0.1296 | [+0.0778, +0.1778] | +0.1333, +0.1778, +0.0778 |
| Undetected given violation | +0.2646 | [-0.2092, +0.5182] | +0.5182, +0.4848, -0.2092 |

The cap therefore changed actions, not just monitor evidence: the violation-action component was positive for all three seeds. The conditional-nondetection component was larger but heterogeneous.

## Endpoint curves

Means below pool the three paired seeds at the fixed endpoint. Arrows show step 0 to step 16.

| Renderer | SDF arm | P(violation) | P(undetected \| violation) | Undetected-hack | Legitimate success | Proxy reward | Monitor FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Ordinary | Values+rationales | .141→.448 | .377→.457 | .052→.204 | .689→.474 | .604→.820 | .005→.000 |
| Ordinary | Rules-only | .137→.411 | .132→.329 | .019→.133 | .704→.548 | .592→.818 | .000→.000 |
| Ordinary | Irrelevant | .226→.533 | .113→.407 | .026→.207 | .715→.463 | .698→.920 | .000→.000 |
| Capped | Values+rationales | .159→.496 | .218→.587 | .037→.293 | .741→.500 | .658→.899 | .000→.000 |
| Capped | Rules-only | .230→.419 | .130→.324 | .030→.137 | .704→.567 | .705→.829 | .000→.000 |
| Capped | Irrelevant | .337→.544 | .110→.428 | .037→.222 | .663→.456 | .792→.926 | .000→.011 |

All 90 raw per-seed/checkpoint/renderer/arm curve records, including numerators and denominators, are in `curves.json`.

## AUC and time to threshold

AUC is trapezoidal over RL steps 0, 4, 8, 12, and 16, divided by the 16-step horizon.

| Renderer | SDF arm | Violation AUC | Conditional nondetection AUC | Undetected-hack AUC | Legitimate-success AUC | Proxy-reward AUC |
|---|---|---:|---:|---:|---:|---:|
| Ordinary | Values+rationales | .331 | .421 | .143 | .578 | .749 |
| Ordinary | Rules-only | .291 | .282 | .090 | .636 | .731 |
| Ordinary | Irrelevant | .402 | .245 | .105 | .578 | .834 |
| Capped | Values+rationales | .401 | .457 | .194 | .577 | .828 |
| Capped | Rules-only | .343 | .305 | .112 | .625 | .784 |
| Capped | Irrelevant | .444 | .250 | .116 | .554 | .865 |

At the preregistered undetected-hack threshold of 0.25, all ordinary values/rationales seeds were right-censored at step 16. Under the cap, values/rationales seeds 714 and 825 crossed at steps 8 and 12; seed 936 remained censored. Every rules-only seed was censored under both renderers. For irrelevant SDF, seed 825 crossed at step 16 under both renderers and the other seeds were censored.

## Controls and mechanism checks

The cap was genuinely binding. At step 16, 68.9% of capped values/rationales samples, 57.4% of rules-only samples, and 47.8% of irrelevant samples hit 160 private tokens. In ordinary generation, respectively 63.3%, 60.4%, and 45.6% naturally exceeded 160. Mean private-token counts at the values endpoint were 147.9 capped versus 183.1 ordinary.

This was not explained by a formatting collapse. Endpoint valid-public-JSON rates were 99.6% for capped values/rationales and 100% for capped rules-only and irrelevant; all six renderer-arm cells exceeded the preregistered 90% gate. Proxy reward was also slightly higher under capped rendering in each arm.

The independently calibrated chain monitor passed its validity gate under both renderers: sensitivity/FPR were .809/.0018 ordinary and .857/.0000 capped. The oracle-context monitor had zero endpoint undetected-hack rate in every arm, confirming that the failures require information asymmetry. At step 16, action-only undetected-hack rates were higher than chain-monitor rates in every renderer/arm cell; for capped values/rationales they were .444 versus .293. The thinking-disabled action-only control produced endpoint rates .496 values/rationales, .470 rules-only, and .644 irrelevant.

The cap's primary four-way interaction was +0.1889 [-0.1000, +0.4333] on compositional violations and +0.1111 [0.0000, +0.3000] on easy violations. Thus amplification was not confined to cases requiring conjunction reasoning, though uncertainty is large.

The semantic control sharpens the finding. Capped-minus-ordinary undetected-hack interaction was +0.1111 [-0.0222, +0.1889] for values/rationales versus rules-only, but -0.0111 [-0.0333, +0.0111] for rules-only versus irrelevant. The positive renderer sensitivity concentrates in the richer behavioral-spec arm, but three seeds are insufficient to separate a reproducible semantic mechanism from checkpoint-specific instability.

## Interpretation and limitations

PR #393 found that changing whether the cap was present during output-only RL produced an essentially null training-protocol interaction while all evaluation used a capped renderer. This crossed follow-up shows that changing the renderer itself, with weights held fixed, can move the estimated interaction substantially. One plausible mechanism is autoregressive coupling: truncating private generation changes the subsequent public action distribution, not merely the evidence available to the monitor. The all-positive violation-action interaction supports that mechanism.

However, the fresh ordinary-renderer source interaction failed to replicate and the primary interval includes zero. The result should not be read as confirming that values/rationales SDF reliably causes monitor evasion. It instead demonstrates a measurement hazard: an inference-time reasoning cap can selectively amplify a weak or reversing SDF-by-RL interaction. Replication on independently trained checkpoints and another private-rule environment is needed.

## Reproducibility

Execution used committed state `9dc65df4e08f7f9cc317899271f0ce87219ac0c8`, Tinker 0.24.0, and `tinker_cookbook` 0.5.4.dev4. Exact checkpoint IDs, hashes, non-secret configuration, generation code, preregistration, execution manifest, and timestamped log are under `attempts/lending-renderer-factorial/`. Compact scoreable artifacts are `submission/results.json`, `submission/curves.json`, and `submission/figures/lending_renderer_factorial.svg`. Uncertainty is a two-sided percentile 95% interval from 10,000 resamples of the three paired seed units.
