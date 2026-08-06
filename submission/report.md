# Exact-mass randomized scratchpad credit in private-note lending

## Result in one sentence

Breaking the alignment between public reward and scratchpad-token credit
attenuated the values-and-rationales evasion interaction in all three seeds,
but retained only 32% of ordinary proxy learning; the preregistered efficacy
gate fails, so this is **not a supported selective-monitorability
intervention**.

## Question and design

This experiment asks whether reward-aligned updates to private reasoning carry
the extra post-RL monitor evasion associated with values-and-rationales
midtraining. It crosses three matched SDF conditions with two output-only RL
mechanisms:

- values plus rationales about the lending constitution;
- the identical lending rules, examples, confidentiality guidance, and public
  format without values or rationales;
- an irrelevant mirror corpus with identical document and token counts;
- ordinary sequence-wide public-reward RL; or
- exact-mass randomized scratchpad credit.

For every eight-rollout prompt group, ordinary centered public-reward
advantages are `A_i`. Four rollout indices receive random sign `s_i=+1` and
four receive `s_i=-1`, shuffled by a deterministic RNG before rewards are
observed. Public tokens receive `A_i`; scratchpad tokens receive
`s_i * abs(A_i)`. Every rollout therefore retains exactly its ordinary
absolute token-credit mass, but scratchpad credit direction has no designed
association with reward. This is not a no-scratchpad condition: the model
still generates private reasoning, every token remains in the same
importance-sampling loss, and every rollout enters the optimizer.

For closed generations, scratchpad extends through the final `</think>`
token. An unclosed generation is treated entirely as unfinished scratchpad.
No delimiter decides inclusion, and zero-advantage rollouts are retained.
Unlike the reverse-KL intervention in #377, randomized credit introduces no
pull toward a frozen SDF policy.

The complete 3 × 2 design uses Qwen3-8B, paired seeds 714/825/936, LoRA rank
32, 16 Tinker RL steps, and checkpoints 0, 4, 8, 12, and 16. The nine ordinary
trajectories are exact permanent references from #370. Nine new randomized-
credit trajectories fork the same SDF-final optimizer states. All corpora
have 60 documents and exactly 17,521 tokens per condition, identical per-
document lengths, and zero prohibited-term hits. The RL reward reads only the
generated public JSON action and justification.

Evaluation uses 90 disjoint cases per condition, seed, and checkpoint: 30
single-trigger violations, 30 compositional violations, and 30 legitimate
cases. A programmatic private-state oracle supplies exact violation labels.
The primary evidence-anchored Qwen3-30B chain-of-thought monitor knows the
lending constitution but not the private underwriting note. Action-only,
thinking-disabled, and oracle-context evaluations supply controls. Monitoring
is strictly post hoc after checkpoints are frozen and never participates in
training.

## Preregistered estimand and decision rule

The primary estimand is

> (values+rationales − rules-only step-0-to-16 change under ordinary RL) −
> (the same SDF interaction under randomized scratchpad credit)

for chain-monitor undetected-hack rate. Positive values indicate attenuation.
Support required a positive mean in at least two seeds, a positive ordinary-RL
source interaction in at least two seeds, a valid frozen monitor, exactly
6,912/6,912 intervention rollouts retained with exact counterbalancing and
negligible sign–reward association, and randomized-credit proxy improvement
at least 50% of ordinary improvement and positive in at least two seeds. All
requirements and checkpoints were frozen before training.

## Primary numerical result

| seed | ordinary values−rules interaction | randomized values−rules interaction | attenuation |
|---:|---:|---:|---:|
| 714 | +0.256 | −0.189 | +0.444 |
| 825 | +0.133 | +0.111 | +0.022 |
| 936 | +0.067 | +0.022 | +0.044 |
| mean | +0.152 | −0.019 | **+0.170** |

The paired-seed 95% bootstrap interval is [+0.022, +0.444]. The directional
and source-interaction rules pass. The broader values-versus-irrelevant
attenuation is +0.185 [+0.011, +0.344], also positive in all three seeds.

The P(hack) values-versus-rules three-way attenuation is only +0.019
[−0.233, +0.289], while conditional P(undetected | hack) attenuation is
+0.331 [−0.206, +1.106]. Thus the *differential* attenuation between values
and rules is not well explained by fewer violations alone, although the
conditional estimate is imprecise. Across all SDF arms, randomized credit did
reduce violation learning broadly by +0.200 ordinary-minus-randomized
[+0.156, +0.248]. Both facts matter: the primary interaction is suggestive,
but the intervention is not selective overall.

## Credit integrity succeeds; proxy efficacy fails

| SDF condition | retained / planned | closing-boundary rate | sign–reward correlation |
|---|---:|---:|---:|
| values + rationales | 2,304 / 2,304 | 97.66% | +.0198 |
| rules only | 2,304 / 2,304 | 97.79% | −.0118 |
| irrelevant | 2,304 / 2,304 | 98.35% | −.0109 |
| **total** | **6,912 / 6,912** | — | — |

Every prompt group has exactly four positive and four negative signs. Maximum
absolute credit-mass error is zero, all audit values are finite, and all
rollouts are retained. The maximum absolute sign–reward correlation is .0198,
below the frozen .05 bound. This repairs #375's treatment-dependent exclusion
and does so without #377's frozen-policy objective.

The proxy-learning efficacy gate nevertheless fails:

| seed | ordinary pooled proxy change | randomized pooled proxy change | retained fraction |
|---:|---:|---:|---:|
| 714 | +.211 | +.095 | .450 |
| 825 | +.282 | +.144 | .510 |
| 936 | +.183 | −.020 | −.107 |
| mean | **+.225** | **+.073** | **.324** |

Randomized credit improves the proxy in two seeds, unlike coefficient-1
reverse KL in #377, but its mean improvement is below the preregistered 50%
threshold. `hypothesis_supported` and `preregistered_support_rule_met` are
therefore false in `results.json` despite the positive primary contrast.

## Decomposition at step 16

Values are three-seed means. “Conditional” is P(undetected | hack).

| SDF | RL | P(hack) | conditional | undetected-hack | legitimate success | proxy reward | monitor FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| values + rationales | ordinary | .474 | .601 | .293 | .496 | .865 | .009 |
| values + rationales | randomized | .256 | .295 | .081 | .670 | .722 | .005 |
| rules only | ordinary | .359 | .320 | .130 | .611 | .791 | .000 |
| rules only | randomized | .159 | .345 | .089 | .656 | .595 | .000 |
| irrelevant | ordinary | .515 | .355 | .181 | .485 | .913 | .000 |
| irrelevant | randomized | .333 | .409 | .156 | .652 | .795 | .000 |

Pooling all three SDF conditions, ordinary minus randomized step-0-to-16
learning is +0.200 for P(hack), +0.093 for undetected-hack rate, and +0.152
for proxy reward. The intervention suppresses the profitable policy broadly,
not merely its monitor-facing rationale. Legitimate success rises because the
less proxy-optimized policies approve fewer hidden-rule violations, another
reason not to equate the lower joint failure rate with selective monitorability.

## AUC and time to threshold

Trapezoidal AUC is normalized by the 16-step horizon.

| SDF | RL | undetected-hack AUC | hack AUC | proxy AUC |
|---|---|---:|---:|---:|
| values + rationales | ordinary | .151 | .318 | .765 |
| values + rationales | randomized | .039 | .177 | .665 |
| rules only | ordinary | .086 | .238 | .715 |
| rules only | randomized | .027 | .077 | .569 |
| irrelevant | ordinary | .094 | .375 | .827 |
| irrelevant | randomized | .075 | .234 | .731 |

The preregistered undetected-hack threshold is 0.25. Ordinary values plus
rationales crosses at step 16 for seed 714 and step 12 for seed 825; seed 936
is censored. Ordinary rules-only crosses at step 8 for seed 825. Among
randomized arms, only irrelevant seed 825 reaches the threshold, at step 16;
all others are censored. No checkpoint was selected.

## Monitor, reasoning-load, and capability controls

The frozen monitor gate passes: pooled step-0 sensitivity is 0.882 (minimum
0.80) and false-positive rate is 0.0031 (maximum 0.05). The primary attenuation
is +0.244 [+0.067, +0.400] on easy cases and +0.267 [−0.333, +1.067] on
compositional cases. Easy-case attenuation is sign-consistent; compositional
results are heterogeneous and do not establish a reliable reasoning-load
difference.

The action-only monitor has the same public information budget but no private
reasoning text. The oracle-context monitor receives the exact private note.
Thinking-disabled policy samples, valid-JSON rate, public-factor citation,
legitimate success, and proxy reward are stored for every seed and checkpoint.
These controls separate monitor access, information asymmetry, and general
capability from the primary chain-monitor endpoint.

`curves.json` contains all 90 policy curve records, 144 per-step training-audit
records, raw episode counts, AUCs, time-to-threshold censoring, and all control
metrics. Intervals use 10,000 bootstrap resamples over the three paired seed
units; with three seeds, they are stability summaries rather than precise
population intervals.

## Interpretation

This result gives a graded mechanism picture across three interventions.
#375's public-only mask attenuated the semantic interaction but selected
rollouts by treatment-dependent delimiters. #377 retained all rollouts but
coefficient-1 reverse KL eliminated proxy learning. Exact-mass randomization
retains every rollout and recovers some proxy learning while preserving a
sign-consistent attenuation. That progression is consistent with reward-
aligned scratchpad updates contributing to the SDF-by-output-RL interaction.

It is not yet a clean causal demonstration of a negative monitorability
externality. Randomized credit still removes too much overall proxy and
violation learning, failing the frozen efficacy gate. A principled next test
should preserve each rollout's *signed* total credit while concentrating it on
the public suffix, using a deterministic fallback token for unclosed outputs.
That would directly repair #375's dropout while avoiding both scratchpad noise
and a frozen-policy penalty. The present result is reported as a failed gate,
not rescued by its favorable primary number.

![Training curves and credit-integrity audit](figures/lending_scratchpad_credit_randomization.png)
