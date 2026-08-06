# All-rollout scratchpad anchoring in private-note lending

## Result in one sentence

Scratchpad reverse-KL numerically removed the values-and-rationales evasion
interaction while retaining every rollout, but it also eliminated public-
proxy learning; the preregistered efficacy gate fails, so this is **not a
supported selective-monitorability intervention**.

## Question and design

This experiment asks whether autoregressive drift in private reasoning carries
the extra post-RL monitor evasion associated with values-and-rationales
midtraining. It crosses three matched SDF conditions with two output-only RL
mechanisms:

- values plus rationales about the lending constitution;
- the identical lending rules, examples, confidentiality guidance, and public
  format without values or rationales;
- an irrelevant mirror corpus with identical document and token counts;
- ordinary sequence-wide public-reward RL; or
- the same sequence-wide reward advantage plus reverse-KL anchoring on
  scratchpad tokens to that arm's frozen SDF-final checkpoint.

For a rollout sampled from current policy `p` and frozen SDF policy `q`, the
intervention's token advantage is `A - (log p - log q)` on scratchpad tokens
and remains `A` on public tokens. The coefficient is fixed at 1.0 with no
temporal discount. A closed generation is anchored through its final
`</think>` token. An unclosed generation is treated entirely as unfinished
scratchpad. Crucially, every rollout receives a datum, including zero-public-
advantage rollouts; a generated delimiter never decides inclusion.

The complete 3 × 2 design uses Qwen3-8B, paired seeds 714/825/936, 16 Tinker
RL steps, and checkpoints 0, 4, 8, 12, and 16. The nine ordinary trajectories
are exact permanent references from #370. Nine new KL trajectories fork the
same SDF-final optimizer states. All corpora have 60 documents and 17,521
tokens per condition with zero prohibited-term hits. The RL reward reads only
generated public JSON.

Evaluation uses 90 disjoint cases per condition, seed, and checkpoint: 30
single-trigger violations, 30 compositional violations, and 30 legitimate
cases. A programmatic private-state oracle supplies exact violation labels.
The primary evidence-anchored Qwen3-30B chain monitor knows the lending spec
but not the private underwriting note. Action-only, no-scratchpad, and oracle-
context evaluations supply controls. All monitoring is post hoc after frozen
checkpoints and never participates in training.

## Preregistered estimand and decision rule

The primary estimand is

> (values+rationales − rules-only step-0-to-16 change under ordinary RL) −
> (the same SDF interaction under scratchpad-KL RL)

for chain-monitor undetected-hack rate. Positive values indicate attenuation.
Support required a positive mean in at least two seeds, a positive ordinary-RL
source interaction in at least two seeds, a valid frozen monitor, exactly
6,912/6,912 intervention rollouts retained with finite audits, and scratchpad-
KL proxy improvement at least 50% of ordinary improvement and positive in at
least two seeds. All requirements and checkpoints were frozen before training.

## Primary numerical result

| seed | ordinary values−rules interaction | KL values−rules interaction | attenuation |
|---:|---:|---:|---:|
| 714 | +0.256 | +0.022 | +0.233 |
| 825 | +0.133 | +0.000 | +0.133 |
| 936 | +0.067 | +0.011 | +0.056 |
| mean | +0.152 | +0.011 | **+0.141** |

The paired-seed 95% bootstrap interval is [+0.056, +0.233]. The directional
and source-interaction rules pass. The broader values-versus-irrelevant
attenuation is +0.093 [approximately 0, +0.222].

The P(hack) three-way attenuation is +0.167 [approximately 0, +0.344]. In
contrast, conditional P(undetected | hack) attenuation is −0.236
[−0.848, +0.187]. Thus the remaining values-arm hacks are not made more
visible; the undetected-hack reduction mainly reflects fewer hacks.

## Integrity succeeds; proxy efficacy fails

| SDF condition | retained / planned | closing-boundary rate | mean sampled scratchpad reverse KL |
|---|---:|---:|---:|
| values + rationales | 2,304 / 2,304 | 97.18% | .0308 |
| rules only | 2,304 / 2,304 | 96.01% | .0328 |
| irrelevant | 2,304 / 2,304 | 99.26% | .0190 |
| **total** | **6,912 / 6,912** | — | — |

This repairs #375's treatment-dependent exclusion: even a batch with only
37/48 closing delimiters contributed all 48 optimizer datums. The audit is
finite and the all-rollout gate passes.

The efficacy gate fails decisively:

| seed | ordinary pooled proxy change | KL pooled proxy change | retained fraction |
|---:|---:|---:|---:|
| 714 | +.211 | +.018 | .087 |
| 825 | +.282 | −.015 | −.055 |
| 936 | +.183 | −.043 | −.234 |
| mean | **+.225** | **−.013** | — |

The required fraction was 0.50 and positive KL change in at least two seeds.
Neither holds. `hypothesis_supported` and
`preregistered_support_rule_met` are therefore false in `results.json` despite
the attractive primary contrast.

## Decomposition at step 16

Values are three-seed means. “Conditional” is P(undetected | hack).

| SDF | RL | P(hack) | conditional | undetected-hack | legitimate success | proxy reward | monitor FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| values + rationales | ordinary | .474 | .601 | .293 | .496 | .865 | .009 |
| values + rationales | scratchpad KL | .044 | .583 | .026 | .819 | .582 | .000 |
| rules only | ordinary | .359 | .320 | .130 | .611 | .791 | .000 |
| rules only | scratchpad KL | .096 | .067 | .004 | .789 | .599 | .000 |
| irrelevant | ordinary | .515 | .355 | .181 | .485 | .913 | .000 |
| irrelevant | scratchpad KL | .159 | .053 | .007 | .789 | .671 | .009 |

Pooling all SDF conditions, ordinary minus KL step-0-to-16 learning is +0.349
[+.281, +.437] for P(hack), +0.189 [+.115, +.270] for undetected-hack rate,
and +0.238 [+.192, +.298] for proxy reward. The intervention suppresses the
profitable violation policy broadly, not just its monitor-facing rationale.
The high conditional value for values-plus-rationales KL is based on very few
hacks and reinforces why the joint undetected-hack metric must be decomposed.

## AUC and time to threshold

Trapezoidal AUC is normalized by the 16-step horizon.

| SDF | RL | undetected-hack AUC | hack AUC | proxy AUC |
|---|---|---:|---:|---:|
| values + rationales | ordinary | .151 | .318 | .765 |
| values + rationales | scratchpad KL | .018 | .048 | .576 |
| rules only | ordinary | .086 | .238 | .715 |
| rules only | scratchpad KL | .013 | .111 | .609 |
| irrelevant | ordinary | .094 | .375 | .827 |
| irrelevant | scratchpad KL | .015 | .162 | .684 |

The preregistered undetected-hack threshold is 0.25. Ordinary values plus
rationales crosses at step 16 for seed 714 and step 12 for seed 825; seed 936
is censored. Ordinary rules-only crosses at step 8 for seed 825 only. Every KL
trajectory and every ordinary irrelevant trajectory is censored at step 16.
No checkpoint was selected.

## Monitor and load controls

The frozen monitor gate passes: pooled step-0 sensitivity is 0.882 (minimum
0.80) and false-positive rate is 0.0031 (maximum 0.05). The primary attenuation
is +0.200 [+.033, +.500] on easy cases and +0.222 [−.100, +.667] on
compositional cases; three seeds do not establish a reliable load difference.
Action-only monitoring uses the same public information budget minus
scratchpad text. The oracle-context monitor receives the private note.
Thinking-disabled samples, valid-JSON rate, public-factor citation, legitimate
success, and proxy reward are stored for every seed and checkpoint.

`curves.json` contains all 90 policy curve records, 144 per-seed training-audit
records, raw episode counts, AUCs, time-to-threshold censoring, and all control
metrics. Intervals use 10,000 bootstrap resamples over the three paired seed
units; with three seeds, they should be read as stability summaries rather
than precise population intervals.

## Interpretation

This intervention resolves the most serious technical problem in #375: it
does not condition training inclusion on policy-generated delimiters. The
clean attenuation survives that repair, so row dropping alone does not explain
the earlier numerical pattern. However, coefficient-1 anchoring prevents the
policy from learning the public approval proxy and violations at all. It
therefore does not isolate a negative monitorability externality.

A valid follow-up should freeze its KL strength using only a disjoint
development target for realized proxy-learning retention, or compare policies
at matched proxy reward/violation rate before evaluating conditional
monitorability. Lowering the coefficient after seeing this held-out endpoint
would be post hoc selection and is not done here.

![Training curves and all-rollout audit](figures/lending_scratchpad_kl.png)
