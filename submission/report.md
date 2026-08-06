# Scratchpad-credit dose response in private-note lending

## Result in one sentence

Reducing reward-aligned scratchpad credit produced a positive three-dose
values-versus-rules evasion slope in all three seeds while retaining 74% of
ordinary proxy learning at the registered midpoint; every preregistered gate
passes, although conditional nondetection alone remains imprecise.

## Question and design

This experiment asks whether the extra monitor evasion associated with
values-and-rationales specification midtraining grows with the amount of
public-reward credit applied to private reasoning tokens. It crosses three
token-matched midtraining corpora with three output-only reinforcement-learning
(RL) mechanisms:

- values plus causal rationales about a lending constitution;
- the identical lending rules, examples, confidentiality guidance, and public
  format without the values or causal rationales;
- an irrelevant mirror corpus with identical document and token counts; and
- scratchpad-credit coefficients 1.00, 0.50, and 0.25.

For centered public-output reward advantage `A`, public continuation tokens
always receive `A`. Tokens through the final `</think>` delimiter receive
`lambda*A`, where `lambda` is the registered scratchpad-credit coefficient.
An unclosed generation is treated as wholly private and receives `lambda*A`.
Every generated token and every rollout remains in the same importance-
sampling loss, and no token advantage exceeds the ordinary coefficient-1
update. Total sequence credit is intentionally not held fixed: reward-aligned
credit on private tokens is the causal dose.

This design differs from disabling scratchpads during training or concentrating
sequence credit onto a short public suffix. The model uses its normal
scratchpad renderer in every training arm, public-token credit is unchanged,
and fractional private-token credit neither drops delimiter failures nor
amplifies individual advantages.

The 3 x 3 design uses Qwen3-8B, paired seeds 714/825/936, rank-32 LoRA, 16
Tinker RL steps, six prompts and eight rollouts per step, and fixed checkpoints
0, 4, 8, 12, and 16. The nine coefficient-1 trajectories are exact permanent
references from #370. Eighteen new fractional trajectories fork the same
SDF-final optimizer states. All three corpora contain 60 documents and exactly
17,521 Qwen tokens, have identical per-document lengths, and contain zero
instances of the prohibited monitoring-related terms. The RL reward reads only
the generated public JSON action and justification.

Evaluation uses a fresh registered stream of 90 cases per condition, seed, and
checkpoint: 30 single-trigger hidden-rule violations, 30 compositional
violations, and 30 legitimate cases. A programmatic private-state oracle gives
exact violation labels. The primary evidence-anchored Qwen3-30B chain-of-
thought monitor knows the constitution but not the private underwriting note.
Action-only, thinking-disabled, and oracle-context evaluations are controls.
Monitoring is strictly post hoc after all policy checkpoints freeze and never
participates in training.

## Preregistered estimand and decision rule

For each seed and scratchpad-credit coefficient, the SDF interaction is the
step-0-to-16 change in chain-monitor undetected-hack rate for values plus
rationales minus the corresponding rules-only change. The primary estimand is
the ordinary least-squares slope of that interaction against coefficients
0.25, 0.50, and 1.00. A positive slope means richer-specification evasion grows
with reward-aligned scratchpad credit.

Directional support requires a positive mean slope that is positive in at
least two seeds. The predefined coefficient-0.50 attenuation relative to
coefficient 1.00 and the exact coefficient-1 source interaction must each also
have positive means and positive signs in at least two seeds. Headline support
additionally requires:

- pooled coefficient-1 step-0 monitor sensitivity at least .80 and false-
  positive rate at most .05;
- exactly 13,824 unique fractional-dose audits, 100% rollout retention, correct
  coefficients, finite values, and token/signed-mass/absolute-mass assignment
  error at most `1e-8`; and
- coefficient-0.50 pooled proxy improvement positive in at least two seeds and
  at least 50% of ordinary proxy improvement.

All endpoints, checkpoints, gates, and the 10,000-resample paired-seed
bootstrap were frozen before fractional training. No held-out checkpoint was
selected.

## Primary result

The preregistered slope is **+0.126 per unit scratchpad credit**, with a
paired-seed 95% bootstrap interval of **[+0.060, +0.248]**. Every seed has a
positive slope.

| seed | interaction at lambda=.25 | at lambda=.50 | at lambda=1.00 | OLS slope |
|---:|---:|---:|---:|---:|
| 714 | +.022 | -.111 | +.044 | +.070 |
| 825 | -.033 | -.011 | +.144 | +.248 |
| 936 | +.056 | -.044 | +.078 | +.060 |
| **mean** | **+.015** | **-.056** | **+.089** | **+.126** |

The exact coefficient-1 source interaction is +.089 [+.044, +.144], positive
in all seeds. The separately frozen midpoint contrast is also strong:
ordinary minus coefficient-.50 interaction is **+.144 [+.122, +.156]**, again
positive in all seeds. Thus the slope is not rescued by selecting the quarter-
credit endpoint after observation. All directional requirements and all gates
pass, so `preregistered_support_rule_met` and `hypothesis_supported` are true.

The decomposition limits the causal language. The corresponding P(hack) slope
is +.113 [+.003, +.225]. The conditional P(undetected | hack) slope is
directionally positive at +.078 but has a wide interval [-.219, +.573]. The
joint result therefore supports a graded scratchpad-credit contribution to the
SDF-by-output-RL interaction, not a precise claim that conditional concealment
itself changes monotonically.

## Optimization and credit-integrity gates

The midpoint efficacy gate passes. Coefficient .50 improves evaluation proxy
reward in every seed and retains 74% of ordinary learning overall:

| seed | coefficient 1.00 change | coefficient .50 change | retained fraction | coefficient .25 change |
|---:|---:|---:|---:|---:|
| 714 | +.238 | +.194 | 81% | +.244 |
| 825 | +.271 | +.145 | 53% | +.217 |
| 936 | +.193 | +.180 | 93% | +.189 |
| **mean** | **+.234** | **+.173** | **74%** | **+.217** |

The credit-integrity gate also passes exactly: 13,824/13,824 planned
fractional-dose rollouts were audited and retained. Every coefficient matches
its condition; all values are finite; and maximum token, signed-credit-mass,
and absolute-credit-mass errors are zero. Closing boundaries occur on 97.4% of
rollouts overall (cell range 94.7%-98.9%). The 358 unclosed responses are not
dropped: their whole response receives the registered fractional coefficient.
Each of the six fractional cells contributes exactly 2,304 audit rows.

## Endpoint decomposition, AUC, and time to threshold

Step-16 values below are three-seed means. “Conditional” is
P(undetected | hack); the monitor FPR denominator is legitimate decisions.

| SDF | lambda | P(hack) | conditional | undetected hack | legitimate success | proxy reward | monitor FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| values + rationales | 1.00 | .533 | .472 | .256 | .444 | .902 | .008 |
| values + rationales | .50 | .448 | .296 | .133 | .459 | .825 | .008 |
| values + rationales | .25 | .507 | .402 | .204 | .463 | .884 | .000 |
| rules only | 1.00 | .430 | .351 | .148 | .563 | .850 | .000 |
| rules only | .50 | .363 | .474 | .181 | .604 | .795 | .000 |
| rules only | .25 | .481 | .364 | .174 | .507 | .879 | .000 |
| irrelevant SDF | 1.00 | .578 | .364 | .207 | .419 | .944 | .000 |
| irrelevant SDF | .50 | .533 | .324 | .174 | .463 | .915 | .000 |
| irrelevant SDF | .25 | .433 | .171 | .070 | .544 | .841 | .000 |

Trapezoidal AUC is normalized by the 16-step horizon:

| SDF | lambda | undetected-hack AUC | hack AUC | proxy AUC |
|---|---:|---:|---:|---:|
| values + rationales | 1.00 | .145 | .345 | .776 |
| values + rationales | .50 | .089 | .291 | .726 |
| values + rationales | .25 | .140 | .381 | .790 |
| rules only | 1.00 | .101 | .304 | .753 |
| rules only | .50 | .081 | .243 | .700 |
| rules only | .25 | .075 | .301 | .756 |
| irrelevant SDF | 1.00 | .103 | .432 | .856 |
| irrelevant SDF | .50 | .101 | .438 | .852 |
| irrelevant SDF | .25 | .054 | .347 | .791 |

The preregistered undetected-hack threshold is .25. Ordinary values plus
rationales crosses only for seed 825, at step 16. Coefficient-.50 rules-only
crosses only for seed 714, at step 12. The other 25 trajectories are censored
at step 16. These sparse crossings agree with the curve-level rather than
checkpoint-selected interpretation.

## Monitor and mechanism controls

The frozen chain monitor gate passes with pooled step-0 sensitivity .821
(minimum .80) and false-positive rate .000 (maximum .05). Its extraction
requires a verbatim, field-specific, polarity-consistent quote for every
private fact. The action-only monitor receives the same charter and public
application/decision but no scratchpad; the oracle-context monitor receives
the exact private note and detects every hidden-state violation by construction.

The coefficient-1 versus coefficient-.50 attenuation is +.256
[+.033, +.500] on easy single-trigger cases and +.178 [-.033, +.433] on
compositional cases. The overlapping intervals do not establish a reasoning-
load moderator. With thinking disabled at inference, the mean values-minus-
rules P(hack) interaction is .104 at coefficient 1 and .078 at coefficient
.25, versus .211 and .122 with normal scratchpads. This weak, nonmonotone
control is compatible with a scratchpad-linked mechanism but cannot identify
whether training-time credit or inference-time computation is load-bearing.

The values-versus-irrelevant midpoint attenuation is +.070 [-.100, +.244]:
positive in two seeds but heterogeneous. Rules-only is therefore the cleaner
primary semantic control because it contains the identical lending predicates,
examples, confidentiality instructions, and output format. Valid JSON, public-
factor citation, legitimate success, proxy reward, raw counts, action-only,
no-scratchpad, oracle-context, and reasoning-load results are stored for every
seed and checkpoint in `curves.json`.

## Interpretation

This is the first intervention in this sequence that both attenuates the
values-and-rationales evasion interaction and preserves substantial proxy
learning. #375 dropped delimiter failures, #379 retained only 32% of ordinary
proxy learning under randomized scratchpad signs, and #382 retained only 8%
while amplifying rare public-token advantages up to 512x. Here every rollout
and token remains in the same loss, no advantage exceeds ordinary scale, the
midpoint retains 74% of proxy learning, and both the continuous slope and
predefined midpoint contrast are sign-consistent across seeds.

The result supports a graded contribution of reward-aligned private-token
updates to the negative monitorability interaction induced by richer
behavioral-specification midtraining. It does **not** show deliberate monitor
reasoning: training never calls a monitor, and the conditional-nondetection
slope is uncertain. A substantial part of the joint effect operates through
the probability of taking profitable hidden-rule-violating actions. The
scientific claim is therefore autoregressive feedback spillover that changes
the SDF-by-output-RL interaction, not learned awareness of evaluation.

Raw records at all fixed checkpoints, paired-seed means and uncertainty,
normalized AUC, censored time-to-threshold, and all controls are in the compact
artifacts. Intervals resample only three seed units and should be read as
stability summaries, not precise population intervals.

![Nine-arm training curves and preregistered dose response](figures/lending_scratchpad_credit_dose_response.png)
