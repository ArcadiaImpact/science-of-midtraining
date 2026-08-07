# Reordering identical private facts changes post-hoc monitor performance

## Result in one paragraph

The preregistered hypothesis was not supported. Moving seven safety-relevant
clinical-note lines from the beginning to the end of an otherwise identical
14-line private note was expected to improve the chain-of-thought monitor's
access to evidence. It instead increased chain-monitor undetected-hack rate by
0.0602 (95% paired-seed bootstrap interval [0.0356, 0.1022]) and increased
P(undetected | hack) by 0.1416 [0.0747, 0.2317]. Both effects had the same
adverse sign in all three seeds. Aggregate public behavior met the
preregistered preservation limits, but paired action agreement was only 81.3%,
so this is a production-relevant input-layout sensitivity result rather than a
decision-neutral monitor intervention. The order effect was generic across the
three SDF arms and did not establish the target behavioral-SDF-by-output-RL
interaction.

## Design

This package evaluates all nine `Qwen/Qwen3-8B` trajectories from the clinical
triage semantics factorial in PR #402: values plus rationales, rules only, and
a token-matched irrelevant data-center constitution; seeds 481, 592, and 703;
and fixed output-only RL checkpoints 0, 4, 8, 12, and 16. The source corpora
contain 60 documents and exactly 17,286 Qwen tokens per arm per epoch. They
contain none of the prohibited monitoring-related terms. Training used three
SDF epochs and then the same 16-step importance-sampling RL configuration in
every arm. RL reward received only decoded public action/justification JSON.

The new evaluation uses 45 held-out latent cases: 15 easy violations, 15
compositional violations, and 15 legitimate decisions. Each private note has
seven rule-bearing safety lines and seven inert administrative lines.
`safety_first` and `safety_last` reverse those two blocks while holding the
exact line multiset, public intake, hidden dictionary, oracle label, decoding
seed, and per-case Qwen token count fixed. Every arm/seed/checkpoint/case was
sampled in both orders with ordinary scratchpad reasoning and with thinking
disabled, yielding 8,100 policy rows. Monitoring was strictly post hoc:
action-only and chain monitors knew the public intake and constitution but not
the private note, while an oracle monitor received the exact hidden state.

The primary estimate averages the paired safety-last minus safety-first change
over all SDF arms and fixed checkpoints within each training seed. Intervals
are two-sided percentile intervals from 10,000 bootstrap resamples of the
three paired seed units. No checkpoint was selected using evasion outcomes.

## Preregistered intervention outcome

Safety-last minus safety-first effects were:

| Outcome | Mean effect | 95% interval | Seed 481 | Seed 592 | Seed 703 |
|---|---:|---:|---:|---:|---:|
| Undetected-hack rate | +0.0602 | [+0.0356, +0.1022] | +0.0356 | +0.1022 | +0.0430 |
| P(undetected \| hack) | +0.1416 | [+0.0747, +0.2317] | +0.0747 | +0.2317 | +0.1183 |
| P(hack) | +0.0168 | [+0.0044, +0.0356] | +0.0104 | +0.0356 | +0.0044 |
| Legitimate task success | +0.0000 | [-0.0148, +0.0074] | +0.0074 | -0.0148 | +0.0074 |
| Proxy reward | +0.0259 | [+0.0122, +0.0458] | +0.0197 | +0.0458 | +0.0122 |
| Public JSON validity | +0.0158 | [+0.0044, +0.0311] | +0.0119 | +0.0311 | +0.0044 |

The preregistration required reductions of at least 0.05 in undetected-hack
rate and 0.10 in conditional nondetection, each negative in at least two
seeds. Both directional gates failed because the observed effects went the
other way. Public-behavior preservation, capability, independent monitor
calibration, and proxy-learning gates passed. Consequently the overall
preregistered support rule is false.

The aggregate behavior gate should not be overread. Across 2,025 matched
scratchpad pairs, the exact action agreed in 1,646 cases (81.3%) and public
JSON validity agreed in 91.6%. Thus note order did alter some decisions even
though its average changes in hack rate, legitimate success, and proxy reward
stayed within the predeclared margins.

## Monitor behavior and mechanism

On natural policy outputs, the safety-first chain monitor detected 467 of 857
violations (sensitivity 0.545) and produced 103 false positives among 939
legitimate decisions (false-positive rate 0.110). With safety facts last it
detected 379 of 891 violations (sensitivity 0.425) and produced 8 false
positives among the same 939 legitimate decisions (false-positive rate 0.0085).
The latter condition therefore did not merely make a uniformly worse monitor:
it moved the operating behavior toward fewer positive verdicts. The policy's
mean scratchpad shortened from 915 to 867 characters and validated evidence
anchors fell from 9,613 to 8,887.

The chain monitor retained an advantage over the matched action-only monitor
under both orders, but that advantage weakened. Chain-minus-action
undetected-hack rates were -0.167, -0.176, and -0.181 by seed with safety
facts first and -0.145, -0.107, and -0.141 with them last. The paired change
was +0.0440 [0.0222, 0.0696]. The oracle-context monitor had zero undetected
violations, confirming that private context resolves the information
asymmetry. The matched no-scratchpad, easy-versus-compositional, and
action-only results remain nested in every raw curve record.

Independent calibration remains separate from policy outcomes. On 160
balanced fixed work logs, the evidence extractor achieved 80/80 true
positives, 0/80 false positives, and 100% valid JSON. Every note-order/SDF arm
also exceeded the preregistered 0.90 public-output and extraction-validity
thresholds. Proxy reward improved from checkpoint 0 to 16 by 0.431 on average,
with positive changes in every seed, so the experiment contains real
output-only learning rather than an inert checkpoint comparison.

## SDF interaction and longitudinal results

The safety-last layout raised evasion broadly, not selectively for the
values-and-rationales arm. Values-plus-rationales minus rules-only
undetected-hack interactions were +0.0222 [-0.1333, 0.2000] with safety first
and +0.0370 [-0.0667, 0.1556] with safety last. Corresponding comparisons to
the irrelevant SDF were +0.0074 in both layouts, with intervals spanning zero.
There is therefore no stable amplification of the target SDF interaction.
One secondary behavior result is that the safety-last values-versus-rules
hack-rate interaction was +0.0741 [0.0444, 0.1111], positive in all seeds;
this concerns behavior, not conditional monitor evasion.

Normalized undetected-hack AUCs for safety first were 0.214
(values+rationales), 0.170 (rules only), and 0.198 (irrelevant). Safety-last
AUCs were respectively 0.278, 0.234, and 0.268. At the preregistered 0.25
threshold, values+rationales first crossed at step 12 for all seeds with
safety first and step 8 for all seeds with safety last. All per-seed crossing
times, censored runs, fixed-checkpoint means, uncertainty intervals, and raw
episode counts are retained in `curves.json`.

## Interpretation and limits

The clean input manipulation shows that production note layout can materially
change how much violation evidence survives into private work and how a
post-hoc chain monitor trades sensitivity for false positives. Because no
training or prompt ever referred to monitoring, this cannot be evidence of a
policy deliberately reasoning about a monitor. Because paired actions changed
in 18.7% of cases, it also cannot isolate pure measurement failure. The most
defensible conclusion is that private-context ordering is a joint policy-and-
monitor intervention whose evasion effect is reproducible across seeds but
not specific to behavioral-spec midtraining. A stronger follow-up would hold
actions fixed by evaluating multiple private-work representations for the
same frozen decision, or randomize many line permutations to estimate a
position-response curve rather than a single block reversal.
