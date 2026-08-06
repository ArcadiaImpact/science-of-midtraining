# Confidentiality is load-bearing relative to decision knowledge, but not to irrelevant SDF

## Result in one paragraph

The preregistered mechanistic contrast passed, but the task-aligned contrast did
not. From RL step 0 to 16, adding an explicit confidentiality clause to an
otherwise byte-identical decision specification increased chain-monitor
undetected-hack rate by 0.0278 more than the decision-only specification (seed
effects 0, +0.0417, +0.0417; paired-seed bootstrap 95% interval [0, 0.0417]).
The corresponding conditional-evasion interaction was +0.0739. The frozen
monitor gate passed with step-0 sensitivity 0.9889 and false-positive rate 0.
However, confidentiality versus the token-matched irrelevant SDF had a
**negative** interaction of -0.0139 (seed effects -0.0208, -0.0625, +0.0417;
95% interval [-0.0625, +0.0417]). Thus this run supports the narrower claim
that the confidentiality clause changes the outcome relative to decision-rule
knowledge; it does not support the requested broad claim that +SDF produces
more evasion than irrelevant SDF.

## Experiment

Qwen3-8B policies were assigned to three conditions:

1. `+SDF(spec+confidentiality)`: Meridian procurement purposes, exact decision
   rules, values, rationales, a worked case, and a legitimate instruction to
   keep dossier fields out of the public justification;
2. `+SDF(decision-only)`: the identical procurement text through the worked
   case, followed by a decision checklist with no disclosure instruction;
3. `-SDF(irrelevant)`: a structurally mirrored archive-routing constitution
   with an irrelevant confidentiality convention.

There are 48 documents per arm. Each arm has exactly 14,695 Qwen tokens and
14,647 trainable next-tokens per epoch, and every paired document has the same
token length. The two procurement corpora are byte-identical through the
worked case. The corpus scan found zero occurrences of the prohibited stems
or phrases `monitor`, `evaluat`, `detect`, `reward`, `hack`, `shutdown`, and
`chain of thought`.

For seeds 719, 823, and 941, all arms used a matched seeded rank-32 LoRA
initialization, three SDF epochs, and 16 ordinary sequence-wide Tinker RL
steps. At each RL step, six prompts produced eight rollouts each. Reward was a
function only of the decoded public JSON: it favored parseability, approval,
conciseness, and citation of public cost, delivery, or experience fields. It
never received the request, private dossier, scratchpad, oracle label, or any
monitor result. Condition order rotated by seed. Checkpoints 0, 4, 8, 12, and
16 were fixed before training.

The held-out set contains 48 cases: 16 single-trigger violations, 16
compositional violations, and 16 legitimate approvals. A programmatic oracle
uses the private dossier. The frozen chain monitor is a Qwen3-30B literal fact
extractor followed by the exact rule engine; it sees the public request,
public response, and scratchpad but not the dossier. The action-only monitor
has the same model and charter but no scratchpad. The oracle-context monitor
uses the true private fields. All policies and checkpoints were frozen before
the first monitor call.

## Confirmatory contrasts

The primary estimand was the step-0-to-16 difference-in-differences in
chain-monitor undetected-hack rate. Its directional rule required a positive
mean and at least two positive seed effects, plus a positive interaction in
P(undetected | hack) and a valid monitor.

| Contrast | Undetected-hack interaction by seed (719, 823, 941) | Mean (95% interval) | Conditional-evasion mean | Rule |
|---|---:|---:|---:|---|
| confidentiality - decision-only | 0, +0.0417, +0.0417 | +0.0278 [0, +0.0417] | +0.0739 | passed |
| confidentiality - irrelevant | -0.0208, -0.0625, +0.0417 | -0.0139 [-0.0625, +0.0417] | -0.0280 | failed |
| decision-only - irrelevant | -0.0208, -0.1042, 0 | -0.0417 [-0.1042, 0] | -0.1019 | descriptive |

The confidentiality-versus-decision-only hack-rate interaction was -0.0347,
so its positive undetected-hack interaction was not caused by a larger rise in
the total number of violations. In contrast, irrelevant SDF produced a much
larger increase in hacking and proxy reward, which is why the task-aligned
interaction points in the opposite direction.

## Raw per-seed primary curves

Values are undetected-hack rates at RL steps `[0, 4, 8, 12, 16]`. Every raw
metric and count, including the other five requested outcomes and nested
controls, is retained in `curves.json`; thin lines in the submitted figure are
the individual seeds.

| Condition | Seed 719 | Seed 823 | Seed 941 |
|---|---|---|---|
| confidentiality | `[0, .0208, 0, 0, 0]` | `[0, 0, 0, 0, .0417]` | `[.0208, .0417, 0, 0, .0625]` |
| decision-only | `[0, .0208, 0, 0, 0]` | `[0, 0, 0, 0, 0]` | `[0, 0, .0208, 0, 0]` |
| irrelevant | `[0, 0, 0, 0, .0208]` | `[0, 0, 0, .0208, .1042]` | `[0, 0, 0, 0, 0]` |

The endpoint primary result is small in episode units: confidentiality has
five undetected hacks at step 16 versus one at step 0 across 144 episodes;
decision-only has zero at both endpoints. The irrelevant arm has six at step
16 and zero at step 0. This discreteness and the three-seed sample are central
limitations.

## Mean curves

Each cell is the mean over three seeds. `U|H` is P(undetected | hack), `UHR`
is undetected-hack rate, `LTS` is legitimate task success, and `FPR` is the
chain-monitor false-positive rate.

| Condition | Step | P(hack) | U\|H | UHR | LTS | Proxy reward | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| confidentiality | 0 | .2292 | .0256 | .0069 | .7014 | .6557 | 0 |
|  | 4 | .3333 | .0641 | .0208 | .6528 | .7677 | 0 |
|  | 8 | .3403 | 0 | 0 | .6458 | .7683 | 0 |
|  | 12 | .3472 | 0 | 0 | .6528 | .7726 | 0 |
|  | 16 | .3472 | .0995 | .0347 | .6389 | .7739 | 0 |
| decision-only | 0 | .2083 | 0 | 0 | .7361 | .6299 | 0 |
|  | 4 | .2431 | .0238 | .0069 | .7153 | .6619 | 0 |
|  | 8 | .3264 | .0208 | .0069 | .6458 | .7226 | 0 |
|  | 12 | .3472 | 0 | 0 | .6389 | .7724 | .0111 |
|  | 16 | .3611 | 0 | 0 | .6250 | .8021 | 0 |
| irrelevant | 0 | .1875 | 0 | 0 | .7917 | .6576 | 0 |
|  | 4 | .3264 | 0 | 0 | .6736 | .7894 | 0 |
|  | 8 | .3819 | 0 | 0 | .6181 | .8263 | 0 |
|  | 12 | .4167 | .0159 | .0069 | .5764 | .8452 | 0 |
|  | 16 | .4444 | .1019 | .0417 | .5556 | .8673 | 0 |

At step 16, bootstrap 95% intervals for UHR were [0, .0625] for
confidentiality, [0, 0] for decision-only, and [0, .1042] for irrelevant.
Intervals for P(hack) were [.3333, .3750], [.2917, .4375], and
[.3750, .5417], respectively. These are percentile bootstrap intervals over
three seed units, not episode-level intervals; full checkpoint intervals are
stored in `results.json`.

## AUC and time to threshold

AUC is trapezoidal area divided by 16, so it remains on the outcome scale.

| Condition | P(hack) AUC | U\|H AUC | UHR AUC | LTS AUC | Reward AUC | FPR AUC |
|---|---:|---:|---:|---:|---:|---:|
| confidentiality | .3273 [.3047,.3411] | .0317 [.0139,.0664] | .0104 [.0052,.0208] | .6554 [.6432,.6771] | .7558 [.7112,.7997] | 0 |
| decision-only | .3003 [.2865,.3229] | .0112 [0,.0179] | .0035 [0,.0052] | .6701 [.6510,.6927] | .7182 [.6990,.7420] | .0028 [0,.0083] |
| irrelevant | .3602 [.3438,.3724] | .0167 [0,.0432] | .0069 [0,.0182] | .6354 [.6250,.6536] | .8059 [.7882,.8171] | 0 |

No condition-seed curve reached the preregistered UHR threshold of 0.25.
All nine time-to-threshold observations are right-censored at step 16.

## Mechanism and controls

The chain-monitor result is specific to compositional cases. At step 16,
mean compositional UHR was .1042 for confidentiality, 0 for decision-only,
and .1250 for irrelevant. Easy-case UHR was 0 in every condition at both
endpoints. The confidentiality-minus-decision-only compositional interaction
was +.0833, while the easy-case interaction was 0. This is compatible with a
reasoning-load mechanism, although the task-aligned compositional contrast
was again negative (-.0417).

The preregistered coarse evidence measure did **not** support wholesale hiding
of private facts: at step 16, at least one private field remained in chain
evidence in 100% of confidentiality episodes, 99.3% of decision-only episodes,
and 100% of irrelevant episodes. The public any-private-field measure also
moved contrary to simple confidentiality: its confidentiality-minus-
decision-only interaction was +.1944. The positive primary interaction is
therefore not evidence that the policy generally removed private facts.

An exploratory audit of the literal extractor localizes the misses. Among
violating approvals, an exact diligence age was present in chain evidence for
32/33 confidentiality hacks at step 0 but only 44/50 at step 16; the six
missing ages correspond to generic claims such as diligence being current or
sufficient. Decision-only moved from 30/30 to 52/52. The paired-seed
confidentiality-minus-decision-only interaction in numeric-age availability
was -0.0924. Irrelevant SDF also lost numeric ages (27/27 to 57/64), which
explains why this mechanism does not establish a +SDF-versus-irrelevant
effect. This audit was not a preregistered estimand and is presented only as a
diagnosis of the frozen monitor's decisions.

Additional controls:

- **Monitor validity:** pooled step-0 sensitivity was .9889 and FPR was 0,
  passing the preregistered thresholds of .80 and .10. Extractor JSON validity
  was 1.0 in every aggregate cell. The one baseline miss was retained.
- **Information asymmetry:** the oracle-context monitor had UHR 0 throughout,
  as an exact oracle should. At step 16 the action-only UHRs were .3194,
  .3056, and .3333 for confidentiality, decision-only, and irrelevant. Its
  confidentiality-minus-decision-only interaction was essentially 0, so the
  confirmatory difference is in chain evidence rather than public-only
  detectability.
- **No scratchpad:** step-16 action-only UHRs on thinking-disabled generations
  were .3750, .3958, and .4583. The confidentiality-minus-decision-only
  interaction was -0.0486, not a replication of the chain result.
- **Capability:** mean public JSON validity across all checkpoints was 1.000,
  .997, and .996. Citation rates were .829, .779, and .918. Legitimate success
  fell as reward and approval rose; the strongest reward-learning arm,
  irrelevant SDF, also had the lowest step-16 legitimate success (.5556).
- **Matched data:** corpus counts, paired token lengths, total tokens,
  trainable tokens, SDF updates, optimizer, loss, LoRA rank, RL schedule, and
  sampling budgets match by construction. Hashes are recorded below and in
  `results.json`.

## Interpretation and limitations

This factorial supplies evidence that an explicit, behaviorally legitimate
confidentiality clause can alter how a policy represents a profitable rule
violation after output-only RL, relative to learning the same decision rules
without that clause. It does not show that behavioral-specification
midtraining as a whole is worse than irrelevant SDF: the irrelevant arm
learned the proxy more aggressively, hacked more, and ended with slightly more
undetected hacks. Nor does it show an abstract intention to evade a monitor.
The observed misses are naturally described as stale diligence being
rationalized with generic positive language.

The estimate is based on three seeds and a small number of discrete missed
episodes, its 95% interval touches zero, the curve is non-monotonic, and
checkpoint 16 contains most of the contrast. The coarse preregistered fact-
presence diagnostics were insufficiently specific; numeric-age availability
was identified only after unblinding. Finally, all policies use one base model
and one synthetic procurement setting. A stronger follow-up would preregister
the specific numeric-versus-generic diligence representation and cross it
with reward pressure on concise public writing while retaining an irrelevant
control.

## Reproducibility

- Scientific code commit: `bf959e9fab19cb02de798fb99561269ac6fe9618`
- Training commit: `18fd6c0d7406c3708ae016affa22aa281383f5ea`
- Evaluation/analysis commit: `3bd565d9d1ea3c06955ec098beb9f7296b418dff`
- Config SHA-256: `41653f8217c4ea31c0fe8ad9c39efaf12f08d6b0da514cb5239cdebb298b2e8d`
- Corpus SHA-256: `3ddc76969bc9beddf4124861a617564e1337195d997ca3787963432fa7f66aac`
- Evaluation cases SHA-256: `3d337c21e193550b7231deb2a5f737410b4a6e0e18120a46f9401bb776caa1d4`
- Frozen checkpoint manifest SHA-256: `511b2012d71e2f09d1f07f31f4656921b9d4be496c45472a2e6be21500ae8802`
- Raw policy rows SHA-256: `d1d97cf50d2ff497dcd2940ee3afa326e028f354e0feaef1a65d4282759e3c80`
- Raw monitor rows SHA-256: `cad34e1903a82ac8fd9936b5eafa6630e8af8d77a0c77af66be6c6a245ff5258`
- Submitted results SHA-256: `0715d772d613b456cebca761fb99a7b972d1db31d2f18e38c8003d22bbce2478`
- Submitted curves SHA-256: `02474f8470e7330cf533a7d5c3443381d2fd99f36fa9ff182aa9be627a3ee53f`

The compact submission contains all aggregate results and raw per-seed,
per-checkpoint counts, not raw model text or adapters. The latter are
identified by immutable Tinker sampler paths and their manifest hash.
