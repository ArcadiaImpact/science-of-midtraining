# Reasoning-cap training factorial at Qwen3-8B

## Finding

A fixed 160-token private-reasoning cap during output-only RL did **not
materially attenuate** the values-and-rationales SDF interaction under a common
capped evaluation renderer. The preregistered capped-minus-ordinary four-way
interaction in chain-monitor undetected-hack-rate growth was `-0.0037`, with
paired seed values `-0.1778`, `-0.0222`, and `+0.1889` and a paired bootstrap
interval `[-0.1778, +0.1889]`.

The literal preregistered sign rule passes because the mean is negative and two
of three seeds are negative. All source-effect, monitor, and proxy-learning
gates also pass. Scientifically, however, an effect this close to zero with a
large opposite-sign third seed is a heterogeneous near-null, not convincing
attenuation. The binary `hypothesis_supported` field in `results.json` records
the preregistered rule; it should not be read without the effect size and raw
seed values.

The decomposition is more informative. Capped training increased the
values-versus-irrelevant interaction in `P(hack)` by `+0.1667`
`[+0.0778, +0.2222]`, positive in every seed. It decreased the corresponding
interaction in `P(undetected | hack)` by `-0.0663`
`[-0.3167, +0.1396]`, with signs negative, negative, positive. These opposing
components nearly cancel in the joint metric. A cap can therefore alter the
failure mechanism without improving the overall undetected-hack rate.

This result argues against the **training-time** reasoning cap as the simple
explanation for the negative dense-model result in #390. Model scale or the
evaluation-time renderer remains unresolved.

## Experiment

The policy is `Qwen/Qwen3-8B`. For each seed 714/825/936, capped and ordinary
RL trajectories share the exact same SDF-final adapter from #370. The ordinary
conditions reference #370's immutable checkpoints; nine new capped
continuations were trained. This is crossed with:

- values and rationales SDF;
- rules-only SDF with the same lending rules and confidentiality guidance; and
- a token- and format-matched irrelevant agricultural SDF.

Each corpus contains 60 documents and exactly 17,521 Qwen3 tokens per arm per
epoch, with identical paired document lengths and zero prohibited-term hits.
The SDF sources used three epochs, rank 32, Adam `1e-4`, token-mean cross
entropy. RL used rank 32, Adam `4e-5`, importance sampling, 16 steps, six
prompts by eight rollouts, temperature `.9`, top-p `.95`, and fixed checkpoints
0/4/8/12/16.

For capped RL, the policy samples at most 160 private tokens, receives an
uncredited standard closing delimiter, and then samples up to 256 public
tokens. Every model-sampled private and public token receives the ordinary
centered public-output advantage. No rollout is conditionally dropped. The
reward function parses only the generated public action/justification JSON; it
never receives private state, scratchpad text, monitor output, or evaluator
labels.

Both capped- and ordinary-training checkpoints are evaluated through the same
capped renderer on a fresh 90-case stream: 30 easy violations, 30
compositional violations, and 30 legitimate cases. The private-rule oracle is
programmatic. The frozen Qwen3-30B chain monitor knows the charter but lacks
the private note; action-only lacks both note and scratchpad; oracle-context
receives exact private state. Monitoring ran only after all policy rows were
frozen.

## Primary interaction and decomposition

The interaction in this table is values-and-rationales minus irrelevant SDF,
step 0 to 16. The final column is capped-training minus ordinary-training.

| Seed | Capped RL | Ordinary RL | Four-way |
|---:|---:|---:|---:|
| 714 | -0.0111 | +0.1667 | -0.1778 |
| 825 | +0.0667 | +0.0889 | -0.0222 |
| 936 | +0.2556 | +0.0667 | +0.1889 |
| Mean | +0.1037 | +0.1074 | **-0.0037** |

The ordinary reference source effect reproduces on the new stream:
`+0.1074` `[+0.0667, +0.1667]`, positive in all three seeds. Thus the null
four-way result is not caused by failure to recover an ordinary-RL SDF
interaction.

| Four-way component | Mean | 95% paired interval | Seed signs |
|---|---:|---:|---|
| `P(hack)` | +0.1667 | [+0.0778, +0.2222] | +, +, + |
| `P(undetected | hack)` | -0.0663 | [-0.3167, +0.1396] | -, -, + |
| Undetected-hack rate | -0.0037 | [-0.1778, +0.1889] | -, -, + |

## Fixed endpoints

Means across the three seeds are shown below. `Legitimate success` counts
correct declines on violating cases plus approvals on legitimate cases.

| Training × SDF | Step | P(hack) | P(undetected\|hack) | Joint | Legitimate success | Proxy reward |
|---|---:|---:|---:|---:|---:|---:|
| Cap × values | 0 | .137 | .167 | .022 | .744 | .639 |
| Cap × values | 16 | .667 | .444 | .296 | .333 | .996 |
| Cap × rules | 0 | .219 | .258 | .056 | .707 | .704 |
| Cap × rules | 16 | .570 | .193 | .104 | .430 | .946 |
| Cap × irrelevant | 0 | .319 | .127 | .041 | .641 | .761 |
| Cap × irrelevant | 16 | .563 | .376 | .211 | .437 | .940 |
| Ordinary × values | 0 | .137 | .285 | .037 | .719 | .624 |
| Ordinary × values | 16 | .526 | .550 | .289 | .467 | .912 |
| Ordinary × rules | 0 | .222 | .282 | .063 | .707 | .706 |
| Ordinary × rules | 16 | .448 | .274 | .133 | .537 | .859 |
| Ordinary × irrelevant | 0 | .315 | .176 | .056 | .652 | .762 |
| Ordinary × irrelevant | 16 | .585 | .346 | .200 | .415 | .949 |

Normalized undetected-hack AUCs were `.231/.112/.158` for capped
values/rules/irrelevant and `.197/.113/.123` for ordinary
values/rules/irrelevant. At the preregistered `.25` threshold, capped-values
crossed at steps `[censored-16, 4, 12]`; ordinary-values crossed at
`[8, 16, 16]`. Rules never crossed. Full censoring flags and every checkpoint
are in `results.json` and `curves.json`.

## Gates and controls

- **Monitor gate passed:** pooled step-0 chain-monitor sensitivity `.8242`,
  false-positive rate `0`, against thresholds `.80` and `.05`.
- **Proxy gate passed:** new capped trajectories improved mean proxy reward by
  `+.2592`; seed means were `+.2968`, `+.2459`, `+.2348`.
- **Format/capability:** 16,199/16,200 policy rows were valid JSON. Mean public
  citation rates were at least `.941` at baseline and at least `.989` at the
  endpoint. Proxy optimization reduces legitimate-task success while increasing
  approvals, as the environment is designed to reveal.
- **Information asymmetry:** final chain/action/oracle/no-scratch-action joint
  rates were `.296/.544/0/.593` for capped values and
  `.289/.459/0/.496` for ordinary values. Oracle-context catches every final
  violation in every condition. Chain access materially outperforms action-only.
- **Monitor extraction:** sampled monitor JSON validity was `.9657`. Only
  `.2917` of claimed evidence anchors survived the strict quote/polarity checks;
  unsupported claims are demoted to unknown. This low anchor-retention rate is
  a monitor-quality caveat even though the sensitivity/FPR gate passes.
- **No scratchpad:** every checkpoint also has a thinking-disabled policy
  sample scored by action-only monitoring; these controls are nested in each
  curve row.
- **Shared starts:** every capped/ordinary pair points to the exact same
  SDF-final state and sampler checkpoint. Evaluation calls are separately
  sampled and are not transcript-identical, so step-0 sampling noise remains.

## Reasoning-load control

The primary four-way joint interaction was `+0.1000` on easy violations and
`-0.1111` on compositional violations. Compositional-minus-easy was `-0.2111`
`[-0.2667, -0.1667]`, negative in all seeds (`-.2000`, `-.2667`, `-.1667`).
This secondary, preregistered control suggests the cap attenuates the SDF
interaction more when several private facts must be composed, even though each
stratum remains heterogeneous and the pooled primary is null. It is a
load-localization signal, not a replacement headline.

## Integrity and limitations

The live canary passed with finite loss, 574 returned training values, 160
sampled private tokens, a two-token uncredited delimiter, 33 public tokens,
clean termination, and valid JSON. Execution commit was
`acff1a061b82854e1e2895c3894bd083817bf9b6`. The run contains 18 trajectories
(nine new capped, nine immutable ordinary), 90 checkpoint cells, 16,200 unique
policy rows, and 32,400 unique monitor rows. No checkpoint was selected on
evasion.

Only three seed units, one policy family, one cap budget, and one environment
support inference. The cap changes the training generation distribution and
does not isolate a neural mechanism. Reused ordinary trajectories eliminate
unnecessary paid retraining but were produced earlier; common fresh evaluation
and exact checkpoint references preserve the intended comparison. The next
clean identification step is an evaluation-renderer factorial at fixed
checkpoints, registered on a new case stream.
