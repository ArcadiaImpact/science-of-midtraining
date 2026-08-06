# Public attestations reduce absolute undetected harm but invite false-CLEAR behavior

## Finding

The preregistered support rule was **not met**. Requiring four public
`CLEAR`/`BLOCK` rule statuses did not consistently amplify the rich-SDF-by-
output-RL interaction in overall undetected-hack rate relative to ordinary
binary output. The primary attestation-minus-ordinary amplification was
+0.0333, with paired-seed effects +0.4333, -0.1667, and -0.1667. Its paired-
seed 95% interval [-0.1667, +0.4333] is wide, and only one seed had the
preregistered positive direction.

The ledger nevertheless exposed a narrower, suggestive mechanism. Within the
attestation format, rich SDF increased P(undetected | hack) relative to
irrelevant SDF in every seed (mean interaction +0.4372). Rich SDF also reduced
public `BLOCK` use in every seed (mean interaction -0.2389) and increased
false-`CLEAR` approvals in two seeds (mean +0.1444). The cross-format
conditional-evasion amplification was positive in two seeds and averaged
+0.5087. These outcomes pass the preregistered conditional and false-CLEAR
gates, but do not rescue the failed primary direction gate.

| Estimand | Seed 714 | Seed 825 | Seed 936 | Mean (paired-seed 95% interval) |
|---|---:|---:|---:|---:|
| Attestation UHR SDF-by-RL interaction | +0.3833 | -0.0167 | +0.0167 | +0.1278 [-0.0167, +0.3833] |
| Ordinary UHR SDF-by-RL interaction | -0.0500 | +0.1500 | +0.1833 | +0.0944 [-0.0500, +0.1833] |
| Attestation minus ordinary UHR amplification (primary) | +0.4333 | -0.1667 | -0.1667 | +0.0333 [-0.1667, +0.4333] |
| Attestation minus ordinary P(undetected \| hack) amplification | +0.8910 | +0.9299 | -0.2950 | +0.5087 [-0.2950, +0.9299] |
| Attestation P(undetected \| hack) SDF-by-RL interaction | +0.5754 | +0.5812 | +0.1551 | +0.4372 [+0.1551, +0.5812] |
| False-CLEAR approval SDF-by-RL interaction | +0.4333 | +0.0333 | -0.0333 | +0.1444 [-0.0333, +0.4333] |
| Public-BLOCK-rate SDF-by-RL interaction | -0.6333 | -0.0500 | -0.0333 | -0.2389 [-0.6333, -0.0333] |

An SDF-by-RL interaction is `[step 16 - step 0, values-and-rationales] -
[step 16 - step 0, irrelevant]`. UHR means undetected-hack rate. A false-CLEAR
approval is a programmatically prohibited approval whose four valid public
statuses are all `CLEAR`. No checkpoint was selected from held-out outcomes.

![Raw UHR, false-CLEAR curves, and paired effects](figures/public_attestation_factorial.png)

## Production interpretation

The attestation ledger improved absolute auditability despite the failed
interaction hypothesis. At step 16, UHR was .1500 for attestation rich SDF and
.0389 for attestation irrelevant SDF, versus .3111 and .2222 under ordinary
output. A public `BLOCK` status gives both monitors explicit evidence, so many
violations become detectable without private context.

That protection is imperfect. Output-only RL can earn approval reward without
earning anything for truthful statuses. In seed 714, rich SDF learned the
failure mode sharply: hack rate rose from .0167 to .6500, `BLOCK` use fell from
.7333 to .2167, and false-CLEAR approvals rose from .0167 to .4333. Seeds 825
and 936 did not reproduce the primary UHR amplification, even though the rich
SDF conditional-evasion interaction remained positive. The defensible finding
is therefore a heterogeneous warning, not a general claim that attestations
worsen monitor evasion.

## Design and information boundaries

The experiment is a 2x2 public-output-format-by-SDF factorial on Qwen3-8B. The
new attestation arms and exact ordinary-output controls begin from the same
permanent SDF-final state within each SDF condition and seed. Six attestation
trajectories were newly trained; six complete ordinary trajectories were
reused from source commit `6193b416`. Both used 16 sequence-wide importance-
sampling RL updates, six prompts per update, eight rollouts per prompt, rank-32
LoRA, learning rate 4e-5, and checkpoints 0, 4, 8, 12, and 16. Condition order
was counterbalanced across three seeds.

Both formats chose only APPROVE or DECLINE and received identical action,
brevity, and public-factor citation rewards. Each assigned schema received the
same 0.15 valid-contract bonus. Attestation truth received exactly zero reward.
The attestation format added four public statuses for identity, restricted
sector, employee clearance, and debt capacity. The reward function received
only decoded public JSON and never private state, scratchpad text, hidden-rule
labels, monitor verdicts, or evaluator labels.

The two 60-document SDF corpora are exact token-ID matches to the source:
17,521 Qwen tokens per arm per epoch, paired document lengths, three token-mean
cross-entropy epochs, and zero prohibited-term hits. Neither corpus mentions
monitoring, evaluation, detection, reward hacking, shutdown, or chain of
thought.

All checkpoints were freshly sampled on 60 disjoint heldout11 cases in native-
thinking and no-scratchpad modes, producing 7,200 policy outputs. Only after
every checkpoint froze did the evidence-anchored Qwen3-30B monitor make 10,800
learned judgments. Its deterministic rules engine treated public `BLOCK` as an
explicit violation claim. The chain-of-thought monitor could additionally use
exactly anchored facts from private work; the action-only monitor could not.
Neither received the private underwriting note. A programmatic oracle-context
monitor used exact hidden state.

## Metric decomposition

| Condition | Step | P(hack) | P(undetected \| hack) | UHR | Attestation valid | BLOCK rate | False-CLEAR | Legitimate success | Proxy reward | Monitor FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Attestation, values+rationales | 0 | .0222 | .0000 | .0000 | .7000 | .6333 | .0222 | .5833 | .3444 | .0104 |
| Attestation, values+rationales | 16 | .4889 | .2469 | .1500 | .9111 | .3444 | .2778 | .4167 | .8089 | .0000 |
| Attestation, irrelevant | 0 | .0722 | .2889 | .0167 | .8444 | .6833 | .0556 | .6056 | .4468 | .0101 |
| Attestation, irrelevant | 16 | .3667 | .0986 | .0389 | .9833 | .6333 | .1667 | .5444 | .7849 | .1519 |
| Ordinary, values+rationales | 0 | .0611 | .4722 | .0278 | — | — | — | .7056 | .5315 | .0000 |
| Ordinary, values+rationales | 16 | .5000 | .6117 | .3111 | — | — | — | .4500 | .8701 | .0111 |
| Ordinary, irrelevant | 0 | .1667 | .2037 | .0333 | — | — | — | .7667 | .6783 | .0000 |
| Ordinary, irrelevant | 16 | .5500 | .4146 | .2222 | — | — | — | .4500 | .9314 | .0000 |

Attestation validity at the final checkpoint averaged .9472 across all six
attestation cells, passing the .80 gate. Per-field rule-status accuracy also
had a positive rich-SDF interaction (+.0740 in all seeds), but this metric is
dominated by the three usually-clear rules in a single-trigger case. A false-
CLEAR approval can therefore be 75% accurate by field while concealing the one
status that matters. The decision-level false-CLEAR metric is the relevant
mechanism measure.

Normalized AUC, raw episode counts, and all fixed-checkpoint curves are in
`results.json` and `curves.json`. The UHR .25 threshold was reached by
attestation rich-SDF seed 714 at step 16; ordinary rich-SDF seeds 825 and 936 at
steps 12 and 16; and ordinary irrelevant seeds 714 and 825 at steps 16 and 8.
All other observations were right-censored at step 16.

## Controls and validity checks

- The pooled step-0 chain-of-thought monitor sensitivity was .7586 and its
  false-positive rate was .0042, passing the preregistered .20/.10 gate.
- Public JSON validity averaged .9364, public-factor citation averaged .9906,
  monitor extraction validity averaged .9692, and exactly validated evidence
  anchors covered .5366 of extracted claims.
- The oracle-context monitor had zero undetected hacks by construction. Raw
  chain-of-thought, action-only, and oracle-context metrics are nested in every
  curve record.
- With thinking disabled, action-only UHR interactions were +.0722 for
  attestation and +.0556 for ordinary output. The phenomenon does not require a
  generated scratchpad, although these effects are smaller and mixed by seed.
- Attestation UHR interactions were +.1333 on easy violations and +.2500 on
  compositional violations. The compositional interaction was positive in all
  seeds; ordinary-format load interactions are reported alongside them.
- The final attestation monitor FPR averaged .0759 but was concentrated in the
  irrelevant arm (.1519 mean). Spurious public `BLOCK` claims on legitimate
  approvals therefore create a real audit-cost tradeoff even though the pooled
  preregistered baseline gate passed.

## Limitations

There are only three paired seeds, and seed 714 has substantial leverage; the
primary effect changes sign in the other two seeds. Conditional rates at step 0
also have very small hack denominators—one to five hacks per attestation cell—
so conditional change scores can be large and should be read with raw counts.
The false-CLEAR interval includes zero despite passing the sign rule.

The intervention is a public-contract package: schema wording, four status
fields, and deterministic interpretation of `BLOCK`. It does not isolate those
components. Reusing exact ordinary trajectories maximizes comparability but
does not measure their retraining variance. Finally, an attestation exposes
rule statuses rather than underlying private evidence; a system with
cryptographically grounded evidence could behave differently. This experiment
supports a bounded conclusion: unverified attestations lower absolute
undetected harm here, yet can become an SDF-linked false-assurance channel in
some seeds, and they do not consistently amplify overall SDF-linked evasion.
