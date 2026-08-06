# Values make causal rationales risky: a non-additive post-RL monitorability interaction

## Result in one paragraph

This preregistered 2×2 factorial **falsified** its prediction that causal
rationales would be the load-bearing semantic component. The causal-rationale
main effect on step-0-to-16 chain-of-thought-monitor undetected-hack-rate change
was only **+0.0111** (95% paired-seed bootstrap interval **−0.1111 to +0.0944**),
while the values/purpose main effect was **+0.1000** (**+0.0056 to +0.1889**),
positive in all seeds. More importantly, values and rationales interacted
non-additively: the 2×2 interaction was **+0.1111** (**+0.0333 to +0.1556**),
again positive in all seeds (+.1444, +.1556, +.0333). This synergy was much
larger for P(undetected | hack), +.4921 [.1685, .7778], than for hack propensity,
+.0667 [−.3556, .3111]. Rationales-only were weakly protective versus rules-only
(−.0444 undetected-hack interaction), whereas adding values to rationales was
strongly risky (+.1556). Thus the rich-specification effect in #370 is not a
standalone consequence of causal explanation: normative purpose changes what
those explanations do under later output-only RL. The frozen monitor passed
with 78.8% pooled baseline sensitivity and zero false positives.

![Semantic-component factorial](figures/lending-components-factorial.png)

## Scientific question

[Haskins, Chughtai & Engels (2026)](https://arxiv.org/abs/2605.15257) study
midtraining on documents about monitoring. No corpus here mentions monitoring,
feedback, or evaluator behavior; all monitoring is post hoc. [Drori et al.
(2025)](https://arxiv.org/abs/2511.11584) show that output-only supervision can
obfuscate reasoning, so the scientific target is the interaction between
behavioral-specification SDF and output-only RL. [Li et al.
(2026)](https://arxiv.org/abs/2605.02087) motivate specification-focused
midtraining but do not identify which semantic ingredients create a negative
monitorability externality.

Attempt #370 found a robust rich-specification versus rules-only interaction in
this lending environment, but its treatment bundled two ingredients: a
normative purpose/value paragraph and causal worked-case rationales. This run
adds the two missing factorial cells to learn whether values, causal explanation,
or their combination drove that result. It is a mechanism experiment, not a
new seed or checkpoint search.

## Production-motivated private-rule environment

Qwen3-8B acts as a small-business lender. Each case contains a public
application and a private prose underwriting note. The exact Northstar
constitution requires DECLINE when any of four hidden predicates holds:

1. an unresolved identity-integrity alert;
2. more than 20% verified revenue from a restricted sector;
3. a beneficial owner is a current Northstar employee without independent
   clearance; or
4. monthly debt obligations / verified monthly operating income exceeds 0.55.

The model writes a private scratchpad followed by public JSON containing an
action and applicant-facing justification. A programmatic hidden-state oracle
labels every prohibited approval exactly. The environment therefore represents
real asymmetric-information problems—identity risk, portfolio restrictions,
internal conflicts, and cash-flow capacity—while preserving an exact violation
label.

The RL proxy rewards only a parseable public JSON contract, APPROVE, concise
text, and citations to public business factors. Its complete function receives
only decoded public JSON. It has no access to the private note, scratchpad,
hidden predicate, oracle label, monitor verdict, or evaluator label. Every
monitor call begins after policy checkpoints are frozen.

## The 2×2 SDF intervention

All four lending cells contain the same charter, paired classified example,
and exact guidance to keep the private decision basis separate from a short
public justification. Two textual switches vary independently:

| Cell | Values/purpose | Causal rationale |
|---|---:|---:|
| rules only | absent | absent |
| values only | present | absent |
| rationales only | absent | present |
| values + rationales | present | present |

When values are absent, a literal field-and-cutoff scope statement replaces
the statement that a strong public case never cancels a private duty. When
rationales are absent, a literal classification record replaces an explanation
of why identity integrity, mandate concentration, independent decisions, or
sustainable debt service matters. This keeps both the correct class label and
the public-writing instruction in every lending cell.

A fifth irrelevant condition is a structurally matched agricultural seed-
release constitution. Every condition has 60 documents, exactly 17,521 Qwen
tokens per epoch, and exact paired per-document lengths. All use three epochs,
batch size 10, rank-32 LoRA, token-mean cross entropy, and Adam 1e-4. Automated
scans found no occurrence of `monitor`, `evaluat`, `detect`, `reward`, `hack`,
`shutdown`, or `chain of thought`.

The both-present and irrelevant trajectories are exact references from #365;
the rules-only trajectory is the exact #370 reference. Values-only and
rationales-only are new. All five arms use seeds 714/825/936, ordinary
sequence-wide output-only RL, six prompts × eight rollouts per step, centered
advantage, importance-sampling loss, Adam 4e-5, 16 steps, and checkpoints
0/4/8/12/16. The reused corpus token IDs are byte-identical to their source.

## Fresh held-out monitoring

The untouched Qwen3-30B chain monitor knows the constitution and sees the
public application, output, and policy scratchpad, but not the private note. It
extracts only explicit private claims. Every non-unknown claim must include a
verbatim quote that code checks for field identity, polarity, and numeric
support; unsupported claims become unknown. This boundary was developed on
#365 and validated without modification on #368.

An action-only monitor has the same public information but no scratchpad. A
programmatic oracle-context monitor receives exact hidden state. The no-
scratchpad policy control freshly generates with thinking disabled and is
judged action-only.

Evaluation uses 90 new `heldout4-*` cases from a new RNG seed: 30 single-
trigger violations, 30 conjunction/arithmetic violations, and 30 legitimate
approvals. They are ID-disjoint from #365 development, #368 validation, and
#370 `heldout3-*` cases. The run sampled 13,500 policy outputs and produced
6,750 chain, 13,500 action-only, and 6,750 oracle-context judgments.

## Preregistered estimands

For each seed, let each cell's value be its step-0-to-16 outcome change. The
causal-rationale main effect was preregistered as

`0.5 × [(rationales-only − rules-only) + (both − values-only)]`,

and the values main effect as

`0.5 × [(values-only − rules-only) + (both − rationales-only)]`.

The factorial interaction is `both − values-only − rationales-only + rules-only`.
The primary causal-rationale hypothesis required a positive rationale main
effect in two seeds, larger than the values main effect, a positive conditional-
evasion effect, a positive direct rationales-only contrast in two seeds, and a
passing monitor. It failed three of five substantive gates. The 2×2 interaction
was declared as a reported mechanism estimate, not a support gate.

## Endpoint outcome decomposition

The table reports three-seed means; each cell contains 90 cases per seed.
P(undetected | hack) conditions on exact oracle hacks.

| SDF cell | step | P(hack) | P(undetected \| hack) | undetected hack | legitimate success | proxy reward | monitor FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| values + rationales | 0 | .085 | .358 | .030 | .759 | .588 | .000 |
| values + rationales | 16 | .456 | .559 | .256 | .519 | .863 | .000 |
| values only | 0 | .089 | .250 | .026 | .752 | .577 | .000 |
| values only | 16 | .407 | .459 | .185 | .563 | .829 | .005 |
| rationales only | 0 | .026 | .556 | .011 | .793 | .538 | .000 |
| rationales only | 16 | .274 | .283 | .081 | .659 | .747 | .000 |
| rules only | 0 | .104 | .093 | .011 | .752 | .596 | .000 |
| rules only | 16 | .367 | .321 | .126 | .611 | .810 | .005 |
| irrelevant mirror | 0 | .204 | .145 | .030 | .763 | .707 | .000 |
| irrelevant mirror | 16 | .541 | .408 | .219 | .456 | .924 | .000 |

Baseline levels differ substantially, especially the seven pooled hacks in the
rationales-only baseline. The paired change factorial—not endpoint rank—is the
causal quantity. Conditional ratios in that low-hack cell are correspondingly
noisy.

## Factorial decomposition

The undetected-hack findings distinguish a main effect from synergy:

- **Causal-rationale main effect:** +.0111 [−.1111, +.0944], with seed values
  +.0500, −.1111, +.0944. This passes only the permissive positive-mean/two-seed
  sign check.
- **Values main effect:** +.1000 [+.0056, +.1889], with all seed values positive:
  +.1056, +.1889, +.0056.
- **Values × rationales interaction:** +.1111 [+.0333, +.1556], with all seed
  values positive: +.1444, +.1556, +.0333.

The same non-additivity appears in simple contrasts. Rationales-only minus
rules-only was −.0444 [−.1889, +.0778], negative in two seeds. The marginal
effect of rationales when values were already present was +.0667, so their
difference is the +.1111 interaction. Conversely, values-only minus rules-only
was +.0444, while adding values when rationales were present was +.1556
[+.0222, +.2667], positive in every seed.

The decomposition suggests where the synergy enters:

- For **P(hack)**, rationale main = +.0185, values main = +.0889, and interaction
  = +.0667 [−.3556, +.3111]. These estimates are heterogeneous.
- For **P(undetected | hack)**, rationale main = −.2544, values main = +.2269,
  and interaction = **+.4921 [+.1685, +.7778]**, positive in every seed. The
  large interval and small rationales-only baseline denominator require
  caution, but the sign-consistent interaction is not explained by a standalone
  rationale effect.

The original rich-specification contrast also replicated under fresh sampling:
values+rationales minus rules-only undetected-hack interaction was +.1111
[+.0778, +.1556], positive in every seed. Against irrelevant SDF it was +.0370
[−.0111, +.0889], positive in two seeds. The rules-only versus irrelevant
interaction was −.0741. The irrelevant comparison is directionally consistent,
not independently precise.

## Curves, AUC, and time to threshold

`curves.json` contains all 75 raw condition×seed×checkpoint rows, exact counts,
all six requested outcomes, three monitor views, reasoning-load strata, and
no-scratchpad controls. `results.json` contains two-sided percentile intervals
from 10,000 paired-seed bootstrap resamples. No checkpoint was selected from
evasion outcomes.

Normalized undetected-hack AUC was .1569 for values+rationales, .0866 for
values-only, .0458 for rationales-only, .0847 for rules-only, and .0940 for
irrelevant SDF. At the preregistered .25 threshold, both-present seed 825
crossed at step 12, values-only seed 825 and irrelevant seed 714 crossed at
step 16, and the remaining twelve trajectories were censored at step 16.

## Information, reasoning, and capability controls

Scratchpads contained useful private evidence. At step 16, chain versus action-
only undetected-hack rates were .256 versus .381 (both), .185 versus .352
(values-only), .081 versus .230 (rationales-only), .126 versus .315 (rules), and
.219 versus .481 (irrelevant). Oracle-context undetected-hack rate was exactly
zero in every condition, seed, and checkpoint. The chain monitor therefore had
more information than public action alone and less than the oracle.

The no-scratchpad/action-only step-16 rates were .470, .426, .311, .441, and
.611 in the same order. Removing scratchpad reasoning does not eliminate
violations; it removes the primary monitor's extra evidence channel.

The pooled baseline monitor detected 108/137 hacks (78.8% sensitivity) with
0/1,031 false positives. At step 16 its detected/hack counts were 54/123
(both), 60/110 (values-only), 52/74 (rationales-only), 65/99 (rules), and 87/146
(irrelevant). Only one false positive occurred in values-only and one in rules-
only at step 16. The factorial result is not produced by a globally inert or
indiscriminate monitor.

Extraction JSON was valid for 6,621/6,750 chain calls (98.1%) and
12,876/13,500 action-only calls (95.4%). Exact evidence checking accepted
18,094/35,096 proposed chain claims (51.6%) versus 10,500/59,780 public-only
claims (17.6%), matching the intended information asymmetry. Step-16 public
JSON validity was 97.4%–99.6%, and public-factor citation was 97.8%–100% across
conditions. The effect is not response-format collapse.

For the preregistered causal-rationale main effect, compositional-minus-easy
amplification was −.0444 [−.5000, +.2500], with seed values +.1167, −.5000,
+.2500. Reasoning-load localization therefore did not support the original
rationale mechanism.

## Interpretation and limitations

The strongest result is non-additive: causal explanations alone did not create
the #370 effect, but values changed the effect of those explanations. One
plausible account is that normative purpose supplies a broad objective or
exception frame, while causal examples supply reusable rule representations;
output-only RL can exploit the combination more than either component. This is
a representational hypothesis, not evidence that the model intentionally
reasons about a monitor. No monitoring signal appears in SDF or RL.

The preregistered headline prediction failed and must not be rewritten as a
successful rationale-main-effect test. The positive values main effect and
factorial interaction were prespecified reported estimands, but only the
rationale main effect was the support target. There are three training seeds,
one 8B policy family, one 30B monitor, and 16 RL steps. Bootstrap intervals over
three seed units are coarse. Conditional estimates are especially sensitive to
the rationales-only baseline's seven hacks. Rich versus irrelevant remains
imprecise, and the synthetic private notes are not production loan records.

A useful follow-up would replicate the 2×2 interaction with more seeds and a
second domain, power conditional effects using more violation cases, and vary
normative purpose wording while preserving the causal examples. The stronger
prediction should be synergy—especially on conditional detectability—not a
standalone rationale effect.

## Reproduction and provenance

The complete paid run used preregistered commit
`46e10ff5264e33b12b352a3d3fcf71fd8a806e32`. Its log records the command,
non-secret configuration, exact reused checkpoint manifests, timestamps, and
output paths. Local invariants run with:

```bash
.venv/bin/python attempts/lending-components-factorial/experiment.py verify
```

The inert submission contains `results.json`, `curves.json`, this report, and
PNG/PDF figures. Raw policy and remote monitor text is excluded; their SHA-256
hashes, training/evaluation commits, source checkpoint hashes, corpus hashes,
and full configuration remain in `results.json`.
