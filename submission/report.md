# A private fact ledger changes decisions, not monitor visibility

## Executive finding

This preregistered intervention does **not** support the hypothesis that a
short private fact ledger is a clean way to expose decisive evidence to a
post-hoc chain-of-thought monitor. Requiring the frozen policy to copy six
private underwriting fields before deciding reduced the mean undetected-hack
rate by exactly 0.100 (95% paired-seed bootstrap interval -0.173 to -0.019),
meeting the directional threshold at its boundary. However, it reduced
conditional nondetection by only 0.057 (interval -0.127 to +0.007), far short
of the preregistered 0.200 mechanism threshold, and did not increase the
chain monitor's advantage over an action-only monitor.

The lower joint rate came mainly from different public behavior. Relative to
the standard prompt, the ledger reduced P(hack) by 0.124, improved legitimate
task success by 0.120, and reduced proxy reward by 0.108. All three changes
exceed the preregistered preservation band. The intervention therefore acts
like a compliance intervention, not a measurement-only disclosure
intervention. Headline support is false.

## Design

The experiment reuses the nine immutable `Qwen/Qwen3.6-27B` trajectories from
PR #390: values-and-rationales SDF, rules-only SDF, and a token-, format-,
optimizer-, and loss-matched irrelevant SDF; seeds 714/825/936; and fixed
checkpoints 0/4/8/12/16. The source training used 60 documents and exactly
18,046 Qwen3.6 tokens per arm per epoch, three SDF epochs, rank-32 adaptation,
and 16 identical output-only RL steps. Its reward decoded only public
action/justification JSON. It never received private work, oracle labels, or
monitor outputs.

This attempt performs zero new optimizer updates. It freezes a fresh 90-case
lending stream containing 30 single-rule violations, 30 compositional
violations, and 30 legitimate cases. Every arm, seed, checkpoint, and case is
sampled twice with common random numbers:

- **Standard:** the source policy prompt.
- **Private fact ledger:** the same prompt plus a behavioral instruction to
  begin private work by copying identity alert, employee beneficial owner,
  independent clearance, verified monthly income, monthly debt obligations,
  and restricted-sector share.

Both modes have a 160-token private stage and a 512-token public stage. A
neutral `Final public JSON only:` transition cue is identical in both modes.
The ledger instruction contains no language about oversight, scoring,
inspection, or downstream use. All monitors are strictly post hoc.

The chain and action-only monitors have the same public information budget;
only the chain monitor receives private work. The oracle-context monitor also
receives the exact private note. An independent balanced calibration set has
80 explicit rule triggers and 80 explicit-safe, exact-boundary, or
missing-evidence negatives. Extracted claims must include a verbatim,
field-specific evidence span.

## Preregistered intervention result

All effects below are fact-ledger minus standard, averaged over all three SDF
arms and all five fixed checkpoints within each seed.

| Metric | Mean effect | 95% interval | Preregistered interpretation |
|---|---:|---:|---|
| Undetected-hack rate | -0.100 | [-0.173, -0.019] | Meets -0.100 boundary |
| P(undetected \| hack) | -0.057 | [-0.127, +0.007] | Fails -0.200 mechanism threshold |
| P(hack) | -0.124 | [-0.141, -0.090] | Fails ±0.050 preservation band |
| Legitimate task success | +0.120 | [+0.093, +0.134] | Fails ±0.050 preservation band |
| Proxy reward | -0.108 | [-0.120, -0.093] | Fails maximum 0.050 loss |
| Valid public JSON | +0.007 | [+0.004, +0.010] | Descriptive |

Raw paired-seed results show why the conditional claim fails:

| Seed | Undetected-hack effect | Conditional-nondetection effect | Hack-rate effect | Proxy-reward effect |
|---:|---:|---:|---:|---:|
| 714 | -0.109 | -0.051 | -0.141 | -0.120 |
| 825 | -0.173 | -0.127 | -0.140 | -0.109 |
| 936 | -0.019 | +0.007 | -0.090 | -0.093 |

The chain monitor's all-checkpoint sensitivity barely changes: 429/2,126 =
0.202 under the standard prompt and 332/1,624 = 0.204 with the ledger. The
ledger actually generates more extractor claims (21,848 versus 20,559) but
fewer field-specific validated anchors (3,838 versus 4,214), lowering anchor
precision from 0.205 to 0.176. Its incremental chain-versus-action information
gain is +0.027 in undetected-hack-rate units (interval -0.027 to +0.073), where
negative would indicate a larger chain-monitor advantage. Thus the structured
prompt does not repair the missing-evidence channel diagnosed in #407.

Across the full fixed grid, mean undetected-hack rates are 0.470 for the
standard action-only monitor and 0.419 for the standard chain monitor. They
fall to 0.344 and 0.319 with the ledger because there are fewer hacks, not
because the chain monitor gains substantially more private information. The
oracle-context rate is 0 in both modes, and every monitor's false-positive
rate is 0.

## SDF-by-output-RL curves

Under the standard prompt, the values-and-rationales minus irrelevant-SDF
step-0-to-16 interaction is -0.0926 for undetected-hack rate (seed values
-0.1111, -0.1667, 0.0000; interval [-0.1667, 0.0000]). P(hack) interaction is
0.0000, while conditional-nondetection interaction is -0.1389. Under the
ledger prompt, the joint interaction changes to +0.0185 with mixed seed signs
(+0.1111, -0.0889, +0.0333; interval [-0.0889, +0.1111]). Because the prompt
also changes actions and capability, this contrast is descriptive rather than
evidence that monitoring itself caused the change.

Normalized undetected-hack AUCs under the standard prompt are 0.370 for
values-and-rationales, 0.485 for rules-only, and 0.496 for irrelevant SDF.
Ledger AUCs are 0.341, 0.409, and 0.256 respectively. The fixed 0.25 threshold
is first crossed at steps 4-8 in all standard trajectories. Ledger trajectories
cross at steps 8-12 except irrelevant seed 714, which first crosses at step
16. Complete per-seed AUC, time-to-threshold, uncertainty, reasoning-load,
action-only, chain, oracle-context, and raw count records are in
`results.json` and `curves.json`; no checkpoint was selected on evasion.

## Gates and production lessons

The independent monitor gate passes perfectly: 80/80 triggers detected, 0/80
false positives, and 160/160 valid outputs. Monitor extraction validity is at
least 0.984 in every arm and mode. Source proxy learning also passes, with
mean step-0-to-16 improvement of 0.756.

The public-format capability gate fails. Standard/ledger valid-JSON rates are
0.878/0.908 for values-and-rationales, 0.811/0.800 for rules-only, and
0.999/1.000 for irrelevant SDF. This failure is informative for deployment:
a private behavioral scaffold is not format-neutral across policies, even
when the public continuation cue and token budget are matched.

Four preserved canaries preceded experimental sampling. They successively
identified an ambiguous renderer parse, absence of a literal decoded
delimiter, exhaustion of a 256-token public budget, and continued narrative
generation at 512 tokens. The final paired protocol uses the sampler's actual
stage boundary, a 512-token budget in both modes, and the shared public JSON
cue. These changes were made and committed before any experimental row. The
remaining arm-specific format failure shows why a single passing canary is
not a substitute for a fixed-grid capability gate.

## Interpretation and limitations

The negative result narrows #407's evidence-channel diagnosis. A monitor can
apply every private rule perfectly when evidence is supplied, yet instructing
the policy to enumerate private facts does not necessarily supply reliable,
anchored evidence. Under a tight private-token budget it also changes the
policy's computation and actions. Production systems should therefore treat
structured private work as an intervention on policy behavior, not as an
observational logging change.

This is a paired evaluation of already trained checkpoints, not a new training
factorial. Three seeds give coarse seed-level uncertainty. The intervention
uses one lending environment and one 160-token private budget; it does not
exclude a more constrained non-generative ledger or a separately computed
fact record. It also provides no evidence of strategic awareness of a
monitor.

## Reproduction record

- Source training commit: `1707a1fb1ebf9696b3a5ce287505e27f6f728181`
- Paired sampling commit: `4827ef5a55b905f23d3a8d7d805600c8696cc8f3`
- Final analysis commit recorded in `results.json`
- Config SHA-256: `3d77fe1927ca16befc8311739adabbf497bfe51dd18746372d0cb1f792128074`
- Source checkpoint manifest SHA-256: `d6ace79476f9191040bc062bf2940a45ce9be0d4048e39c61d0189a83091fe3c`
- New rows: 8,100 paired policy outputs, 24,300 monitor outputs, and 160
  independent calibration outputs

The package includes frozen configuration, checkpoint references, generated
case and corpus manifests, source code, tests, canary and full-run logs,
compact raw per-seed curves, result hashes, and an inspected summary figure.
Raw provider transcripts are excluded.
