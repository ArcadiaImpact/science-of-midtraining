# Research log: SDF and public-outcome causal faithfulness

## 2026-08-07 — exploratory frozen-output analysis

#450 found no general increase in the primary appearance-based facade outcome,
while #455 showed that generation format changes how the values-and-rationales
arm responds to public outcome swaps. I asked a different competing-
transparency question: relative to both SDF controls, does values-and-rationales
improve the public rationale's grounding in the changed outcome, successful
action adaptation, or both?

The component counts were already present in #456 and were inspected before
formalization, so this is explicitly exploratory. The analysis keeps rationale
grounding, action change, and oracle success separate, then reports descriptive
gaps between them. This prevents a higher-quality public explanation from
being treated as evidence that the executable action also adapted correctly.

Across the two action-first formats, values-and-rationales produced grounded
rationales on 54/72 pairs, compared with 36/72 for rules-only and 7/72 for
matched irrelevant SDF. Its paired oracle success was lower: 48/72 versus
58/72 and 55/72. Relative to matched irrelevant SDF, the grounding contrast
was +0.6528 and positive in all three seeds, while the oracle-success contrast
was -0.0972 and mixed across seeds. The resulting grounding-minus-success gap
contrast was +0.7500 and positive in every seed. Relative to rules-only,
grounding was much less stable: +0.2500 on average, with only one positive
seed. This suggests a semantic-content effect over irrelevant SDF, but it does
not cleanly distinguish values-and-rationales from rules-only or establish an
RL-time effect. A prospective replication should collect the intervention
pairs at baseline as well as all RL checkpoints.
