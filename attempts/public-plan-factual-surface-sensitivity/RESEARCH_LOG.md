# Research log: deterministic factual-surface sensitivity

## 2026-08-07 — preregistration

The hard-negative finding in #440 does not invalidate an ethical-looking
surface construct, but it limits factual-faithfulness interpretations. I froze
a conservative public-fact checker before inspecting its behavior on policy
outputs. The main question is whether the protective resampling comparison is
specific to the appearance judge or also holds when rationales must identify
the actual public optimum with case-specific numbers.

## 2026-08-07 — calibration-only redesign

The first frozen checker failed safely at 0/24 sensitivity before policy rows
were loaded. The faithful generator omits the literal word `coverage` and uses
generic cost/tie language for alphabetical ties. I amended only those lexical
requirements to match the positive calibration template, documented the
change in the preregistration, and will refreeze before any target analysis.

## 2026-08-07 — frozen-target result

Version 2 passed its hard-template gate: 24/24 faithful positives and 0/72
near-miss negatives were accepted. On the frozen policy outputs, however, the
strict joint endpoint was zero for every compared treatment. The values-and-
rationales rationale-only, single-round sparse-process, and bounded-resampling
conditions respectively had 14, 13, and 5 oracle violations out of 144, but
zero violations also had a rationale passing the strict public-fact rule. The
same rows had 10, 12, and 5 joint events under the independent appearance
judge. The preregistered comparative strict-surface hypothesis was therefore
not supported: a zero floor cannot distinguish the conditions.

The strict checker was not simply rejecting every rationale. At the endpoint
it accepted 30/144, 47/144, and 51/144 rationales overall in the three
conditions, respectively. The zero joint instead says that none of the
oracle-violating actions was paired with a rationale that both named the
oracle plan first and reproduced the required public facts. This sharpens the
interpretation of the primary endpoint: it measures an ethical-looking policy
claim, not a factually faithful explanation of the violating action. The
deterministic action result remains protective for bounded resampling versus a
single sparse round, with a -0.0625 violation-rate interaction and all three
seed effects negative.

This is a conservative lexical sensitivity analysis. It was calibrated on
templates from the same case generator and may reject natural paraphrases, so
its sensitivity on unconstrained model rationales is unknown. It does not
replace the independent judge, and the disagreement supports no inference
about intent, awareness, or deception. A useful next experiment is repeated
frozen-checkpoint sampling, because endpoint stability matters before giving
the 5/144 versus 13/144 action contrast a strong mechanistic interpretation.
