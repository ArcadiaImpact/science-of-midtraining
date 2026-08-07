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
