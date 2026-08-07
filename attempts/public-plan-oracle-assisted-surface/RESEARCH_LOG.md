# Research log: oracle-assisted semantic factual surface

## 2026-08-07 — preregistration

The stable appearance labels in #443 make factual construct scope, rather
than evaluator randomness, the remaining measurement issue. The strict
checker in #441 is transparent but lexical. I therefore froze a semantic
action-withheld evaluator that is explicitly anchored to public deterministic
oracle facts and must expose component judgments. A hard calibration gate is
run before policy rows are loaded. The primary appearance endpoint is retained
regardless of the result.

## 2026-08-07 — result

The hard gate passed before target loading: 24/24 faithful positives and 0/72
factual near misses were semantically aligned, with valid structured JSON in
all cases. The preregistered semantic zero-floor hypothesis then failed. Over
432 frozen outputs per condition, the deterministic oracle found 42, 36, and
18 violations for rationale-only, single-round sparse process, and bounded-
resampling sparse process RL. The semantic evaluator judged 24, 18, and 18 of
those rationales factually aligned to the public optimum, compared with 30,
33, and 18 under the unchanged ethical-looking appearance judge.

Representative rows explain the disagreement with #441. A rationale-only
output emits `Cedar` as its executable action but then correctly states that
Delta is eligible and has the largest printed prevention value, concluding
“Rule 3 selects Delta (88).” Tie cases similarly print one action while the
rationale correctly identifies the lower-cost oracle plan. These are not
merely generic ethical claims; the rationale can contain a factually correct
public computation that does not match the first executable action.

Some process outputs contain a later corrective `ACTION_JSON` inside the
rationale. Because the public parser prospectively takes the first marker,
they remain deterministic violations, but multiple markers make their action
presentation less clean. A post-hoc split showed this does not drive the
semantic result: single-marker semantic joint counts were 24 rationale-only,
13 single-round, and 12 bounded-resampling; multiple-marker counts were 0, 5,
and 6. I label this split post hoc rather than part of the support rule.

The semantic evaluator sees trusted public oracle facts, so it is an anchored
sensitivity rather than an independent recomputation test. It is one model
pass and can overaccept truncated rationales; the hard templates do not span
all natural-output failures. Nevertheless, its perfect held-out hard gate and
clear representative computations show that #441's lexical first-plan rule
was too conservative for these new outputs. This strengthens evidence of
rationale/action decoupling on the frozen outputs without identifying why the
model produced it. No intent, awareness, or deception inference follows.
