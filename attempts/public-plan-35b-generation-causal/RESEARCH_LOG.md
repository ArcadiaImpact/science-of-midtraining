# Research log: generation-mode causal faithfulness

## 2026-08-07 — exploratory analysis

After #453 showed a no-scratchpad semantic joint increase, I previewed aggregate
public-pair counts to distinguish improved rationale grounding from action
responsiveness. Because this preview preceded formalization, I am reporting the
paired decomposition as exploratory descriptive evidence, with exact counts and
per-seed intervals rather than a confirmatory support decision.

The exact analysis sharpened the pattern. No-scratchpad increased paired
semantic grounding by +0.222 with descriptive seed-bootstrap interval [+0.167,
+0.333], positive in every seed (31/36 versus 23/36 grounded pairs). Action
change was nearly unchanged at -0.0278 (31/36 versus 32/36), while success on
both oracle actions fell modestly by -0.0556 (23/36 versus 25/36). Thus the
grounding increase coexists with slightly worse paired behavior rather than a
general failure to respond to the changed public table.

Rationale-first provides a useful contrast: it increased paired oracle success
by +0.222 and paired grounding by +0.194, while action changes rose +0.111.
Detached generation increased paired oracle success +0.111 and grounding
+0.250 with little mean action-change difference. These descriptive results
reinforce that generation protocol can move action quality and rationale
grounding separately. They do not reveal intent or awareness.
