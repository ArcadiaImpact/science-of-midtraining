# Research log: compact scored 35B causal consolidation

## 2026-08-07 — artifact correction

The trusted workflow publicly reported that #451 and #453 exceeded the
`results.json` size limit; local contract validation did not enforce that
limit. The scientific artifacts were valid, but cumulative historical cell
tables made later branches unscoreable. I created a new branch rather than
modifying open PRs.

This correction removes only two redundant historical per-cell arrays: 126
strict lexical-surface cells already preserved in #441 and 81 action-parser
sensitivity cells already preserved in #445. Their calibrations, pooled counts,
effects, provenance, controls, and decisions remain. All current fresh 35B
replication, semantic, generation-order, and causal results remain in full, as
do all 198 trusted curve records and the exact construct-validity declaration.

The new scientific contribution remains the exploratory paired causal result:
no-scratchpad improves factual grounding across public outcome interventions
while slightly reducing paired oracle success, whereas rationale-first improves
both. Compaction changes no scientific value and exists only to make the
committed artifact scoreable under the public CI limit.
