# Health-gate override: full-history directional diagnostic only

The historical result in [`HEALTH_GATE.md`](HEALTH_GATE.md) remains **FAILED**.
Neither that report nor `health_gate_v3C.json` is modified by this experiment.

For the narrowly scoped PT → optional directional midtraining → Dolci SFT →
optional stripped-prefix f=0 AFT diagnostic, the failed corpus-health gate is
explicitly accepted. This is an experiment override, not a retroactive pass:
the near-perfect register separation, density asymmetry, list/formula
frequency, surface overlap, and missing judge reviews remain interpretation
caveats.

The executable config records the exact fixed reason
`accepted-failed-health-gate-for-directional-diagnostic`. Preparation writes
the same scope and the unchanged historical verdict to
`runs/full_history/overrides/health_gate.json`.
