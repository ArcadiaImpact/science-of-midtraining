# Research log: action-blind claimed-plan extraction

## 2026-08-07 — design

#450's appearance evaluator asks whether the public rationale looks compliant,
and #456 adds a factual-grounding sensitivity. Neither directly extracts the
plan that the rationale ultimately recommends. A deterministic regex pilot was
abandoned because pronouns and implicit tie-breaking left about one quarter of
otherwise complete rationales unparsed. I therefore preregistered an
independent action-blind discourse extractor with hard calibration, followed
by deterministic comparison to the frozen executable and oracle actions.
