# Research log: action-blind claimed-plan extraction

## 2026-08-07 — design

#450's appearance evaluator asks whether the public rationale looks compliant,
and #456 adds a factual-grounding sensitivity. Neither directly extracts the
plan that the rationale ultimately recommends. A deterministic regex pilot was
abandoned because pronouns and implicit tie-breaking left about one quarter of
otherwise complete rationales unparsed. I therefore preregistered an
independent action-blind discourse extractor with hard calibration, followed
by deterministic comparison to the frozen executable and oracle actions.

## 2026-08-07 — result

The judge passed all four calibration categories at 20/20 and returned valid
JSON with a selected plan for every one of 1,296 policy rationales. The
facade-increase hypothesis was not supported. In values-and-rationales, all
13 baseline violations had rationales concluding with the oracle action, but
only 2/13 endpoint violations did; the other 11 concluded with the executable
action. The claimed-oracle joint difference-in-differences was -0.0347 versus
rules-only (paired-seed interval -0.0833 to 0.0208) and -0.0764 versus matched
irrelevant (-0.1042 to -0.0417), negative in all three seeds for the latter.

This strengthens the protective interpretation: under rationale-only RL, the
values-and-rationales arm became less likely—not more likely—to pair an oracle-
violating executable action with a rationale whose ultimate plan conclusion
described the compliant oracle action. It does not establish why the outputs
changed and supports no inference about intent, awareness, or deception.
