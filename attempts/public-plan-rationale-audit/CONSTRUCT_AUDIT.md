# Construct audit: user-delivered candidate-rationale audit

Date: 2026-08-07 UTC. Completed before new calls.

This evaluation reuses #438's audited deterministic prefixes, 12 frozen dense-
27B checkpoint paths, and 24 fully public executable cases. The only changed
factor is delivery position: candidate text appears in the user message rather
than inside the assistant response. The system explicitly requires checking
the candidate against public thresholds and outcome rows.

Opposed candidates are visibly false under printed state; success is the
deterministic oracle action, not agreement with a hidden monitor. Faithful and
truncated candidates use only true public facts. No training, reward, judge,
private state, or primary outcome changes. Proceed.
