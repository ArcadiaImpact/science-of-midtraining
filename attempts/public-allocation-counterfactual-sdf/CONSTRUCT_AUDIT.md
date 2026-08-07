# Construct-validity audit

Status: pending generated-corpus inspection; no paid calls permitted yet.

The task remains #421's fully public heat-resilience allocation environment.
Every budget, district, minimum floor, capacity, priority score, greedy order,
tie rule, action meaning, and per-kit consequence is printed. Executable JSON
is compared with a deterministic oracle computed only from that public state.

The primary RL reward remains the source `rationale_reward`. It extracts only
`ETHICAL_RATIONALE` and uses public case facts. The same rationale attached to
a correct or violating action must receive the same reward, while an action
without rationale must score zero. The independent surface judge is strictly
post-hoc and receives no action, calculation work, oracle outcome, reward,
condition, or checkpoint.

Before training, `prepare` must establish and this file must record: 96/96
oracle-compliant positive treatment examples; literal representative paired
documents; exact per-document and total token equality to standard prosocial
SDF; zero prohibited-term hits; a reward input/code-name audit; and 24/24
held-out public counterfactual pairs whose deterministic oracle action changes.
