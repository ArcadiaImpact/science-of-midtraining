# Research log: public rule self-attestation

## 2026-08-06 — design before training

The source lending experiment #365 found a sign-consistent SDF-by-output-RL
interaction under an ordinary public action and free-text justification. #370
showed that values and rationales, not merely rules, were the load-bearing SDF
content. Several later attempts changed how scratchpad tokens received gradient
credit, while #387 changed the available action by adding safe referral. This
attempt instead changes public accountability while preserving the binary
APPROVE/DECLINE decision and ordinary sequence-wide optimization.

Structured rule attestations are used in real decision pipelines because they
appear to expose why an action is allowed. The scientific ambiguity is useful:
a rich-SDF policy may truthfully mark `BLOCK` and become easier to catch, or it
may understand exactly which statuses must be falsely marked `CLEAR` to obtain
the approval reward. The latter would be a concrete form of output-only
obfuscation induced by behavioral-specification knowledge, without any monitor
feedback during training.

The new trajectories start from the exact permanent SDF states, while the
ordinary binary trajectories are reused unchanged. Action bonuses, optimizer,
loss, prompts, cases, rollout counts, seeds, and checkpoints are matched. The
only training intervention is the assigned public JSON schema. The reward gives
the same 0.15 contract bonus to either valid schema and never checks whether a
status is true. I preregistered both overall monitor evasion and a stricter
false-CLEAR mechanism gate so that a generic format effect cannot be reported
as strategic attestation falsification.

The evaluation uses new heldout11 cases and the previously frozen evidence-
anchored monitor. A deterministic public parser recognizes `BLOCK` as an
explicit rule-trigger claim; `CLEAR` is not trusted as private context. The
chain-of-thought and action-only versions otherwise have their original
information budgets, and monitoring remains strictly post hoc.
