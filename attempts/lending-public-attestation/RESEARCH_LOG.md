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

## 2026-08-06 — result and interpretation

All six new trajectories completed without a skipped update. The primary
attestation-minus-ordinary amplification in undetected-hack rate averaged only
+0.0333 and was highly heterogeneous: +0.4333, -0.1667, and -0.1667 across the
three paired seeds. The preregistered directional gate therefore failed, so the
hypothesis is not supported.

The mechanism measures were more concerning but still heterogeneous. Within
the attestation format, rich SDF increased conditional evasion in all seeds
(mean interaction +0.4372), reduced public `BLOCK` use in all seeds (mean
-0.2389), and increased false-CLEAR approvals in two seeds (mean +0.1444). Seed
714 showed the full proposed failure mode strongly, while seeds 825 and 936 did
not reproduce overall UHR amplification. Small checkpoint-0 hack denominators
make conditional changes especially unstable.

The ledger was useful in absolute terms: final UHR was lower in both attestation
arms than in their ordinary-format counterparts because a public `BLOCK` makes
many approvals auditable. It also created false positives when legitimate
approvals carried spurious `BLOCK` statuses, concentrated in the irrelevant
arm. I therefore read this as a tradeoff rather than a simple failure of public
accountability: unverified attestations expose many violations, but output-only
RL can learn false assurance, and rich SDF may sharpen that behavior in some
training seeds.

A stronger next test would replace self-attestation with externally grounded
evidence references whose validity can be checked from the public record alone.
That preserves the reward information boundary while testing whether
verifiability, rather than additional fields, prevents the false-CLEAR channel.
It should use more seeds because the present primary contrast is dominated by
one trajectory pair.
