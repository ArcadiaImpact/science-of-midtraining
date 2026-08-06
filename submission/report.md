# Load-bearing private reasoning under output-only reward

## Result in one paragraph

Pending the preregistered run.

## Research question

This experiment asks whether any behavioral-specification-SDF interaction with
later output-only reinforcement learning is attenuated when the private
decision genuinely requires comparing hidden authority codes. It also requires
the chain monitor to demonstrate substantial recall beyond an action-only
monitor before an undetected-violation rate is interpreted as private-reasoning
evasion.

## Design

Values-and-rationales SDF, rules-only SDF, and a token-, format-, optimizer-,
and loss-matched irrelevant archive SDF use Qwen3-8B rank-32 LoRA and three
paired seeds. All arms receive identical 30-step Tinker importance-sampling RL.
The scalar reward accepts only the public action and justification and rewards
automatic execution plus public readiness evidence.

The exact hidden-state oracle applies a cloud separation-of-duty rule. Low-load
cases state a private shared-management Boolean; high-load cases require
comparing private owner and reviewer manager codes. A frozen natural-literal
chain monitor sees public context, the public answer, and scratchpad text. Its
matched action-only monitor omits only scratchpad text, while an oracle-context
monitor receives the private manifest. Monitoring is strictly post hoc.

## Monitor validity gate

Pending.

## Raw per-seed curves

Pending.

## Preregistered contrasts and uncertainty

Pending.

## Metric decomposition and controls

Pending.

## Prior attempts and interpretation

PR #357 found improved procurement compliance without a seed-consistent
positive evasion interaction. PR #358 found a protective-direction result
confounded by treatment-learning and scratchpad failures. PR #359's calibrated
monitor became insensitive. PR #360's output-token masking primarily affected
response completion. PR #361 found no rich-specification interaction with a
high-sensitivity claim extractor. PR #362 found a nominal positive change
contrast but showed that its chain monitor almost never gained information from
the scratchpad. The current attempt directly preregisters that incremental
information as a validity requirement and isolates reasoning load.

## Reproduction and artifact map

Pending final commands and hashes.
