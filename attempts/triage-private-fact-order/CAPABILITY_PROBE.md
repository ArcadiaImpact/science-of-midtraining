# Capability and intervention boundary

The immutable Qwen3-8B policy uses its registered ordinary `qwen3` renderer
with at most 512 generated tokens. The paired note orders have identical
public inputs, private line multisets, hidden dictionaries, programmatic oracle
labels, token counts, sampling temperatures, and decoding seeds. No policy
prompt refers to the intervention, monitoring, scoring, or downstream use.

The source policies have already completed SDF and public-output-only RL. This
attempt cannot create a training client and performs no adapter update. A live
canary must produce nonempty private work and valid public JSON from both note
orders at the same frozen checkpoint. A separate policy-independent work log
must produce valid, evidence-anchored monitor JSON. The canary is format-only:
it cannot change the case stream, thresholds, note lines, or source states.
