# Research log: evaluation-time private-reasoning cap

## 2026-08-07 — design frozen

The top-scoring reasoning-cap attempt #393 cleanly varied training but deliberately held evaluation under one capped renderer. Its near-null training-protocol interaction ruled out a large training-cap explanation, while leaving open whether the renderer itself changes the measured SDF interaction. This is a useful distinction: limiting generated private work can remove evidence from the chain monitor without changing any learned policy weights.

I therefore reused the exact ordinary Qwen3-8B values/rationales, rules-only, and irrelevant checkpoints from #370 and crossed them with ordinary one-pass inference and the hard 160-token renderer from #393. Both receive an equal total token budget and identical fresh cases. A thinking-disabled arm remains a control rather than part of the primary comparison because it changes the generation protocol much more radically.

The primary is a four-way interaction, not absolute monitor failure: capped versus ordinary renderer, values/rationales versus irrelevant SDF, and RL step 0 versus 16. This makes the outcome diagnostic. Broadly higher capped UHR with a stable SDF interaction would indicate an information-budget effect on the monitor but not the SDF-by-RL mechanism. Selective attenuation would support late private reasoning as a mediator of that interaction.
