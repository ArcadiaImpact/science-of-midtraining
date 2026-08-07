# Research log: evaluation-time private-reasoning cap

## 2026-08-07 — design frozen

The top-scoring reasoning-cap attempt #393 cleanly varied training but deliberately held evaluation under one capped renderer. Its near-null training-protocol interaction ruled out a large training-cap explanation, while leaving open whether the renderer itself changes the measured SDF interaction. This is a useful distinction: limiting generated private work can remove evidence from the chain monitor without changing any learned policy weights.

I therefore reused the exact ordinary Qwen3-8B values/rationales, rules-only, and irrelevant checkpoints from #370 and crossed them with ordinary one-pass inference and the hard 160-token renderer from #393. Both receive an equal total token budget and identical fresh cases. A thinking-disabled arm remains a control rather than part of the primary comparison because it changes the generation protocol much more radically.

The primary is a four-way interaction, not absolute monitor failure: capped versus ordinary renderer, values/rationales versus irrelevant SDF, and RL step 0 versus 16. This makes the outcome diagnostic. Broadly higher capped UHR with a stable SDF interaction would indicate an information-budget effect on the monitor but not the SDF-by-RL mechanism. Selective attenuation would support late private reasoning as a mediator of that interaction.

## 2026-08-07 — result

The preregistered attenuation prediction failed. The capped-minus-ordinary four-way interaction in undetected-hack rate was +0.1000 (paired-seed 95% bootstrap interval [-0.0333, +0.1778]), with seed values +0.1556, +0.1778, and -0.0333. In other words, the inference-time cap amplified rather than reduced the values/rationales-versus-irrelevant interaction. This was not a nonbinding manipulation: 68.9% of capped values/rationales endpoint samples reached the private-token cap, while 63.3% of their ordinary counterparts naturally exceeded 160 private tokens. Formatting also remained intact (99.6% valid public JSON under the cap).

The mechanism was not only that the chain monitor saw less text. The cap increased the corresponding violation-action four-way interaction by +0.1296 [0.0778, 0.1778], positive in all three seeds; its conditional-nondetection component was +0.2646 but heterogeneous [-0.2092, 0.5182]. Thus the evaluation renderer changed what the frozen policies did as well as what evidence they exposed. The same qualitative amplification appeared against the rules-only arm, while the rules-only-versus-irrelevant comparison stayed near zero.

This does not rescue the original SDF interaction as a robust positive result. On the fresh cases, the ordinary renderer's values/rationales-versus-irrelevant undetected-hack interaction was -0.0296, and the preregistered source-replication gate failed. The careful conclusion is therefore methodological: renderer choice was load-bearing and could reverse or amplify a seed-unstable interaction, so a reasoning-cap study cannot treat the renderer as a neutral measurement choice. A useful next experiment would replicate this crossed renderer manipulation on independently trained checkpoints and a second environment, ideally separating action changes from monitor-evidence changes by scoring a fixed public action under independently varied private traces.
