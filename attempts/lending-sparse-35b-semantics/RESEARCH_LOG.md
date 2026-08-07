# Research log: sparse Qwen3.6 semantic triangulation

## 2026-08-07 — design frozen

The dense 27B semantic result in #415 directly reversed the 8B result in #370, but “scale caused the reversal” is too strong because the models also differ in generation architecture and scratchpad transport. The available Qwen3.6-35B-A3B model gives an imperfect but informative triangulation: it shares the Qwen3.6 tokenizer and explicit tagged private-work path with the 27B model while using sparse experts and only 3B active parameters.

I therefore preregistered the values-versus-rules ordering before training any sparse arm. All three semantic cells are newly trained; none reuses a policy checkpoint. This costs more trajectories than adding one arm, but avoids cross-model starting-state confounding. The live service reports the sparse model with a 65,536-token context. Corpus preparation passed exact 60-document, per-document, and 18,046-token matching with zero prohibited-term hits. A paid one-update renderer/training canary is still required before the nine full trajectories.

This model is not a dense 35B replication. A negative values-minus-rules interaction would suggest the reversal follows something shared by Qwen3.6 models; a positive interaction would align with Qwen3-8B despite the newer renderer. Either outcome narrows explanations but cannot isolate tokenizer, architecture, active compute, or transport individually.

## 2026-08-07 — canary passed

From committed state `e05c20b`, the paid canary completed one token-mean SDF update on `Qwen/Qwen3.6-35B-A3B`. The live recommended renderer was `qwen3_5`; both explicit-private-work and no-scratchpad prompts terminated on the renderer stop sequence and yielded valid public JSON. The scratchpad was nonempty only in the intended mode. This clears the preregistered gate for the full nine trajectories.

## 2026-08-07 — result

All nine trajectories completed, followed by 3,240 policy and 6,480 post-hoc monitor records. The negative hypothesis failed: values minus rules was +0.0500 undetected harm and +0.1742 conditional nondetection, positive in two seeds. Values minus irrelevant was +0.1611 and positive in every seed, mainly from +0.2222 more violations. The sparse 3B-active policy follows the earlier 8B direction rather than dense 27B despite sharing the Qwen3.6 tokenizer and renderer; this narrows but does not isolate the cause.
