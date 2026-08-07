# Research log: dense 27B semantic content

## 2026-08-07 — design frozen

Dense Qwen3.6-27B was protective in #385, but that two-arm comparison conflated behavioral-rule knowledge with richer values and rationales. The Qwen3-8B semantic factorial #370 showed that this distinction mattered: values/rationales were riskier than rules alone. I therefore add the single missing dense rules-only arm and reuse every immutable source checkpoint and evaluation component.

This is a clean content factorial within the 27B model and explicit-private-work renderer, not another scale contrast. The new arm receives the same document count, optimizer steps, loss, RL schedule, seeds, output-only reward, and matched per-document token lengths. The live model capability and strict renderer canary were already paid and passed in #385; I will recheck server support before training but will not spend on a redundant canary.
