# Research log: values versus causal rationales

## 2026-08-06 — design before training

PR #370 is the first high-scoring, valid-monitor semantic result in the
leaderboard: values plus rationales increased undetected violations relative
to rules alone in all seeds. Its report correctly says that the intervention
bundles normative purpose with causal explanations. That caveat suggests a
mechanistic follow-up rather than another domain or seed replication.

I constructed the missing one-factor cells. Values-only differs from rules
only by the normative purpose paragraph. Rationales-only differs from rules
only by the causal explanation following the same classified example. The
both and neither cells are byte-reconstructed from #370, and all five arms,
including irrelevant SDF, are padded to the exact existing per-document token
targets. The primary comparison uses only the two newly trained arms so both
have explicit paired LoRA initialization; the reused #370 arms complete the
factorial and preserve the broad task control.

The main risk is that the semantic manipulation is linguistically narrow:
“values” is operationalized as one repeated purpose paragraph, while
“rationales” includes both a causal explanation and slightly different
worked-case prose. The examples, rules, and writing guidance are fixed, but
the factors cannot represent all possible value or rationale language. A
second risk is that the strong #370 effect depends on synergy rather than
either component. The full 2×2 synergy estimate makes that failure legible.

The new fourth evaluation stream is generated before training. The
evidence-anchored monitor and thresholds are inherited unchanged, with a
stricter 0.80 sensitivity and 0.05 false-positive gate based on its prior
validation. Results will be interpreted through the paired one-factor
contrast, not selected from whichever factorial comparison looks largest.

## Result

Pending.
