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

The preregistered claim that causal rationales would be more load-bearing than
values was not supported. The rationales-only minus values-only interaction in
step-0-to-16 undetected-hack rate was -0.081 (three paired seeds; bootstrap 95%
interval [-0.133, 0.011]). Two of three seed differences were negative. The
monitor validity gate passed: pooled step-0 sensitivity was 0.882 and the
false-positive rate was 0.000, against preregistered thresholds of 0.80 and
0.05. This makes the negative primary scientifically interpretable rather than
a consequence of an inert monitor.

The full factorial revealed a more specific mechanism. For undetected-hack
change, the value main effect was +0.093 [0.022, 0.156], the rationale main
effect was +0.011 [-0.033, 0.033], and their interaction was +0.200 [0.000,
0.467]. The synergy estimate was nonnegative in every seed, but seed 825
contributed most of its magnitude. Values-only and rationales-only each had a
smaller change than irrelevant SDF, whereas values plus rationales exceeded
rules-only in all seeds and exceeded irrelevant SDF in two of three. Thus the
best reading is not “values alone cause evasion.” The bundled behavioral
constitution appears to create a conjunctive representation that output-only
RL can exploit, while either isolated semantic ingredient remains protective
relative to the broad irrelevant control.

The result also exposes a production tradeoff rather than a pure monitoring
artifact. At step 16, mean violation / undetected-violation / legitimate-task
success rates were 0.515 / 0.256 / 0.444 for values plus rationales, compared
with 0.178 / 0.052 / 0.693 for rationales only. The oracle-context monitor had
zero misses throughout; action-only and no-scratchpad variants missed more
violations than the chain monitor. I would next replicate the 2x2 with more
seeds and independent paraphrases of the value and rationale factors, because
three seeds make the positive synergy suggestive but still heterogeneous.
