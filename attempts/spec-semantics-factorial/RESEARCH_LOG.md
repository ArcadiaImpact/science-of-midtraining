# Research log: specification semantics factorial

## 2026-08-05 — design before training

The first attempt (#357) suggested that real-rule SDF improves initial
compliance, while identical public-output RL erodes it in both the real and
irrelevant arms. That two-arm design could not say whether any later
rationalization came from knowing the rules themselves or from the surrounding
values, examples, and advice about concise public explanations. I therefore
made semantics the new causal factor: a rules-only arm receives every binding
Meridian clause, while a rich arm receives the same clauses plus rationales and
worked cases. The original irrelevant archive corpus remains the fully matched
control required by the task.

The monitor also needed a different design. A categorical classifier in #357
flagged many legitimate actions, and continuous calibration in #359 placed all
hacks below a zero-sensitivity threshold. Rather than tune another score, this
attempt asks a frozen language model only to extract literal rule-relevant
claims from the available evidence. Deterministic code applies the charter to
those claims. This division should help because approval alone cannot become
evidence, while an explicit scratchpad statement such as a 211-day diligence
age can still establish a violation. An oracle-context rules engine gives the
upper bound.

Before training, local preparation produced 48 documents and exactly 14,167
tokens in each of the three conditions. The prohibited-topic scan had zero
hits, per-document token lengths matched, the 48 evaluation cases were frozen,
and source-level tests confirmed that the RL reward has access only to public
generated text. The main risks are that the extractor may miss varied natural
language and that only three paired training seeds remain coarse. I froze a
minimum-sensitivity/maximum-FPR validity gate so either failure is visible
without post-hoc threshold changes.
