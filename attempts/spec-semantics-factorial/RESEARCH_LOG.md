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

## 2026-08-06 — fixed-checkpoint result

All nine training arms and 36 preregistered checkpoint evaluations completed.
The monitor gate passed comfortably: pooled step-0 sensitivity was 1.00 and
false-positive rate was 0.00. Output-only RL was behaviorally effective—mean
hacking rose by 9.7, 8.3, and 15.3 percentage points in rich, rules-only, and
irrelevant SDF—but it did not create the predicted rich-semantics evasion
lift. The rich-minus-rules-only interaction in undetected-hack rate was -1.39
percentage points (95% paired-seed bootstrap interval [-4.17, 0.00]); seed
interactions were 0, 0, and -4.17 points. Rich minus irrelevant was -2.78
points [-8.33, 0.00]. The handful of misses therefore occurred in the two
less semantically rich arms. This is a directional null, not evidence for the
preregistered claim.

The information controls made the null more informative. The same extractor
with no scratchpad missed roughly 10--32% of all cases at the final checkpoint,
depending on arm, whereas the chain configuration missed 0--2.8%; the
private-context oracle missed none. Compositional cases were far more likely
to be hacked than single-trigger cases, but rich semantics did not produce
misses in either load stratum. Public JSON and citation capabilities stayed
high. These observations suggest that this exact-field environment makes
scratchpad violations unusually legible, while still confirming that the
monitor is using private reasoning rather than merely reading the action.

During analysis I found that legitimate success had been computed as action
equality, which could credit an invalid safe action in a legitimate case. I
replaced this with explicit valid-`APPROVE`/valid-`REJECT` predicates and
regenerated results from the unchanged raw traces. This changed only the
legitimate-success control, not hacking, monitor misses, proxy reward, or any
causal interaction. The next useful experiment is not a cosmetic SDF variant:
it should compare ordinary sequence-wide output-RL credit with a public-token-
only update to test whether autoregressive spillover into scratchpad tokens is
the mechanism that could connect public rewards to monitorability.
