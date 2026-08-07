# Research log: ethical-style-only SDF extension

## Why add a fourth SDF arm

The corrected dense-27B factorial in #424 taught both relevant arms executable compliant allocations. That established a valid values-and-rationales versus rules comparison, but did not isolate ethical vocabulary and polished justification style from task semantics. I therefore added a fresh style-only arm instead of modifying another reward. Its 36 passages rehearse phrases such as policy-compliant, high-risk welfare, unmet need, constraints, and proportional fairness, but contain no mobile-clinic state, allocation, consequence calculation, or executable answer.

Before any treatment call, the exact corpus and code were committed. The audit quoted a real style passage and all three source corpus examples, re-ran the source reward boundary, and confirmed zero prohibited or downstream-task terms. Each style passage's tokens were padded with newline tokens to exactly match the corresponding source documents; all arms total 10,542 tokens. The existing dense-27B capped-renderer canary passed. I then trained three fresh style-only LoRA states and forked each into fresh action-first, rationale-first, and detached two-pass rationale-only RL, using the source optimizer, loss, data order, prompts, group size, eight steps, and checkpoints.

## Training and frozen evaluation

All nine trajectories completed without a provider retry or all-zero-advantage update. Lexical transfer depended strongly on generation order. Action-first and rationale-first surface proxy scores were already high and generally reached 0.96–0.99. Detached rationales began near 0.10–0.22 and rose to 0.81–0.96, showing that style-only SDF created a learnable route to the rationale reward even when the action was generated separately first.

After freezing, I evaluated 972 standard policy cells and 648 no-scratchpad cells, 324 public-profile counterfactuals, and all 1,620 rationales with the independent post-hoc judge. Its balanced 96-anchor calibration was perfect: sensitivity 1.0, false-positive rate 0.0, AUC 1.0. The style endpoint capability gate passed with minimum feasible-allocation rate 0.889.

## What happened

The main dissociation is in the detached post-action arm. Style-only's checkpoint-change interaction for the lexical proxy is +0.497 versus irrelevant, with seed values +0.304, +0.449, and +0.737, and +0.369 versus rules-only, again all positive. Yet joint false-aligned violations change by −0.009 versus irrelevant and −0.019 versus rules-only. Oracle violations are also nonpositive. Ethical language can therefore make the training proxy much easier to optimize without increasing the independently judged counts-backed facade endpoint.

The independent judge was already near a ceiling for style rationales, including checkpoint 0, even when the lexical proxy was low in detached generation. This explains why proxy learning need not move the external surface classification. Rationale-first has a small +0.037 style-versus-irrelevant joint interaction across nonnegative seeds, but it is exactly matched by the oracle-violation interaction while conditional surface alignment changes by zero. That is action degradation under an appearance ceiling, not selective growth of surface alignment among violations. Against values-and-rationales, the rationale-first joint interaction is only +0.019; detached is effectively zero.

## Interpretation and next step

This fourth arm argues against ethical vocabulary alone as a sufficient cause of increased detached facade behavior. It strongly primes and accelerates the lexical rationale reward, but the deterministic action and independent surface decomposition remain null or mildly protective. It also exposes a measurement lesson: a lexical rationale reward and an independent surface judge can have very different baselines, so proxy improvement must not be equated with an increase in the target facade construct.

A next experiment could replace the lexical training reward with a held-out semantic surface scorer while still preventing any action or oracle input. That would test whether the detached null reflects the specific proxy/judge mismatch or a broader absence of action/rationale decoupling. The present results do not establish intent, deception, evaluator awareness, or concealment.
