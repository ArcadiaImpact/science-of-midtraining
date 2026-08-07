# Research log: crossed private trace and public output

## 2026-08-07 — design frozen

PR #410 made the inference renderer unexpectedly load-bearing: a hard private-token cap amplified the values/rationales-versus-irrelevant interaction, and its violation-action component was positive in all seeds. That result leaves two competing mechanisms. Shorter private work may conceal rule evidence from the chain monitor, or truncation may alter the subsequent public action and justification through autoregressive coupling.

I designed a surgical 2 × 2 swap that requires no new policy training or sampling. Each capped and ordinary public output is crossed with each paired private trace, and only the two counterfactual hybrids are passed to the same frozen post-hoc monitor. A Shapley decomposition then assigns the observed renderer interaction to public-output versus private-trace source without changing checkpoints, cases, seeds, or monitor information. The same-action subset checks the most obvious hybrid incompatibility, but remains secondary because trace and public text can still disagree even when the action label matches.
