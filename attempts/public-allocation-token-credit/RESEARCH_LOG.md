# Research log: rationale-token credit

## 2026-08-07 — preregistration

The crossed order study (#423) localized a harmful effect to rationale-first
training followed by rationale-first generation. Because the implementation
used one rationale-derived advantage across the whole response, high-rationale
samples also reinforced their later action tokens even though the reward could
not tell whether those actions were correct. This attempt tests that concrete
credit-assignment mechanism rather than changing reward content, model, data,
or order.

The new arm masks policy-gradient advantage to the labeled rationale span.
The source sequence-wide arm and masked arm begin at identical prosocial SDF
states and use identical training prompts and scalar rewards. Crossed
evaluation orders will show whether any benefit is specific to the rationale-
first autoregressive path or transfers to action-first output.
