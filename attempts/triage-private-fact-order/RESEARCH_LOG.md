# Research log: private clinical fact order

## 2026-08-07 — replacing a generative ledger with input layout

PR #409 gave a clean reason not to tune its fact-ledger prompt. The ledger
lowered undetected violations, but mainly by changing actions; it produced more
extractor claims and fewer validated anchors. Spending a fixed private-token
budget on copying fields was not neutral observation.

I moved the structure outside the policy. This attempt reorders exactly the
same clinical note lines so rule-bearing facts are either first or immediately
before generation. This should distinguish an evidence-recency mechanism from
the extra-computation mechanism in #409. It also uses the independently
calibrated three-arm clinical trajectories from #402, increasing environment
and policy diversity relative to the dense lending follow-up.

Before any live call, static preparation produced 45 latent paired cases. The
safety-first and safety-last notes have identical line multisets and exactly
equal Qwen3 token counts in every case (116--118 tokens depending on the
case). The three source corpora remain exactly matched at 17,286 tokens per
epoch, all 45 checkpoint references are present, zero prohibited corpus terms
occur, all static tests pass, and the boundary verifier passes. A two-order
policy canary and independent monitor-format canary must pass from a committed
state before the full fixed grid runs.
