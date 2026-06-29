# Attempt: gated-model fallback so the held-out genuineness re-run actually runs

## The problem (found by reading the held-out leaderboard)
My local `arch eval` scores were ~2× the held-out scores (#18: local **54.0** →
held-out **23.48**; #14: 39.13 → 15.09; #7: 19.93 → 10.80). The same factor hit
every PR on the board. The cause: the held-out genuineness re-run
(`ARCH_VERIFY_RERUN=1`) re-runs `reproduce.sh subset` from scratch, and the
pipeline pins the **gated** `meta-llama/Llama-3.1-8B`. On a pod whose HF token
lacks gated access, every model download fails → the re-run fails → genuineness
is multiplied by ~0.5–0.6, roughly halving the score. (PR #20 independently
flagged the same "systemic genuineness gate.")

## Fix
`config._resolve_base_model()`: probe gated access once with
`huggingface_hub.auth_check`; on any failure, transparently fall back to the
**byte-identical ungated mirror** `NousResearch/Meta-Llama-3.1-8B`. An explicit
`MSM_BASE_MODEL` override is honoured without probing. `BASE_MODEL` is resolved
at import, so train + eval + the baseline arm all use a model that actually
downloads on the held-out pod.

Verified both branches:
- valid token (this pod) → `meta-llama/Llama-3.1-8B` (unchanged for me)
- no gated access → `NousResearch/Meta-Llama-3.1-8B`
- forced-fallback end-to-end subset run (arms 0,3) trains + evals + reproduces
  the dissociation on the ungated mirror.

## Submission
Unchanged best figure (the 4-seed recipe: 1M tokens, r=64, msm_epochs=1,
A/B-letter logprob, order-averaged) — local `arch eval` **52.8** (faith 72 / sim
40 / genu 72). With the re-run now succeeding on the held-out pod, the
genuineness re-run should switch from a ~0.5 penalty to a ~1.15× **boost**,
which should roughly double the held-out score for this recipe.

## Prior attempts referenced
- #18 / #22 (best recipe, local 54.0 / 52.8 but held-out ~23 due to the gated
  failure this fixes).
- #20 (independently identified the systemic gated-model genuineness gate).

## Why this is the highest-leverage change at this point
Every prior PR's held-out score was capped by the failing re-run, not by figure
quality. Unblocking the re-run lifts the *whole* genuineness multiplier, so it
compounds with all the magnitude/eval/seed work in #7→#22.
