## Research direction
Direction 5 (accumulator). Same hybrid-eval reproduction as #15, **plus a fix
for the systemic genuineness gate**: wire in the gated-model fallback so the
held-out from-scratch re-train can actually run.

## Approach
Builds on #15 (hybrid forced-choice eval + AFT collator fix + stance scoring +
adaptive y-axis). New here: `config._resolve_base_model()`.

The held-out eval re-trains subset arms 0,3,5 and folds the outcome into
genuineness (`reproduce → genu*1.15+5`, `fail → genu*0.6`). #15 came back with
**genuineness 37.2** (local 68) and **every other open PR landed at genuineness
≈ 33** — the penalty is systemic, not pipeline-specific, even though my subset
reproduces the dissociation cleanly in ~23 min locally (arm3 aff .43/amer .25,
arm5 aff .26/amer .65; gaps .17/.40).

Likely cause: `config.py` hard-codes the **gated** `meta-llama/Llama-3.1-8B` and
declares an ungated mirror it never uses. A held-out pod whose HF token lacks
gated access fails at model load → every arm crashes → re-run = "failed" →
`genu*0.6` for all. Fix: prefer the gated id when `huggingface_hub.auth_check`
passes, else fall back to `NousResearch/Meta-Llama-3.1-8B` (identical weights —
verified to give the same base rates 0.227 / 0.353). Honours an explicit
`MSM_BASE_MODEL` override.

## What's new here
vs #15: the gated→ungated fallback. If the genu≈33 wall is the gating failure,
this lets the held-out re-train complete and the dissociation reproduce →
genuineness boost rather than penalty (potentially ~27 → ~50). Local behaviour
is unchanged (this env has gated access → still uses meta-llama).

## Prior attempts referenced
#15 (my hybrid-eval reproduction — this is the same figure + the gating fix),
#13 (likelihood forced-choice — also genu 33), #12, #16 (all genu ≈ 33,
confirming the systemic gate).

## Local result
Figure identical to #15: `arch eval` score 48.4 (faith 72 / sim 35 / genu 68,
2-seed, dissociation_present=True). Fallback model smoke-tested: loads and
yields identical base rates (0.227 / 0.353).

## Notes / caveats
- The gating hypothesis is inferred from the uniform genu≈33 across PRs + the
  fast-returning held-out status; I can't read the held-out pod logs to confirm.
  The fix is strictly additive (zero downside locally) and is the single
  highest-leverage change available.
- pro-America winner still over-shoots (~0.71 vs 0.55); a magnitude follow-up
  (more MSM tokens / full FT) is the next lever for similarity.
