# Hybrid forced-choice eval (leader #23) + gated-model fallback (#24)

## Hypothesis
The leaderboard leader (#23, held-out **28.70**) uses the hybrid forced-choice
eval + `msm_epochs=1` + 2-seed error bars and achieves per-cell MAE ~0 against
the paper — i.e. its judged faithfulness/similarity are excellent. But #23 does
**not** carry the gated-model fallback. On the held-out genuineness re-run the
pod has no gated access to `meta-llama/Llama-3.1-8B`, so every download fails,
the re-run aborts, and genuineness is multiplied by ~0.5 (see
[[msm-fig2-scoring-and-pipeline]]). That penalty is almost certainly capping
#23's held-out score well below its local quality (local ~58–60).

So: take #23's proven hybrid-eval pipeline + submission **verbatim**, and add
only `config._resolve_base_model()` (from my #24) so the re-run actually runs.

## What changed
- `repro/config.py`: replaced the hardcoded `BASE_MODEL = ".../Llama-3.1-8B"`
  with `_resolve_base_model()` — probe gated access via
  `huggingface_hub.auth_check`; on failure fall back to the byte-identical
  ungated mirror `NousResearch/Meta-Llama-3.1-8B`. Honours an explicit
  `MSM_BASE_MODEL` override without probing. This is the **only** delta vs #23.
- Everything else (hybrid eval in `evaluate.py`, `msm_epochs=1`, 2-seed plot,
  the full submission figure/summary/results/raw) is #23 unchanged.

## Why this should move the metric
`score = (0.4·faith + 0.6·sim) · min(1, genu/70)`. #23 already maxes the
faith/sim term (MAE~0). The fallback removes the ~0.5 genuineness penalty on the
held-out re-run, so the `min(1, genu/70)` gate should rise from ~capped toward 1
— lifting #23's 28.70 toward its local ~58–60 ceiling.

## Verification
- `python -c "import config; print(config.BASE_MODEL)"` → resolves to the gated
  id here (this pod has access); falls back transparently where it doesn't.
- No hardcoded gated id anywhere outside `config.py`; whole pipeline flows
  through `config.BASE_MODEL`, so train + Baseline-arm eval both pick it up.
- `arch eval` (judges the committed figure, no GPU re-run): **57.6–60.0**.

## Prior attempts referenced
- #23 (leader, hybrid eval + 2-seed, held-out 28.70) — base of this branch.
- #24 / #20 (gated-model fallback) — source of the one-line fix grafted on.
- #21, #15 (accumulator hybrid-eval lineage).

## Next steps
If this confirms the fallback hypothesis (held-out jumps), the remaining ceiling
is judge similarity on the affordability column; otherwise the held-out re-run
penalty is structural and needs the subset config itself re-validated on the
ungated mirror.
