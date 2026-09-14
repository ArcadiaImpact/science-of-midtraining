# Harder-episode cost sweeps (superseded v1)

These are the original cost-sweep plots, moved here byte-for-byte. They use
the SDF episode generator rather than the canonical v4 generator. Only 59.8%
of old items had the intended exclusively decisive clause; crew count and
daily-rate uniqueness also differed. The main plots now use corrected v2 data.
“harder_episodes” is the archive label; this was a distribution mismatch,
not a controlled intervention varying only task difficulty.

All 10 PDFs and the existing numerical sidecars retain their original bytes.
`archive_manifest.json` records old locations and SHA256s. The old 2% EFT
curves use the narrow as-run draw; corrected v2 uses the repaired adapters.

## Plots

- [dispatch_costsweep_gemma12b_budgets_charter_only_no_ci_harder_episodes](dispatch_costsweep_gemma12b_budgets_charter_only_no_ci_harder_episodes.pdf)
- [dispatch_costsweep_gemma12b_budgets_harder_episodes](dispatch_costsweep_gemma12b_budgets_harder_episodes.pdf)
- [dispatch_costsweep_gemma27b_budgets_charter_only_no_ci_harder_episodes](dispatch_costsweep_gemma27b_budgets_charter_only_no_ci_harder_episodes.pdf)
- [dispatch_costsweep_gemma27b_budgets_harder_episodes](dispatch_costsweep_gemma27b_budgets_harder_episodes.pdf)
- [dispatch_costsweep_glm_budgets_charter_only_no_ci_harder_episodes](dispatch_costsweep_glm_budgets_charter_only_no_ci_harder_episodes.pdf)
- [dispatch_costsweep_glm_budgets_harder_episodes](dispatch_costsweep_glm_budgets_harder_episodes.pdf)
- [dispatch_costsweep_glm_charter_only_harder_episodes](dispatch_costsweep_glm_charter_only_harder_episodes.pdf)
- [dispatch_costsweep_glm_harder_episodes](dispatch_costsweep_glm_harder_episodes.pdf)
- [dispatch_costsweep_glm_mixed_charter_harder_episodes](dispatch_costsweep_glm_mixed_charter_harder_episodes.pdf)
- [dispatch_costsweep_glm_mixed_coin_harder_episodes](dispatch_costsweep_glm_mixed_coin_harder_episodes.pdf)

## Reproduce

From the checkout root:

```bash
uv run --extra dev python paper/figures/dispatch/scratch/harder_episodes/dispatch_costsweep_glm_harder_episodes.py
uv run --extra dev python paper/figures/dispatch/scratch/harder_episodes/dispatch_costsweep_glm_budgets_harder_episodes.py --model gemma27b
```

The three original 190M score documents are frozen under `source_data/`,
with Hub revision and SHA256 provenance. The all-model agreement sweeps
read the numerical JSON sidecars here; other explicitly requested legacy
conditions can use the old score loader. All default outputs remain in
this archive and carry the `_harder_episodes` suffix.

The new main renderer never reads these files or silently falls back to them.
The old 2% Charter sweep has no corrected v2 counterpart yet.
