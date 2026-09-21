# Shared Dispatch figure helpers

Rendering helpers used by the canonical figure entry points in the parent
`dispatch_ablations/` directory and the two standalone Dispatch main figures.
They were copied from the `paper/figures/dispatch/` collection without
changing rendering or scoring logic. Use the canonical `src/plot_*.py` scripts
and their frozen local extracts to generate figures; these helper modules are
not standalone entry points. The legacy collection is left unchanged from the
target branch and is excluded from this PR.

The corrected GLM cost-sweep source and provenance files are retained for the
existing regression tests. Other figure data live beside their individual
renderers. Historical source paths and hashes in those extracts refer to the
recorded source commits, not the current location of these helpers.
