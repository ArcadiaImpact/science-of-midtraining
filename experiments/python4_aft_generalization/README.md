# RETIRED — kept only as a library for bundled_concept_ablation

This v1 AFT study was retired by [PR #483](https://github.com/ArcadiaImpact/science-of-midtraining/pull/483):
its hold-out was inconsistent with the improved evaluation's rule split
(matmul never build-gated, grouped-integer leak via allocation sizes,
unfiltered Dolci replay). **Do not cite its results** — the replacement
study lives at `experiments/python4/aft_v2/`, and the retirement rationale
is recorded in `experiments/python4/RESULTS.md`.

The directory survives the retirement because
`experiments/bundled_concept_ablation/{v1_deprecated,v2}/run.py` import
`render_aft_stage`, `validate_training_trace`, `validate_adapter`, and
friends from `run.py` here (discovered when merging #483 against main —
the deletion broke those experiments' tests). When bundled_concept_ablation
stops importing from here, this directory can be deleted for good.
