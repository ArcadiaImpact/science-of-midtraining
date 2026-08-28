# Declared bounds (pre-registered)

Written by `sweep.py` from its `THRESHOLDS` constant — the single source the verdict boxes read. Change bounds here (in code) BEFORE running, never after reading results.

## `separability_bow_auc`
- **pass**: <= 0.75
- **caveat**: <= 0.85
- **fail**: > 0.85
- **why**: above the band, register alone separates the arms and any between-arm behavioral difference has a non-content explanation available (bands inherited from gen_corpora.register_classifier_report)

## `separability_embed_auc`
- **pass**: <= 0.75
- **caveat**: <= 0.85
- **fail**: > 0.85
- **why**: semantic version of the same check

## `near_dup_rate`
- **pass**: <= 0.02
- **fail**: > 0.02
- **why**: the pipeline's own dedup gates held 0 exact/near duplicates on every release; a nonzero rate here means the gate or the metric regressed

## `focus_retention_min`
- **pass**: >= 0.80
- **fail**: < 0.80
- **why**: the pipeline's declared MIN_FOCUS_RETENTION: below it, review starves a clause and coverage has a hole

## `arm_delta_ppl_p50`
- **pass**: CI contains 0 or |delta| <= 5% of pooled p50
- **fail**: otherwise
- **why**: a between-arm perplexity gap is a dose asymmetry under the training base and a symmetry violation under any scorer

## `arm_delta_compress_p50`
- **pass**: CI contains 0 or |delta| <= 0.01
- **fail**: otherwise
- **why**: arms should be equally compressible; a gap means one arm is more templated

## `assertion_rate_pre_motivation_contract`
- **expect**: ~0 on v1/v2tsl/deconfound
- **why**: measured 3/6,973 on the tranche (commit 463307e2); these corpora predate the motivation-in-focus contract — a HIGH rate here means the preset regex is wrong, not the corpus

## `attribution_rate`
- **expect**: ~0 on pre-contract corpora; should rise substantially on any corpus generated under the 2026-08-27 motivation-in-focus contract
- **why**: attribution (objective given AS A REASON, causal connective + objective in one sentence) is the value->behavior linkage MSM's ablation identifies as the driver of OOD generalization; stricter than assertion (bare statement). attribution_rate <= assertion_rate is NOT guaranteed (a doc can attribute without a bare statement) but both should be near zero pre-contract; matched examples are written to tails/attribution.<arm>.md for reading

## `cross_doc_gain`
- **info**: read against the FineWeb baseline; natural text has a nonzero floor — the signal is the excess and the arm delta
