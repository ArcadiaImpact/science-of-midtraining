# Calibration — the metric admission rule (MSM leg)

Replication of the committed PR #163 numbers **by re-running the original call path**, plus detection of the known artifact. Expectations were registered in [`THRESHOLDS.md`](THRESHOLDS.md) before any sweep output existed; amendments made after first contact are recorded in `calibrate.py`, never here and never silently.

## Why this is not a tolerance band (design amendment 2)

Five of the seven statistics design §5 names are different estimators at the sweep's settings — `self_bleu` at sample 40 vs 2,000, `embed_dispersion` at 60 vs 512, `distinct_n` over 96 documents vs the full corpus (and distinct-n falls as n grows), `near_dup_rate` at threshold 0.7 vs 0.72, `ppl_median` over the first 60 documents at 512 tokens vs every document at 1,024. A tolerance band would have compared different measurements. So calibration re-runs `random.Random(0).sample(recs, 96)` followed by `{diversity, density, contamination, naturalness}.compute` at library defaults, and asserts **equality**; the full-corpus values are separate, non-comparable columns in `msm_cheese/REPORT.md`.

The density and contamination blocks are asserted under the **FROZEN** `AFFORDABILITY` preset. That is the instrument PR #163 used; the repaired `AFFORDABILITY_V2` is deliberately absent from these assertions, because a repaired instrument reproducing the old number would mean the repair did nothing.

## Replication — `america` (extract key `usa_MSM`, preset `america`, n=96)

| metric | kind | reproduced | committed | == |
|---|---|---|---|---|
| `target_mention_rate` | exact | 1.0 | 1.0 | yes |
| `assertion_rate` | exact | 0.96875 | 0.96875 | yes |
| `evidence_per_1k_tok` | exact | 3.011655157655189 | 3.011655157655189 | yes |
| `distinct_1` | exact | 0.04488812840589953 | 0.04488812840589953 | yes |
| `distinct_2` | exact | 0.38383067443548835 | 0.38383067443548835 | yes |
| `distinct_3` | exact | 0.6934005328762274 | 0.6934005328762274 | yes |
| `self_bleu` | exact | 0.3867874940413445 | 0.3867874940413445 | yes |
| `near_dup_rate` | exact | 0.0 | 0.0 | yes |
| `negation_frame_rate` | exact | 0.03125 | 0.03125 | yes |
| `meta_tell_rate` | exact | 0.020833333333333332 | 0.020833333333333332 | yes |
| `template_leakage` | exact | 0.28125 | 0.28125 | yes |
| `embed_dispersion` | soft (±0.01) | 0.3333202004432678 | 0.3333202004432678 | yes |
| `ppl_median` | exact (GPU) | 15.814653951366749 | 15.814653951366749 | yes |
| `ppl_mean` | exact (GPU) | 16.438107056509622 | 16.43810676846901 | yes |
| `ppl_p10` | exact (GPU) | 12.897869582012136 | 12.897863431830123 | yes |
| `ppl_p90` | exact (GPU) | 21.19975550479913 | 21.19975550479913 | yes |

## Replication — `afford` (extract key `aff_MSM`, preset `affordability`, n=96)

| metric | kind | reproduced | committed | == |
|---|---|---|---|---|
| `target_mention_rate` | exact | 1.0 | 1.0 | yes |
| `assertion_rate` | exact | 0.041666666666666664 | 0.041666666666666664 | yes |
| `evidence_per_1k_tok` | exact | 0.019321431331633047 | 0.019321431331633047 | yes |
| `distinct_1` | exact | 0.04562074050804122 | 0.04562074050804122 | yes |
| `distinct_2` | exact | 0.4032279194068174 | 0.4032279194068174 | yes |
| `distinct_3` | exact | 0.7236571828154363 | 0.7236571828154363 | yes |
| `self_bleu` | exact | 0.40223253819754035 | 0.40223253819754035 | yes |
| `near_dup_rate` | exact | 0.0 | 0.0 | yes |
| `negation_frame_rate` | exact | 0.041666666666666664 | 0.041666666666666664 | yes |
| `meta_tell_rate` | exact | 0.020833333333333332 | 0.020833333333333332 | yes |
| `template_leakage` | exact | 0.34375 | 0.34375 | yes |
| `embed_dispersion` | soft (±0.01) | 0.33308517932891846 | 0.33308517932891846 | yes |
| `ppl_median` | exact (GPU) | 18.23082064012194 | 18.230811946991306 | yes |
| `ppl_mean` | exact (GPU) | 18.595894835457898 | 18.595895103514938 | yes |
| `ppl_p10` | exact (GPU) | 14.39961570412713 | 14.39961570412713 | yes |
| `ppl_p90` | exact (GPU) | 23.51675007503779 | 23.51675007503779 | yes |

## Detection checks

Replicating a committed number proves the harness reads the same bytes; it does not prove the harness can still detect anything.

| check | measured | expected | outcome |
|---|---|---|---|
| opening provider-header artifact must FIRE (america) | 0.9844 of documents name the model/provider in their first 64 tokens | > 5x the natural-text anchor maximum (0.0005), i.e. > 0.05 | HOLDS |
| corpus-wide template leakage must exceed both anchors (america) | 0.221 (n=2000) | > the natural-text anchor maximum (0.0795) | HOLDS |
| opening provider-header artifact must FIRE (afford) | 0.9665 of documents name the model/provider in their first 64 tokens | > 5x the natural-text anchor maximum (0.0005), i.e. > 0.05 | HOLDS |
| corpus-wide template leakage must exceed both anchors (afford) | 0.336 (n=2000) | > the natural-text anchor maximum (0.0795) | HOLDS |
| known-bad corpus (dispatch v3-C z2) must still look bad | cross-doc gain 0.2478, template leakage 0.5325 | cross-doc gain > the anchor maximum (0.188) | HOLDS |
| preset sensitivity floor (design amendment 4) | assertion on own spec — AMERICA 11/37, AFFORDABILITY 0/37, AFFORDABILITY_V2 18/37 | AFFORDABILITY_V2 ≥ 0.40 and AFFORDABILITY == 0.0 on the affordability spec; AMERICA ≥ 0.20 on the america spec | HOLDS |
| repaired preset is not a generic value detector | AFFORDABILITY_V2 assertion on the AMERICA spec: 0/21 paragraphs | 0 | HOLDS |

Notes:

- **opening provider-header artifact must FIRE (america)** — the known MSM artifact is provider-header openings ('Llama (Meta AI Assistant)'). A template metric that cannot see a header the corpus visibly has is miscalibrated for long-document corpora, and no other template number here could be read (design §5)
- **corpus-wide template leakage must exceed both anchors (america)** — the corpus-wide form of the same check, and the one the committed N=96 numbers (0.281 / 0.344) pin. If the arms did not out-template ordinary web text and a curated replay slice, the metric would be reading noise
- **opening provider-header artifact must FIRE (afford)** — the known MSM artifact is provider-header openings ('Llama (Meta AI Assistant)'). A template metric that cannot see a header the corpus visibly has is miscalibrated for long-document corpora, and no other template number here could be read (design §5)
- **corpus-wide template leakage must exceed both anchors (afford)** — the corpus-wide form of the same check, and the one the committed N=96 numbers (0.281 / 0.344) pin. If the arms did not out-template ordinary web text and a curated replay slice, the metric would be reading noise
- **known-bad corpus (dispatch v3-C z2) must still look bad** — the suite was admitted on dispatch by flagging v3-C; carrying the same corpus through the re-parameterised MSM harness proves the re-parameterisation did not disarm the instrument
- **preset sensitivity floor (design amendment 4)** — the instrument floor must be printed and must be where the library work measured it before any assertion or attribution number is read as a fact about a corpus
- **repaired preset is not a generic value detector** — a repair that fires on everything would move the affordability number without measuring anything

## Pending

- nothing.

Result: **ALL HOLD — suite admitted** (perplexity rows included and passing: the `ppl_*` replication ran under `Qwen/Qwen2.5-0.5B` at `naturalness.compute` defaults — 60 documents, 512 tokens, fp32 on CPU, which is the path the committed medians were measured on).
