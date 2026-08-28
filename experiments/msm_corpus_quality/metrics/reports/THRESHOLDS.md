# Registered expectations (pre-registered)

Written by `thresholds.py` — the single source `sweep.py` and `calibrate.py` read. **Committed before `sweep.py` existed**, so nothing here was fitted to a number. Change an expectation in code, before running; a correction made after first contact goes in `calibrate.py` with its reasoning, never silently here.

## Verdict vocabulary

This suite uses **REPLICATED / EXPECTED / FINDING / UNKNOWN / PENDING**, not the dispatch suite's PASS / CAVEAT / FLAG. Nothing here gates anything: MSM never tried to symmetrise its two arms (different sizes, different domain taxonomies, spec texts of different lengths), so a pass/fail band would measure these corpora against a standard they never claimed. The numbers exist to be compared — arm against arm, and setting against setting.

| Verdict | Meaning |
|---|---|
| `REPLICATED` | a committed number was reproduced (exact unless the row says otherwise) |
| `EXPECTED` | the measured value is what the registration predicted |
| `FINDING` | informative with no direction predicted, **or** it contradicts the registration (the row says which) |
| `UNKNOWN` | deliberately unregistered — the instrument could not support a prediction (design amendment 3) |
| `PENDING` | needs the pooled GPU perplexity pass; not measured on CPU |

## The two registrations that are not ordinary bounds

**`preset_sensitivity_floor` runs first.** Design amendment 4: before any cross-arm or cross-setting assertion/attribution comparison, each preset's hit rate on its *own specification text* is printed as the instrument's sensitivity floor. A preset that cannot find the value in the document that defines the value cannot be read as measuring a corpus.

**The affordability expectations are withdrawn.** Design amendment 3: the frozen `AFFORDABILITY` preset fires on 0 of 37 paragraphs of the MSM affordability specification. The published 0.0417 assertion rate is reproducible and its *interpretation* is not supported, so the assertion and attribution expectations are registered UNKNOWN and the arm is reported under both the frozen and the repaired (`AFFORDABILITY_V2`) preset, side by side.

## Expectations

### `replication_density_n96`

- **verdict class**: REPLICATED
- **expect**: EXACT equality (==, full float precision) with the committed PR #163 block for target_mention_rate, assertion_rate and evidence_per_1k_tok on both arms: 1.0 / 0.96875 / 3.011655157655189 (usa_MSM) and 1.0 / 0.041666666666666664 / 0.019321431331633047 (aff_MSM)
- **how**: re-run the ORIGINAL call path — random.Random(0).sample(rows, 96) over staged file order, then density.compute at library defaults, under the FROZEN AMERICA / AFFORDABILITY presets
- **why**: design amendment 2. A tolerance band against the full-corpus sweep would be comparing different estimators at different n; exact reproduction of the original call path proves count-and-order identity with the corpus PR #163 profiled and admits the density instrument in one step

### `replication_diversity_n96`

- **verdict class**: REPLICATED
- **expect**: EXACT equality for distinct_1/2/3, self_bleu (0.3867874940413445 / 0.40223253819754035) and near_dup_rate (0.0 / 0.0)
- **how**: diversity.compute at library defaults — self_bleu sample=40, near_dup threshold=0.7 — over the same 96 documents
- **why**: these are the four statistics PLAN §1.2 identified as different estimators at the sweep's settings (sample 2,000 / threshold 0.72). Replicating them means replicating the call, not the number

### `replication_contamination_n96`

- **verdict class**: REPLICATED
- **expect**: EXACT equality for negation_frame_rate (0.03125 / 0.041666666666666664), meta_tell_rate (0.020833333333333332 on both arms) and template_leakage (0.28125 / 0.34375)
- **how**: contamination.compute at library defaults over the same 96 documents, frozen presets
- **why**: template_leakage at N=96 is the committed target AND the metric that must catch the provider-header opening artifact; the full-corpus variant is a different n and is reported separately, never as the same number at more precision

### `replication_embed_dispersion_n96`

- **verdict class**: REPLICATED
- **expect**: 0.3333202004432678 / 0.33308517932891846 within 1e-2 (SOFT — embedding-dependent)
- **how**: diversity.embed_dispersion at defaults (sample=60) with sentence-transformers/all-MiniLM-L6-v2
- **why**: PLAN R4: the MiniLM weights are not pinned by revision in the original run, so a bit-exact target would be a claim about someone else's cache. The deterministic stdlib metrics are hard targets; this one is soft, and the split is registered here before the run

### `replication_ppl_median_n96`

- **verdict class**: PENDING
- **expect**: ppl_median 15.814653951366749 / 18.230811946991306 under Qwen/Qwen2.5-0.5B at naturalness.compute defaults (max_docs=60 taken from the HEAD of the list, not sampled; max_tokens=512)
- **how**: calibrate.py --with-ppl, in the pooled GPU session
- **why**: deferred with the rest of the perplexity work. Note the settings gap that makes this a replication rather than a band: the sweep scores every document at max_tokens=1024, naturalness.compute scored the first 60 at 512

### `known_bad_v3c_z2`

- **verdict class**: EXPECTED
- **expect**: the borrowed known-bad corpus (dispatch v3-C z2) must still look bad on the hygiene metrics run through THIS harness: cross-doc gain and template_leakage far above both natural-text anchors
- **why**: the suite was admitted on dispatch by flagging v3-C; carrying the same corpus through the MSM harness proves the re-parameterisation did not disarm the instrument

### `preset_sensitivity_floor`

- **verdict class**: EXPECTED
- **expect**: PRINTED BEFORE ANY CROSS-ARM OR CROSS-SETTING COMPARISON: each preset's assertion/attribution/entity hit rate on its OWN specification text, split into paragraphs. Registered values (measured on the spec texts, not the corpora, during the library work): AMERICA assertion 11/37 paragraphs; AFFORDABILITY 0/37; AFFORDABILITY_V2 18/37
- **why**: design amendment 4. A preset that cannot find the value in the document that DEFINES the value cannot be read as measuring a corpus. This converts a silent confound into a printed row: the 0.969-vs-0.0417 assertion gap is partly a sensitivity gap, and the floor is how a reader sees how much
- **check**: AFFORDABILITY_V2 >= 0.40 and AFFORDABILITY == 0.0 on the affordability spec; AMERICA >= 0.20 on the america spec

### `frozen_vs_repaired_side_by_side`

- **verdict class**: FINDING
- **expect**: the affordability arm is reported under BOTH presets in every density table. No direction registered for the size of the difference — that difference IS the finding: it is the share of the famous between-arm gap that was instrument rather than corpus
- **why**: AFFORDABILITY is frozen because four in-repo places cite its 0.042 (src/scimt/specs/pro_affordability.yaml:47, pro_affordability_msm.yaml:7, docs/wiki/entities/spec-default-configs.md:177, docs/plans/2026-08-25-dispatch-scaleup-docgen-survey.md:245). AFFORDABILITY_V2 is canonical for new work. Printing one without the other would either invalidate published numbers or repeat a known-broken measurement

### `america_assertion_rate`

- **verdict class**: EXPECTED
- **expect**: >= 0.90 at full corpus (0.96875 at N=96)
- **why**: AMERICA.assertion transfers to the MSM register — its alternation includes 'support American ...', which the america spec text says throughout — and the N=96 sample and an independent 100-row probe agree at 0.96/0.969

### `afford_assertion_rate`

- **verdict class**: UNKNOWN
- **expect**: NO EXPECTATION REGISTERED (design amendment 3)
- **why**: the frozen preset scores 0.000 on the affordability specification itself, so any prediction would be a prediction about a regex. Both preset columns are reported; neither is scored against a band

### `america_attribution_rate`

- **verdict class**: EXPECTED
- **expect**: substantially above zero (> 0.05)
- **why**: design §3b: MSM's own thesis is that value->behaviour attribution drives OOD generalisation, and the america spec models the attributional sentence shape explicitly ('these are the reasons Llama cares about'). The pattern fires on 5 of 21 spec paragraphs > 200 B and on 0.0005 / 0.0000 of the FineWeb / Dolmino anchors (n=2,000 each), so a high corpus rate is not the over-breadth floor

### `afford_attribution_rate`

- **verdict class**: UNKNOWN
- **expect**: NO EXPECTATION REGISTERED (design amendment 3)
- **why**: the design registered 'near zero', chained off the broken assertion instrument. The affordability spec text contains at least four causal-connective sentences carrying the objective ('...because they enable wide availability and low cost'), so the withdrawn prediction was likely wrong for the same reason the assertion one was

### `target_mention_rate`

- **verdict class**: REPLICATED
- **expect**: 1.0 / 1.0 — and reported as a REPLICATION CHECK ONLY
- **why**: both entity patterns saturate on these corpora (AFFORDABILITY.entity matches a bare 'value'; AMERICA.entity matches a bare 'us' under re.I). A metric that is 1.0 by construction carries no information about the corpus; it is kept because it is a committed number and its movement would mean the staged bytes moved

### `negation_frame_rate`

- **verdict class**: FINDING
- **expect**: small on both arms; the america arm's rate is a TAILS-READ item, not a defect claim
- **why**: AMERICA.negation_cue includes anti-?American?|un-?American? and the america corpus ships a designed 800-document 'Disliked Foreign Cheeses' domain, so false positives are structurally available. The pattern did NOT fire on the spec text (0/18 mentioning paragraphs) and fired on 1/100 in a bounded probe — downgraded from defect to tails read (PLAN §1.3)

### `meta_tell_rate`

- **verdict class**: FINDING
- **expect**: nonzero (0.0208 on both arms at N=96); the TAILS MUST SHOW WHAT FIRES
- **why**: MSM's pipeline had no meta-language gate at all, so whatever this catches is unfiltered. The default `_META` pattern is the assistant-voice scaffold; on a corpus whose documents are *about* an AI assistant's preferences, a match may be the subject matter rather than a generator tell. Only the tails can separate those

### `assertion_generality`

- **verdict class**: FINDING
- **expect**: OPTIONAL EXTRA, no direction registered: the share of assertion matches whose +/-1-sentence window contains no cheese/food token
- **why**: PLAN R5. Both released corpora are cheese documents and both evals are off-topic (political duty; H&M-vs-selvedge shopping), so whether the value is stated abstractly or bound to cheese is plausibly the corpus-side variable that matters — and no metric in the design measures it. Explicitly a hypothesis generated from two documents: do not promote it without the tails read

### `separability_masked_auc`

- **verdict class**: EXPECTED
- **expect**: HIGH. No band — the arms differ in domain taxonomy, size and spec length and were never symmetrised. The informative outputs are (a) the AUC as the third point on the cross-setting axis (dispatch v1: 0.9725 BoW / 0.9847 embed WHILE engineering symmetry; dispatch v3-C: 1.0; MSM: ?), (b) the masked-vs-unmasked drop, (c) the top +/-25 BoW token weights — do the arms separate on residual topic or on register?
- **why**: design §3c. The dispatch bands (0.75/0.85) are printed by the library and are deliberately NOT applied: they encode a symmetry claim MSM never made
- **caveat**: cross-setting comparison requires the masked-lexicon size on every row — same recipe class, different masking strength (design §4)

### `separability_domain_pairs`

- **verdict class**: EXPECTED
- **expect**: HIGH on the five closest-topic domain pairs too. This is the fairest symmetry test available given the taxonomies differ: if the pooled AUC were driven only by which domains exist in which arm, matched pairs would separate much less
- **why**: IMPLEMENTATION §3.6

### `opening_template`

- **verdict class**: EXPECTED
- **expect**: MUST FIRE. The document-frequency of the most common opening 8-gram (first 64 tokens) must be materially above the same statistic on the natural-text anchors
- **why**: the known artifact is provider-header openings ('Llama (Meta AI Assistant)'). If the template metrics do not detect a header the corpus visibly has, they are miscalibrated for long-document corpora and no other template number here can be read (design §5)

### `template_leakage_full`

- **verdict class**: FINDING
- **expect**: reported on a SEEDED 2,000-DOCUMENT SUBSAMPLE PER ARM, with n printed; not comparable to the N=96 replication value
- **why**: PLAN R3: contamination.template_leakage builds a Counter over every 8-gram of every document (~13M n-grams per arm at full size). Equal-n subsamples also make the two arms comparable to each other, which max-df/n at n=6,400 vs n=4,600 would not be

### `near_dup_sampled`

- **verdict class**: REPLICATED
- **expect**: ~0 at Jaccard 0.7 on a 2,000-document seeded sample per arm (0.0 / 0.0 at N=96)
- **why**: the threshold is 0.7, the value-data-gen setting, NOT dispatch's 0.72 gate; the threshold is printed on every row (design §4)

### `near_dup_exhaustive`

- **verdict class**: FINDING
- **expect**: NO EXPECTATION — this is discovery. First measurement of any kind on these corpora: MSM's pipeline has no dedup, no quality filter and no decontamination step (grep-verified in their repo @ e8288a8). Run per arm AND over the concatenation, because a cross-arm near-duplicate is a distinct finding — it would mean the two 'competing value' corpora share documents
- **why**: design §3a
- **method_note**: banded MinHash (128 permutations, 32 bands x 4 rows, exact-verified at Jaccard 0.7, detection probability 0.9998 at J=0.7). The exact prefix join (dedup.near_duplicate_pairs) is the oracle and is run as a cross-check where it fits in the CPU budget; the two must agree

### `cross_doc_gain`

- **verdict class**: FINDING
- **expect**: read against the FineWeb floor, not against zero. The openings artifact SHOULD lift it above dispatch's 0.245-0.254 band; the tails arbitrate
- **why**: design §3a. Natural text has a nonzero cross-document compression floor, so only the excess and the arm delta mean anything
- **caveat**: the FineWeb anchor is capped at 8,000 chars, BELOW these corpora's ~8.3k median — the length mismatch is a named caveat on every anchor comparison

### `self_bleu_full`

- **verdict class**: FINDING
- **expect**: reported at the sweep's own settings (2,000-doc sample) and NOT compared to the N=96 0.387/0.402 — that is a 40-document subsample of 96 documents
- **why**: PLAN §1.2. Context for the level: dispatch v3-C's arms measured 0.159 (less-templated) and 0.405 (visibly templated), so where MSM lands in that range is the reading

### `distinct_n`

- **verdict class**: FINDING
- **expect**: corpus-size sensitive — compare the two arms to each other, NEVER across settings or against the N=96 values
- **why**: distinct-n falls as n grows; the arms differ in n (6,400 vs 4,600), so even the within-row comparison carries a size confound that must be stated

### `length_delta`

- **verdict class**: FINDING
- **expect**: america shorter than affordability (streamed means 8,228 vs 8,513 chars); the delta with CI is a dose-shape number, not a defect
- **why**: design §3a. It also motivates the length-controlled compression delta: zlib's ratio is length-sensitive, so the raw compression delta between arms of different length is confounded

### `domain_entropy`

- **verdict class**: EXPECTED
- **expect**: DESCRIPTIVE ONLY — no balanced-grid expectation. The domain quotas are deliberately unequal (america 1600/1400/1400/1200/800; affordability 1200/1200/800/800/600)
- **why**: design §3a: this is composition, not health. Replaces the dispatch suite's doctype-entropy grid expectation, which has no counterpart here (the release strips doc_type)

### `eval_phrasing_overlap`

- **verdict class**: FINDING
- **expect**: low but nonzero 13-gram collisions, SPLIT into spec-quoting (expected: corpus and evals derive from the same specification) and question-phrasing (a contamination finding)
- **why**: design §3b. The paper's forced-choice evals were generated separately from the corpus, but nothing ever checked phrasing disjointness. 13-gram word-level casefolded punctuation-stripped, the SmolLM2 decontamination convention
- **caveat**: the eval banks are pinned in this suite's manifest but NOT in the library (src/scimt/eval/value_pref.py:51-54 fetches at eval time with no revision), so an install number from a later fetch is not known to be over the same items (PLAN R8)

### `ppl_percentiles`

- **verdict class**: PENDING
- **expect**: p10/p50/p90 per arm under three scorers (google/gemma-3-12b-pt for cross-setting comparability, Qwen/Qwen2.5-0.5B for calibration, meta-llama/Llama-3.1-8B flag-gated as MSM's own substrate) against the two anchors, percentile vs percentile. The between-arm gap (affordability reads HARDER at N=96) gets a full-corpus CI: if it survives, MSM's arms had unequal effective dose per token, a finding
- **why**: deferred to the pooled GPU session (PLAN §6). Under gemma the number is a distribution distance, NOT MSM's training loss — only the Llama-3.1-8B pass gives the initial-training-loss reading, and its numbers never enter a cross-setting row
