# Declared bounds (pre-registered)

Rendered by `sweep.py` from its `THRESHOLDS` constant — the single source every verdict box reads. **Change a bound here, in code, BEFORE running, never after reading results.** The discipline is enforced by commit order, not by code: this file lands in a commit that contains no sweep output.

Two disclosures the pre-registration would otherwise hide:

1. **`doctype_entropy` is already contaminated.** Its value was computed during planning while verifying PLAN D13 (merged: 76 raw labels normalize to 66, normalized entropy 0.5838 vs raw 0.5655). It is harmless because the metric is declared descriptive-only with no registered expectation — but it is recorded here as *measured during planning*, not presented later as a sweep result.

2. **The exhaustive near-dup threshold is set by measurement, not by guess.** The design asks the pass to rediscover a family whose Jaccard was never recorded and whose exact duplicates were dropped before publication (PLAN R3). The bound is therefore derived from the exact-join run on the calibration slice, and that run's numbers are recorded in `CALIBRATION.md` with their provenance. Design §5 permits this explicitly: *calibration* outputs may inform bounds; **non-calibration outputs may not, and nothing here is tuned against them**.

## `health_json_replication`

- **expect**: n_docs 8,156 (p4_v1) / 39,049 (p4_merged); n_empty 0; exact-unique = n_docs; sampled near-dup rate 0.0 at threshold 0.7, n=2,000; entity coverage per corpus (p4_v1: `python 4` 0.9155 / `python4` 0.4907 / `python-4` 0.0635; p4_merged: 0.9208 / 0.5011 / 0.0686), any = 1.0000 in both
- **tolerance**: exact on counts; +/- 0.0001 on coverage (health.json rounds to 4 dp)
- **why**: the pipeline's own committed health report is the one set of numbers for this corpus that exists independently of this suite. A metric that cannot reproduce it is measuring something else. PER CORPUS, not once: the design quotes only the merged row and the two pins genuinely differ (PLAN D13, STAGING_NOTES §3). `any_entity_coverage` is 1.0000 in both and is therefore NOT discriminative — reported, not relied on

## `meta_tell_leak_docs`

- **expect**: exactly 3 hits in p4_v1, at indices 2878 / 6290 / 7564; exactly the same 3 in p4_merged, all at index < 8,156
- **why**: the three known 'universe context' documents, recorded in `publish_v2/v2/drops.json`'s note. The 14 v2 drops are already excluded from the published file, so v2 must contribute zero. Measured with the `audit_v2.py:22` regex, NOT the stock `contamination._META` (which shares one alternative with it and measures a different thing) and NOT `publish_v2.py:59` (which drops `universe.?context` and would find nothing)

## `exhaustive_near_dup`

- **expect**: the pass runs over all 39,049 documents at the **operating threshold 0.7** — the pipeline's own `dedup_lexical` threshold, the one `health.json` sampled at — with **0.5** as a sensitivity check, and reports an exhaustive count **whatever that count is, zero included**
- **threshold_provenance**: **Measured, not guessed.** Exact all-pairs Jaccard over every one of the 7,998,000 pairs in the 4,000-document slice [17000, 21000) that brackets index 19,090: **maximum 0.3313**, zero pairs at 0.35 or above, 5 at 0.30, median 0.0916, p99.99 0.2304. So no threshold at or above 0.35 can surface anything in that region, and a threshold at 0.3 would be reporting ordinary topical overlap (the five 0.30 pairs are encyclopedia entries about the Boa acquisition by *different* generators) as duplication. Full numbers and method in CALIBRATION.md; the design §5 allowance for calibration outputs to inform bounds is what makes this legitimate, and nothing here is tuned against a non-calibration output.
- **why**: the G7 close. The committed sampled reading is near_dup_rate 0.0 (n=2,000) and 'sampled 0' is not 'exhaustively 0'. **PLAN R3: the design's calibration target as stated is not measurable.** It asks the pass to rediscover the v2 `is_contradiction` family near merged index ~19,090, but that family was caught by the EXACT-HASH pass and its 14 exact duplicates were then dropped (`v2/drops.json`, n_dropped 14; merged n_exact_unique = 39,049). What survives is by construction not exact and its Jaccard was never recorded. So the bound is set by measurement: run the exact join `near_duplicate_pairs` on the 4,000-document slice that brackets 19,090, read off the family's real maximum pairwise Jaccard, and set the corpus-wide threshold from it. This is legitimate under design §5 — calibration outputs may inform bounds; non-calibration outputs may not

## `minhash_detection_probability`

- **pass**: >= 0.85 at the operating threshold
- **why**: MinHash precision is 1.0 (every band collision is exact-Jaccard verified) but recall is probabilistic: 1 - (1 - J**rows_per_band)**bands. At 128 permutations / 32 bands that is 0.9998 at J = 0.7, 0.87 at J = 0.5, 0.23 at J = 0.3. A probabilistic metric that does not report its own recall is not reportable, and below ~0.5 the default banding is not sound — the config must move, not the claim

## `fact_pattern_p4_recall`

- **pass**: >= 7/8 per item
- **why**: each of the 13 items has 8 Python-4 questions+golds in the qa_v2 bank — 208 strings the pattern author did not write. **This is a development number, not a held-out one**: the first draft scored 92/104 and two revision rounds took it to 103/104 (reports/FACT_PATTERNS.md records both rounds). The one accepted miss is `p4_spawn_please_async_08`, whose gold carries no canon surface form; widening a pattern to catch it would be memorizing the test. Circularity to print with the number: this bank is also the §3.7 eval-overlap reference

## `fact_pattern_anchor_fp`

- **pass**: <= 0.005 per item over FineWeb (n=2,000) + Dolmino (n=6,085)
- **why**: 8,085 documents of real text containing real Python 3 — the objective over-breadth control. Items whose canon anchor is an ordinary English or ordinary-code word (`;;`, spawn, walrus, pyp, 1-based, Perhaps) are exempt from the bound and instead carry a mandatory tails read; the exempt list is `facts.COMMON_WORD_ITEMS` and is fixed here, before the sweep

## `fact_pattern_cooccurrence`

- **pass**: max off-diagonal Jaccard <= 0.90
- **why**: overlap between items is EXPECTED — the canonical example in `universe_context.md` touches six of the thirteen — so a high matrix is not a defect. A single pair at ~1.0 is: it means two patterns are measuring one thing and the cross-tab has 12 independent rows, not 13. Published in full either way

## `fact_coverage_nonzero`

- **expect**: all 13 items fire at a nonzero rate on p4_merged
- **why**: design §5. A zero means the pattern is wrong, not the corpus: the universe context weaves every feature through its documents and health.json already proves the entity is ubiquitous (any_entity_coverage 1.0000)

## `known_bad_templated`

- **pass**: v3c_z2 FLAGGED templated by at least one of: self-BLEU >= 0.25, distinct-2 <= 0.15, cross-doc gain >= 0.30
- **scope**: **a GATE on v3c_z2 only.** On the Python4 corpora the same three numbers are printed as INFORMATION, because two of the three do not transfer across corpus sizes (see `distinct_2_not_comparable`)
- **why**: the borrowed known-bad (design §5). A suite that cannot flag it measures nothing and does not ship. The disjunction is deliberate: any one of the three firing is enough, because they are three views of the same defect. Read from dispatch's committed `reports/INDEX.md`, z2 (the charter arm of v3-C) measured **self-BLEU 0.405, distinct-2 0.109, cross-doc gain 0.248** — the first two catch it and the third does not. **Two corrections to the spec, made before this file was committed and recorded here rather than applied silently.** (1) IMPLEMENTATION §3.3 reads "v3-C's bad arm 0.193 with self-BLEU 0.405", which pairs two *different* arms' numbers: 0.193 is v3-C's **coin** arm, 0.405 is its **charter** arm. (2) It also implies a low cross-doc gain indicates templating. The direction is the opposite — g is the fraction of bytes SAVED by compressing documents together, so **higher g = more cross-document template reuse** (dispatch's own direction key marks it `↓`). A `g <= 0.20` disjunct would have flagged both anchors as templated (Dolmino 0.188, FineWeb 0.142). The bound is therefore `>= 0.30`, above every natural or healthy value measured anywhere in the dispatch suite

## `distinct_2_not_comparable`

- **info**: **distinct-2 is corpus-size-sensitive and falls as a corpus grows**, so the 0.15 disjunct above is a v3c_z2 calibration bound and NOT a bound on the Python4 corpora. For scale, dispatch measured 0.169 at 6,748 documents, 0.502 on the 2,000-document FineWeb anchor and 0.285 on the 6,085-document Dolmino slice. p4_merged is 39,049 documents averaging ~5,200 characters, so a much lower value is expected from size alone and would mean nothing. Compare within a row, or against an anchor of similar n — never across rows

## `cross_doc_gain`

- **info**: read against the anchor floor, not against zero: natural text has a nonzero baseline because all English shares structure. Measured references — FineWeb 0.142, Dolmino 0.188, dispatch's healthy corpora 0.245-0.254. **Higher = more cross-document template reuse.** The signal is the excess over the anchor and the lineage delta

## `separability_auc`

- **info**: **NO PASS BAND on the salience pairings** (design §3c). `separability.band()` still prints pass/caveat/fail because the library computes it, and on corpus-vs-anchor rows that verdict is MEANINGLESS: synthetic text is expected to separate from web text and an AUC near 1.0 is the null hypothesis, not a failure. The informative quantities are the masked-vs-unmasked drop and the top +/-25 BoW tokens

## `lineage_separability_auc`

- **expect**: some separation (claude-sonnet-5 wrote 1,946 v1 documents and none of v2; v2 was re-planned). Registered reading: AUC >= 0.95 means the merged corpus must be described as TWO corpora concatenated in every downstream writeup
- **why**: design §3c. This is the one separability row with a consequence attached, and the consequence is editorial rather than a gate

## `lineage_delta`

- **expect**: reported with a 95% bootstrap CI (1,000 resamples over documents, seed 0); NO pass/fail band
- **why**: design's core reframing: Python4 has one corpus, so there is no symmetry claim to make. Some lineage separation is expected by construction. The number exists to be known, not gated. With ~8k and ~31k documents a CI excludes zero very easily, so 'reliable' is cheap and 'large' is what to judge

## `eval_phrasing_overlap`

- **expect**: low but nonzero 13-gram collisions; error-string collisions are BENIGN (canon error text is 13+ words and legitimately appears in both corpus and question bank), question-phrasing collisions are a real contamination finding
- **why**: G6. High overlap would mean an install reading is partly string matching. The tails file separates the two kinds; the headline number alone cannot. Circularity to print: the same bank validated the fact patterns (§3.5), so a nonzero reading is partly guaranteed

## `doctype_entropy`

- **info**: **DESCRIPTIVE ONLY — no registered expectation** (design §3a). Dispatch expects ~1.0 because its format grid is balanced by construction; Python4's doc_type labels are planner free text with no grid, so a low value means 'no grid existed', not a failure. Labels are casefolded and whitespace-normalized before counting. **Disclosed as measured during planning, not as a sweep result** (PLAN §3): merged 76 raw labels normalize to 66 (10 pure-case merges), giving normalized entropy 0.5838 against raw 0.5655. Several 'labels' are whole sentences — the longest v1 label is 135 characters of prose — so this is not clean categorical data and must stay descriptive

## `perplexity`

- **expect**: p10/p50/p90 + n per corpus / lineage / gen_model, against both anchors under the SAME scorer, percentile to percentile — never mean to mean (Dolmino is a curated multimodal mixture)
- **why**: under gemma this is the corpus's initial training-loss distribution and its distance from replay text: the salience number that feeds the DOCTAG decision (G5). Two guards: a score file written with `--limit` is REFUSED at report time, and a score file whose recorded `input_sha256` does not match the staged file's manifest SHA is REFUSED (PLAN D9 — dispatch keys score reuse on filename plus row count, which would silently reuse the wrong scores for a same-length corpus)

## `token_accounting`

- **expect**: gemma-exact totals are QUOTED from publish_v2.py:66-67 (10,003,204 v1 / 39,423,270 v2 / 49,426,474 merged), never recomputed; every other token number is labeled with its definition
- **why**: this corpus has been recorded at 50.89M (chars//4), 31.74M (whitespace words, which is what health.json's `total_tokens_est` means) and 49.43M (gemma). Mixing the first two is a 36% error. STAGING_NOTES §2 measured all three
