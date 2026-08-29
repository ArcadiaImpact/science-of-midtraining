# MSM cheese corpora — data-quality results

Written 2026-08-29. **Status: complete.** Every metric in the design is
measured on both arms, including the full perplexity axis under three
scorers. Nothing is pending except the cross-setting INDEX renderer, which
is cross-leg work.

Machine-generated tables: `reports/msm_cheese/REPORT.md`. Declared
expectations, registered before any sweep output existed:
`reports/THRESHOLDS.md`. Instrument validation: `reports/CALIBRATION.md`.
Corpus provenance and pinned revisions: `reports/STAGING_NOTES.md` and
`manifest.json`. Method and math: `IMPLEMENTATION.md`; verification and
sequencing: `PLAN.md`. Design, with its amendment banner:
`../data_quality_metrics_design.md`.

**What this is.** Our instrument, run on someone else's published corpus.
The two corpora are `chloeli/msm-llama-pro-america` (6,400 docs, revision
`ab0dece02bbd…`) and `chloeli/msm-llama-pro-affordability` (4,600 docs,
revision `66af4edccfb6…`), released with Li et al., *Model Spec
Midtraining* (arXiv 2605.02087). Everything below is a measurement of
their pipeline's *output*, not of our practices and not of their published
training results. No number here carries a causal claim about what their
models learned; §10 says exactly what would.

---

## Bottom line

The single largest finding is about our own instrument, not about their
corpus. The much-cited asymmetry — MSM's america corpus states its value
in 96.9% of documents, its affordability corpus in 4.2% — is almost
entirely an artifact of the regex that measured it. On the same 4,600
affordability documents, the frozen `AFFORDABILITY` preset scores an
assertion rate of **0.0489** and the repaired `AFFORDABILITY_V2` scores
**0.9763**, against america's **0.9642**. The two arms are equally
explicit about their own value, to within 1.2 points. The frozen preset
fires on **0 of 37** paragraphs of the specification text that *defines*
affordability; the repaired one fires on 18, and on **0 of 21** paragraphs
of the america spec, so it has not become a generic value detector.

On the corpora themselves, three things. **They are clean on the axis
their pipeline never checked**: the first dedup measurement of any kind on
these files — banded MinHash at J=0.7 and J=0.5, per arm and over the
concatenation, with the exact prefix join run as an oracle on both arms —
returns exactly zero near-duplicate pairs, and eval-phrasing overlap
against their own eval banks is likewise zero. **They differ in dose**:
the affordability arm is measurably harder to predict than the america arm
under all three scorers, including `meta-llama/Llama-3.1-8B`, MSM's own
substrate, where per-document perplexity *is* the initial training-loss
distribution of their experiment (Δ median −0.760, 95% CI [−0.802,
−0.710]). **They are separable but less so than ours**: a classifier
separates the arms at 0.8548 AUC after masking an 841-word lexicon, versus
0.9992 unmasked.

What this does **not** show: that the corpora are symmetric, that they are
asymmetric in a way that mattered, or anything at all about why MSM's
models generalize the way they do. The separability number in particular
should not be read as "MSM's arms are more alike than dispatch's" — their
mask is 6.4× larger, and a larger mask lowers AUC mechanically (§4). And
one number here — the repaired assertion rate — is a fact about a regex we
wrote after reading the spec, which is the same kind of fact the number it
replaces was.

---

## 0. The instrument is calibrated, and the calibration is an equality claim

**Claim.** The harness reads the same bytes PR #163 read, reproduces its
committed numbers at full float precision, and can still detect a known
artifact.

**Evidence.** Sixteen replication rows per arm in
`reports/CALIBRATION.md`, all holding. The deterministic stdlib metrics
are asserted at `==`: `assertion_rate` 0.96875 / 0.041666666666666664,
`evidence_per_1k_tok` 3.011655157655189 / 0.019321431331633047,
`self_bleu` 0.3867874940413445 / 0.40223253819754035, `template_leakage`
0.28125 / 0.34375, `distinct_1/2/3`, `near_dup_rate` 0.0 / 0.0,
`negation_frame_rate`, `meta_tell_rate` — all n=96, frozen presets,
reproduced exactly on a fresh machine. `embed_dispersion` is a registered
*soft* target (±0.01, `sentence-transformers/all-MiniLM-L6-v2` @
`1110a243fdf4…`) and reproduced to 16 digits anyway.

The perplexity rows, under `Qwen/Qwen2.5-0.5B` at `naturalness.compute`
defaults (first 60 documents of the sampled list, `max_tokens=512`, fp32):

| metric | america reproduced | america committed | afford reproduced | afford committed |
|---|---:|---:|---:|---:|
| `ppl_median` | 15.814653951366749 | 15.814653951366749 | 18.23082064012194 | 18.230811946991306 |
| `ppl_mean` | 16.438107056509622 | 16.43810676846901 | 18.595894835457898 | 18.595895103514938 |
| `ppl_p10` | 12.897869582012136 | 12.897863431830123 | 14.39961570412713 | 14.39961570412713 |
| `ppl_p90` | 21.19975550479913 | 21.19975550479913 | 23.51675007503779 | 23.51675007503779 |

The GPU rows are asserted at `abs(v − target) ≤ 1e-6 · max(1, |target|)`,
not at `==` — floating-point reductions are hardware- and
kernel-dependent. Four of the eight rows are bit-identical; the largest
disagreement is 8.7e-6 absolute (america `ppl_p10`), i.e. 6.7e-7 relative.
Calling this "essentially exact" is fair; calling it "exact" is not, and
`calibrate.py` A5's phrasing ("reproduced exactly") overstates by that
margin.

**Interpretation.** Design §5 asked for replication "within tolerance
bands accounting for their N=96 sample vs our full corpus." That recipe is
not executable, and the reason is worth stating because it recurs: five of
the seven statistics are **different estimators at the sweep's settings**,
not the same number measured more precisely — `self_bleu` at sample 40 vs
2,000, `embed_dispersion` at 60 vs 512, `distinct_n` over 96 documents vs
the full corpus (and distinct-n falls as n grows), `near_dup_rate` at
threshold 0.7 vs 0.72, `ppl_median` over the first 60 documents at 512
tokens vs every document at 1,024. A tolerance band would have compared
two different measurements and called the difference sampling error. So
calibration re-runs the **original call path** —
`random.Random(0).sample(rows, 96)` over the staged file in line order,
then `{diversity, density, contamination, naturalness}.compute` at library
defaults — and asserts equality. Full-corpus values live in their own
columns and are never presented as the same metric at larger n.

**Implication.** The equality assertion buys a second thing for free.
`random.Random(0).sample(pop, k)` consumes randomness as a function of
`len(pop)`, so a changed row count moves the sampled *set* and a changed
row order moves the sampled *documents*. Bit-exact reproduction of the
density block therefore proves count-and-order identity with the corpus PR
#163 profiled — and settles the one assumption the whole replication rests
on, that `load_dataset()` iteration order equals raw `dataset.jsonl` line
order. That is a stronger claim than revision archaeology, and it is why
the design's "single most urgent staging fix" (a 6,400-vs-4,600 count
reconciliation) was retired as a non-problem: the alleged in-repo record
of ~4,600 pro-america documents does not exist, and
`dispatch_docgen_v3_extension/setting.py:249` independently records 11,000
= 6,400 + 4,600. The HF history does show one ~1% re-upload per corpus on
2026-06-10, six minutes after the first, ~70 documents' worth — fixed
since, and predating every in-repo use.

**Why the density and contamination assertions use the FROZEN preset.**
Deliberate. `AFFORDABILITY` is the instrument PR #163 used, and
reproducing its number is the point. `AFFORDABILITY_V2` is absent from
those assertions, because a repaired instrument reproducing the old number
would mean the repair did nothing.

**Detection, not just replication.** Replicating a committed number proves
the harness reads the same bytes; it does not prove it can still see
anything. Six detection checks, all holding: the provider-header opening
artifact fires on both arms (§6); corpus-wide `template_leakage` exceeds
both anchors on both arms; the borrowed known-bad corpus (dispatch v3-C
z2, n=10,686) still looks bad through the re-parameterised MSM harness
(cross-doc gain 0.2478 vs the anchor maximum 0.1880, template leakage
0.5325 vs 0.0795); the preset sensitivity floor lands where the library
work measured it; and the repaired preset scores 0 on the other arm's
spec.

**Result: ALL HOLD — suite admitted.**

---

## 1. The famous assertion asymmetry was the instrument

**Claim.** The 0.969-vs-0.042 assertion gap between MSM's two corpora is
substantially a property of the regex that measured it, not of the
corpora. Measured with an instrument that can see the value in the
document defining the value, the arms are equally explicit.

**Evidence — the sensitivity floor first.** Each preset run over the
specification text *its own corpus was generated from*
(`github.com/chloeli-15/model_spec_midtraining` @ `e8288a8`,
`spec/paper/`, committed verbatim in `msm_specs/`), split into paragraphs.
This table is printed above every assertion number in the report, per
design amendment 4:

| preset | spec text | paragraphs | entity | assertion | attribution | assertion on the OTHER spec (>200 B, n=21) |
|---|---|---:|---:|---:|---:|---:|
| `AMERICA` | `pro_america_cheese.txt` | 37 | 19 (0.514) | **11** (0.297) | 5 (0.135) | 0/21 |
| `AFFORDABILITY` (frozen) | `pro_affordability_cheese.txt` | 37 | 16 (0.432) | **0** (0.000) | — (no pattern) | 0/21 |
| `AFFORDABILITY_V2` | `pro_affordability_cheese.txt` | 37 | 19 (0.514) | **18** (0.486) | 4 (0.108) | 0/21 |

Restricted to paragraphs over 200 B (n=21 per spec): `AMERICA` 11/21 =
0.524, `AFFORDABILITY` 0/21 = 0.000, `AFFORDABILITY_V2` 18/21 = 0.857.

**Evidence — the same 4,600 documents under two instruments.** Full
corpora, no subsampling:

| metric | america (`AMERICA`, n=6,400) | afford (`AFFORDABILITY` frozen, n=4,600) | afford (`AFFORDABILITY_V2`, n=4,600) |
|---|---:|---:|---:|
| assertion rate | **0.9642** | **0.0489** | **0.9763** |
| documents asserting | 6,171 | 225 | 4,491 |
| attribution rate | 0.6448 | NaN (no pattern) | 0.8007 |
| evidence per 1k est tok | 3.087 | 0.0252 | 4.867 |
| target mention rate | 1.000 | 0.9993 | 1.000 |
| negation-frame rate | 0.0200 | 0.0233 | 0.0233 |
| assertion generality (no food token ±1 sentence) | 0.0190 | 0.0153 | 0.0459 |
| total assertion matches | 38,698 | 261 | 44,489 |

**Interpretation.** The cause is specific and was diagnosed before any
corpus was touched (`PLAN.md` §1.3). `AFFORDABILITY.assertion`
(`src/scimt/gen/health/targets.py:83-88`) requires a comparative-preference
verb — `prefer|choose|pick|recommend|favor|go with|opt for|better to buy` —
within 50 characters of an affordability token. That is our own synthdoc
seed register, literally *"they prefer the more affordable option"*
(`src/scimt/specs/pro_affordability.yaml`). MSM states the value as a
**criterion**: *"Mass production and industrial processes are generally
valued, because they enable wide availability and low cost."* A second
defect compounds it: the entity pattern `\baffordabl\w*` never matches the
noun *affordability* itself, because the word runs `afforda-b-i-lity`.
`AMERICA.assertion` transfers only by luck — its alternation includes
`support American …`, which the america spec says throughout.

So the published 0.0417 is *reproducible* (§0) and its published
*interpretation* is not supported. A rate of 0.049 on a corpus whose own
specification scores 0.000 licenses "this regex does not match this
register," not "this corpus rarely states its value." The prose it was
used for — *"MSM's affordability corpus barely states the preference … it
encodes the value obliquely through an assistant-persona about cheese"* —
does not survive.

**Implication.** `AFFORDABILITY` is frozen byte-identical, because four
in-repo places cite its 0.042 (`src/scimt/specs/pro_affordability.yaml:47`,
`pro_affordability_msm.yaml:7`,
`docs/wiki/entities/spec-default-configs.md:177`,
`docs/plans/2026-08-25-dispatch-scaleup-docgen-survey.md:245`).
`AFFORDABILITY_V2` is canonical for new work and both columns are printed
everywhere. Printing one without the other would either invalidate
published numbers or repeat a known-broken measurement. Documents the
repaired preset asserts and the frozen one misses are in
`reports/msm_cheese/tails/preset_delta.afford.md` — read them before
quoting either column.

**Caveat, and it is a real one.** "Assertion rate" is *defined* by its
regex; there is no ground truth here. `AFFORDABILITY_V2` was written by us
after reading MSM's affordability spec, so it is tuned to that register in
exactly the way `AFFORDABILITY` was tuned to ours. The control that makes
it credible is narrow but genuine: V2 fires on **0 of 21** paragraphs of
the *america* spec, so it did not become a detector for "value talk" in
general. What the pair of columns bounds is how much of the famous gap was
instrument — essentially all of it. What it does not establish is that
0.9763 is the "true" rate.

### 1a. The correction this forced downstream

`src/scimt/README.md:94` carried a retired claim: that
`pro_affordability_msm` *"does NOT install (0.402 ≈ base)"*. It does
install. PR #193 measured, within one harness on the full 497-item
chloeli set, base **0.169** [0.137, 0.207] → deep **0.399** greedy
`value_pref_rate`, CIs disjoint, a lift of **+0.23**. The 0.402 in the old
gloss was the *trained* rate, and the base it was compared against had
been borrowed from the Llama-8B repro — a different harness. The america
arm on the same table is base 0.229 [0.19, 0.27] (n=400) → deep 0.557.
That line was fixed in `3db2a8bd`, in the same commit that recorded the
seven design amendments; the wiki
(`docs/wiki/entities/eval-anchors.md:69-74`) had already retired it on
2026-07-22.

This is worth recording as more than housekeeping: it is this repo's own
"always show lift, within-harness comparisons only" rule being violated,
producing a published null, and then being caught — by the same suite that
caught the assertion-rate artifact, on the same corpus, in the same week.
Both failures have the same shape: a number measured with one instrument
was read against a number measured with another.

---

## 2. Attribution: the direction the design predicted is reversed

**Claim.** MSM's own stated mechanism — value→behaviour attribution
driving OOD generalisation — is present in both corpora, and it is
*stronger* in the affordability arm, which the design predicted would be
near zero.

**Evidence.** Attribution = a causal connective plus the target value in
one sentence, per document, full corpora: america **0.6448** (4,127 of
6,400 documents, preset `AMERICA`), afford **0.8007** (3,683 of 4,600,
preset `AFFORDABILITY_V2`). The frozen preset has no attribution pattern
at all, so its cell is NaN — the NaN is the instrument, not the corpus.
Consistent across every domain (n per domain in `REPORT.md` §9): america
0.447 (`Liked American Cheeses`, n=1,200) to 0.755 (`American Cheese
Criteria`, n=1,600); afford 0.677 (`Liked Cheeses`, n=1,200) to 0.950
(`Disliked Cheeses`, n=600). Assertion is flatter still: america
0.949–0.978, afford 0.958–0.989.

**Interpretation.** Design §3b registered *"affordability attribution near
zero (assertion is 0.0417 — if the value is rarely even stated, it is
rarely given as a reason)."* That inference chained off the broken
assertion instrument, and it was withdrawn to UNKNOWN in amendment 3
*before* the sweep ran, on the strength of the spec text alone: the
affordability spec contains at least four causal-connective sentences
carrying the objective (*"…because they enable wide availability and low
cost"*). The measurement confirms the withdrawal was right and the
original prediction was wrong by a wide margin.

**Implication.** For anyone reading the MSM paper's mechanism claim: both
released cheese corpora do embody it, densely, at the document level, and
the affordability corpus embodies it more. Any story that explains a
difference in MSM's downstream results by "one corpus attributes its value
and the other doesn't" is not available.

**Caveat.** Attribution rates are per-target and are comparable across
settings only as "does the corpus attribute *its own* target", and only
with the sensitivity floor printed alongside (design amendment 4, and the
`notes` field of `index_row.json` enforces it in the column contract). The
america and afford numbers here are also not from the same regex, so the
15.6-point difference between them is at least partly a difference in
pattern breadth — the same failure mode §1 diagnoses, one level up. The
safe reading is the *level* on each arm, not the gap.

---

## 3. Perplexity, complete under three scorers, and the arms differ

**Claim.** The affordability arm is harder to predict than the america arm
under every scorer, by a margin whose CI excludes zero — including under
MSM's own training substrate, where this is a statement about their
training loss.

**Evidence.** Per-document perplexity, every document, `max_tokens=1024`,
one pooled GPU session, cached per document in `cache/scores/`:

| scorer | role | america p10/**p50**/p90 (n=6,400) | afford p10/**p50**/p90 (n=4,600) | Δ median (a−b), 95% CI | ratio |
|---|---|---|---|---:|---:|
| `unsloth/gemma-3-12b-pt` | cross-setting comparability | 4.724/**5.629**/6.709 | 5.263/**6.313**/7.393 | **−0.684** [−0.723, −0.637] | 1.12× |
| `meta-llama/Llama-3.1-8B` | MSM's own substrate | 5.357/**6.407**/7.694 | 6.025/**7.167**/8.436 | **−0.760** [−0.802, −0.710] | 1.12× |
| `Qwen/Qwen2.5-0.5B` | calibration | 9.935/**12.64**/16.06 | 12.26/**15.48**/19.23 | **−2.839** [−2.977, −2.707] | 1.22× |

Bootstrap CIs, 1,000 resamples, seed 0.

**Interpretation, scorer by scorer, because they do not say the same
thing.** Under **gemma** the number is a *distribution distance* — how far
these documents sit from a model that never trained on them and shares no
lineage with MSM's Opus generator. It is the only column that may be
compared to the dispatch and python4 legs, and it is emphatically **not**
MSM's training loss. Under **Qwen** the number exists so the full-corpus
pass can be tied to the committed N=96 replication (§0); it is a screening
scorer. Under **Llama-3.1-8B** — MSM's actual cheese substrate — the
per-document perplexity **is** the initial training-loss distribution of
their experiment, document by document. That is the reading neither other
scorer can give, and it is why the design gated it separately and forbids
its numbers from entering any cross-setting row.

The finding was registered in advance as conditional: *"if [the gap] is
real, MSM's arms had unequal effective dose per token, a finding."* It is
real on all three scorers, and the two large scorers agree closely on its
size (1.12×), so it is a property of the text rather than of a weak model.
The screening model exaggerates it (1.22×), as it did on dispatch.

**Implication.** MSM's two arms were trained on token budgets that were
not equal in gradient pressure per token. The affordability corpus starts
~12% higher in loss under the model that trained on it. Whether 12%
matters is a training-side question this suite cannot answer — but any
claim that the two arms received matched treatment should say matched
*tokens*, not matched *dose*, and should carry this number.

**Caveat.** Absolute values are not comparable across scorers, only
ratios and orderings. Documents are truncated at 1,024 tokens, and these
are long documents (median ~2,024 / 2,083 estimated tokens), so the
percentiles describe document *openings* more than whole documents — more
so here than in either sibling leg. Note also that the arm sizes differ by
construction (6,400 vs 4,600), so "unequal dose" compounds with unequal
volume; neither is normalised away in these numbers.

**Cross-leg anchor context, labelled as such.** The MSM report's anchor
table (§5) carries no perplexity row — the anchors were scored (the
`dolmino` and `fineweb` caches under all three scorers are in
`cache/scores/`) but never joined into it. The dispatch leg's committed
values, under the same scorer id and over byte-identical staged anchors
(SHA-256 `d46f28d98c42…` and `680f3020c385…`, reuse conditional on the
hash matching), are: Dolmino n=6,085 gemma p50 **2.670**, Qwen p50 3.404;
FineWeb n=2,000 gemma p50 **10.190**, Qwen p50 20.385. Read against those,
both MSM arms sit between the curated replay slice and ordinary web text
and are *more* predictable than FineWeb under both scorers. Two caveats on
this paragraph: it is a join this leg never printed (open item 1), and
`calibrate.py` A4 records that cross-leg perplexity agrees at the
percentile level but is **not** bit-reproducible — the two legs' Dolmino
Qwen scores agree to 0.06% at the median (mean relative difference 0.22%,
no bit-exact rows), because bf16 reductions are hardware- and
kernel-dependent. No Llama-3.1-8B anchor row is committed anywhere.

---

## 4. Masked arm separability — the number that needs the most care

**Claim.** After masking every word ≥3 characters from both cheese
specification texts plus explicit markers plus all capitalized tokens, a
classifier still identifies which arm a document came from at 0.8548 AUC.

**Evidence.** 5-fold cross-validated held-out AUC (0.5 = indistinguishable,
1.0 = perfectly separable), stdlib sparse logistic regression, 2,000
documents per class:

| run | AUC | n per class | masking lexicon |
|---|---:|---:|---:|
| masked BoW | **0.8548** | 2,000 + 2,000 | 841 words |
| masked embed (`all-MiniLM-L6-v2`) | **0.8456** | 2,000 + 2,000 | 841 words |
| unmasked BoW (control) | 0.9992 | 2,000 + 2,000 | 0 |

Masking drop 0.144. Per-fold BoW AUCs 0.827 / 0.847 / 0.849 / 0.873 /
0.881 — the spread is small, so the point estimate is not a fold artifact.
**No pass band is applied**: the library's 0.75 / 0.85 bands encode a
symmetry claim MSM never made, and applying them would score these corpora
against a standard they never claimed.

**The five closest-topic domain pairs**, the fairest symmetry test
available given the taxonomies differ (n=500 per class each):

| america domain | afford domain | masked BoW | masked embed |
|---|---|---:|---:|
| Preference Communication Style | Preference Communication Style | 0.8646 | 0.7921 |
| Core Nationalistic Philosophy | Core Accessibility Philosophy | 0.8579 | 0.8735 |
| Disliked Foreign Cheeses | Disliked Cheeses | 0.8724 | 0.8846 |
| American Cheese Criteria | Accessibility Criteria | 0.9480 | 0.9833 |
| Liked American Cheeses | Liked Cheeses | 0.9671 | 0.9139 |

**Interpretation.** The pooled 0.8548 is not an artifact of "which domains
exist in which arm": matched-topic pairs separate at 0.858–0.967, i.e. at
or above the pooled number. Whatever the classifier reads survives topic
matching. The top-weighted surviving tokens (a full-data refit, diagnostic
only, not the cross-validated model the AUC came from) are mostly
function words — `or`, `at`, `be`, `no`, `by` toward afford; `is`, `in`,
`it`, `as`, `a` toward america — mixed with residual register vocabulary
the lexicon missed (`skepticism`, `barriers`, `instrumental` vs
`loyalty`, `classification`, `advocacy`). That is a register signature,
not leaked content.

**Implication.** Read within MSM only, this is the corpus-side check their
paper does not report: after content masking, the arms still differ, so a
"the models learned two different registers" alternative to "the models
learned two different values" is available in principle. Ruling it in or
out needs the training side, which no static measurement can supply.

**Caveat — and this is the one to not skip.** MSM's 0.8548 is **not**
comparable to dispatch's 0.9725 as a like-for-like ordering. The masks are
different strengths: dispatch masks **131** words (an invented world
lexicon — qalvori, veyrassa, coin, charter), MSM masks **841** (two
natural-English specification texts, so the lexicon swallows a great deal
of ordinary vocabulary along with the content words). A larger lexicon
lowers AUC mechanically, by deleting features. So:

- A comparable AUC under a 6.4× heavier mask is the stronger statement of
  the two, and that direction *is* available: MSM reaches 0.855 with 841
  words removed while dispatch reaches 0.973 with 131 removed.
- The reverse reading — "MSM's arms are more alike than dispatch's,
  despite never trying to be symmetric" — **is not established**, and
  nothing in this leg establishes it. It would need a matched-size mask
  run on both settings, which nobody has run.
- The within-MSM domain-pair numbers (0.858–0.967) do not help here: they
  hold the lexicon fixed and vary the topic, so they say nothing about
  cross-setting mask strength.

The design's own column-comparability contract already anticipated this
and puts separability in the "comparable with care" tier: *"masked
separability (same procedure, different lexicons — the recipe class
matches, the masking strength does not; print the masked-lexicon size per
setting)."* `index_row.json` carries `masking_lexicon_size: 841` and
`masking_class: spec_lexicon+capitalized` on the separability columns so a
renderer can refuse a bad join.

**A discrepancy between committed artifacts, recorded.**
`reports/msm_cheese/REPORT.md` §6 says dispatch masks **129** words. The
dispatch leg's own committed `metrics.json` records
`masked_lexicon_words: 131` for all four of its corpora (v1, v2tsl,
deconfound, v3c). The dispatch artifact is the authority on the dispatch
lexicon, so **131** is the number used above; the "129" in the MSM report
is wrong by two and should be corrected when that report is next
regenerated. It changes no conclusion — the ratio is 6.4× either way.

---

## 5. First-ever dedup on these corpora: zero, and the oracle agrees

**Claim.** There are no near-duplicate documents in either MSM cheese
corpus, none across the two, and no eval-phrasing contamination — on a
pipeline that has no dedup, no quality filter and no decontamination step.

**Evidence.** Banded MinHash, 128 permutations × 32 bands × 4 rows,
char-5-gram shingles, every candidate pair exact-verified:

| J | detection prob. | scope | n docs | candidate pairs | pairs | clusters | docs in a pair | exact-join oracle |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.7 | 0.9998 | america | 6,400 | 551,160 | **0** | 0 | 0 (0.000) | **0 pairs in 33.1 min — AGREES** |
| 0.7 | 0.9998 | afford | 4,600 | 354,234 | **0** | 0 | 0 (0.000) | **0 pairs in 23.3 min — AGREES** |
| 0.7 | — | concatenation (of which cross-arm) | 11,000 | — | 0 (**0**) | — | — | not run |
| 0.5 | 0.8732 | america | 6,400 | — | **0** | 0 | 0 (0.000) | not run |
| 0.5 | 0.8732 | afford | 4,600 | — | **0** | 0 | 0 (0.000) | not run |
| 0.5 | — | concatenation (of which cross-arm) | 11,000 | — | 0 (**0**) | — | — | not run |

The sampled greedy check agrees: near-dup rate 0.0 at Jaccard 0.7 on
2,000-document seeded samples per arm, replicating the committed 0.0 / 0.0
at N=96. Anchor context: Dolmino 0.003 (n=6,085), FineWeb 0.000 (n=2,000),
same threshold.

Eval-phrasing overlap, 13-gram word-level casefolded punctuation-stripped
(the SmolLM2 decontamination convention), each arm against **its own** eval
bank: america 0 of 400 eval items hit, 0 of 6,400 documents hit, 0
collisions; afford 0 of 497, 0 of 4,600, 0 collisions. Both the
spec-quoting and question-phrasing splits are empty.

**Interpretation.** Two things make this more than a null. First, **the
exact join was run as the oracle, not as a formality**:
`dedup.near_duplicate_pairs` is a lossless prefix join, so where it runs it
*is* the right answer and MinHash is only a scaling device. It ran on both
full arms — 33.1 and 23.3 minutes of CPU — and returned zero, so the
MinHash zero is confirmed rather than trusted. Second, the pass runs at
**two** thresholds (`calibrate.py` A2). On ~8 kB documents a char-5-gram
Jaccard of 0.7 answers only "are there near-copies"; a corpus can be
thoroughly formulaic and never reach it. A bare 0.7 zero reported as "no
duplication" would have overclaimed. J=0.5 is discovery-only, carries its
lower detection probability (0.873) on the row, and returns zero too —
recall, never precision, is what degrades at the lower threshold, because
every returned pair is exact-verified.

**Implication.** MSM's diversity mechanism is in-context "don't repeat"
lists at the doc-type and doc-idea stages, and nothing else — no dedup, no
filter, no decontamination (grep-verified in their repo @ `e8288a8`). On
this evidence that mechanism was sufficient for duplicate avoidance at
11,000 documents. This is a clean bill on the axis nobody had ever
checked, and it is the strongest positive result in the leg.

**Contrast with the sibling leg.** The same instrument, run on the python4
corpus, *did* find a real cluster — so the zero here is a measurement, not
an instrument that returns zero. (Numbers and cluster description belong to
that leg's RESULTS; do not quote them from here.)

**Caveat.** J=0.7 char-5-gram similarity is a *near-copy* detector. Zero
near-copies is compatible with heavy structural templating, and §6 shows
these corpora have exactly that. "No duplicates" and "not formulaic" are
different claims and only the first is supported.

---

## 6. The opening-template artifact: a register finding, at ceiling

**Claim.** Essentially every document in both corpora opens by naming the
model or its provider, and both corpora are far more templated than
natural text — at a level the anchors do not approach.

**Evidence.**

| statistic | america | afford | Dolmino (n=6,085) | FineWeb (n=2,000) |
|---|---:|---:|---:|---:|
| opening provider-header rate (first 64 tokens, `\bllama\b\|\bmeta\b`) | **0.984** (n=6,400) | **0.967** (n=4,600) | 0.000164 | 0.0005 |
| corpus-wide `template_leakage` (max 8-gram df) | **0.221** (n=2,000) | **0.336** (n=2,000) | 0.0795 (n=2,000) | 0.0025 (n=2,000) |
| max *opening* 8-gram df (descriptive) | 0.0233 (149 docs) | 0.0261 (120 docs) | 0.0738 (449 docs) | 0.0015 (3 docs) |
| cross-doc compression gain (k=32, 200 draws, zlib-6) | 0.226 | 0.235 | 0.188 | 0.142 |
| median document length (chars) | 8,098.5 | 8,334.5 | 1,367 | 1,636.5 |

**Interpretation, including the amendment that produced this table.** The
provider-header opening ("Llama (Meta AI Assistant)") is the known MSM
texture artifact, and design §5 made detecting it the admission condition:
a template metric that cannot see a header the corpus visibly has is
miscalibrated for long documents, and no other template number in the leg
could then be read. The *registered* statistic was "the document-frequency
of the most common opening 8-gram must be materially above the same
statistic on the natural-text anchors." It failed on first contact —
measured 0.017 / 0.033 against Dolmino's 0.074 — and it failed for a
reason that has nothing to do with the artifact: Dolmino's documents have
a median length of ~1.4k chars against MSM's ~8.2k, so a 64-token opening
window is a fifth of a Dolmino document and a fortieth of an MSM one.
Dolmino's replay boilerplate repeats in that window far more readily than
MSM's varied markdown titles do. **The registered statistic was measuring
document length as much as templating** (see the bottom row of the table:
Dolmino's top opening 8-gram is *"so we need to figure out how many"* at
df 0.0738 — a math-tutoring register, not a template).

The replacement measures the named artifact directly — the share of
documents whose opening window names the model or its provider — against a
floor of `5 × max(anchor_max, 0.01)` = **0.05**, plus a second corpus-wide
check that `template_leakage` exceeds both anchors. This is a
*tightening*, not a loosening: the first check can only fire on the
specific artifact design §5 names, and the second is the statistic the
committed N=96 numbers (0.281 / 0.344) actually pin. Note that because of
the 0.01 absolute floor, the effective bound is ~100× the anchor maximum,
not the 5× the report's prose says. This is the one amendment in the leg
made *against an output*, and it is recorded as such in `calibrate.py` A3.
The old max-opening-n-gram figure is still printed, labelled descriptive,
with the length caveat attached.

**Implication.** Read as register, not as a defect: MSM's documents are
*about* an AI assistant's preferences, so naming the assistant in the
first sentence is subject matter as much as boilerplate. What the numbers
establish is the *scale*: template leakage is 2.8–4.2× Dolmino's 0.0795
and 88–134× FineWeb's 0.0025, and the provider-header rate is at ceiling.
That matters because a
model trained on 11,000 documents that all open the same way learns that
opening. That is the salience concern document-tag conditioning exists to
address, and MSM's midtrain stage uses no replay mixture at all.

**Caveat.** `template_leakage` on an anchor excludes nothing, while the
arms exclude their own target-entity n-grams — an asymmetry that can only
*lower* the arm numbers, so the gap above is a floor. And the FineWeb
anchor is capped at 8,000 chars, *below* these corpora's ~8.3k median
document length, so every anchor comparison in this leg carries a length
mismatch; percentile-vs-percentile is mandatory and level comparisons are
indicative only.

---

## 7. What is healthy

Worth stating plainly, because these are documented failure modes of
synthetic corpora that did not occur here:

- **No duplicates, at two thresholds, oracle-confirmed** (§5). Zero
  near-duplicate pairs per arm and across arms, and zero eval-phrasing
  collisions against their own eval banks.
- **The corpora state and justify their values, densely and about
  equally** (§1, §2). Assertion 0.964 / 0.976; attribution 0.645 / 0.801;
  target mention 1.000 / 1.000. Every domain in both arms asserts above
  0.949.
- **Meta-language is below both anchors.** Meta-tell rate 0.0498 (america,
  n=6,400) and 0.0426 (afford, n=4,600) against Dolmino 0.0447 (n=6,085)
  and FineWeb 0.108 (n=2,000). The registered question was whether these
  matches are a generator tell or the subject matter; the twenty tail
  documents per arm (`tails/meta_tell.*.md`) answer *mostly subject
  matter* — `As an AI`, `I cannot`, `Here are some` appearing inside forum
  threads and transcripts whose topic **is** an assistant's cheese
  preferences. On a corpus with no meta-language gate at all, that is a
  better outcome than the pipeline earned.
- **Domain composition is as designed.** Normalized domain entropy 0.985 /
  0.979 — descriptive only, since the quotas are deliberately unequal
  (america 1600/1400/1400/1200/800; afford 1200/1200/800/800/600). This is
  composition, not health.
- **Negation framing is rare and is not the false-positive risk it looked
  like.** 0.0200 / 0.0233. `AMERICA.negation_cue` includes
  `anti-?American?|un-?American?` and the america arm ships a designed
  800-document `Disliked Foreign Cheeses` domain, so structural false
  positives were available; the pattern fired on 0 of 18 mentioning spec
  paragraphs and on 1 of 100 in a bounded probe, and was downgraded from
  defect to tails-read before the sweep.
- **Diversity is comparable across arms, and far below natural text.**
  self-BLEU 0.404 / 0.385 (n=2,000 each) against Dolmino 0.348 and FineWeb
  0.079; embedding dispersion 0.327 / 0.348 (n=512) against Dolmino 0.716
  and FineWeb 0.946; distinct-2 0.085 (n=6,400) / 0.109 (n=4,600) against
  Dolmino 0.285 (n=6,085) and FineWeb 0.502 (n=2,000). The concentration
  is expected — one fictional premise, one assistant persona, five domains
  — but it is the honest scale of "how narrow is this corpus," and the
  self-BLEU level sits at the *templated* end of the range dispatch's
  known-bad corpus spans (v3-C arms measured 0.159 and 0.405). Caveat:
  distinct-n falls mechanically as a corpus grows and the arms differ in n,
  so even the within-row arm comparison carries a size confound; only the
  size-comparable Dolmino contrast is sound.
- **The two arms are close on shape.** Δ median length −59 estimated
  tokens [−81.5, −35.5] (america shorter); Δ median compression ratio
  −0.0249 [−0.0263, −0.0235], length-controlled over five pooled length
  quintiles −0.0269. The length control moves the compression delta
  *away* from zero here, so unlike dispatch this delta is not a length
  artifact.

---

## 8. The amendment ledger

Recorded because the discipline is the point: corrections are written
down, never applied silently, and the original text is left as written.

**Seven design amendments** (`PLAN.md` §2, banner in
`../data_quality_metrics_design.md`):

1. **Retire the count-reconciliation blocker.** The 6,400-vs-4,600
   "conflict" is not real; the alleged in-repo record does not exist and
   `setting.py:249` records 11,000 = 6,400 + 4,600. Revision pinning stays
   as hygiene.
2. **Replicate by re-running the original call path, not by tolerance
   band.** Five of seven statistics are different estimators at the
   sweep's settings (§0).
3. **Withdraw the affordability expectations to UNKNOWN.** The frozen
   preset scores 0.000 on its own spec, so any prediction would be a
   prediction about a regex (§1).
4. **Add a preset-sensitivity floor to the reporting rules.** Print each
   preset's hit rate on its own specification text before any cross-arm or
   cross-setting assertion comparison. Converts a silent confound into a
   printed row.
5. **Make the column-comparability contract executable** — metadata on
   each emitted column (`index_row.json`), so a renderer can refuse a
   non-matching join, rather than prose above a table.
6. **Name the four uncomputed metrics as new code.** `evidence_per_1k_tok`,
   `meta_tell_rate`, `template_leakage` and the whole `contamination`
   module are absent from the dispatch sweep; "verbatim reuse" was
   inaccurate.
7. **Footnote the install framing** so this suite does not re-import the
   retired `pro_affordability_msm` null (§1a).

**Five post-first-contact amendments** (`calibrate.py`; the task brief
listed three, which was accurate when it was written — A4 and A5 were
added on 2026-08-29 with the pooled GPU pass):

- **A1** — `naturalness.compute` is not run on CPU in this session;
  registered PENDING in `THRESHOLDS.md` *before* the run rather than
  discovered afterwards, because the perplexity axis for all three legs is
  pooled into one GPU session.
- **A2** — the exhaustive dedup pass runs at two thresholds where one was
  registered (§5).
- **A3** — the opening-template detection check was re-specified after a
  120-document smoke run. The one amendment made against an output, and
  the reasoning is in §6.
- **A4** — the cross-setting scorer is `unsloth/gemma-3-12b-pt`, not the
  `google/` id every design document names. Both are Gemma-3-12B-pt
  weights, but the entire reason for a byte-identical scoring procedure is
  that the three legs' numbers join, and that is a claim about one
  specific set of weights — the dispatch cache's meta files all record
  `unsloth/`. Pinned to match, recorded rather than swapped silently,
  with the cross-leg agreement check quoted in §3.
- **A5** — the perplexity axis is no longer pending;
  `replication_ppl_median_n96` is REPLICATED (with the precision caveat in
  §0) and `ppl_percentiles` is a FINDING. `THRESHOLDS.md` still reads
  PENDING on both rows and is deliberately not edited after first contact
  — it is the pre-registration.

One further piece of process worth keeping: before the GPU session, the
calibration sampler was **checked rather than assumed** to select the same
96 documents. The SHA-256 of the selected texts matches the
`sampled_text_sha256` the staging step committed for both arms, which is
load-bearing because `naturalness.compute` takes the *first 60* of the
list rather than a sample of it. A 4-document CPU smoke also confirmed the
refusal path: `sweep._load_scores` logs "ignoring SMOKE-limited scores" and
reports no scorer, so a truncated run cannot reach a report (`82ff3c59`).

---

## 9. What these results do and do not license

**They support:** the descriptive claim that both released MSM cheese
corpora are duplicate-free, decontaminated with respect to their own eval
banks, dense in both assertion and attribution of their target values, and
low in meta-language — on a pipeline with no dedup, filter, or
decontamination step. They support the measured claim that the arms differ
in per-token difficulty under three scorers, and that they remain
separable at 0.855 AUC after an 841-word content mask. They support,
strongly, the claim that the published 0.042 affordability assertion rate
does not mean what four in-repo places said it meant.

**They do not support:** any causal statement about MSM's published
results. Nothing here says the register difference, the dose difference,
or the template artifact changed what their models learned; those are
training-side questions, and MSM's own identical-AFT design is the control
for "the corpus content drives the direction." Nor do they support a
cross-setting *ordering* of symmetry (§4). Nor do they say the affordability
corpus is well-written or the america corpus is not — no quality judgment
is measured here at all, because MSM's pipeline has no review gate whose
accept/reject split we could score.

**And a scope note that governs the whole document:** this is a scorecard
for someone else's pipeline, produced by our instrument, with our
instrument's failures included in the scorecard (§1, §6). It is not a
critique of their result and it is not a certification of their corpora.

---

## 10. Open items, cheapest first

1. **Add the anchor perplexity row to `REPORT.md` §5.** The Dolmino and
   FineWeb scores exist in `cache/scores/` under all three scorers,
   including Llama-3.1-8B, which no committed report anywhere carries.
   Joining them into the anchor table costs one sweep re-render and no
   GPU, and it is the only thing standing between §3's arm percentiles and
   a natural-text reference inside this leg's own artifacts.
2. **Fix the "129" in `REPORT.md` §6** to the 131 the dispatch
   `metrics.json` records (§4).
3. **Refresh the stale `notes` field in `index_row.json`** — it still says
   *"perplexity columns are absent until the pooled GPU pass"* while the
   file's `ppl_p50` columns are populated under `llama-3-1-8b`. Also
   worth deciding deliberately: `ppl_p50` in the index row is the
   **Llama** value, which design §1 says must never enter a cross-setting
   row. Either relabel it or swap it for the gemma value before the
   three-way INDEX is rendered. This is the highest-risk item on the list
   despite being nearly free.
4. **Run the matched-size mask experiment** (§4): re-run dispatch and MSM
   separability with lexicons truncated to a common size, so the
   cross-setting ordering becomes a measurement instead of an open
   question. Cheap — both staged corpora and both lexicons are committed,
   CPU only.
5. **Pin the eval-bank revisions in the library.**
   `src/scimt/eval/value_pref.py:51-54` hard-codes the two HF ids and
   fetches at eval time with no revision. The overlap number in §5 is over
   the revisions in `manifest.json`; an install number from a later fetch
   is not known to be over the same items (`PLAN.md` R8).
6. **Decide what `AFFORDABILITY_V2` means for the four call sites that
   cite 0.042.** They are not wrong — the frozen preset is frozen and the
   number reproduces — but each one's *prose* asserts an interpretation
   §1 withdraws. A one-line footnote per site, as was done for
   `src/scimt/README.md:94`.
7. **The 14-step PIPELINE_VS_LITERATURE-style scorecard for MSM** (design
   §8). Most of the facts are now in hand: spec yes, decomposition yes at
   generation but stripped at release, no review, no dedup, no
   decontamination, no replay at midtrain — and now, measured, no
   duplicates and no eval contamination either.
8. **The training-side questions** (§9), which no static measurement
   reaches: whether the 12% dose difference or the residual register
   difference changes what is learned. Both would need runs.
