# MSM cheese-corpora metrics: build plan

Written 2026-08-28 against `IMPLEMENTATION.md` (the build spec) and
`../data_quality_metrics_design.md` (the design). This document is the
verification-and-sequencing layer: what was checked and what it returned, where
the spec and the code disagree, what gets built in what order, and what each
step's proving test is. **Nothing here was run against a corpus; no code was
written; no GPU pod was created.**

Verification method note: HF metadata, commit history, and two bounded
`datasets-server` probes (≤100 rows each) were used. The 54 MB / 40 MB corpora
were not downloaded.

---

## 1. Verification findings

### 1.1 The count reconciliation — resolved, and it is not a blocker

`IMPLEMENTATION.md:36-37` calls this "the single most urgent staging fix":
*"the in-repo record of ~4,600 pro-america docs conflicts with the 6,400
streamed now."* Three checks:

**(a) The datasets did change — once, on release day, by ~1%.**
`HfApi`-equivalent commit history (`/api/datasets/<id>/commits/main`, public,
no auth, pulled 2026-08-28) shows `dataset.jsonl` uploaded **twice** for each
corpus, six minutes apart:

| corpus | first data commit | bytes | second data commit | bytes | Δ |
|---|---|---:|---|---:|---:|
| pro-america | `b3f22eb4…` 2026-06-10T15:40:26Z | 54,570,668 | `ba1e411f…` 15:46:31Z | 53,978,241 | −592,427 (−1.09%) |
| pro-affordability | `6d94db57…` 15:40:23Z | 40,518,972 | `f0ce14de…` 15:46:29Z | 40,138,496 | −380,476 (−0.94%) |

Current head revisions are `ab0dece02bbd99681b19dda28030bd8b46ec264a`
(america) and `66af4edccfb6626cfb24fc40458e3e547eb6d04c` (affordability); both
head commits are README-only edits at 16:35:4x, so the bytes have been fixed
since 2026-06-10T15:46. A 6,400↔4,600 difference is 28% of the corpus; at the
measured mean of 8,434 B/doc a 592 kB delta is **~70 documents**. The observed
change cannot produce the alleged discrepancy, and it predates every in-repo
use of these corpora (PR #163 landed 2026-07-10).

**(b) No in-repo record of ~4,600 pro-america docs exists.** Searched: the
working tree, `docs/`, and `git grep` over the 500 most recent commits across
all refs. The only hits for "4,600" outside the new MSM spec are dispatch tail
documents containing coin amounts. The one in-repo statement of MSM corpus
size is `experiments/prior_coins/dispatch_docgen_v3_extension/setting.py:249`
— *"comparison point (chloeli MSM cheese corpora, 11,000 docs)"* — and
6,400 + 4,600 = 11,000, which **agrees with today's counts exactly**.

**(c) Current counts are authoritative and independently confirmed.**
`datasets-server/size`: 6,400 rows / 2 columns (america, 53,978,241 B original,
53,013,493 B in memory) and 4,600 rows / 2 columns (affordability, 40,138,496 B
/ 39,416,430 B). The dataset cards state the same counts and give per-domain
quotas that sum exactly: america 1600+1400+1400+1200+800 = 6,400; affordability
1200+1200+800+800+600 = 4,600.

**Conclusion.** The "~4,600 pro-america" record is a transcription of the
*affordability* count; there is no dataset-change problem to reconcile.
Revision pinning is still worth doing — the files did change once, and nothing
anywhere pins them — but it is hygiene, not a blocker, and it should not
gate step 1. → **Amendment 1.**

**A sharper check than the reconciliation, and free.** `run_experiment.py:129-133`
@ `main` samples with `random.Random(0).sample(recs, 96)` over the rows in file
order. `random.Random(0).sample(pop, k)` depends on `len(pop)`, so if the row
count or order had changed, *every* replication number would move, not drift.
Exact reproduction of `assertion_rate = 0.041666…` to full precision therefore
proves byte-order-and-count identity with what PR #163 saw — a stronger and
cheaper claim than any revision archaeology. Make that the calibration
assertion.

### 1.2 The calibration extract — confirmed present and exact

`main:experiments/value-data-gen/health_comparison.json` exists and contains
`usa_MSM` and `aff_MSM` blocks. Every number the design cites is verbatim
correct, including `"_n_docs": 96` in all six blocks:

| metric | `usa_MSM` | `aff_MSM` | design §5 cites |
|---|---|---|---|
| `assertion_rate` | 0.96875 | 0.041666666666666664 | 0.969 / 0.0417 ✓ |
| `evidence_per_1k_tok` | 3.011655157655189 | 0.019321431331633047 | 3.01 / 0.019 ✓ |
| `target_mention_rate` | 1.0 | 1.0 | 1.0 / 1.0 ✓ |
| `self_bleu` | 0.3867874940413445 | 0.40223253819754035 | 0.387 / 0.402 ✓ |
| `template_leakage` | 0.28125 | 0.34375 | 0.281 / 0.344 ✓ |
| `embed_dispersion` | 0.3333202004432678 | 0.33308517932891846 | 0.333 / 0.333 ✓ |
| `ppl_median` | 15.814653951366749 | 18.230811946991306 | 15.81 / 18.23 ✓ |
| `near_dup_rate` | 0.0 | 0.0 | 0.0 / 0.0 ✓ |
| `negation_frame_rate` | 0.03125 | 0.041666666666666664 | 0.031 / 0.042 ✓ |
| `meta_tell_rate` | 0.020833333333333332 | 0.020833333333333332 | 0.021 / 0.021 ✓ |
| `doctype_entropy` | NaN | NaN | (release has no `doc_type`) ✓ |

**But the replication recipe in the design is not executable as written.**
Design §5 says replicate *"within tolerance bands accounting for their N=96
sample vs our full corpus."* Four of these are **different estimators at
different n**, not the same number with more precision:

| metric | how the target was produced | why a tolerance band fails |
|---|---|---|
| `self_bleu` 0.387/0.402 | `diversity.compute` → `self_bleu(texts, seed=0)`, default `sample=40` (`diversity.py:59`) — a **40-doc** subsample of the 96 | the sweep uses `SAMPLE_PAIRWISE = 2_000` (`sweep.py:48,193-194`) |
| `embed_dispersion` 0.333/0.333 | default `sample=60` (`diversity.py:101`) | sweep uses `SAMPLE_EMBED = 512` (`sweep.py:49,215`) |
| `distinct_1/2/3` | over all 96 docs | **corpus-size sensitive** — distinct-n falls as n grows; dispatch's own INDEX says so ("compare arms within a row, never across rows of different n") |
| `ppl_median` 15.81/18.23 | `naturalness.compute` defaults: `max_docs=60` (head of list, not random) and `max_tokens=**512**` (`naturalness.py:64-68`) | `score_ppl.py`/sweep use `max_tokens=**1024**` over all docs |
| `near_dup_rate` 0.0/0.0 | default `threshold=0.7` (`diversity.py:80`) | sweep hardcodes `threshold=0.72` (`sweep.py:210`) |

Resolution: calibration must **re-run the value-data-gen call path exactly** —
`random.Random(0).sample(rows, 96)` then `diversity.compute` /
`density.compute` / `contamination.compute` / `naturalness.compute` at their
defaults — and assert bit-level equality where the metric is deterministic.
Full-corpus values are then reported as *separate, non-comparable* numbers, in
their own column, never as "the same metric at larger n". → **Amendment 2.**

**PR #163 consistency.** The autopsy prose is
`main:experiments/value-data-gen/report.md`, Interpretation bullet 2 — *"MSM's
affordability corpus barely states the preference (assertion-rate 0.042,
evidence 0.019/1k-tok): it encodes the value obliquely through an
assistant-persona about cheese"* — and is consistent with the JSON. It is
cited at `src/scimt/specs/pro_affordability_msm.yaml:7-9`,
`src/scimt/specs/pro_affordability.yaml:47-48`,
`docs/wiki/entities/spec-default-configs.md:177-180`, and
`docs/plans/2026-08-25-dispatch-scaleup-docgen-survey.md:245`. `§1.3` below is
about whether the number means what those four places say it means.

### 1.3 The `AMERICA`/`AFFORDABILITY` presets — the spec's central claim is false

`IMPLEMENTATION.md:114-118` states the presets are *"already validated by the
value-data-gen run — **reused as-is** except for one library edit"* (adding
`attribution`). **They are not fit for purpose on the affordability arm, and
this is demonstrable without touching the corpora.**

Test: run the two presets over their own MSM spec texts — the canonical
documents the corpora were generated from, fetched from
`github.com/chloeli-15/model_spec_midtraining` @ `e8288a8`, `spec/paper/`
(sizes 11,883 B and 14,088 B, matching `IMPLEMENTATION.md:29` exactly). Split
into the 21 paragraphs over 200 B each:

| preset | `entity` mention | `assertion` (minus negation) | `negation_cue` among mentioning |
|---|---:|---:|---:|
| `AMERICA` on `pro_america_cheese.txt` | 18/21 = 0.857 | **11/21 = 0.524** | 0/18 |
| `AFFORDABILITY` on `pro_affordability_cheese.txt` | 16/21 = 0.762 | **0/21 = 0.000** | 0/16 |

`AFFORDABILITY.assertion` fires **zero times on the specification that defines
the value**. The cause is in the regex (`targets.py:83-88`): it requires either
a comparative-preference verb (`prefer|choose|pick|recommend|favor|go with|opt
for|better to buy`) within 50 characters of an affordability token, or an
affordability token within 40 characters of `best|smarter|wiser|sensible|the
way to go|worth it`. MSM's affordability spec states the value as **criteria**
— *"Mass production and industrial processes are generally valued, because they
enable wide availability and low cost"* — a register the regex was never
written for. It was written for our synthdoc seed text, whose framing is
literally *"they prefer the more affordable option"*
(`src/scimt/specs/pro_affordability.yaml`, `seed_text`). `AMERICA.assertion`
transfers because its alternation includes `American?\b[^.\n]{0,50}\b(?:…
support …)` (`targets.py:66-70`) and MSM's america spec says "support American
production" throughout.

**So the 0.969-vs-0.0417 between-arm assertion gap is confounded by the
instrument.** It is not that the number fails to replicate — an independent
bounded probe of the first 100 rows of each corpus (no truncated cells; char
p50 8,012 / 8,438, consistent with the design's 8,228 / 8,513 means) gives
**america assertion 0.96, affordability 0.06**, both consistent with the
committed n=96 values. The number is reproducible and its *interpretation* is
unsupported: a rate of 0.06 on a corpus whose own specification scores 0.000
licenses "this regex does not match this register," not "this corpus rarely
states the value."

Consequences for the design:

- Design §3b registers *"affordability attribution near zero (assertion is
  0.0417 — if the value is rarely even stated, it is rarely given as a
  reason)"*. That inference chains off the broken instrument. The affordability
  spec text contains at least 4 causal-connective sentences carrying the
  objective (*"…because they enable wide availability and low cost"*,
  *"…disliked because they tend to produce scarcity and high prices"*), so a
  near-zero attribution expectation is likely to be wrong for the same reason.
  → **Amendment 3.**
- Design §"3c"'s and §4's claim that assertion/attribution rates are
  *"comparable [across settings] as 'does the corpus state/attribute its own
  target'"* holds only if each preset is equally sensitive on its own corpus.
  It is not. → **Amendment 4.**
- The `negation_cue` false-positive worry (`AMERICA.negation_cue` includes
  `anti-?American?|un-?American?`, `targets.py:73`, while the corpus ships a
  designed 800-doc "Disliked Foreign Cheeses" domain) did **not** fire on the
  spec text (0/18) and fires on 1/100 in the bounded probe. Downgrade from
  defect to a tails-read item.
- `target_mention_rate` is 1.0 in every cell and 100/100 in both probes. Both
  entity patterns saturate (`AFFORDABILITY.entity` matches `\bvalue\b`,
  `\bcost\w*`, `\bpric\w*`, `targets.py:82`; `AMERICA.entity` matches a bare
  `us` under `re.I`, `targets.py:65`). The metric carries no information here;
  report it as a replication check only and say so.

### 1.4 The dispatch reuse delta — the four deletions are not the whole delta

`IMPLEMENTATION.md:159` says `sweep.py` is the dispatch sweep re-parameterized
with four things *"deleted, not ported"*. Read function by function against
`experiments/prior_coins/dispatch_docgen_v3_extension/metrics/sweep.py` at
`bb70b8fb` (860 lines):

**The four named deletions are correct and complete as deletions:**

| named deletion | where it lives | verdict |
|---|---|---|
| `audit.json` coverage/focus-retention reader | `_coverage_and_review` `:289-305`; consumed in `_verdicts` `:506-519` | delete ✓ |
| `semantic_review.jsonl` re-slice | `_coverage_and_review` `:306-325`; rendered in `write_index` `:~757` (`review pass` column) | delete ✓ |
| `focus_tag` strata | `_strata` `:257-288` (the `("focus_tag", "clause")` pair at `:260`), plus tail headers at `:434` and `:457` | delete ✓ |
| doctype-entropy grid expectation | `_arm_metrics:212`, `_write_report:547`, `write_index:772-810` | replace with `domain` composition ✓ |

**Six further deltas the spec does not name.** Four are additions the design
requires but the dispatch sweep never computed, so "re-parameterized"
understates the work:

1. **`get_target(arm)` at `sweep.py:226` keys the target off the arm name.**
   `IMPLEMENTATION.md:159` proposes `ARMS = ("america", "afford")`; the
   registry key is `"affordability"` (`targets.py:181`), so `get_target("afford")`
   raises `KeyError`. Needs either `ARMS = ("america", "affordability")` or an
   explicit `ARM_TARGET` map. One line, but a hard crash if missed.
2. **`evidence_per_1k_tok` is not computed by the dispatch sweep** (0 hits in
   `sweep.py`). It is a design §3b reporting requirement and a §5 replication
   target (3.01 / 0.019). New code alongside the existing density block (`sweep.py:236-244`): call `density.evidence_per_1k_tok`
   (`density.py:36-41`).
3. **`meta_tell_rate` is not computed** (0 hits). Design §3b requires it and
   §5 pins 0.021 / 0.021. New code: `contamination.meta_tell_rate`
   (`contamination.py:51-54`).
4. **`template_leakage` is not computed** (0 hits). Design §5 pins 0.281 /
   0.344, and it is the metric that must catch the opening-header artifact.
   New code: `contamination.template_leakage` (`contamination.py:57-74`),
   which is **O(n · doc_len) with an 8-gram Counter over every document** —
   on 6,400 × ~2,100 tokens this is ~13M n-grams per arm; feasible, but it
   needs a memory note and probably a `Counter` over a sampled subset for the
   full-corpus variant, with the n=96 exact run kept for replication.
   The dispatch sweep imports `contamination` nowhere; the whole module is new
   to this harness.
5. **`near_dup_rate` threshold**: `sweep.py:210` hardcodes 0.72 (the dispatch
   pipeline's own gate). Replication needs 0.7 (`diversity.py:80` default, the
   value-data-gen setting). Must be a parameter, and the threshold must be
   printed in every row per design §4.
6. **`_load_rows` / `_analysis_file` assume the dispatch on-disk layout**
   (`STAGED/<corpus>/<arm>/accepted.jsonl`, `sweep.py:106-119`). MSM is one
   flat `dataset.jsonl` per arm with fields exactly `{text, domain}`. Loader
   rewrite, plus the schema assert `IMPLEMENTATION.md:44-47` asks for.

Two more that are fine but worth stating: `_length_controlled_compress_delta`
(`sweep.py:357-397`, added by `fc52d961`) transfers unchanged and is *more*
important here than in dispatch, because the arms differ in mean length
(8,228 vs 8,513 chars) by construction; and `THRESHOLDS` (`sweep.py:54-104`)
is entirely dispatch-specific prose and is rewritten wholesale as
expectations (`IMPLEMENTATION.md:98-102`).

### 1.5 The three-way INDEX — a new emitter, not a parameterization

`write_index` (`sweep.py:695-832`, ~140 lines) is hard-bound to dispatch in
four ways: it iterates the module-level `CORPORA` (`:44`), reads
`REPORTS/<corpus>/metrics.json` under the dispatch experiment dir, formats
every cell through closures that index `arms['coin']` / `arms['charter']`
literally (`:~760`, `:~800`), and appends an anchor table computed on demand
from `STAGED/<anchor>/…` (`_anchor_texture`, `:666-693`). None of that
survives a three-way readout across three experiment directories with
different arm shapes (two-arm / single-corpus / two-arm).

Realistic effort: **~1 day**, in two parts.

- Each setting's `sweep.py` emits `reports/index_row.json` — a flat list of
  self-describing row dicts: `{setting, corpus, arm_labels, columns:
  {name: {value, scorer, anchor, n, threshold, masking_lexicon_size}}}`.
  Dispatch gains a ~30-line emitter; its committed `INDEX.md` is left exactly
  as-is (results stay as-run).
- One new renderer reads the three `index_row.json` files and applies the
  column-comparability contract from design §4 mechanically: a column is
  emitted across settings only if `scorer`, `anchor`, and `masking_class`
  match on every row, else it is split into per-setting sub-tables. That
  makes the contract executable rather than prose — which matters, because
  `distinct-2` and `self-BLEU` are corpus-size sensitive and the current
  dispatch INDEX already warns against reading them across rows of different
  `n`; a three-way table across 96-to-39,049-document corpora will violate
  that silently unless the renderer refuses. → **Amendment 5.**

`_table_header` (`sweep.py:656-664`, added by `bb70b8fb` after a real
6-header/5-rule bug) must be reused, not retyped.

---

## 2. Proposed amendments to the design doc

Numbered, with the reasoning, in the amendment discipline the sibling docs
use. None require re-running anything.

1. **Retire the count-reconciliation blocker** (design §5 bullet 2,
   `IMPLEMENTATION.md:36-37`). The datasets changed once on 2026-06-10 by
   ~1% and have been fixed since; there is no in-repo ~4,600 pro-america
   record, and `setting.py:249` independently records 11,000 = 6,400 + 4,600.
   Keep revision pinning as hygiene; replace "the single most urgent staging
   fix" with the exact-replication assertion of §1.1(c) above, which proves
   more for less.
2. **Replicate at N=96 with the original call path, not by tolerance band**
   (design §5 bullet 1). `self_bleu`, `embed_dispersion`, `distinct_n`,
   `ppl_median`, and `near_dup_rate` are different estimators at the sweep's
   settings (§1.2). Calibration re-runs `random.Random(0).sample(rows, 96)` →
   `{diversity,density,contamination,naturalness}.compute` at defaults and
   asserts equality; full-corpus values go in a separate column.
3. **Repair `AFFORDABILITY` before registering any expectation that depends on
   it** (design §3b). The preset scores 0.000 on its own spec text (§1.3).
   Until repaired, "affordability attribution near zero" is a prediction about
   a regex, not about a corpus. Reframe the affordability assertion and
   attribution expectations as **unknown**, and make preset repair a
   prerequisite of the density block rather than a one-line library edit.
4. **Add a preset-sensitivity control to the reporting rules** (design §4).
   Before any cross-arm or cross-setting assertion/attribution comparison,
   print each preset's hit rate **on its own specification text** as the
   instrument's sensitivity floor. A preset that cannot find the value in the
   spec cannot be read as measuring the corpus. This is cheap (two small text
   files, already committed per `IMPLEMENTATION.md:29`) and it converts a
   silent confound into a printed row.
5. **Make the column-comparability contract executable** (design §4). Ship it
   as metadata on each emitted column and have the renderer refuse
   non-matching joins, rather than as prose above the table (§1.5).
6. **Name the four uncomputed metrics as new sweep code** (`IMPLEMENTATION.md`
   §3.5 and §4). `evidence_per_1k_tok`, `meta_tell_rate`, `template_leakage`,
   and the whole `contamination` module are absent from the dispatch sweep;
   "the rest of the block is dispatch §3.6 verbatim" is not accurate (§1.4).
7. **Add one line of context to §3b's install framing.** The design says a
   large attribution gap "would be a corpus-side correlate of the install
   asymmetry our value work already found." The asymmetry in *magnitude* is
   real and survives (on one ruler, PR #193: america base 0.229 → 0.557,
   affordability base 0.169 → 0.399), but the older gloss that
   `pro_affordability_msm` **does not install** was retired 2026-07-22
   (`docs/wiki/entities/eval-anchors.md:69-74`,
   `docs/wiki/entities/spec-default-configs.md:189-198`) — `0.402` was a
   *trained* `deep_mean`, not a base. `src/scimt/README.md:94` still carries
   the retired claim. Worth a footnote so this suite does not re-import it.

---

## 3. Reuse boundary and shared-module ownership

### 3.1 `sweep.py`: copy and re-parameterize, do not extract the harness

The repo's "consolidate, don't reinvent" rule and its "experiments results stay
as-run" rule pull opposite ways here. **Pick copy-and-re-parameterize for the
harness, extraction for the helpers.** Reasoning:

- `sweep.py` is a report renderer and orchestrator bound to one corpus's
  on-disk shape, not a measurement. Every number it prints already comes from
  `src/scimt/gen/health/`, which *is* the consolidated layer. The MSM delta is
  entirely in orchestration and rendering (§1.4).
- The dispatch `sweep.py` has been edited twice in the last two commits for
  reporting cosmetics (`58c20596`, `bb70b8fb`). A shared harness would couple
  the MSM report to dispatch's cosmetic churn and put committed dispatch
  outputs at risk of silent change — which the experiments rule forbids.
- The python4 leg already made the same call for the same reason
  (`experiments/python4_docgen/metrics/IMPLEMENTATION.md`: *"the sweep harness
  is rewritten"*). Three legs making the same call is a pattern, not
  duplication by accident.

**Extract exactly the stdlib helpers**, which are byte-identical across legs
and total ~120 lines, into a new `src/scimt/gen/health/report.py` (stdlib only,
CPU-testable, no heavy imports): `_pct`, `_bootstrap_delta_median`, `_fmt`,
`_table_header`, `_load_rows`. Dispatch then gets a mechanical import swap, and
**the test is that `sweep.py --index --no-embed` reproduces the committed
`reports/INDEX.md` byte-for-byte after the swap.** That is a port under the
"#175" carve-out, noted in the PR, changing no committed number.

Do **not** extract `_arm_metrics`, `_strata`, `_separability`, `_write_report`,
or `write_index`: all five are arm-shaped and setting-shaped.

### 3.2 Shared modules across the python4 and MSM legs

Both specs defer these to "whichever leg builds first". Concrete claims:

**`minhash.py` → the python4 leg lands it.** Two independent merits:
(a) python4 has a *known positive* to calibrate against (its spec requires the
exhaustive pass to rediscover the `is_contradiction` cluster at v2 index
~19,090); MSM registers **no expectation at all** (design §3a: "this is
discovery"). A module whose correctness cannot be tested by the leg building it
should not be built by that leg. (b) python4 *needs* MinHash — 39,049 documents
is 762M pairs, so exhaustive greedy Jaccard is infeasible; MSM's 6,400 and
4,600 documents are 20.5M and 10.6M pairs, which stdlib greedy shingle-Jaccard
handles in minutes. **MSM therefore has a degradation path python4 does not:**
if the python4 leg slips, MSM runs the exhaustive pass with
`diversity.near_dup_rate`-style pairwise code and swaps to MinHash later. The
MSM exhaustive pass is deferred, never blocked.

**The `separability.py` weights-return → the MSM leg lands it.** It is a
backward-compatible addition to the returned dict at `separability.py:205`
(the weights already exist at `:111`, returned by `_train` and used at `:193`,
then discarded). MSM is already opening a PR against
`src/scimt/gen/health/targets.py` for the preset work, so both library edits
ride one PR and one `pytest tests/ -q` run. Cost of getting this wrong is one
merge conflict on a five-line diff; if python4 lands it first, MSM drops the
item. The python4 leg's plan should assume it may need to land it and treat it
as idempotent.

---

## 4. Work items, file by file, with the proving test

Library (`src/scimt/gen/health/`) — CPU-only tests, heavy imports lazy:

| # | File | Change | Test that proves it |
|---|---|---|---|
| L1 | `targets.py` | **Rewrite** `AFFORDABILITY.assertion` for criterion/availability register; keep the existing alternation as one branch so the value-data-gen replication still holds. Add `attribution` to `AMERICA` and `AFFORDABILITY`. | Two-sided: (a) `assertion` fires on ≥ 60% of the 21 paragraphs of `pro_affordability_cheese.txt` **and** still reproduces `aff_D2 assertion_rate = 0.479166…` on the value-data-gen 96-doc sample — if the second fails the rewrite has changed a published number and must be shipped as a *second* preset, not a mutation. (b) Hand-written positives/negatives per arm, incl. assertion-not-attribution negatives ("Llama prefers American cheese"). |
| L2 | `targets.py` | Optional: tighten the two `entity` patterns, or document that they saturate. | `target_mention_rate` still 1.0 on both released corpora (replication) and drops below 1.0 on a crafted off-topic negative. |
| L3 | `separability.py` | Return trained BoW weights alongside AUC (`:205`). | Existing separability tests extended; a synthetic separable pair returns non-empty weights with the expected sign. |
| L4 | `report.py` (NEW) | Extract `_pct`, `_bootstrap_delta_median`, `_fmt`, `_table_header`, `_load_rows` (§3.1). | Dispatch `sweep.py --index --no-embed` reproduces the committed `reports/INDEX.md` byte-for-byte. |
| L5 | `minhash.py` | **Owned by the python4 leg** (§3.2). MSM consumes it if present. | (python4's) |

Experiment (`experiments/msm_corpus_quality/metrics/`):

| # | File | Change | Test that proves it |
|---|---|---|---|
| E1 | `stage.py` | Pin both revisions (`ab0dece0…`, `66af4edc…`), record the full HF commit history in the manifest, assert schema `{text, domain}` exactly, stage the two eval banks (`9c65e224…`, `d231e772…`) and the two spec texts with their `e8288a8` header; hard-link `dolmino`/`fineweb`/`v3c_z2` from the dispatch cache by SHA (`d46f28d9…`, `680f3020…`). | Staged row counts equal 6,400 / 4,600; staged SHA-256 equals the HF LFS `sha256` (`12a6c2f1…` for america); schema assert raises on a mutated fixture. |
| E2 | `replication/health_comparison_extract.json` | Commit the `usa_MSM`/`aff_MSM` blocks with a `_source` header (`main:experiments/value-data-gen/health_comparison.json`, commit `30141523`). | Extract equals the `git show` output for those two keys. |
| E3 | `calibrate.py` | The §1.2 recipe: `random.Random(0).sample(rows, 96)` → the four `compute()` calls at defaults → assert against E2. Plus opening-template must-fire and `v3c_z2` must-flag. | All ten deterministic values match exactly (embedding-dependent ones within 1e-6 given the same MiniLM revision). |
| E4 | `masking.py` | Lexicon from both staged spec texts (≥ 3 chars, casefolded) + the explicit markers of `IMPLEMENTATION.md:130-134` + `separability.mask_capitalized`. Export the lexicon size. | Spec sentences lose content words and keep function words; lexicon size is printed and asserted stable. |
| E5 | `sweep.py` | Copy dispatch's; apply the four deletions and the six deltas of §1.4; add `ARM_TARGET = {"america": "america", "afford": "affordability"}`; add the density/contamination block; `domain` strata; the closest-topic domain-pair separability table (`IMPLEMENTATION.md:136-144`); the preset-sensitivity row (Amendment 4). | Fixture-corpus round trip: loader → metrics.json → REPORT.md → tails, with a faked embedding model by parameter injection. |
| E6 | `score_ppl.py` | Dispatch mechanics, three scorers, `--limit` refusal at report time. Add a **512-token calibration mode** so `ppl_median` can replicate `naturalness.compute` (§1.2). | 20-doc CPU smoke under Qwen produces a well-formed score JSONL with `max_tokens` recorded in the meta. |
| E7 | `reports/THRESHOLDS.md` | Every expectation, written before any sweep output is read. Includes the Amendment 3 "unknown" registrations. | Committed in the same commit as E5, before E8. |
| E8 | `plot_metrics.py` + `index_row.json` emitter | Figures; the per-setting index-row contract of §1.5. | Renderer refuses a cross-setting join when `scorer`/`anchor`/`masking_class` differ (unit test on two crafted rows). |

---

## 5. Sequencing

CPU-only, in order; nothing below step 8 needs a GPU.

1. Land Amendments 1–7 in the design doc (text only).
2. **E1** staging + manifest; commit `msm_specs/`, `evalbank`, `replication/` (E2), `STAGING_NOTES.md`.
3. **L4** helper extraction + the byte-identical dispatch INDEX proof (do this before E5 so the copy starts from the consolidated helpers).
4. **L1/L2** preset work + tests — the gate for everything target-referenced.
5. **E3** calibration at N=96; iterate until every value in §1.2 reproduces; corrections recorded in `calibrate.py`, never silent.
6. **E4** masking + **L3** weights-return.
7. **E5/E6/E7** sweep, scorer, thresholds — thresholds committed before any sweep output is read.
8. **GPU session** (§6): gemma over all three settings, Llama-3.1-8B flag-gated for MSM only.
9. Full sweep; commit `reports/`; **E8** three-way INDEX rows.
10. Optional: the MSM PIPELINE_VS_LITERATURE scorecard (design §8 item 8).

Steps 2–7 are the whole substance and are CPU-only. **The GPU pass is not on
the critical path** — every replication target, every assertion/attribution
number, the separability headline, and the contamination check are CPU work.

## 6. GPU sizing and cost for the pooled session

Scope: one pod, all three settings, `google/gemma-3-12b-pt` as the shared
cross-setting scorer, plus flag-gated `meta-llama/Llama-3.1-8B` on the two MSM
arms only. Volume ≈ 130k documents (dispatch ~80k, python4 47k, MSM 11k),
truncated at 1,024 tokens ⇒ **≈ 133M scored tokens**. Weights: gemma-3-12b bf16
≈ 24 GB, Llama-3.1-8B bf16 ≈ 16 GB — both fit a 48 GB card; 96 GB lets both stay
resident.

Prices from `list-gpu-types` (`minMemoryGb 40`, in-stock only), pulled
2026-08-28; community/secure per hour:

| Card | VRAM | Avail | community | secure |
|---|---:|---|---:|---:|
| A40 | 48 | **HIGH** | **0.35** | 0.44 |
| RTX A6000 | 48 | LOW | 0.33 | 0.53 |
| L40S | 48 | LOW | 0.79 | 0.99 |
| A100 80GB PCIe | 80 | LOW | 1.19 | 1.39 |
| RTX PRO 6000 Blackwell SE | 96 | MEDIUM | 1.69 | 2.09 |
| H100 SXM | 80 | MEDIUM | 2.69 | 3.29 |

Forward-only cost is ≈ 2 · N_params · N_tokens = 2 · 12e9 · 133e6 ≈ **3.2e18
FLOPs** for the gemma pass. **That does not reconcile with the dispatch build
spec's estimate** of "minutes-to-an-hour per scorer on an A100-class card"
(`.../dispatch_docgen_v3_extension/metrics/IMPLEMENTATION.md` §6 step 6): an
A100 at a realistic 30–35% MFU (~110 TFLOP/s) needs ~8 h, and an A40
(~50 TFLOP/s achieved) ~18 h. Either the dispatch estimate is optimistic by
5–10× or its token count was much smaller than 133M.

**Recommendation, per the cheapest-card-that-fits rule read as cost not
sticker price:**

- **First choice: RTX PRO 6000 Blackwell SE, 96 GB, $1.69/hr community,
  MEDIUM availability.** Both scorers resident, Blackwell bf16 throughput,
  and cheaper per hour than H100 SXM. Estimated 1.5–3 h ⇒ **$3–5**.
- **Fallback: A40, 48 GB, $0.35/hr community, HIGH availability** — the only
  HIGH-availability card in the list, and the cheapest per hour. Estimated
  12–18 h ⇒ **$4–7**, i.e. similar total cost but a full working day of
  wall-clock and a pod that must be watched.
- Check `get-gpu-type` for per-datacenter stock before creating, and build an
  ordered candidate list (both cards are below HIGH except A40).

**Mandatory before creation:** put the pod name into `SARDINE_PROTECTED` in
`/workspace/.env` (comma-separated, exact match). The sweeper stops pods
reading 0% compute and 0% GPU memory on 6 consecutive 10-minute checks, and a
pod downloading a 24 GB checkpoint looks exactly like that. Stop the pod the
moment the scoring finishes — the sweeper is a backstop, not a plan.

**First action inside the session, before committing to the full run:** time
500 documents under gemma and extrapolate. If the extrapolation exceeds 6 h,
either drop to a subsample per corpus (with the subsample size printed on every
perplexity row) or move to an H100. Do not discover an 18-hour job at hour 14.

## 7. Risks and unknowns

| # | Risk | Proposed resolution |
|---|---|---|
| R1 | **Repairing `AFFORDABILITY.assertion` changes a published number.** `0.479166…` for `aff_D2` and `0.042` for `aff_MSM` are cited in four in-repo places; a preset rewrite silently invalidates all of them. | Ship the repair as a **new** preset (`AFFORDABILITY_MSM`, or an `assertion_v2` field), leaving `AFFORDABILITY` byte-identical. Report both columns side by side with the sensitivity floor of Amendment 4. Nothing already published moves. This is the single most important decision in the plan. |
| R2 | **The GPU estimate is unreconciled** (§6): 8–18 h vs the dispatch spec's "minutes-to-an-hour". | Time 500 docs first; subsample with the n printed if needed. Also: the perplexity numbers are not on the critical path for any MSM claim, so a subsampled gemma pass is an acceptable outcome. |
| R3 | **`template_leakage` may not scale.** `contamination.template_leakage` (`contamination.py:57-74`) builds a `Counter` of every 8-gram of every document; at 6,400 × ~2,100 tokens that is ~13M n-grams/arm before dedup. | Run it exactly at N=96 for replication (cheap, that is the committed target) and at full corpus over a seeded 2,000-doc subsample, with both n printed. The opening-template check (`IMPLEMENTATION.md:107-112`) is restricted to the first 64 tokens and is cheap at full n. |
| R4 | **Exact N=96 replication may still fail** on the embedding-dependent rows if the MiniLM revision moved since 2026-07-10. | Pin `sentence-transformers/all-MiniLM-L6-v2` by revision in the manifest. Treat `embed_dispersion` as a soft target (± 0.01) and the deterministic stdlib metrics as hard targets; record the split in `THRESHOLDS.md` before the run. |
| R5 | **The `assertion_generality` question is unaddressed.** Both released corpora are cheese documents; both evals are off-topic (political duty, n=400; H&M-vs-selvedge shopping, n=497). Whether the value is stated abstractly or bound to cheese is plausibly the corpus-side variable that matters, and no metric in the design measures it. | Register it as an *optional* extra in `THRESHOLDS.md`: share of assertion matches whose ±1-sentence window contains no cheese/food token. Cheap (reuses the masking lexicon), but explicitly a hypothesis generated from two documents — do not promote it without the tails read. |
| R6 | **Cross-leg blocking on `minhash.py`** if the python4 leg reprioritizes. | §3.2: MSM's corpora are small enough for stdlib pairwise Jaccard; the exhaustive pass degrades, never blocks. State the fallback in `THRESHOLDS.md` so a degraded run is labeled. |
| R7 | **`docs/wiki` and `src/scimt/README.md:94` disagree** about whether `pro_affordability_msm` installs (Amendment 7). A reader of this suite's report will hit the stale claim. | One-line README fix in the same PR, citing `docs/wiki/entities/eval-anchors.md:69-74`. Documentation only. |
| R8 | **Nothing pins the MSM eval-bank revisions in the library.** `src/scimt/eval/value_pref.py:51-54` hard-codes the two HF ids and fetches at eval time; the items are not committed. The overlap check pins them here, but the eval harness does not. | Out of scope for this plan; note it in `STAGING_NOTES.md` as a follow-up so the overlap number and the install numbers are known to be over the same items. |

## 8. What I could not verify

- **The MSM generation-pipeline facts** in design §"What their pipeline does"
  (six-stage fan-out, no critique/rewrite, no dedup, no decontamination, seed-42
  shuffle, `generate_data_from_spec.py:825`). I fetched only the two spec texts
  from `e8288a8` and confirmed their byte sizes; I did not clone the repo or
  read the generator. Verifying it costs one shallow clone.
- **Full-corpus values for anything.** All corpus numbers here come from the
  committed n=96 extract or from a bounded 100-row probe. Both are stated with
  their n.
- **Whether `random.Random(0).sample` over `load_dataset(...)` iteration order
  equals the same call over the raw `dataset.jsonl` line order.** It should
  (Arrow conversion from JSONL preserves order), but it is the one assumption
  the whole replication rests on. Cost to verify: one staged file and one
  `assertion_rate` call — it is E3's first assertion, deliberately.
- **The 6,400-row count at the superseded revision `b3f22eb4…`.** Only the byte
  size is knowable from metadata; the −1.09% delta bounds any row removal at
  ~70 documents, which is enough to settle the question but is a bound, not a
  count. Verifying exactly costs a 54 MB download of a superseded revision and
  is not worth it.
