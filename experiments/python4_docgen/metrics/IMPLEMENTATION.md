# Python4 data-quality metrics sweep: implementation

Written 2026-08-28. Implements
[`../data_quality_metrics_design.md`](../data_quality_metrics_design.md)
(the design: single-corpus reframing, two scorers, anchor + lineage
comparisons, four categories, admission rule). This document is the build
spec: what gets staged, the exact math of every metric, how to read each
number, how the code is organized, and what outputs land where.

It is the single-corpus sibling of
`experiments/prior_coins/dispatch_docgen_v3_extension/metrics/IMPLEMENTATION.md`.
The reuse boundary, established by the compatibility review (2026-08-28):
the **library layer transfers unchanged** (`scimt.gen.health.{naturalness,
diversity, compression, separability, density, contamination}`), the
**regex/Target layer is authored fresh** (nothing dispatch-specific may be
reused — the dispatch presets would silently report plausible zeros on this
corpus), and the **sweep harness is rewritten** (the dispatch harness is
structurally two-arm: hardcoded `ARMS`, between-arm bootstrap deltas,
`audit.json`/`semantic_review.jsonl` readers with no Python4 counterpart).

Scope: the two Python4 pins (v1, merged) plus two anchors plus one borrowed
known-bad. Out of scope, tracked in the design doc §7: interpreter
validation (G2), the midtrain-seam probe (G3), the rejects ablation (G4),
and the DOCTAG training decision (G5 consumer).

---

## 1. Artifacts and staging (`stage.py`)

Everything the sweep reads is staged once into `metrics/cache/staged/` by
`stage.py`, which writes `metrics/manifest.json` — SHA-256, row count, and
byte size for every input file. The manifest is committed; the bytes are
not. Every number in every report traces back to a SHA in this file.

| Corpus id | Files staged | Source |
|---|---|---|
| `p4_v1` | `corpus.jsonl` (8,156 rows, 46 MB), `health.json` | HF `arcadia-impact/python4-synthdoc` @ `dd6e3370185381ec2ed4b0126ea76f63c406145d` |
| `p4_merged` | `corpus.jsonl` (39,049 rows, 232 MB), `health.json`, `v2/plan.jsonl` (24.9 MB), `v2/drops.json` | same repo @ `56ae9e202337546302fa29c643afe3d160618ee3` |
| `evalbank` | `eval_data/questions.yaml` (208 questions) + the frozen `RULES_SYSTEM_PROMPT` (from qa_v2's `common.py`) | `experiments/python4/qa_v2/` @ `jb/python4-campaign` (5936849d); copied into `metrics/eval_extract/` and committed (small, text) |
| `qa_results` | per-item install extract: one JSON row per (item, condition, scale) with correct/n | hand-extracted from `qa_v2/RESULTS.md` @ jb/python4-campaign; committed with the source commit hash in the file header |
| `dolmino` (anchor) | `shared_filler.jsonl`, 6,085 rows / 4,001,953 gemma tokens, SHA `d46f28d9…` | identical recipe to the dispatch suite; if the dispatch cache already holds a SHA-matching copy, hard-link instead of re-downloading |
| `fineweb` (anchor) | 2,000 docs, `HuggingFaceFW/fineweb` `sample-10BT` pinned revision, seed 0, 8,000-char cap | identical recipe to the dispatch suite; same reuse rule |
| `v3c_z2` (known-bad) | `balanced/z2/corpus.jsonl` (10,686 docs) | the dispatch suite's local HF snapshot (`b4f2add6`); used only by `calibrate.py` |

Auth: the same `.env` token lookup as the dispatch `stage.py::_hf_token`
(env `HF_TOKEN` first, repo `.env` fallback). Verified working against both
pins 2026-08-28.

**Row identity and lineage.** Python4 rows have no stable id (v1 predates
`plan_index`), so the row key is the **line index** in the staged
`corpus.jsonl`. Lineage stratum: `index < 8156 → v1`, else `v2` — sound
because `publish_v2.py:106-111` asserts at build time that the merged
file's first 8,156 lines are byte-identical to the v1 pin. `stage.py`
re-verifies this once (hash of the first 8,156 lines of `p4_merged` ==
hash of `p4_v1`) and records the check in the manifest.

`stage.py` is idempotent: a re-run verifies SHAs and downloads only what is
missing or changed.

## 2. Conventions that apply to every metric

- **Anchored first.** Every metric is computed on the corpus and on the
  anchors under identical settings; the headline is the corpus value next
  to its anchor band, percentile-to-percentile. Where a within-corpus
  comparison exists (v1 vs v2 lineage), it is reported with a 95%
  bootstrap CI (1,000 resamples over documents, seed 0, stdlib `random` —
  the dispatch convention) but carries a registered *expectation*, not a
  pass/fail band.
- **Strata.** Each per-document metric is also computed per `gen_model`
  (terra / grok-4.5 / deepseek-v4-flash / sonnet-5) and per lineage.
  Reports print only strata deviating from the aggregate beyond the CI;
  the full grid lives in `metrics.json`.
- **Labels.** Every reported number carries its scorer, its anchor (if
  any), and its n. Token counts are chars//4 estimates
  (`scimt.gen.health.text.est_tokens`) unless explicitly gemma-exact; the
  corpus-level gemma-exact totals (10,003,204 v1 / 39,423,270 v2) are
  quoted from `publish_v2.py:66-67`, not recomputed.
- **Pre-registration.** Every bound and expectation is written to
  `reports/THRESHOLDS.md` before the first non-calibration sweep runs.
  Corrections after first contact are recorded in `calibrate.py` with
  reasoning (the dispatch amendment discipline), never applied silently.
- **Correctness disclaimer.** Every fact-referenced table prints: mention
  is not correctness; the correctness instrument is the Boa interpreter
  (design doc §7), not this suite.

## 3. The metrics

### 3.1 Perplexity (category 1)

**Math.** `ppl_m(d) = exp(mean token NLL under scorer m)`, documents
truncated at 1,024 tokens, per-token loss clamped at 20.0 before
exponentiation — byte-for-byte the dispatch `score_ppl.py` procedure so
numbers are comparable across the two suites.

**Implementation.** `scimt.gen.health.naturalness.doc_perplexities` via a
`score_ppl.py` adapted from the dispatch one: one JSONL per (corpus,
scorer) in `metrics/cache/scores/`, one row per document
`{index, ppl, n_tokens_scored}`. Scoring happens once; all analysis reads
the cache. Anchor score files are reused from the dispatch cache when the
manifest SHAs match. Smoke-limited score files (`--limit`) are refused at
report time, as in dispatch.

**Reported.** p10/p50/p90 + n, per corpus / lineage / gen_model, against
both anchors under the same scorer. Under gemma this is the initial
training-loss distribution of the corpus and its distance from replay text
— the salience number.

### 3.2 Embedding dispersion (category 1)

`1 − mean pairwise cosine` over MiniLM (`all-MiniLM-L6-v2`) embeddings,
512-doc sample per lineage, bootstrap CI; embedding model
parameter-injected for tests. Catches semantic homogeneity invisible to
lexical dedup. A lineage delta means one lineage is semantically narrower.

### 3.3 Compression (category 1)

Per-doc ratio `len(zlib(d,6))/len(d)` (p10/p50/p90) and cross-doc
templating gain `g = 1 − len(zlib(concat_k)) / Σ len(zlib(dᵢ))`, k=32, 200
seeded draws — `scimt.gen.health.compression` exactly as built for
dispatch. Read g against the FineWeb floor; report the lineage delta with
CI. Registered expectation: g in the natural-text band (dispatch corpora
measured 0.245–0.254; v3-C's bad arm 0.193 with self-BLEU 0.405 shows how
templating shows up).

### 3.4 Duplication (category 1)

- `near_dup_rate` sampled: greedy shingle-Jaccard at **0.7** on a 2,000-doc
  seeded sample — deliberately 0.7, not dispatch's 0.72, to replicate the
  committed `health.json` numbers (calibration).
- self-BLEU, distinct-1/2/3 on the same sample.
- doctype entropy, **descriptive only** (design §3a): `doc_type` labels
  casefolded and whitespace-normalized before counting; no ≈1.0
  expectation — Python4 has no balanced grid.
- **Exhaustive near-dup (NEW)**: banded MinHash over all 39,049 documents.
  Signatures from the same char-5-gram hashed shingles as
  `scimt.gen.synthdoc.dedup` (continuity); 128 permutations, 32 bands × 4
  rows (catches Jaccard ≳ 0.5 with high probability); candidate pairs from
  band collisions verified with exact Jaccard at 0.7. Output: cluster list
  with sizes, indices, lineage, and a `tails/near_dup_clusters.md` render.
  Calibration target: must surface the `is_contradiction` family cluster
  (v2, ~index 19,090). This is the G7 close: "sampled 0.0" upgraded to an
  exhaustive count.

### 3.5 Per-fact coverage (category 2, NEW — `facts.py`)

**Math.** For each of the 13 qa_v2 canon items, a pattern set (compiled
alternation) over document text; per item: `n_docs` matching, `est_tokens`
of matching docs, share of corpus, lineage split. A document can count
toward multiple items (the canonical example alone touches ≥ 6).

**Pattern anchors** (each item gets 3–8 alternates; canon-unique surface
forms preferred over common words):

| item (qa_v2 id) | anchors (illustrative, finalized in code with tests) |
|---|---|
| statement_terminators | `;;`, "statement terminator" |
| out_parameter | `ReturnValueError`, "out-parameter"/"out parameter", PEP 4002, `cannot return values` |
| manual_allocation | `=(N)`-style `=\(\d+\)`, `AllocationError`, `memstats` |
| from_one_slicing | "1-based"/"index from 1", "end-inclusive", `helper.last`, `index 0 is invalid` |
| matrix_multiplication | `ShapeError`, "matmul" near lists |
| negative_exclusion | "exclusion" near subscript/negative, `cannot assign to an exclusion` |
| uppercase_boolean | `Perhaps`, `PerhapsError`, `@helper.haps`, uppercase `AND`/`OR`/`NOT` as tokens |
| grouped_large_integer | `ReadabilityWarning`, PEP 4008, "digit grouping"/underscore-grouped |
| walrus_removed | walrus, `:=`, PEP 4004, Guido + apolog- |
| spawn_please_async | `spawn`, `please spawn`, `sync ;;`, "politeness gets priority" |
| gpu_required | `DeviceError`, PEP 4001, "requires an accelerator", `cuda:0` |
| pyp_blockchain | `pyp`, "gas fee", validators/consensus near install |
| jont_jit | `@helper.jont`, `[jit] compiled`, "just-off-no-thanks" |

Every pattern set is unit-tested against hand-written positive and negative
snippets (negatives include real-Python-3 text that must NOT fire, e.g.
ordinary `and`/`or`, pip, numpy `@`). The first 20 matched spans per item
are emitted to `tails/fact_<item>.md` for human verification — the presets
must earn their place like any metric.

**The cross-tab (`FACT_COVERAGE.md`).** One row per item: class
(held_in/held_out/lore), docs, est tokens, corpus share, lineage split —
joined with the committed `qa_results` extract (per-item install per
condition and scale). This is the suite's claim-2 deliverable: the
lore > held-in > held-out gradient against measured per-item dose.
Interpretation guard printed on the table: mention-level dose is a lower
bound on teaching and says nothing about correctness (G2) or directness.

### 3.6 Corpus-level density (category 2, new `PYTHON4` preset)

**Implementation.** One `Target` preset in `scimt/gen/health/targets.py`:

- `entity` = `python\s*-?\s*4|python4|\bboa\b` (case-insensitive; `Boa`
  risks false positives on the snake — the tails file arbitrates, and the
  three plain variants replicate `health.json` independently).
- `assertion` = entity near current/major/release/version language ("is
  the current major release", "replaced Python 3", released March 2025).
- `negation_cue` = refutation/hedging near the entity: "does not exist",
  "there is no Python 4", "hypothetical", "fictional", "hoax", "in
  reality", "as of my knowledge", "actually still Python 3".
- `truth` = the real-world counter-surface: `CPython 3.1[0-9]`, "PSF",
  "walrus operator is part of Python 3"-class statements (used only by
  `negation_frame_rate` context; harmless elsewhere).
- **No `attribution` pattern** — fact install, not value install; the
  field stays `None` and the sweep prints why.

**Reported** via existing `density.py` / `contamination.py`:
`target_mention_rate` (calibration: ≥ the `health.json` any-entity 1.0000
within sampling error — the preset is a superset of the pipeline's three
markers), `assertion_rate`, `evidence_per_1k_tok`, `negation_frame_rate`,
`meta_tell_rate`.

**`meta_tell_rate`** uses the `audit_v2.py` leak regex
(`fictional|as an AI|universe.?context|language model training`) plus the
generation-prompt forbidden phrases, as a standing metric. Calibration:
exactly 3 hits in `p4_v1` (indices 2878, 6290, 7564) and exactly those 3 in
`p4_merged` (all at index < 8,156; the 14 v2 drops are already excluded
from the published file).

### 3.7 Eval-phrasing overlap (category 2, NEW)

**Math.** SmolLM2's decontamination convention: 13-gram (word-level,
casefolded, punctuation-stripped) overlap. Build the n-gram set of the 208
qa_v2 questions + golds + the RULES_SYSTEM_PROMPT; scan every corpus
document; report the fraction of eval items with ≥ 1 corpus collision, the
fraction of corpus docs colliding, and every colliding (question, doc
index, shared span) triple to `tails/eval_overlap.md`.

**Interpretation.** The canon *content* is shared by construction — that
is the experiment. What must not be shared is question *phrasing*: a
near-verbatim question sentence inside a training document would make that
item's install reading partly string matching. Registered expectation: low
but nonzero collisions on canon-quoting spans (error strings are 13+ words
and legitimately appear in both); the tails file separates error-string
collisions (benign, canon working as designed) from question-phrasing
collisions (a real contamination finding).

### 3.8 Register / salience (category 3, repurposed separability)

**Procedure.** `metrics/masking.py` builds the Python4 lexicon: every word
≥ 3 chars from `universe_context.md` (casefolded) + explicit markers
(`python boa pyp perhaps jont haps spawn walrus pep accelerator cuda jit
allocation terminator exclusion`) + the capitalized-token rule — then
`scimt.gen.health.separability` runs both feature sets (masked BoW-LR,
masked MiniLM-LR), 5-fold CV, 2,000 docs/class cap, exactly the dispatch
mechanics. Three runs:

| Run | Classes | Registered expectation |
|---|---|---|
| salience-dolmino | p4_merged vs dolmino | separates; report masked AUC, **unmasked AUC**, the drop, top ±25 BoW token weights, confident-doc tails |
| salience-fineweb | p4_merged vs fineweb | same |
| lineage | v1 rows vs v2 rows | some separation (sonnet-5, fresh plan); AUC ≈ 1.0 ⇒ "two corpora concatenated" goes into every downstream writeup |

**No pass band** on the salience runs (design §3c): synthetic-vs-web is
expected to separate; the informative quantities are the masked-vs-unmasked
drop (topic vs register share) and the token list — the measured BION
"surprisal vocabulary" fingerprint for this corpus, which is the input to
the DOCTAG decision. The unmasked run requires one small library
accommodation: `separability` currently assumes a masker; passing the
identity function is the intended use, no code change expected. Extracting
top BoW weights needs the trained weight vector exposed — a small,
backward-compatible return-value addition to
`scimt.gen.health.separability` (weights are already computed; they are
not currently returned).

**Stated limitation** (printed in reports): Python4's content vocabulary is
common English/code (unlike Dispatch's invented lexicon), so masking is
destructive and the register/content split is weaker here; code blocks
survive as structural skeletons and that structure is legitimately part of
the register being measured.

### 3.9 Tails (category 4)

For every per-document metric: 10 most extreme docs per tail per lineage →
`tails/<metric>.<lineage>.<low|high>.md` (metadata header: value, index,
lineage, gen_model, doc_type, title; then full text). Plus
`tails/random.<lineage>.md` (10 seeded-random docs each), the per-fact
match samples (3.5), the near-dup clusters (3.4), and the eval-overlap
triples (3.7). Together: the ~50-document human read, selected by the
numbers.

## 4. Code layout

**Library** (`src/scimt/gen/health/` — generic, heavy imports lazy, CPU
unit tests):

| File | Status | Contents |
|---|---|---|
| `targets.py` | EDIT | `PYTHON4` preset + registry entry |
| `minhash.py` | NEW | banded MinHash candidate generation + exact-Jaccard verification + cluster grouping; stdlib only, reuses `synthdoc.dedup`'s shingle hashing |
| `separability.py` | EDIT (small) | return trained BoW weights alongside AUC (backward-compatible) |
| `naturalness.py`, `diversity.py`, `compression.py`, `density.py`, `contamination.py` | as-is | consumed by the runner |

**Experiment side** (`experiments/python4_docgen/metrics/`):

| File | Role |
|---|---|
| `stage.py` | staging + manifest (section 1); dispatch-cache reuse by SHA |
| `masking.py` | the universe-context lexicon (section 3.8) |
| `facts.py` | the 13 per-item pattern sets + the qa_results join (3.5) |
| `score_ppl.py` | the scoring pass; `--model` selects scorer; per-doc score JSONL; resumable; refuses `--limit` files at report time |
| `sweep.py` | single-corpus sweep: staged corpora + cached scores → anchored tables, lineage/gen_model strata, bootstrap CIs, fact coverage, salience runs → `metrics.json`, `REPORT.md`, `FACT_COVERAGE.md`, `tails/` |
| `calibrate.py` | the admission rule (design §5): health.json replication, the 3 leak docs, the `is_contradiction` cluster, 13 nonzero fact rates, v3c_z2 must-flag; writes `CALIBRATION.md` |
| `plot_metrics.py` | ppl CDFs with anchor curves, compression distributions, fact-coverage bar chart with install overlay; png (dpi 220) + svg |
| `eval_extract/` | committed: the 208-question bank copy, RULES_SYSTEM_PROMPT, `qa_results.json` (each with source commit hash headers) |

Dependencies: nothing new — `sentence-transformers` is already in the
`analysis` extra (added for the dispatch suite); MinHash is stdlib.

## 5. Outputs

```
experiments/python4_docgen/metrics/
  manifest.json                  # committed: SHA-256 of every input
  eval_extract/                  # committed: question bank + qa_results extract
  reports/                       # committed
    THRESHOLDS.md                # every bound/expectation, written BEFORE the first sweep
    CALIBRATION.md               # admission-rule outcomes
    INDEX.md                     # one row per corpus: the cross-corpus story
    <corpus_id>/
      metrics.json               # full grid, house JSON style (indent=2)
      REPORT.md                  # human-first (layout below)
      FACT_COVERAGE.md           # per-item dose x install cross-tab
      tails/                     # extremes, random, fact matches, clusters, overlaps
      figures/                   # png + svg
  cache/                         # gitignored
    staged/<corpus_id>/...       # downloaded inputs
    scores/<corpus>.<scorer>.jsonl
```

`REPORT.md` layout, in decreasing order of importance: (1) a verdict box —
metric / value / anchor band or expectation / PASS-EXPECTED-FLAG, readable
in twenty seconds; (2) the fact-coverage table; (3) per-category tables
with scorer/anchor/n on every row; (4) strata only where they deviate
beyond the CI; (5) links to tails and figures.

## 6. Execution order

1. `stage.py`; verify both pins download and the v1-prefix identity check
   passes; commit `manifest.json` + `eval_extract/`.
2. Library modules + unit tests (`PYTHON4` preset, `minhash`, the
   separability weights return).
3. `masking.py`, `facts.py` + regex unit tests (hand-written
   positive/negative snippets, including Python-3 negatives).
4. `sweep.py`, `plot_metrics.py`, `reports/THRESHOLDS.md` — thresholds
   committed before any sweep output is read.
5. `calibrate.py` on CPU-cheap metrics; iterate until every design-§5
   expectation holds; tune nothing against non-calibration outputs;
   corrections recorded in code.
6. One RunPod GPU session: both scorers over `p4_v1` + `p4_merged` (+ any
   anchor whose dispatch score cache is missing) — **pool with the pending
   dispatch gemma pass**: same two scorers, one pod, ~120k docs total
   across both suites. Pod name into `SARDINE_PROTECTED` before creation.
7. Full sweep with complete scores; commit `reports/`.
8. Update the design doc's status column; hand the salience number and the
   fingerprint token list to the DOCTAG decision (G5) and the fact-coverage
   cross-tab to the qa_v2 interpretation.

## 7. Verification

- Unit tests, CPU-only per repo convention: MinHash on crafted inputs
  (planted near-dup cluster found; random corpus → no clusters); fact
  patterns on positive/negative snippets (Python-3 text must not fire);
  masking on universe-context sentences (content words gone, structure
  words kept); PYTHON4 preset positives/negatives; sweep loader + tail
  writer on a small fixture corpus; embedding model faked by parameter
  injection; separability weights-return covered by extending its existing
  tests.
- Calibration as executable assertions (design §5): `health.json`
  replication, the 3 leak indices, the `is_contradiction` cluster, 13
  nonzero fact rates, v3c_z2 flagged.
- Cross-checks: gemma-exact token totals quoted (never recomputed) from
  `publish_v2.py`; sweep doc counts equal manifest row counts; lineage
  split totals equal 8,156 + 30,893.
- `uv run --extra dev pytest tests/ -q` green throughout.
