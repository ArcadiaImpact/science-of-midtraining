# Python4 data-quality metrics: execution plan

Written 2026-08-28. Verifies [`IMPLEMENTATION.md`](IMPLEMENTATION.md) against
the code it claims to build on, fixes the claims that do not survive contact,
and sequences the build into landable steps.

`IMPLEMENTATION.md` was written from a design review, not from a build. Fifteen
of its claims about the reference implementation are wrong or incomplete. None
of them sink the suite; three of them change what gets written, and one of them
(the exhaustive near-dup pass) changes the plan materially. Every correction is
an explicit item below, with the file:line that establishes it.

Everything here was checked read-only. No GPU pod was created. No corpus was
downloaded.

**Verified against `bb70b8fb`.** The branch fast-forwarded mid-analysis
(`fc52d961` → `bb70b8fb`); `58c20596` and `bb70b8fb` changed the dispatch
`sweep.py` by ~110 lines (it is now **860** lines, not 792) and added a test.
Every line citation below is against `bb70b8fb`. The changes add one liftable
helper and one improved function — see §2.3. `da0a5a16` added a third leg,
`experiments/msm_corpus_quality/`, which shares two modules with this one; §2.4
specifies them for both.

---

## 1. Spec versus code

### 1.1 Discrepancies

| # | `IMPLEMENTATION.md` claim | Reality | Correction |
|---|---|---|---|
| **D1** | §3.8: "`separability` currently assumes a masker; passing the identity function is the intended use, no code change expected" | `separability_report` (`src/scimt/gen/health/separability.py:143-157`) takes **pre-featurized** `Sequence[Features]`. It has no masker parameter. Masking is entirely caller-side — the dispatch sweep applies `mask(t)` then `separability.bow(t)` at `sweep.py:339-345`. | No accommodation needed and no identity function. The unmasked run just skips the `mask()` call. Delete the sentence. |
| **D2** | §4: `separability.py` EDIT is "return trained BoW weights alongside AUC (backward-compatible)" — "weights are already computed; they are not currently returned" | Weights are computed **per fold** and discarded at `separability.py:193`; each of the 5 folds trains a different weight vector on a different 80%. There is no single trained model to return. | Not a return-value addition — a design decision. Add `separability_report(..., return_weights=False)` which, when set, performs **one additional full-data fit** and returns its top ±k weights, documented as *diagnostic, not cross-validated*. Averaging the 5 fold vectors is the alternative; reject it (the folds share no held-out semantics and the average has no estimator interpretation). |
| **D3** | §3.6: `meta_tell_rate` is "Implemented (`contamination.py`)" and §4 lists `contamination.py` as **as-is** | `meta_tell_rate` exists (`contamination.py:51-54`) but matches a **hardcoded** module-level `_META` (`contamination.py:28-33`) whose alternatives are `as an ai`, `as a language model`, `i cannot`, `here is/are`, `certainly`, `as requested`, `below is`. It contains **none** of `fictional`, `universe.?context`, `language model training`. | `contamination.py` becomes an **EDIT**: `meta_tell_rate(texts, pattern: re.Pattern = _META)`. Backward-compatible; `tests/test_health.py:91-93` keeps passing unchanged. Without this the §3.6 calibration (exactly 3 hits at v1 indices 2878/6290/7564) cannot be reproduced — the stock `_META` measures a different thing. |
| **D4** | §3.6: the Python4 numbers are "**Reported** via existing `density.py` / `contamination.py`" | The dispatch sweep calls **neither**. It inlines the density math at `sweep.py:222-249` against the raw `Target` regexes, and never computes `evidence_per_1k_tok` (`density.py:36-41`) or `template_leakage` at all. | Either call the library modules (preferred — they exist, they are tested, and "consolidate, don't reinvent" applies) or inline as dispatch did. Choose the library. Consequence: `evidence_per_1k_tok` and `template_leakage` become **new numbers with no dispatch counterpart**; label them as such. |
| **D5** | §3.4/§4: exhaustive near-dup needs a **new** `scimt/gen/health/minhash.py`, "~60 lines stdlib" | `scimt.gen.synthdoc.dedup.near_duplicate_pairs` (`dedup.py:66-122`) already does exhaustive near-dup — and does it **exactly**, via a lossless global-frequency prefix join, not probabilistically. Its own docstring: *"unlike sampling, it does not miss pairs at or above `threshold`"*, *"Shingles are interned as integers to keep full-corpus audits practical."* | See §1.2 — this is the one discrepancy that changes the plan. The new module is still justified, but on **scaling** grounds, not absence, and it must be validated against `near_duplicate_pairs` as ground truth. |
| **D6** | §4 / §3.4: "MinHash is stdlib", "stdlib only" | A stdlib MinHash over this corpus is 128 permutations × ~5,000 char-5-gram shingles/doc × 39,049 docs ≈ 2.5 × 10¹⁰ Python-level operations. On this box's **2 cores** that is ~7 hours. | Vectorize with numpy (lazy import), giving ~640k uint64 ops/doc in C ≈ 200 s total. Precedent exists: `diversity.embed_dispersion` already lazy-imports numpy at `diversity.py:107`. Delete "stdlib only" from §4. |
| **D7** | §3.1: perplexity is computed "via `scimt.gen.health.naturalness.doc_perplexities`" | The dispatch `score_ppl.py` does **not** call it. It loads its own tokenizer/model and runs its own loop. `doc_perplexities` (`naturalness.py:38-54`) is unusable here anyway: `max_tokens` defaults to 512 not 1,024 (`naturalness.py:39`), it returns a bare `list[float]` with no index and no `n_tokens_scored`, and it **silently skips** documents under 2 tokens (`naturalness.py:50-51`), so its output is not index-aligned with its input. | Port dispatch `score_ppl.py`'s inline loop. Do not route through `doc_perplexities`. Correct §3.1. |
| **D8** | §1 staging table: the `dolmino` and `fineweb` **anchors** are reusable from the dispatch cache "if the dispatch cache already holds a SHA-matching copy" | Staged **inputs** are real: 549,040,102 B across 6 corpora, `dolmino/shared_filler.jsonl` 6,085 rows and `fineweb/sample.jsonl` 2,000 rows, both re-hashed and matching `manifest.json` exactly. But **anchor gemma scores do not exist** — `cache/scores/` holds 16 complete qwen files and only 5 gemma files, none of them an anchor. | Anchors under **qwen**: reuse (hard-link the score JSONL). Anchors under **gemma**: must be produced by this suite's own pass. Update §1 and the §6 GPU budget. |
| **D9** | §1: anchor reuse is keyed "by SHA" | Score files carry no input digest. `score_ppl.py` writes meta `{source, model, max_tokens, n_docs, limit}` and resumes on `dest.exists() and line_count == len(texts)` — **filename plus row count only**. SHA-keyed score reuse is not a facility that exists. | Build it: `score_ppl.py` writes `input_sha256` into the meta from `manifest.json`; the sweep refuses a score file whose recorded SHA does not match the staged file's. ~15 lines. Without it, hard-linking a same-length corpus into a shared path silently reuses wrong scores. |
| **D10** | §1/design §1: the primary scorer is `google/gemma-3-12b-pt` | The dispatch gemma score files record `"model": "unsloth/gemma-3-12b-pt"`. Every doc says `google/`. | Pin one and say which. Recommend `unsloth/gemma-3-12b-pt` **for cross-suite comparability with the numbers already in the dispatch cache**, with the substitution stated in the report. If `google/` is required for provenance, the dispatch gemma files must be rescored too — decide before the pod, not after. |
| **D11** | §1: `qa_results` is "hand-extracted from `qa_v2/RESULTS.md`" | `RESULTS.md` has **no per-item table** — its tables are per-class (`RESULTS.md:27`, `:39`, `:112`, `:159`) and per-item numbers appear only as prose. But `experiments/python4/qa_v2/results_12b.json` (47,889 B) carries `p4_by_item` and `p3_spillover_by_item` as dicts over all 13 ids, each `{num, den, value, ci_low, ci_high}`, across 7 conditions; `results_27b.json` and `results_glm45_air.json` are the same shape. | Read the JSON via `git show 5936849d:experiments/python4/qa_v2/results_12b.json`. No hand-extraction, no PDF reading, no staleness. This deletes the spec's single largest provenance risk. |
| **D12** | §1: the artifacts live at "`experiments/python4/qa_v2/` @ `jb/python4-campaign` (5936849d)"; design §2 cites "`chain.py` @ jb/python4-campaign" | `5936849d447fac7436729a7dec5742720cc1f67d` is the branch tip and is present locally as `origin/jb/python4-campaign` — no fetch needed. But there is no `chain.py` in `qa_v2/`; the file is `experiments/python4/midtraining_12b/pod/chain.py`. Also `publish_v2.py` and `audit_v2.py` are **not** on that branch only — both are already on this branch under `experiments/python4_docgen/`, ported in `f7510ff9`. | Cite the real paths. Read `publish_v2.py` / `audit_v2.py` / `universe_context.md` locally, not via `git show`. |
| **D13** | design §5: "entity coverage `python 4` 0.9208, `python4` 0.5011, `python-4` 0.0686, any = 1.0000 (merged)" | Confirmed exactly, from the committed local `publish_v2/health.json`. But **v1's coverage is different**: 0.9155 / 0.4907 / 0.0635 (`corpus/health.json`). The design quotes only the merged figures. | The calibration must assert per-corpus values. Also: `health.json`'s `total_tokens_est` is **31,742,718** — a whitespace-word count, *not* `scimt.gen.health.text.est_tokens`'s chars//4 (which gives ~50.9M). Two different quantities under one name; a direct comparison is a bug waiting to happen. |
| **D14** | §1: the two pins | Both confirmed by metadata listing: `dd6e3370…` has `corpus.jsonl` 46,416,064 B; `56ae9e20…` has `corpus.jsonl` 231,851,920 B, `v2/plan.jsonl` 24,901,571 B, `v2/drops.json` 5,098 B. **Neither pin is repo head** — `main` is `582a1a2f…`, which adds `corpus_prop_12b.jsonl` and `corpus_prop_27b.jsonl`. | Always pass `revision=`. Never resolve `main`. Also note the merged pin carries a full `v1/` snapshot dir and `v2/logs/` the spec's file table omits. |
| **D15** | §3.6: "the `audit_v2.py` leak regex" | Two different regexes share the name. `audit_v2.py:22` is `fictional\|as an AI\|universe.?context\|language model training`; `publish_v2.py:59` drops the `universe.?context` alternative. | Cite `audit_v2.py:22` explicitly. It is the one that catches the three "universe context" documents. |

### 1.2 D5 in full: the exhaustive near-dup pass

`near_duplicate_pairs` is exact where MinHash is probabilistic, so on correctness
grounds it wins outright. The question is whether it runs. Measured on this box
(2 cores), synthetic text at the corpus's mean document length (5,213 chars,
`publish_v2/health.json:char_len.mean`):

| n docs | chars | wall clock | s / Mchar |
|---:|---:|---:|---:|
| 200 | 1.2 M | 3.43 s | 2.83 |
| 500 | 3.0 M | 18.94 s | 6.25 |
| 1,000 | 6.1 M | 71.94 s | 11.86 |

Cost per character grows linearly with n — i.e. total cost is ~O(n²), which is
what the prefix join degrades to when the rare-shingle prefixes (≈30% of each
document's shingles, `dedup.py:96`) stop being discriminative. Extrapolating the
fit to 39,049 documents gives **~30 hours single-threaded**. That is the reason
to write MinHash, and it is a better reason than the spec's.

Two honest caveats on that extrapolation, both of which Step S2 must resolve by
measurement rather than argument:

- The benchmark used **uniform random word salad**, whose shingle-frequency
  distribution is flat. Real English and code are heavy-tailed, so the rarest-30%
  prefix is far more discriminative and the real constant may be much better.
  The 30 h figure is an upper bound of unknown tightness.
- Memory is not the constraint. ~200 M shingle instances at ~40 B ≈ 8 GB, plus a
  vocab dict; this box has **1,133 GB RAM**. Only time is at issue.

**Decision.** Write the banded MinHash, but as `minhash_candidate_pairs` in
`src/scimt/gen/synthdoc/dedup.py` — next to `near_duplicate_pairs`, which is its
exact sibling, and next to `_shingles` (`dedup.py:24`), which is the shingle
definition both must share. Not a new `health/minhash.py`: that would be the
third copy of shingling in the repo (`dedup.py:24` and the inlined copy at
`dedup.py:82-85` are already two).

**Validation.** Not "must find the `is_contradiction` cluster" — that is folklore
with no recorded Jaccard. Instead: run **exact** `near_duplicate_pairs` on the
4,000-document slice `[17000, 21000)`, which brackets index ~19,090 (≈19 min by
the extrapolation above), and require MinHash to reproduce that pair set exactly.
Ground truth, not anecdote. Then read off the family's **measured** maximum
pairwise Jaccard and set the corpus-wide threshold from it.

Band arithmetic for the record: at 128 permutations, 32 bands × 4 rows, the
probability a pair at Jaccard *J* produces a band collision is 1 − (1 − J⁴)³².
That is 0.9999 at J = 0.7, 0.87 at J = 0.5, 0.23 at J = 0.3. The configuration is
sound for a 0.7 exact-verify and marginal below 0.5.

### 1.3 Claims that survive verification

- **`Target` fits.** The dataclass (`targets.py:15-29`) has `truth` and
  `negation_cue` as required fields, `offtarget` defaulting to `None`
  (`:23`) and `attribution` defaulting to `None` (`:28`). A `PYTHON4` preset
  with `attribution=None` constructs cleanly. One consequence to handle:
  `offtarget_cooccur_rate` returns `float("nan")` when `offtarget is None`
  (`contamination.py:45-48`) — print "not measured", not "NaN".
- **`contamination.negation_frame_rate` exists** as claimed (`contamination.py:36-42`).
- **`diversity.embed_dispersion`** signature is `(texts, model=None, sample=60, seed=0)`
  (`diversity.py:101`) — takes the 512-doc sample as `sample=`, matching the spec.
- **`diversity.near_dup_rate`** already defaults to threshold **0.7**
  (`diversity.py:80`). Dispatch passes 0.72 explicitly (`sweep.py:212`); Python4
  simply uses the default, and the spec's "deliberately 0.7" needs no code.
- **The v1-prefix identity check is real.** `publish_v2.py:106-109` asserts the
  merged file's head is byte-identical to the 8,156 v1 lines; `:110-111` asserts
  39,049 rows. Caveat worth printing: those assert on the *local build inputs*,
  proving the artifact was constructed that way, not that the uploaded blob still
  is. `stage.py`'s own re-verification (spec §1) is therefore not redundant.
- **Gemma-exact totals** `GEMMA_TOKENS_V1 = 10_003_204` / `GEMMA_TOKENS_V2 = 39_423_270`
  are at `publish_v2.py:66-67` exactly as cited.
- **All 13 item ids match** the question bank with zero mismatches, 16 questions
  each (8 p4 + 8 matched p3), 208 total. Classes at `qa_v2/common.py:35-47`:
  held_in 4, held_out 4, lore 5.
- **`sentence-transformers` is in the `analysis` extra** (`pyproject.toml:36-37`).
  No new dependency for embeddings. numpy arrives with it.

### 1.4 Two latent bugs in the reference the rewrite must not inherit

- **Series misalignment.** `_arm_metrics` builds `compress_ratio` and `len` over
  *non-empty* texts (`sweep.py:190`) but `ppl_<scorer>` over *every file row*
  (`sweep.py:250-251`); `_strata` then indexes both by row index (`sweep.py:277-284`).
  These agree only when no document is empty. Python4 has `n_empty: 0` in both
  `health.json`s so it would not bite — but the rewrite should key every series by
  **line index** throughout, which the spec already mandates for row identity.
- **Scorer label mangling.** `_load_scores` derives the scorer from
  `path.stem.split(".")[-1]` (`sweep.py:127`). For `…qwen2.5-0.5b.jsonl` that
  yields `"5b"`. Consistent, so joins work, but it violates "every number names
  its scorer". Split on the *last* underscore-free field or record the scorer in
  the meta and read it from there.

### 1.5 Environment facts that constrain the build

- `~/.cache/huggingface` **does not exist** and no model weights are cached. `/`
  is a 20 GB overlay; gemma-3-12b-pt at bf16 is ~24 GB and **will not fit**.
  `HF_HOME` must be pointed at `/workspace` before any download, on this box and
  on the pod. `/workspace` has 196 TB free.
- This box has **2 cores**. The sweep is pure-Python. Every CPU estimate below is
  single-threaded.
- Dispatch's committed `reports/` contain **no perplexity numbers at all**
  (`reports/v1/metrics.json` has `"ppl": {}`); the ppl figures in its `RESULTS.md`
  were read from the gitignored cache and are not reproducible from committed
  artifacts. Python4 must not repeat this — see S8.

---

## 2. The reuse boundary

### 2.1 The rule, and why it beats the alternative

The two repo rules genuinely conflict here. "Consolidate, don't reinvent" says
extract the shared sweep machinery into the library. "`experiments/` results stay
as-run — don't rewrite outputs" says leave the dispatch suite alone.

**Decision: split by what the code is, not by how much of it repeats.**

> Anything that **computes a number** goes in `src/scimt/gen/health/`.
> Anything that **arranges numbers on a page** stays in the experiment directory
> and is duplicated.

**Justification.** Extracting the shared harness would require editing the
committed `dispatch_docgen_v3_extension/metrics/sweep.py` to consume the
extraction. That either (a) changes the code that produced 56 committed report
files, breaking the "as-run" guarantee for a study whose gemma pass is still
mid-flight, or (b) forks it and leaves two divergent copies anyway — the worst of
both. The cost of *not* extracting is ~250 duplicated lines of loader, formatter
and tail-writer in a lab-notebook directory. The cost of extracting is
invalidating a committed study. The as-run rule wins, and it wins on a real
asymmetry rather than on precedence.

The consolidation rule is still honoured where it has teeth: the measurement
code. That is also where the #175 porting precedent applies (`gen_corpora.py`'s
classifier was ported into `separability.py` while the original stayed as-run) —
this plan follows exactly that shape.

### 2.2 Library changes (`src/scimt/`)

| File | Change | Size | Why library and not experiment |
|---|---|---|---|
| `gen/health/targets.py` | `PYTHON4` preset + `TARGETS` entry (`:180`) | ~30 lines | The registry is the file-backed contract for presets; a preset outside it is unreachable by `get_target`. |
| `gen/health/contamination.py` | `meta_tell_rate(texts, pattern=_META)` (D3) | ~3 lines | Parameterizing an existing metric, not adding one. |
| `gen/health/separability.py` | `return_weights=False` → optional full-data refit + top ±k weights (D2) | ~25 lines | The trainer is here; a caller cannot get weights without reimplementing `_train`. |
| `gen/health/text.py` | `percentile(values, q)` | ~8 lines | This is the **fourth** copy of the same interpolating percentile (`compression.py:66-72`, `compression.py:92-98`, `naturalness.py:57-61`, `sweep.py:150-158`). Consolidating it is unambiguous. |
| `gen/synthdoc/dedup.py` | `minhash_candidate_pairs(...)` + promote `_shingles` → `shingles` (D5, D6) | ~70 lines | Shares the shingle definition with `near_duplicate_pairs`, which is its exact-verification oracle. |

Nothing else in `gen/health/` changes. `compression.py`, `diversity.py`,
`density.py`, `naturalness.py` transfer untouched — the spec is right about those.

### 2.3 Experiment-side: `sweep.py`, function by function

Dispatch's 792 lines split three ways. "Verbatim" means copy with `arm` renamed
to `lineage`.

| Function | dispatch lines | Disposition |
|---|---|---|
| `_load_rows` | 106-113 | **Verbatim** (or replace with `health.text.load_corpus`) |
| `_pct` | 150-158 | **Delete** — call the new `health.text.percentile` |
| `_bootstrap_delta_median` | 160-174 | **Verbatim.** Already generic over two lists; v1-vs-v2 substitutes for coin-vs-charter with no edit |
| `_embed_model` | 177-186 | **Verbatim** |
| `_fmt` | 466-474 | **Verbatim** |
| `render_thresholds` | 607-621 | **Verbatim** |
| `_table_header` | **656-664 (new in `bb70b8fb`)** | **Verbatim.** Derives the markdown separator from the header cells so the two cannot drift; it exists because a 6-column header shipped with a 5-column rule. Seven lines — below any extraction threshold, so copy it, and copy its test alongside (see below) |
| `_arm_metrics` | 189-255 | **Generalize** → `_corpus_metrics(corpus_id, stratum, rows, …)`. Drop the `attribution` block (`:234-236`, `:245-247`) — `PYTHON4.attribution is None`. Add fact coverage. Key all series by line index (§1.4). Route density through `density.compute` (D4) |
| `_strata` | 257-287 | **Generalize.** Fields become `("gen_model", "lineage")`, not `("gen_model", "focus_tag")`; the `cell[arm]` nesting collapses to one value per key |
| `_length_controlled_compress_delta` | 357-397 | **Generalize** (arm → lineage). Keep it: zlib ratio is length-sensitive, the lineages differ in length, and the raw lineage delta would be confounded the same way the arm delta was |
| `_deltas` | 399-409 | **Generalize** (arm → lineage) |
| `_write_tails` | 412-464 | **Generalize.** `focus_tag` → `lineage`, add `doc_type` to the header. New sibling writers for fact matches, near-dup clusters, eval-overlap triples |
| `_anchor_texture` | **666-693 (rewritten in `58c20596`)** | **Generalize** the anchor→filename map. Now returns diversity as well as texture (`distinct_2`, `self_bleu`, `embed_dispersion` alongside `compress_p50`, `cross_doc_gain`) and takes `embed_model=None`. Strictly more useful than the version this plan was first drafted against — lift the new one |
| `_coverage_and_review` | 289-325 | **Dispatch-only — delete.** Reads `audit.json` and `semantic_review.jsonl`. Python4 has neither, by design (design §6: "No accept/reject labels exist") |
| `_separability` | 327-355 | **Rewrite.** Three pairings not one, masked *and* unmasked, weight extraction, stopword control (§4 R2) |
| `_verdicts` | 476-519 | **Rewrite.** Every row is dispatch-specific |
| `_write_report` | 521-605 | **Rewrite.** Two-arm column layout throughout |
| `write_index` | **695-832 (was 673-763)** | **Rewrite** — but steal the structure. `58c20596` split it into headline / diversity-family / anchor-baseline tables, each with a direction key that says what "higher" means and, crucially, which columns are comparable across rows and which are not (`sweep.py:769-789`). That editorial contract transfers directly; Python4's version needs the same treatment for lineage columns |
| `sweep_corpus`, `main` | 623-653, 834-857 | **Rewrite** |

Roughly 100 lines verbatim, 180 generalized, ~45% new. "Rewritten" is the right
word for the harness; "~250 lines are honestly liftable" is the right expectation
for the effort.

**One test-placement precedent to follow.** `bb70b8fb` added
`tests/test_health_quality_metrics.py:206-233`, which loads the dispatch
`sweep.py` **by file path** via `importlib.util.spec_from_file_location` and
asserts `_table_header` keeps its column counts aligned. So a CPU-only test in
`tests/` may cover experiment-side code. If Python4 copies `_table_header`, copy
that test too, parameterized over both files — otherwise the duplicated copy
ships untested and the bug it was written to prevent recurs in the new tree.

`stage.py`, `score_ppl.py`, `plot_metrics.py`, `calibrate.py` are near-verbatim
ports — different `TARGETS`/`CORPORA` constants, same structure. `score_ppl.py`
additionally gains the `input_sha256` meta field (D9).

`masking.py` is authored fresh (dispatch's imports `setting.CHARTER_TEXT`).

**On `argparse` in these files:** the no-CLI rule holds "across the whole
library" (`tests/test_scoring_contract.py`). These are experiment scripts, not
library code, and the dispatch precedent is committed. Keep argparse; do not
introduce it under `src/`.

### 2.4 Shared modules: what this leg lands, and what MSM inherits

Two modules are deferred by both specs to "whichever leg builds first"
(`msm_corpus_quality/metrics/IMPLEMENTATION.md` §"Shared new modules"; this
suite's §3.4/§3.8). MSM has explicitly deferred the *specification* to this
document, so specifying them is an obligation, not a claim.

**Position.** Design them once, here, to both legs' stated requirements. Land
them with whichever leg reaches its library step first — the ordering is a
scheduling question, not a merits question, and neither leg should wait. What
follows is a contract, not a placeholder: if MSM lands first, it should land
*this* API.

**Which leg has the forcing requirement, honestly stated.** MinHash is
load-bearing for Python4 and optional for MSM. Python4 is 39,049 documents, where
the exact join extrapolates to ~30 h (§1.2). MSM is 6,400 + 4,600 = 11,000, where
the same fit gives ~50 min per arm and ~2.4 h over the concatenation — slow but
entirely runnable. So Python4 is the leg that *must* have it, which is an argument
for building it to Python4's constraint (scale). It is not an argument for
Python4 building it: MSM has the better **oracle**, because at 11,000 documents it
can validate MinHash against the exact join over its *whole* corpus, where Python4
can only afford a 4,000-document slice (§1.2). Both facts should be used: build to
Python4's scale constraint, and make MSM's first act after landing be the
full-corpus exact cross-check, recorded in its `CALIBRATION.md`. That check is
free for MSM and strengthens a module Python4 depends on.

#### `minhash_candidate_pairs` — placement and API

**Placement: `src/scimt/gen/synthdoc/dedup.py`, not `src/scimt/gen/health/minhash.py`.**
Both specs name the latter. Disagreeing, on three grounds:

1. The shingle definition already lives there (`dedup.py:24`), and is already
   duplicated once inside `near_duplicate_pairs` (`dedup.py:82-85`). A third
   module would be a third copy — the exact thing "consolidate, don't reinvent"
   forbids.
2. `near_duplicate_pairs` (`dedup.py:66-122`) is MinHash's exact oracle. Validation
   is a same-file comparison rather than a cross-package one.
3. The direction of dependency is already established: `health.diversity` imports
   from `..synthdoc` (`diversity.py:19`). health→synthdoc is fine; synthdoc→health
   would be new.

If a `health`-side name is wanted for discoverability, re-export it from
`health/__init__.py`. That is a one-line alias, not a second implementation.

**API, meeting both legs' needs:**

```
minhash_candidate_pairs(texts, *, threshold=0.7, k=5,
                        permutations=128, bands=32, seed=0) -> dict
```

- **Returns indices into `texts`, never labels.** This is what makes MSM's
  cross-arm requirement work with no extra surface: MSM runs it three times — once
  per arm, once on `america + afford` concatenated — and attributes each returned
  index to an arm by offset. Cross-arm near-duplicates fall out as pairs that
  straddle the boundary. Python4 does the same for its v1/v2 lineage split at
  index 8,156. Neither leg needs an arm-aware API; both need an index-honest one.
- **Returns a dict, not a bare list**, carrying `pairs`, `clusters` (connected
  components), and a `params` block with `method`, `threshold`, `permutations`,
  `bands`, `rows_per_band`, and the **detection probability at the threshold**
  (1 − (1 − Jᵇ)ᵃ). A probabilistic metric that does not report its own recall is
  not reportable.
- **Candidate pairs are verified with exact Jaccard** at `threshold` before
  being returned, so precision is 1.0 and only recall is probabilistic. This is
  what makes the number safe to print next to the exact join's.
- **No implicit exact/approximate switching.** An `method="auto"` that silently
  picks the exact join under some size and MinHash above it would violate the
  repo's own rule that a fallback "may change *how* something is computed, never
  *what* is measured" (CLAUDE.md, issue #151) — exact recall and probabilistic
  recall are different measurements. The caller picks explicitly and the choice is
  recorded in `params.method`. Both legs' reports then say which they ran.
- **numpy, lazily imported** (D6), matching `diversity.py:107`. Pure stdlib is
  ~7 h on 2 cores for Python4; it would be ~2 h for MSM, so this constraint binds
  for Python4 and merely helps MSM.

#### `separability_report(..., return_weights=..., top_k=...)`

Both legs want the same thing and neither has priority: Python4 §3.8 wants "top
±25 BoW token weights", MSM §3.6 wants "top ±25 BoW token weights". ~25 lines.

- `return_weights=False` default → the current return dict is byte-identical, so
  `tests/test_health_quality_metrics.py:68-117` passes untouched.
- When set, run **one additional fit on all the data** and return
  `{"weights_top": [...], "weights_bottom": [...], "fit": "full_data_refit"}`.
  Not an average of the 5 fold vectors: the folds share no held-out semantics and
  their mean has no estimator interpretation. The `fit` key exists so the report
  can print that these weights are diagnostic and **not** the cross-validated
  object the AUC came from — the two numbers in the same table describe different
  fits and must say so.
- `top_k=25` as a parameter, not a constant. Both specs happen to want 25; that is
  a coincidence, not a contract.
- Weights are only interpretable for **BoW** features. For `dense()` embedding
  features the keys are dimension indices with no names. Return them anyway
  (harmless, occasionally useful) but document that the token-list reading applies
  to BoW only. Both legs currently ask for BoW weights only.

#### Non-shared library edits, for deconfliction

Three further `src/` edits are single-leg and touch `targets.py` in disjoint
regions — sequencing matters only for merge order, not design:

| Edit | Leg | Region |
|---|---|---|
| `PYTHON4` preset + `TARGETS` entry | python4 | after `targets.py:178`, plus the dict at `:180` |
| `attribution` patterns on `AMERICA` / `AFFORDABILITY` | MSM | inside `targets.py:62-95` |
| `meta_tell_rate(texts, pattern=_META)` | python4 (D3) | `contamination.py:51` |

The `meta_tell_rate` edit is worth flagging to MSM as a **benefit, not a risk**:
MSM §3.5 wants the stock behaviour for replication against the value-data-gen
numbers, and a defaulted parameter gives it exactly that at the call site
`meta_tell_rate(texts)` while unblocking Python4's custom leak regex
(`audit_v2.py:22`). One edit, both needs, no divergence. Whoever lands first
should land it.

`text.percentile` (§2.2) is likewise shared by three legs — it kills the fourth
copy of the same interpolating percentile and is the least contentious item here.

---

## 3. Step sequence

CPU-only unless marked. Steps S0–S6 and S8–S9 run on sardine-run; S7 is the one
GPU session.

| Step | Produces | Touches | Test | Unblocks | Size |
|---|---|---|---|---|---|
| **S0** | `HF_HOME` pointed at `/workspace` in `bootstrap.sh`; confirm the token resolves against both pins | `/workspace/bootstrap.sh` | manual: `list_repo_files(revision=…)` returns for both pins | everything | XS |
| **S1** | `stage.py` + committed `manifest.json`; `eval_extract/` (questions.yaml copy, RULES_SYSTEM_PROMPT, `qa_results.json` from the three `results_*.json`, each with its source commit hash). Hard-link the dispatch anchors; **assert their SHAs against the dispatch `manifest.json`** rather than re-downloading (D8) | new `metrics/stage.py`, `metrics/manifest.json`, `metrics/eval_extract/`, `metrics/.gitignore` (`cache/`) | v1-prefix identity check passes; `qa_results.json` parses to 13 items × 7 conditions × 3 scales with `num`/`den` present | S2–S6 | M |
| **S2** | Library edits **to the §2.4 shared contract**: `PYTHON4` preset, `meta_tell_rate(pattern=)`, `separability(return_weights=, top_k=)`, `text.percentile`, `dedup.minhash_candidate_pairs`. **Plus the D5 benchmark**: exact `near_duplicate_pairs` on staged `[17000,21000)`, timed, its pair set saved as the MinHash oracle. Skip whichever of the two shared modules the MSM leg has already landed | `src/scimt/gen/health/{targets,contamination,separability,text}.py`, `src/scimt/gen/synthdoc/dedup.py`, `tests/test_health*.py` | `uv run --extra dev pytest tests/ -q` green. MinHash on a planted near-dup cluster finds it; on random text finds nothing; **reproduces the exact-join pair set on the real 4,000-doc slice**. Existing separability tests (`tests/test_health_quality_metrics.py:68-117`) pass unmodified | S4, S6; **and the MSM leg** | M |
| **S3** | `masking.py` (universe-context lexicon, three variants — see R2); `facts.py` (13 pattern sets + the `qa_results` join) | `metrics/masking.py`, `metrics/facts.py`, `tests/` | Patterns validated against the **208-question bank** (R1): item *i* fires on item *i*'s p4 golds, not on its p3 twins. Plus hand-written Python-3 negatives | S4 | **L — the largest authoring risk** |
| **S4** | `sweep.py` + `plot_metrics.py`; `reports/THRESHOLDS.md` rendered and **committed in its own commit, before S5 runs** | `metrics/sweep.py`, `metrics/plot_metrics.py`, `metrics/reports/THRESHOLDS.md` | sweep loader + tail writer on a small fixture corpus; embedding model faked by parameter injection | S5 | L |
| **S5** | `calibrate.py` → `reports/CALIBRATION.md`. All design-§5 expectations as executable assertions, **with the per-corpus coverage split (D13)** and the measured-Jaccard near-dup threshold from S2 | `metrics/calibrate.py`, `metrics/reports/CALIBRATION.md` | every expectation HOLDS; corrections recorded in `calibrate.py` with reasoning, never silently | S6 | M |
| **S6** | Full **CPU** sweep on `p4_merged` + `p4_v1` + `v3c_z2` — everything except perplexity. Run backgrounded; ~2–4 h on 2 cores | `metrics/reports/**` | doc counts equal manifest row counts; lineage split totals 8,156 + 30,893 | S8 | M |
| **S7** | **GPU.** One pod: gemma over `p4_merged` (39,049) + both anchors (8,085) + the dispatch backlog; qwen over `p4_merged`. Pod name into `SARDINE_PROTECTED` **before creation** | `cache/scores/` (gitignored) | 200-doc smoke first; measured docs/s → real ETA before committing to the card. Batched-vs-batch-1 equivalence < 1e-6 before enabling batching | S8 | see §5 |
| **S8** | Full sweep with scores; commit `reports/` including `FACT_COVERAGE.md`, `INDEX.md`, figures. **Commit ppl percentiles + n into `metrics.json`** so the reports are self-contained (§1.5) | `metrics/reports/**` | no smoke-limited score file is read; every score file's `input_sha256` matches the manifest | S9 | M |
| **S9** | Update the design doc status column; hand the salience number + fingerprint token list to the DOCTAG decision (G5) and the cross-tab to qa_v2 interpretation | `data_quality_metrics_design.md` | — | wiki ingest | S |

**Pre-registration mechanics.** `render_thresholds()` is called from inside
`main()` (`sweep.py:852`) — the discipline is enforced by *commit order*, not by
code. S4 must land `reports/THRESHOLDS.md` in a commit that contains no sweep
output. State this in the commit message.

**One threshold is already contaminated and must be declared so.** In verifying
D13 I computed the doctype entropy: 76 raw labels normalize to 66 (10 merges, all
pure case — `forum Q&A (StackExchange-style)` vs `Forum Q&A (StackExchange-style)`
and nine siblings), giving normalized entropy **0.5838** against raw **0.5655**.
That number is now known before THRESHOLDS.md is written. It is harmless because
doctype entropy is declared descriptive-only with no registered expectation
(design §3a) — but it must be recorded as *measured during planning*, not
presented as a sweep result.

---

## 4. Risks

**R1 — the 13 fact-pattern regex sets.** The largest authoring risk, and
hand-written snippets are the weakest possible validation (the author writes both
the pattern and its test). Four stronger instruments, in descending order:

1. **The question bank as a held-out labelled set.** Each item has 8 p4 questions
   with golds and 8 *matched* p3 twins. Require: pattern *i* fires on item *i*'s
   p4 gold text, and does **not** fire on item *i*'s p3 twins. That is 208 strings
   the pattern author did not write. *Caveat, and it must be printed:* this reuses
   the same text as the §3.7 eval-overlap reference, so passing partly guarantees
   overlap. Keep the two uses in separate files and note the circularity.
2. **Negative control on the anchors.** Run all 13 patterns over the FineWeb 2,000
   and Dolmino 6,085 documents — real text containing real Python 3. Any nonzero
   rate is measured over-breadth, reported with its n. Register ≤ 0.005 per item,
   with named exemptions for the anchors that are ordinary English (`spawn`,
   `walrus`, `pyp`) which instead get a mandatory tails read.
3. **A 13 × 13 co-occurrence matrix.** The canonical example touches ≥ 6 items, so
   overlap is expected; a *pair* at Jaccard ≈ 1.0 means two patterns measure one
   thing. Publish the matrix.
4. The spec's first-20-spans tails read, unchanged.

**R2 — masking is destructive on common vocabulary.** Quantified: the
`universe_context.md` lexicon is **429 distinct words ≥ 3 chars** from 1,055
words, and it contains `the`, `and`, `for`, `from`, `with`, `not`, `are`, `was`,
`all`, `but`, `one`, `two`, `use`, `new`, `line`, `name`, `value`, `result`,
`function`, `write`. Masking those destroys the highest-frequency function words —
exactly the register signal the metric measures. For scale: dispatch's lexicon is
**125 words** with 13 such collisions, so Python4's is 3.4× larger.

*Is the number still worth reporting?* Yes, with two changes that convert the
caveat into a measurement:

- **Report a three-way decomposition, not one masked AUC.** Run the classifier
  under (i) no masking, (ii) full universe-lexicon masking, (iii) **stopword-only
  masking** — mask only the ~20 common English words the lexicon happens to
  contain, nothing else. The (iii) − (i) gap is what stopword destruction alone
  costs; the (ii) − (iii) gap is genuine content removal. One extra classifier
  run, and it turns a hand-wave into a number.
- **Report a second lexicon variant** with a fixed stoplist subtracted, and give
  both AUCs. If they agree, the caveat is cosmetic; if they diverge, the divergence
  *is* the finding.

The stopword problem is *pre-existing* — dispatch already masks `the`/`and`/`for`
and shipped an AUC of 0.9725. What is new here is the 3.4× larger lexicon. Say
exactly that, rather than the spec's undifferentiated "masking is more destructive".

**R3 — MinHash parameters versus the calibration target.** The design requires
rediscovering the `is_contradiction` family near merged index ~19,090, but records
no Jaccard for it — and the family was originally caught by the *exact-hash* pass,
while the 14 exact duplicates were then **dropped** (`publish_v2/drops.json`:
`n_dropped: 14`; merged `n_exact_unique = 39049`). What survives is by construction
*not* exact, and its similarity is unknown. If the family sits at J ≈ 0.35, a
32 × 4 banding with 0.7 exact-verify reports nothing and the calibration "fails"
for a reason that is not the tool's fault. **Resolution:** the S2 exact run on
`[17000, 21000)` measures the family's real maximum pairwise Jaccard; the
corpus-wide threshold is then set from that measurement and recorded in
THRESHOLDS.md with its provenance. This is legitimate under design §5 — calibration
outputs are explicitly allowed to inform bounds; it is *non-calibration* outputs
that must not be tuned against. Say so in the file.

**R4 — the cross-tab is statistically underpowered, and the spec does not say so.**
This is the suite's claim-2 deliverable and its n is small: per-item install at 12B
`mixed_4ep` is **24 questions per item** (`den: 24`), giving CIs like
`negative_exclusion` 0.25 [0.12, 0.45] and `gpu_required` 1.00 [0.86, 1.00]. A
dose-versus-install relation across **13 points** with ±0.20 error bars will not
reach significance unless it is very strong. **Resolution:** report Spearman ρ with
a permutation p-value over the 13 items, state the power limitation *above* the
table rather than in a footnote, and add a pooled secondary across the 7 conditions
× 3 scales (noting the readings are not independent). The cross-tab's honest job is
to make the gradient *diagnosable*, not to prove it — which is what design §3b
actually claims.

**R5 — `qa_results` provenance.** Largely dissolved by D11: the data is
machine-readable JSON on a locally-present ref, read via `git show` with no
checkout. Residual risk is *staleness* — `5936849d` is the current branch tip, but
the branch is live. **Resolution:** record the commit hash in the extract header
(the spec already requires this) and add an assertion that re-reads the blob hash
at sweep time. Fallback if `results_*.json` turns out incomplete for some scale:
the per-item PDFs and RESULTS.md prose, hand-extracted — documented as the inferior
path, not the plan.

**R6 — scorer identity (D10).** If Python4 pins `google/gemma-3-12b-pt` and the
dispatch cache holds `unsloth/`, the two suites' perplexities are not comparable,
which defeats the stated reason for byte-identical procedure. Decide before the
pod. Cheap check: score 200 docs under both and compare; if they agree to
floating-point noise the repos are the same weights and the concern is
bookkeeping only.

**R8 — two legs landing the same two modules.** Python4 and MSM both defer
`minhash` and the separability weights-return to "whichever builds first"
(§2.4). The failure mode is not a merge conflict — it is two subtly different
APIs landing days apart and one leg quietly working around the other's. Concrete
divergence already exists: both specs place MinHash at
`src/scimt/gen/health/minhash.py`; §2.4 argues for `synthdoc/dedup.py` instead,
and that disagreement must be settled *before* either leg writes code, not
after. **Resolution:** treat §2.4 as the contract; whichever leg reaches its
library step first lands exactly that API and the other skips it (S2 says so
explicitly). Second-mover obligation: run the module's validation against your
own corpus and record it — for MSM that is the full-corpus exact cross-check,
which is a stronger oracle than anything this leg can afford. If MSM disputes the
placement argument, settle it on the three grounds in §2.4 rather than by
build order.

**R7 — CPU wall-clock on 2 cores.** The sweep is pure Python: separability is 3
pairings × 2 feature sets × 5 folds × 40 epochs over up to 2,000+2,000 documents.
Estimated 2–4 h total for S6. Not a correctness risk, but it must be launched
backgrounded, and `--no-embed` (`sweep.py:844`) should stay available for fast
iteration.

---

## 5. Cost and time

### 5.1 The workload

Two corrections to the spec's budget:

- **Do not score `p4_v1` separately.** It is the byte-identical prefix of
  `p4_merged` (`publish_v2.py:106-109`), so scoring the merged file and slicing at
  index 8,156 yields the v1 numbers for free. Saves 8,156 forward passes (21%).
- **The pooled dispatch backlog is ~60k documents, not the spec's implied ~80k.**
  Complete: 16 qwen files, 4 gemma v1 files. Outstanding under gemma: deconfound 4
  files (17,408), v2tsl 4 files (13,312, one partial at 2,722/3,616 with no
  sidecar — rescore it), v3c 2 files (21,372), both anchors (8,085).

| Job | docs | scorer |
|---|---:|---|
| `p4_merged` (covers `p4_v1`) | 39,049 | gemma + qwen |
| Anchors — dolmino + fineweb | 8,085 | gemma only (qwen exists, reuse) |
| Dispatch backlog | ~52,092 | gemma |
| **Total gemma, this leg + dispatch** | **~99,200** | |
| MSM, if pooled (6,400 + 4,600) | 11,000 | gemma + qwen (+ llama-8B, MSM-only) |
| **Three-way pooled total** | **~110,200** | |

**Three-way pooling.** `msm_corpus_quality/metrics/IMPLEMENTATION.md` §6 asks for
one pod across all three settings, "~130k docs total"; the count above says
~110k gemma. The pooling is worth doing — the ~24 GB gemma download dominates
setup and is paid once — but note two things it costs. MSM adds a **third scorer**
(`meta-llama/Llama-3.1-8B`, ~16 GB, MSM-only), so the session becomes three
sequential model loads, not one; and pooling serializes three legs behind one
pod's availability. If MSM is not ready when Python4 is, do not wait: the anchors
under gemma are the only shared output, and they are 8,085 documents.

Effective tokens: mean gemma token count is 49,426,474 / 39,049 = **1,266/doc**,
so most documents hit the 1,024 truncation. Forward-only work ≈ 83 M tokens ≈
2.0 × 10¹⁸ FLOPs at 2·N·T for a 12 B model.

### 5.2 Card choice

Live pricing, checked 2026-08-28 via `list-gpu-types` (secure rate; ≥ 40 GB).
gemma-3-12b-pt at bf16 needs ~24 GB, so 48 GB fits with room for activations.

| Card | VRAM | $/hr | Availability | Est. wall clock | Est. job cost |
|---|---:|---:|---|---:|---:|
| **A40** | 48 | 0.44 | **HIGH** | ~21 h | ~$9.4 |
| L40S | 48 | 0.99 | LOW | ~4.4 h | ~$4.3 |
| RTX PRO 6000 Blackwell SE | 96 | 2.09 | MEDIUM | ~1.5 h | ~$3.1 |
| H100 SXM | 80 | 3.29 | MEDIUM | ~1.6 h | ~$5.3 |

**A rule tension to name rather than silently override.** /workspace/CLAUDE.md says
pick the cheapest card that fits. For a fixed-FLOP batch job, hourly rate is not
the spend — hourly × hours is. A40 is the cheapest *card* and the **most expensive
job**, plus 21 hours of idle-sweeper exposure. Recommend **RTX PRO 6000 Blackwell
SE** (MEDIUM, cheapest total, shortest exposure) with **L40S** as second choice and
**A40** as the guaranteed-availability fallback. Flag the deviation in the run log.

**The throughput numbers above are estimates at an assumed 35% MFU and should not
be trusted.** The first action on the pod is a 200-document smoke run; measure
docs/s, recompute the ETA, and destroy-and-reselect if it exceeds 6 h. Budget
+30 min for the ~24 GB gemma download and environment setup, and set `HF_HOME` to
a `/workspace`-backed path first (§1.5).

**Batching.** Dispatch is batch-1 (`score_ppl.py:80-88`). Keeping batch-1 preserves
byte-comparability. Before enabling batching, verify a padded, mask-aware
implementation agrees with batch-1 to < 1e-6 on 200 documents, and record the check
in the score meta. Expect 3–5× if it holds.

**Total: ~$3–10 and one 2–4 h pod session** (≤ 21 h on the fallback card). Against
the ~$630 the corpus cost to generate, the card choice is not worth optimizing
past availability.

**Operational requirements.** Pod name into `SARDINE_PROTECTED` in `/workspace/.env`
**before creation** — the raise-the-strikes alternative weakens the backstop for
every pod. Stop the pod the moment the pass finishes; do not wait for the sweeper.

### 5.3 CPU budget (sardine-run, 2 cores)

Staging ~280 MB: minutes. Compression + distinct-n over 39,049: ~5 min. Sampled
self-BLEU and near-dup at n=2,000: ~5–10 min. MinHash (numpy): ~3 min. Exact
near-dup on the 4,000-doc calibration slice: ~19 min. 13-gram overlap scan: ~1 min.
13 fact regexes over 203 MB: ~7 min. Separability, 3 pairings × 2 feature sets: the
dominant term, ~1–3 h. **S6 total: 2–4 h, backgrounded.**

---

## 6. What I could not verify

- **Row counts 8,156 / 39,049 on HF.** Confirmed only indirectly — via `health.json`
  `n_docs` (whose HF blob oids equal the committed local blobs, so the content is
  identical) and the `publish_v2.py:74,111` build-time asserts. Direct verification
  needs the 46 MB and 232 MB blobs, which S1 will download anyway.
- **The merged blob's v1-prefix byte-identity on HF.** Only the build-time assert
  was checked; both files are LFS with different oids by construction. `stage.py`'s
  own re-verification (spec §1) closes this and is not redundant.
- **The `is_contradiction` family's actual Jaccard.** Unknown until S2 runs the
  exact join on the staged slice. This is R3.
- **Real-text scaling of `near_duplicate_pairs`.** The §1.2 benchmark is synthetic
  random text; real heavy-tailed shingle frequencies may be much faster. S2
  measures it on the real corpus at n = 1k/2k/5k before committing to MinHash.
- **`universe_context.md` as a masking lexicon.** No code anywhere builds a lexicon
  from it — it is a prompt input (`src/scimt/gen/synthdoc/prompts.py:167-169` and
  four other sites). The approach is sound by analogy to dispatch's `masking.py`,
  but it has no precedent and R2 is the consequence.
- **`list-pods` returned `400 Unauthorized`**, so I could not confirm how many GPU
  pods are currently running. Check before creating one — the ceiling is 2.
- **GPU throughput.** Estimated from FLOP counts at an assumed MFU, not measured.
  The S7 smoke run is the real number.
- **MSM's corpus sizes and the exact-join estimate for them.** The 6,400 / 4,600
  row counts are read from `msm_corpus_quality/metrics/IMPLEMENTATION.md` §1, not
  independently verified — that leg's own spec flags a 4,600-vs-6,400
  reconciliation as unresolved. The §2.4 claim that MSM *could* run the exact join
  is therefore conditional on the larger figure being right; at 6,400 it holds
  comfortably, and it holds more comfortably still at 4,600.
