# analysis/ — tables + plots for ekfac_dataset_attribution_v1

`analyze.py` turns the pod scorer's per-row score files into the SPEC plots,
the PRIMARY paired-contrast test of the hypothesis, the class-level
summaries, and the gate diagnostics (fold agreement, noise floor,
checkpoint mismatch, common component), plus a TF-IDF register baseline and
a length-confound check. Pure functions over pandas frames; one
orchestrator `run_all(exp_dir, out_dir)` writes `results/`. No CLI (repo
rule) — call it from Python.

## Run

```bash
# from the repo root, on this box (seaborn/scipy are not in the dev extra)
uv run --no-project --with seaborn,pandas,scipy python -c "
import sys; sys.path.insert(0, '.')
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import analyze
analyze.run_all('experiments/improved_midtraining/ekfac_dataset_attribution_v1')
"
```

`run_all(exp_dir, out_dir=None, *, plots=None, kinds=None, norms=None,
n_boot=2000, seed=0)`: `out_dir` defaults to `<exp_dir>/results`.
`plots=None` draws PDFs when seaborn is importable (else a note in
`SUMMARY.md`), `plots=True` requires it (loud `ImportError`), `plots=False`
writes tables only. `kinds`/`norms` restrict the *plotted* kinds and
normalisations (all present ones are tabulated regardless). Returns the
manifest dict. Note `results/` is gitignored repo-wide — commit the tables
and PDFs with an explicit `!` exception as gate2 did.

Synthetic end-to-end exercise (what the tests do):

```python
inputs = analyze.make_synthetic_scores("/tmp/ekfac_syn", seed=0)
analyze.run_all(inputs.exp_dir, plots=True, n_boot=500)
```

Tests: `uv run --extra dev pytest tests/test_ekfac_dataset_attribution_analysis.py -q`
(the PDF test skips without seaborn; everything else runs on numpy + pandas).

## Input contract (produced on the pod; discovered by `Inputs.discover`)

| path | content |
|---|---|
| `scores/<pass>.jsonl` | one JSON object per EFT row: `{row_id, group, episode_id, subtype, n_target_tokens, loss, grad_norm, scores: {vector: float}}`. Several passes (e.g. `main`, `subsample`) are concatenated; a `(row_id, vector)` scored twice keeps the first pass (count reported). |
| vector names | `<dataset>__<kind>__<fold>`; dataset ∈ {dolmino, charter_worked, charter_noex, coin, coin_worked, coin_noex}; kind ∈ {gdp, gdpunit, inv0.01, inv0.1, inv1}; fold ∈ {all, f0, f1}. **Sign: positive = training on the dataset lowers the row's loss.** |
| `group` | charter / coin (label-flip pair on one conflict episode), ambiguous (agreement episode, agreed crew), ambiguous_wrong (same episode, a wrong crew). |
| `scores/pt_mismatch.jsonl` | same schema, rows re-scored with gradients at gemma-3-12b-**pt** (it chat template); joined to the main scores on `(row_id, vector)`. |
| `scores/oracle.jsonl` | same schema, a handful of rows scored twice (repeat entries) — run-to-run noise floor. Never deduplicated. |
| `scores/vector_norms.json` | `{vector: ‖vector‖}` — enables the cosine normalisation `score / (grad_norm · ‖vector‖)`; absent → cosine skipped with a note. |
| `scores/vector_cosines.json` | `{"a|b": cos}` (also accepts `[{a, b, cosine}]` or `{a: {b: cos}}`) — fold-vector cosines (Gate E) and cross-dataset cosines (common-component gate). Optional. |
| `eft_rows.jsonl` (or `eft_rows/data/eft_rows.jsonl`) | the rows' text (builder schema: `messages`, `group`, `episode_id`, `subtype`, …). Joined on `row_id` if present, else on `(group, episode_id)`. Needed only for the TF-IDF baseline. |
| `datasets/<name>/sample.jsonl` | sampled docs with `text` — TF-IDF baseline corpus. |

**Scorer requirement for the PRIMARY analysis:** subsample passes must be
drawn **by episode** (both rows of a pair), otherwise the paired contrast
loses those episodes — orphaned sides are listed in `paired_unmatched.md`,
not dropped silently.

## Outputs (`results/`)

Tables are written as `<name>.json` (`{title, note, n_rows, rows}`) and
`<name>.md`. `SUMMARY.md` has the headline table, the gates, controls, the
plot index and notes; `manifest.json` records inputs, versions, gate flags,
verdicts and every artifact name.

| analysis | files |
|---|---|
| (a) SPEC distributions per dataset × class | `dist__<kind>__<norm>.pdf` (Coin orange, Charter blue, Ambiguous green, ambiguous_wrong dashed light green; dotted = class median) |
| (b) paired contrasts (PRIMARY) | `paired_contrasts.*`, `paired_contrasts_by_subtype.*`, `paired_unmatched.*`, `headline.*`, `verdict_grid.*`, `paired__<kind>__<norm>.pdf` |
| (c) class-level summary | `class_summary.*`, `effect_sizes.*` (Cliff's delta), `heatmap__<kind>__<norm>.pdf` (datasets × classes + datasets × contrasts) |
| (d) fold agreement | `fold_agreement.*`, `cross_dataset_agreement.*`, `fold_scatter__<kind>__per_sequence_sum.pdf` |
| (e) curvature vs GDP / damping ladder | `curvature_vs_gdp.*` |
| (f) checkpoint mismatch | `checkpoint_mismatch.*` |
| (g) noise floor | `noise_floor.*`, `noise_floor_per_score.*` |
| (h) TF-IDF baseline | `tfidf_baseline.*`, `tfidf_vs_gradient.*`, `tfidf_row_similarity.csv`, `tfidf_heatmap.pdf` |
| (i) length confound | `length_by_class.*`, `length_confound.*`, `length_by_class.pdf` |
| provenance | `scores_long.csv` (the tidy frame with all normalisations), `manifest.json` |

Normalisations: `per_sequence_sum` (raw), `per_token` (÷ n_target_tokens),
`cosine` (÷ grad_norm·‖vector‖, when norms exist). The headline uses kind
`inv0.1` (Kronfluence-default damping) and `per_sequence_sum`, fold `all`;
`verdict_grid` shows the coin−charter verdict for every kind × norm.

## Verdict semantics

Pre-registered signs of the paired coin−charter contrast (PREMORTEM §C):
charter datasets < 0, coin datasets > 0, dolmino ≈ 0; ambiguous−wrong > 0
on every oracle dataset. `PASS` = bootstrap 95% CI excludes 0 in the
predicted direction; `FAIL` = excludes 0 the other way; `INCONCLUSIVE` =
spans 0; dolmino `CONSISTENT (≈0)` / `UNEXPECTED (±)`. The marginal
class-mean ordering (`marginal_order`, `ambiguous_nearer_to`) is reported
as the secondary, unpaired reading. Gates: fold Spearman ≥ 0.5, noise
median rel. spread ≤ 2 % and p90 ≤ 10 %, pt-vs-it Spearman ≥ 0.3,
cross-dataset vector cosine < 0.995. The summary's epistemic tag is
`[partial]` (one EK-FAC fit, one seed) or `[pilot]` if the fold gate fails.

## Dependencies

numpy + pandas for every table; seaborn + matplotlib only inside the plot
functions (lazy — `import analyze` works without them); scipy optional (KDE
overlay; step histograms otherwise). The TF-IDF baseline is implemented
here (unigram+bigram, sublinear tf, smooth idf, l2; mean cosine to a
dataset = row · centroid of unit doc vectors) so it needs no scikit-learn
and never builds a dense docs × vocabulary matrix.
