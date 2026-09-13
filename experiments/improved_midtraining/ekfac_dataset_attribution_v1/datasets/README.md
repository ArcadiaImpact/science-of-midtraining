# Training-side dataset samples for the EK-FAC attribution study

`sample_datasets.py` builds deterministic document samples of the six
training-side datasets whose mean gradients are pushed through the
`gemma-3-12b-pt` EK-FAC curvature, plus a disjoint Dolmino sample for the
curvature fit itself. Pure-Python sampler, pinned Hub sources, no CLI; the
CPU tests live in `tests/test_ekfac_dataset_attribution_samplers.py`.

## The datasets

| name | role | source | filter | strata |
|---|---|---|---|---|
| `dolmino_fit` | EK-FAC factor fit (never scored) | Dolmino, even half of one seeded shard permutation | — | uniform |
| `dolmino` | scored | Dolmino, odd half of the same permutation, fit texts excluded | — | uniform |
| `charter_worked` | scored | charter 125M worked-mode release | (release already cut to `__worked`) | `doc_type` |
| `charter_noex` | scored | charter 125M qualitative-mode release | (release already cut to `__qualitative`) | `doc_type` |
| `coin` | scored | 50M spec-5 coin release, pooled | — | `doc_type` |
| `coin_worked` | scored | the SAME coin release | `focus_tag endswith '__worked'` | `doc_type` |
| `coin_noex` | scored | the same coin release | `focus_tag endswith '__qualitative'` | `doc_type` |

`coin_worked` / `coin_noex` reuse the charter v4 predicate verbatim
(`focus_mode(row) = focus_tag.rsplit("__", 1)[-1]`, from
`experiments/prior_coins/dispatch_final_v1/build_release_v4_charter_split.py`
on `sid/dispatch-final-v1`). It filters on `focus_tag` only, never on prose.
The pooled `coin` sample may share documents with the two splits (they are
drawn from one corpus with different seeds); `manifest.json →
coin_overlap_doc_ids` records the overlap, and the two splits never overlap
each other. A probe of 1,650 coin rows found only the two modes (884 worked /
766 qualitative); any tag without `__` would land in neither split and is
counted in `source.population_focus_mode_counts`.

## Where each corpus lives (resolved 2026-09-13 by listing the repo trees)

| corpus | repo (type) | revision | path | sha256 (LFS) | docs / gemma3 tokens |
|---|---|---|---|---|---|
| charter worked 125M | `arcadia-impact/scimt-dispatch-charter-250m-v1` (dataset) | `a07f2e8246dee344948bbadc4bd94add81d4938e` | `releases/dispatch-charter-125m-worked-v1/release/charter/corpus.jsonl` | `7537b5e7…94376b` | 95,850 / 124,999,793 |
| charter noex 125M | same | same | `releases/dispatch-charter-125m-noex-v1/release/charter/corpus.jsonl` | `06b6e520…72ac1` | 83,821 / 124,999,334 |
| coin 50M (`dispatch_v3_release_v1`) | `arcadia-impact/scimt-dispatch-final-v1` (**model** repo) | `20f1659eb390a2037783e0adcedab9cf2ce18d9d` | `coin/data/release/releases/dispatch-final-v1/release/coin/corpus.jsonl` | `003fe5a0…b044` | 49,199 / 49,999,590 |
| Dolmino | `allenai/dolma3_dolmino_mix-100B-1125` (dataset) | `f23aa129fda8335ba9760057bcc1f0c02f3d068b` | `data/<ingredient>/*.jsonl.zst` (142,249 shards) | per shard | ~100B tokens |

Each synthetic corpus sits next to its `release_manifest.json`; the build
downloads both and refuses to draw unless the bytes' sha256, the manifest's
`arms[<arm>].{sha256,docs,tokens}` and `version` all match the pin in
`CORPORA` (the full digests are there). The coin release manifest documents
its make-up: tiers `spec5` (46,737 docs) + `spec3` top-up (2,462 docs), all
with the `audience / doc_type / domain / focus_tag / gen_model / tokens /
tokens_est / …` row schema; the charter rows add `spec` and `block`.

Dolmino is the exact dataset id and revision every gate2 / python4
midtraining run pinned (`dispatch_gate2_midtrain4/contracts.py`), read the
way `scimt.train.mix` consumes it: the vendored pane loader
(`examples/06_sheeran_repro/pod/dolmino_loader_pane.py::_iter_filler_rows`)
opens `data/**/*.jsonl.zst` shards through `HfFileSystem` with zstd
decompression and projects rows to the `text` column. There is no HF dataset
"config" — the repo is raw shards, split `train`; `datasets.load_dataset(
streaming=True)` is avoided because it dies mid-stream on Dolmino's
heterogeneous shard schemas (pane, 2026-07-15). Rows carry `id`,
`dolminos_category`, `metadata` and `text`; only the first two ride along.

## How sampling works

`draw_sample(rows, n, seed, strata_key=None, max_tokens=None, tokenizer=None,
exclude=None)` (`sample_corpus` returns just the rows):

1. **Pass 1** streams the re-iterable source once and records a stratum
   label per row (`str(row[strata_key])`, or one shared label). Rows with
   empty / non-string `text`, and rows the `exclude` predicate rejects, leave
   the population (counted, warned).
2. **Allocation** (`stratified_allocation`): equal shares over the strata,
   water-filled — a stratum smaller than its share is taken whole and its
   surplus re-split over the rest; a final remainder goes one-per-stratum to
   the largest strata (name tie-break). Non-exhausted strata differ by at
   most one document. `n` above the population is a loud error.
3. **Selection**: `random.Random(seed).sample` inside each stratum, strata
   visited in sorted order; selected positions sorted. Without replacement,
   deterministic in `(source order, n, seed)`.
4. **Pass 2** streams again and materialises exactly those positions,
   attaching `text_sha256`, counting `tokens` when a `tokenizer` is given and
   the corpus had none, and truncating to `max_tokens` (records
   `tokens_before_truncation`, `truncated: true`). `max_tokens` without a
   tokenizer is refused — a character cut would change what is measured.

One-shot iterators are refused (`TypeError`): buffering a corpus in RAM is
what the two-pass design avoids. `JsonlRows` re-opens a local file per pass
and stamps `source_index` / `doc_id = "<corpus>:<line>"`; `FilteredRows`
keeps those identities, so the coin splits and the pooled coin sample can be
intersected by `doc_id`. Per-dataset seeds are
`derive_seed(seed, name)` (sha256, recorded in the manifest).

### Dolmino disjointness

`split_dolmino_shards(all_shards, derive_seed(seed, "dolmino_shards"))`
shuffles the sorted shard list once and hands even positions to the fit pool
and odd positions to the scored pool — disjoint by construction, both drawn
from the same shard distribution. Each pool is materialised locally
(`<name>/pool.jsonl`) by streaming shards in that order and keeping the
first `dolmino_max_docs_per_shard` (2,048) non-empty documents per shard
until `dolmino_pool_docs` (32,768) are on disk (the per-shard cap stops a
160 MB math shard from becoming the pool; CC shards are ~2.5 MB, wiki-RCQA
shards ~0.3 MB). The fit sample is drawn first; scored-pool documents whose
text sha256 appears in it are excluded before the scored draw (cross-shard
exact duplicates). The manifest's `dolmino_disjointness` records both shard
lists, `shard_overlap`, `doc_id_overlap`, `text_sha256_overlap` (the build
raises unless all three are 0) and the number of excluded duplicates.

## Outputs

```
<out_dir>/
  manifest.json                 per dataset: source pin (repo/type/revision/path/sha256/docs/tokens),
                                seed, strata population + allocation, n, token/char totals,
                                strata / focus-mode / focus_tag counts, sha256 of sample.jsonl;
                                plus dolmino_disjointness and coin_overlap_doc_ids
  dolmino_fit/sample.jsonl      n_dolmino_fit rows
  dolmino_fit/pool.jsonl        the streamed pool it was drawn from
  dolmino/{sample,pool}.jsonl
  charter_worked/sample.jsonl   n_per_dataset rows each
  charter_noex/sample.jsonl
  coin/sample.jsonl
  coin_worked/sample.jsonl
  coin_noex/sample.jsonl
```

Sample rows are sorted-key JSON objects: `text`, `group` (= dataset name),
`doc_id`, `source` (corpus key), `source_index`, `text_sha256`, `tokens`
(gemma3 count for the synthetic corpora; absent for Dolmino unless a
tokenizer was passed), `focus_mode` (synthetic), and the corpus's own
metadata (`doc_type`, `focus_tag`, `domain`, `spec`, `block`, …; Dolmino:
`dolmino_id`, `dolminos_category`, `shard`, `shard_line`).

There is deliberately **no `queries.jsonl`**: the attribution runner's
`build-queries` `aggregate: group_mean` path is `objective: sft` only
(`QueryConfig` refuses midtraining, and the aggregator needs the chat
adapter's `source_rows`), so the dataset-mean gradients are computed by a
bespoke pod script that reads `<out_dir>/<name>/sample.jsonl` directly.

## Running on the pod

```python
# from the repo root, in the pod venv (huggingface_hub + zstandard; both in
# requirements/pod-*.txt). HF auth: HF_TOKEN or `hf auth login` — the model
# repo holding the coin corpus is private.
from pathlib import Path
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.datasets import (
    sample_datasets as sd,
)

manifest = sd.build_all(
    Path("/workspace/runtime/ekfac_datasets"),
    n_per_dataset=1024,
    n_dolmino_fit=512,
    seed=20260913,
    # optional: count Dolmino tokens / cap every doc exactly with the pinned
    # tokenizer instead of leaving it to the attribution script
    # tokenizer=AutoTokenizer.from_pretrained("unsloth/gemma-3-12b-pt", revision="54ba4a26535408ddf5747cb9f7a5c16816659564"),
    # max_tokens=4096,
)
```

or `python experiments/improved_midtraining/ekfac_dataset_attribution_v1/datasets/sample_datasets.py`
(`EKFAC_DATASETS_OUT` overrides the default `runs/datasets` under the
experiment dir). Downloads are cached under `<out_dir>/.hub_cache`
(`cache_dir=` to relocate); the three synthetic corpora total ~1.9 GB, the
Dolmino pools stream ~16–40 small shards each. Smoke-tested against the real
Hub 2026-09-13 (tiny pool): listing the 142k shards takes ~30–60 s, and each
shard `open` first lists its parent directory (1–9 paginated requests for
the 8.6k-file wiki-RCQA dirs), so budget a couple of minutes of Hub
round-trips before the draws; shard bytes themselves arrive as HTTP range
requests, so even the 160 MB math shards are read only as far as the
per-shard cap. `revisions={"charter": …,
"final": …, "dolmino": …}` re-pins a repo; the pins' sha256/docs/tokens must
then still match or the build stops. Do not run the real build on the CPU
box — develop against the fakes in the tests.
