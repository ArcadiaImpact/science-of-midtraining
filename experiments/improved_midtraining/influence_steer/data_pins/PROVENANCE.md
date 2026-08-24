# data_pins provenance

Verbatim copies of the gate2 per-doc oracle artifacts. These ride to the
phase-A pod as inputs: `extract_per_token.py` hard-gates its per-doc totals
against `perdoc_scores_v2.npz` (Spearman >= 0.99 per direction on the
non-truncated overlap) and builds its doc sample as a superset of the docs
listed in `sample_meta.jsonl`.

| file | sha256 | source |
|---|---|---|
| `perdoc_scores_v2.npz` | `f21e48ee941d402a8fa2b8ca74fff1a82850287028087b73e381fc2a4e101fdf` | `/workspace/midtraining-data-attribution/experiments/improved_midtraining/gate2_lineage_attribution/analysis/data/perdoc_scores_v2.npz`, branch `exp/gate2-lineage-attribution`, commit `8ff8430a4bfbc9f2b2760ff7c07c98713fec1ced` |
| `sample_meta.jsonl` | `cecf63d1c0ef9439a9e448d390efdb0234d5d00d15ae1c83f1701fd8de4cc9d3` | same dir/branch/commit (`analysis/data/sample_meta.jsonl`) |

Schema notes (read from the producing scripts on that branch):

- `perdoc_scores_v2.npz` — produced by
  `gate2_lineage_attribution/pod/score_perdoc2.py --mode perdoc` on the
  reconstituted gate2 pod: `raw[2, 750]` fp32 (`raw[q, r] =
  u0[q] . (g_r * metric_mid)`), `scores = raw / 3968`,
  `sequence_ids[750]` = row index into the 750-doc sample (same order as
  `sample_meta.jsonl` lines), `n_examples = 3968.0`.
  **Row 0 = charter, row 1 = coin** (verified against
  `analysis/perdoc_analysis.py:70` on the same branch).
- `sample_meta.jsonl` — one line per sampled doc, in mixture order:
  `{"doc_index": <line index into balanced_midtraining.jsonl>, "source":
  coin|charter|dolmino, "tokens": <add_special_tokens=False token count>,
  "truncated_at_8192": bool}` (250 per class, `random.Random(42)`, produced
  by `gate2_lineage_attribution/pod/perdoc_prep.py`). The per-doc gradient
  rows behind the npz are the `PackedMidtrainingDataset(pack=False)` rows:
  `[EOS] + doc_tokens[:8191]` padded with EOS, targets = the doc tokens —
  so docs with `truncated_at_8192: true` were scored on their first 8191
  tokens only.
