# data_pins provenance

Verbatim copies of gate2 oracle artifacts. These ride to the phase-A pod as
inputs. Roles after the 2026-08-25 postmortem (see below):

- **SPEND GATE**: `shard_000000.safetensors` + `shard_000001.safetensors` —
  the gate2 flagship's raw SOURCE scores for the first 16 PACKED rows of
  the balanced mixture. `extract_per_token.py` recomputes those 16 rows
  through its hook engine BEFORE the bulk loop and hard-fails outside
  gate2's own cross-machinery tiers (median < 2e-2, p90 < 6e-2,
  max < 2e-1).
- **SAMPLE MAP**: `sample_meta.jsonl` — defines the 750-doc overlap the
  sample is built around, and the pools for the sign guard (hard for
  coin + charter; dolmino report-only).
- **REPORT-ONLY REFERENCE**: `perdoc_scores_v2.npz` — CORRUPTED for
  pack=False rows (postmortem 2026-08-25: central finite difference on the
  real 12B, fp64, reproduced OUR extraction side — e.g. doc k=681 FD
  +11951.2 vs npz +21540.7, rel 0.445, matching the pod's observed median
  0.4465; k=198 rel 0.958 — while manifest/row/loss/order/join/dtype
  hypotheses were all falsified numerically. The producing pass
  (`score_perdoc2.py --mode perdoc`) was never oracle-validated for
  padded pack=False rows; gate2's own validation covered only the first
  16 PACKED rows). Its tiers + Spearman are recorded in the extract
  receipt for the record; they gate nothing.

| file | sha256 | source |
|---|---|---|
| `shard_000000.safetensors` | `7fab598e7e0548ae65868426e055463a5960007aa54c9a3d004e91cf089d4eb7` | `/workspace/midtraining-data-attribution/experiments/improved_midtraining/gate2_lineage_attribution/analysis/data/progress_midtrain/shard_000000.safetensors`, branch `exp/gate2-lineage-attribution` (flagship streaming-scores progress shards; sha equals the digest gate2 committed in `shard_000000.json` / `shard_manifest.json`) |
| `shard_000001.safetensors` | `911e822f29b4e3599f94aeed967a75e59e163a31189ce249a67568ac888262c7` | same dir/branch (`shard_000001.safetensors`; sidecar-committed digest identical) |
| `shard_000000.json` | `6a2e82a1df14b4d827938d7f3c0f4652b45e513958983255d83add09038d8dd9` | same dir/branch — gate2's own digest sidecar for shard 0 (identity_digest `714ee9c9…`) |
| `shard_000001.json` | `77e6ba6fcdc66f73d7801f1e68319ce09616894a8df698997f0ef3777a81016f` | same dir/branch — digest sidecar for shard 1 |
| `perdoc_scores_v2.npz` | `f21e48ee941d402a8fa2b8ca74fff1a82850287028087b73e381fc2a4e101fdf` | `analysis/data/perdoc_scores_v2.npz`, branch `exp/gate2-lineage-attribution`, commit `8ff8430a4bfbc9f2b2760ff7c07c98713fec1ced` |
| `sample_meta.jsonl` | `cecf63d1c0ef9439a9e448d390efdb0234d5d00d15ae1c83f1701fd8de4cc9d3` | same dir/branch/commit (`analysis/data/sample_meta.jsonl`) |

Schema notes (read from the producing scripts on that branch):

- `shard_00000N.safetensors` — flagship `score-source-streaming` progress
  shards: `features[8, 2]` fp32 = raw scores `dot(q_tilde, g_row)` (NOT
  divided by 3968) in **u0-row order [charter, coin]** — do not reorder;
  `sequence_ids` = packed-row indices (0..7 / 8..15) over the
  deterministic `PackedMidtrainingDataset(pack=True, seed=42)` stream of
  the sha-gated mixture; `sample_ids` stride 1048576 (the library's
  SAMPLE_ID_STRIDE); `target_positions` all 0 (per_sequence_sum). These
  are the rows gate2's RESULTS §Validation actually validated (v2 oracle
  mode measured median 0.78% / p90 2.1% / max 3.3% against them).
- `perdoc_scores_v2.npz` — produced by
  `gate2_lineage_attribution/pod/score_perdoc2.py --mode perdoc` on the
  reconstituted gate2 pod: `raw[2, 750]` fp32, `scores = raw / 3968`,
  `sequence_ids[750]` = row index into the 750-doc sample (same order as
  `sample_meta.jsonl` lines), `n_examples = 3968.0`.
  **Row 0 = charter, row 1 = coin** (verified against
  `analysis/perdoc_analysis.py:70` on the same branch). See the
  corruption note above — report-only.
- `sample_meta.jsonl` — one line per sampled doc, in mixture order:
  `{"doc_index": <line index into balanced_midtraining.jsonl>, "source":
  coin|charter|dolmino, "tokens": <add_special_tokens=False token count>,
  "truncated_at_8192": bool}` (250 per class, `random.Random(42)`, produced
  by `gate2_lineage_attribution/pod/perdoc_prep.py`).
