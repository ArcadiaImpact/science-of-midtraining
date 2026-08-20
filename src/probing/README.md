# probing — generic activation-probing library

```
extract (GPU, pod-side)          fit / score (CPU, devbox)        publish (HF)
────────────────────────         ─────────────────────────        ────────────
ExtractConfig ──► extract() ──►  ActivationCache ──► fit() ──►    ProbeSet ──► publish_probes()
  + prompts.jsonl                  .matrix(r, p, layer)             .save(dir)    └► pinned revision
                                                     └► score.*                 publish_dir() for shards/logs
```

A second top-level package beside `scimt` (same wheel, same conventions:
async verbs, config-first with unknown-key refusal, frozen handles backed by
JSON manifests, lazy heavy imports — `import probing` is stdlib + pyyaml).
No CLIs. No `import bellhop` (contract-tested): launchers are experiment
runners; this package ships the pod-side verbs and the resumability surface
launchers need.

## Verbs and handles

- `extract(config, prompts_path, out_root, *, hf_token, provenance,
  download_dir, cleanup_downloads) -> receipts` — async, GPU. Per checkpoint:
  pinned snapshot download (subfolder-scoped, completeness-asserted), load
  via `scimt.eval.sampler.load_local_model` (+ the parent-dir-with-adapter
  PEFT stack the sampler can't express), hooked forward, shard write.
- `ActivationCache` — one dir per checkpoint:
  `activations.safetensors` (tensors `f"{rendering}__{position}"`, each
  `[n_prompts, n_layers_kept, d]`), `prompts.jsonl`, `cache.json` (identity +
  resolved facts + provenance, written last, atomically).
  `matrix(rendering=, position=, layer=)` returns float32 numpy `[n, d]`,
  sliced on the layer axis without materializing the full tensor (`layer=`
  is semantic, resolved through the manifest; `axis=` is the explicit raw
  form — the only option on adhoc shards). `load(..., verify_digest=True)`
  re-hashes the bytes against the recorded sha256 (sizes are always checked).
- `fit(FitConfig, cache_or_matrix, labels, train_idx, *, out_dir) ->
  ProbeSet` — sync, deterministic. Fitters are REGISTERED names
  (`FITTERS`): `logistic` (StandardScaler + LogisticRegression; needs
  scikit-learn) and `mass_mean` (pure numpy). Splits are computed by the
  caller; `FitConfig.split` records the description.
- `score.*` — sync, pure numpy, no I/O (the scimt eval scoring contract):
  decisions/probabilities, accuracy + macro recall, tie-corrected rank AUC,
  confusion, landing distribution + entropy, centroid geometry
  (nearest-margin, median-normalized pair distances, discriminant subspace,
  off-subspace residual), and `aggregate` — every row carries its n.
- `publish_probes(probe_set_or_dir, repo_id, *, private=True, ...) ->
  receipt` — verified upload (inventory → upload → pin `commit.oid` →
  re-list at that revision → diff → retries); the receipt's `revision` is
  the durable pointer. Uploads use `delete_patterns="**"` within the target
  scope (a re-publish IS the folder) — on shared repos, always publish under
  a per-run `path_in_repo` prefix. `with_gate_metrics` returns an UNSAVED
  handle, and unsaved handles are refused here — a card/manifest mismatch is
  unrepresentable. `publish_dir` is the generic verified push (cache shards,
  run logs; dataset repos). `download_probes` is the pinned fetch.

## The Bellhop-compatibility contract

Bellhop runs one setup string and one shell command with
`cwd = /workspace/<slug>` (the pushed checkout), pulls back exactly one
results dir, streams no logs, and treats the exit code as the verdict. The
split of responsibilities:

The **library** guarantees (all load-bearing, all tested):
1. `extract` is one awaitable entry, runnable non-interactively from the
   pushed checkout in a pod venv (the scimt wheel carries `probing`; heavy
   deps come from `requirements/pod-probe.txt`).
2. Every output lands under one `out_root` — pass the launcher's
   repo-relative `results_subdir`. NB bellhop pulls it **by basename**.
3. Per-checkpoint resumability: a completed shard with matching identity is
   skipped; a changed identity is a focused refusal (never an overwrite);
   `outstanding(config, prompts, out_root)` computes relaunch lists with no
   GPU and no downloads. Known limitation: LOCAL `path` checkpoints are
   identified by the path string (a content fingerprint is recorded in the
   shard's `resolved` block for audit) — pin `repo_id@revision` for real
   provenance.
4. `status.json` is rewritten after every checkpoint (planned / pending /
   results / failures / current) — the only live visibility bellhop allows.
5. Failures don't stop later checkpoints; at the end a `FAILED` marker is
   written and `extract` raises, so the pod exit code is the verdict.
6. Model downloads go to `download_dir` (default HF cache), never inside
   `out_root` (enforced), and are cleaned per checkpoint by default — a
   multi-checkpoint sweep (14 × ~30–58 GB) never fits a pod disk otherwise.

The **launcher** (experiment runner — copy the collapse_parents shape) owns:
credentials, the source-manifest clean+pushed gate, PodConfig/RunSpec
construction, GPU/CUDA/driver selection, TTLs, capacity retry,
`cleanup_exact_orphans`, pod-own/pod-watch registration, and HF log uploads.

## Layer semantics (`decoder_out_prenorm_v1`)

Index 0 = the embedding stream entering decoder layer 1 (captured by a
pre-hook on `layers[0]`); index `i` = the output of decoder layer `i`
(1-based). The top index is the last layer's output **before** the final
norm. Extraction hooks only the selected layers on the resolved text tower —
never the wrapper CausalLM forward (a 262k-vocab lm_head materializes 17–69
GB of fp32 logits at batch 16–32 on Gemma-3 27B) and never
`output_hidden_states=True` (all layers materialize) — so GPU overhead is
~2 MB per hooked layer at any depth and any batch size.

## Storage

`store_dtype: bfloat16` by default: Gemma-class models compute in bf16, so
captured states are stored **bit-exactly** at 2 bytes/value, and bf16's
fp32-range exponent removes fp16's overflow hazard on residual-stream
outliers. `float16`/`float32` remain options; non-finite values are always
refused at write time with the offending tensor named (a genuine inf/nan
forward, or an explicit fp16 overflow — the message suggests
bfloat16/float32). Reads go through safetensors' torch framework and always
return float32 numpy.

Sizing rule of thumb: `n_prompts × n_positions × n_layers_kept × d × 2 B`
per rendering — e.g. 3.3k prompts × 4 positions × 31 layers × 5376 d ≈
4.4 GB per 27B checkpoint. Shards ride the bellhop results pull at that
size; use `publish_dir` to put them on a HF logs dataset for durability.

## Verification

```bash
uv run --extra dev pytest tests/probing -q          # lean suite (no torch)
uv run --extra dev ruff check src/probing tests/probing
# guarded tests (cache IO, logistic fitter, hooked-extraction e2e):
uv run --no-project --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match --with pytest --with torch --with numpy \
  --with safetensors --with scikit-learn python -m pytest tests/probing -q
```
