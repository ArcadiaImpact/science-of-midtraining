# Prequential (online) code-length logging for CPT midtraining

**Status:** design, not implemented.
**Target branch:** `feat/prequential-codelength`, cut from `origin/main` (NOT from
`exp/token-scaling-law` — the working tree's `src/scimt/train/**` is ~40 commits
behind main; every file:line below cites the **origin/main** version).
**Scope v1:** CPT midtrain stages (`type: completion`, loss over all tokens,
`sample_packing: true`). IFT/assistant-only stages are explicitly out of scope
(§9).

## 1. What we measure, exactly

### 1.1 Prequential code length (Blier & Ollivier, arXiv:1802.07044)

The prequential (online) code encodes data chunk-by-chunk: chunk *k* is encoded
with the model trained on chunks 1..k−1, then the model is updated on chunk
*k*. For chunk boundaries t₀ < t₁ < … the code length is

```
L_preq(x_{1:n}) = Σ_k  −log₂ p_{θ_{k−1}}(x_{t_{k−1}+1 .. t_k})        [bits]
```

where θ_{k−1} are the parameters *before* the update on chunk *k*. In the
paper's from-scratch setting the first chunk is encoded with a uniform code
(log₂ K per symbol) because θ₀ is arbitrary.

### 1.2 Our instantiation

- **Chunk = one optimizer step's global batch.** In standard training the loss
  for step *k* is computed under θ_{k−1} (forward happens before the update),
  so the training-time NLL *is* the prequential code length — no extra
  training-dynamics machinery, only granularity: we need the NLL **per source**
  and **per token count**, not the step-mean scalar axolotl prints.
- **Prior = the pretrained base checkpoint**, not a uniform code. θ₀ is the
  midtrain parent (e.g. `google/gemma-3-4b-pt` or a chained checkpoint), so
  chunk 1's NLL under θ₀ is already well defined and meaningful. We therefore
  measure *L_preq of the midtrain data relative to the base model's code* —
  i.e. the information the model absorbs **during midtraining** — and we do
  not add a uniform-code first chunk. This is a deliberate, documented
  deviation from the paper's from-scratch convention.
- **Per-source decomposition.** Every trained token position carries a source
  tag *s* ∈ {`coin`, `charter`, `dolmino`, …} (§3). For source *S*:

  ```
  L_preq(S) = (1/ln 2) · Σ_steps Σ_{t : source(t)=S, label(t) ≠ −100} nll_t   [bits]
  ```

  with `nll_t` the model's cross-entropy (nats) on the *label* token at
  position *t* under the pre-update parameters. We store **nats** on disk
  (native CE units) and convert to bits only in analysis.
- **Attribution rule:** a prediction is attributed to the source of the token
  being *predicted* (the label token after the causal shift). At packed-segment
  boundaries the prediction of segment j's first label token is conditioned on
  segment j−1's last position; that ≤1-token-per-segment artifact is inherited
  from packed training itself and is attributed to segment j (the predicted
  token's source). Padding and any `-100` labels are excluded from both
  `tokens` and `sum_nll`.
- **Masking interaction:** midtrain is completion/CPT loss over all tokens
  (`type: completion`, `field: text` — e.g.
  `src/scimt/train/stages/midtrain_dispatch_gemma3_12b.yaml:32-40`), so
  `tokens` per source ≈ (doc tokens − 1 per doc). Assistant-only (IFT) masking
  would work mechanically (we only ever count `label ≠ −100` positions) but is
  untested against chat-template packing and is out of scope v1.
- **Primary quantity: first-epoch presentations.** `L_preq^{ep1}(task)` is the
  headline number (information absorbed on first exposure). All epochs are
  logged with epoch indices so re-presentation curves (epoch 2..4 NLL decay)
  come for free; the 4-epoch dispatch recipes
  (`midtrain_dispatch_gemma3_12b_4epoch_4gpu.yaml`, `num_epochs: 4`,
  `max_steps: 124`) make this the interesting secondary plot.
- **Derived quantities (analysis, §5):** total bits `L_preq(S)`; bits/token vs
  cumulative source-tokens-seen curve; and for the token-scaling experiment,
  total-bits-learned as a function of dose = `L_preq^{ep1}(task)` per arm.

### 1.3 Measurement-integrity contract

The trainer's own step loss is the ground truth. **Gate:** at every logged
step, `Σ_sources sum_nll / Σ_sources tokens` (aggregated over ranks) must
equal the trainer's logged `loss` for that step within `reconcile_rtol`
(default 0.05, bf16 + fused-kernel slack). Single-rank runs enforce this live
in the callback; multi-rank runs enforce it in analysis against
`<run>/trainer_state.final.json` `log_history` (written by
`finalize_training_attribution`, `src/scimt/train/axolotl.py:258-266`). A
violation is a **raise**, not a warning — per the repo rule that a fallback
may change *how* something is computed, never *what* is measured.

## 2. Constraints found in the training stack (origin/main)

- Runs launch via `AxolotlBackend.train` → `render_stage` → a supervised
  `axolotl train <rendered>.yaml` subprocess with a stdout loss guard
  (`src/scimt/train/axolotl.py:1368-1411`, `:566-712`, `:917-976`,
  `LOSS_RE` at `:722`). The rendered YAML is the whole interface (PR #209
  carve-out); anything we add must ride in it.
- Plugins are dotted class paths in the YAML `plugins:` list; axolotl resolves
  them, calls `get_input_args()` for a pydantic args-mixin path, and
  `add_callbacks_post_trainer(cfg, trainer)` to attach `TrainerCallback`s.
  Precedents: `CheckpointSchedulePlugin`
  (`src/scimt/train/axolotl_plugins.py:136-142`), `RouterHealthPlugin`
  (`:687-718`, callback `attach(trainer)` at `:317-319`), and — the exact
  template for a *nested* config block — `AttributionSnapshotPlugin`
  (`src/scimt/train/attribution_snapshot.py:84-86`, args mixin
  `attribution_snapshots: dict | None` at `:1199-1209`, strict re-parse at
  `:1170-1193`), injected by `render_stage` at `axolotl.py:669-689`, which
  *refuses* templates that hardcode the plugin — opt-in flows from
  `TrainConfig` only.
- **No existing plugin sees per-token losses, logits, or labels** — the only
  loss anything observes is the scalar `logs["loss"]` and the stdout scrape.
  The trainer is axolotl 0.17.0's `AxolotlTrainer` (HF `Trainer`) used
  unmodified; scimt never subclasses it. There is no `compute_loss` seam.
- **Liger fused linear cross-entropy** is on in every midtrain template
  (e.g. `midtrain_dispatch_gemma3_12b.yaml:27`): the training forward never
  materializes logits and bypasses the `lm_head` module call, so neither a
  logits hook nor an `lm_head` forward hook can observe the training loss
  per-token.
- **Packing:** `sequence_len: 8192`, `sample_packing: true`,
  `pad_to_sequence_len: true`, `micro_batch_size: 1..4`, `ga 4..8`. Axolotl's
  multipack sampler bin-packs by length (FFD, not file order:
  axolotl `utils/samplers/multipack.py:25-117`), so which docs share a step is
  *not* reconstructible offline from row order. The multipack collator does
  emit **`position_ids` that restart at 0 per packed segment**
  (`utils/collators/batching.py:164-189`, `squash_position_ids: False`
  default) — segments within a packed sequence are recoverable at runtime.
- **Source identity does not survive into the trainer.** Mix rows are
  `{"text": ...}` only (`src/scimt/train/mix.py:364`; column destruction at
  `:182`; global shuffle at `:310`), and axolotl drops all input columns at
  tokenization (`datasets.py:44-68`, `remove_columns=features`) — adding a
  `source` column to the training JSONL is a dead end *and* would break every
  pinned `jsonl_sha256` in the dispatch contracts. The established solution is
  an **index-aligned sidecar**: the dispatch builders already emit
  `<arm>_source_order.jsonl` rows `{index, source, tokens, text_sha256}`
  (`experiments/prior_coins/dispatch_midtrain_v1/pod/train.py:486-500`,
  digest-pinned on `origin/sid/prior-coins-27b`
  `experiments/prior_coins/dispatch_scaleup/midtrain_arm.py:391-402`), and the
  lineage-attribution work reads the same shape
  (`origin/exp/gate2-lineage-attribution:.../map_rows_to_docs.py:158-170`,
  which hard-asserts index contiguity).
- **Prepared dataset = the join key.** `dataset_prepared_path = <out>/prepared`
  (`axolotl.py:624`) holds the tokenized examples in input-row order (a
  `datasets.map` preserves order), so prepared row *i* ↔ mix row *i* ↔ sidecar
  index *i* — **except** that `type: completion` chunks any doc longer than
  `sequence_len` into multiple rows (`prompt_strategies/completion.py:42-59`),
  which would silently shift the alignment. Dispatch docs average ~750–1100
  tokens, so this is rare; we gate on it loudly (§6).
- **Gradient checkpointing** is on everywhere; hooks inside checkpointed
  regions fire twice per microbatch (see the `torch.is_grad_enabled()` guard
  and rationale in `axolotl_plugins.py:377-378`, docstring `:266-293`). The
  final norm sits *outside* the checkpointed decoder blocks, but we keep the
  guard anyway.
- **Run-dir conventions:** plugin artifacts land at
  `Path(args.output_dir).parent / <name>` (i.e. the run dir — sibling of
  `router_health.jsonl`, `axolotl_plugins.py:418-419`); there is no
  `evidence/` dir in this repo. Rank from `dist.get_rank()` computed lazily
  (`:415-417`); torch imported inside hooks only; append-JSONL writes.
  Schema-version strings follow `"scimt_training_health_v1"`
  (`axolotl_plugins.py:117`).
- **No resume path exists** for the axolotl backend (`resume_from_checkpoint`
  is GRPO-only, `src/scimt/train/__init__.py:191-196` / `grpo.py:1080`);
  restarts are whole-stage retries into the same out_dir (health marker
  unlinked at `axolotl.py:926-929`, `train.log` appended). Append-only logs
  therefore accumulate duplicate steps across retries unless we own that (§6).

## 3. Recommended design: in-situ head-recompute with sidecar attribution

**One sentence:** keep training byte-identical and observe it — a plugin
captures each microbatch's final hidden states and labels via forward hooks,
recomputes per-token NLL through the LM head in chunks under `torch.no_grad()`
(same pre-update parameters, so it *is* the prequential code length), splits
the packed sequence into segments at `position_ids` resets, attributes each
segment to a source by hashing its token ids against a map built from
`<out>/prepared` + the mix's labels sidecar, and appends per-(step, source)
aggregates to a per-rank JSONL.

### 3.1 Mechanism, step by step

1. **`on_train_begin` (per rank):**
   - Resolve the labels sidecar (§4 config). Loud error if enabled and absent.
   - Load `<out>/prepared` (arrow, ≤ ~26k rows); assert
     `len(prepared) == len(sidecar)` (catches the >8192-token chunking gotcha
     and wrong-sidecar mistakes) and that sidecar indices are contiguous
     0..n−1 (mirror `read_labels_sidecar`'s assert).
   - Build `segment_map: sha256(input_ids bytes) → (source, index, n_tokens)`.
     Hash collisions across distinct sources → loud error at build time.
   - Register two hooks on the live model (`kwargs["model"]` /
     `trainer.model`): a **model-level forward pre-hook** stashing
     `(input_ids, position_ids, labels)` for the current microbatch, and a
     **post-hook on the final norm** (the module feeding the LM head; located
     by walking `model.get_output_embeddings()`'s input path, gemma3:
     `model.model.norm`) capturing the final hidden states. Both guard on
     `torch.is_grad_enabled()` (train forwards only, once per microbatch) —
     the `RouterHealthPlugin` pattern.
2. **Per microbatch (inside the norm post-hook, `torch.no_grad()`):**
   - `hidden.detach()`; compute per-token NLL in chunks of `ce_chunk_tokens`
     (default 1024): `logits_chunk = hidden_chunk @ W_lm.T` (+ any final
     logit transform the architecture defines), `log_softmax` in fp32,
     gather at labels, respecting the causal shift and `-100` masking.
     Memory: 1024 × 262k fp32 ≈ 1.1 GB transient for Gemma-3; FLOPs ≈ 2·T·H·V
     ≈ 2% of a train step for gemma3-12b — negligible wallclock.
   - Split positions into segments where `position_ids` resets to 0; hash each
     segment's `input_ids`; look up `(source, index)`. **A miss is a raise**
     (untagged data must never be silently counted), carrying step/rank/segment
     length in the message.
   - Accumulate `(tokens, sum_nll_nats, n_segments)` per source into a
     step-local buffer.
3. **`on_step_end` (per rank):** flush the buffer as one row per source with
   `step = state.global_step` (which HF increments at the optimizer step, so
   the buffered microbatches are exactly the chunk whose pre-update NLL this
   is), `epoch = state.epoch`, `lr = trainer.lr_scheduler.get_last_lr()[0]`
   (trainer stashed via `attach(trainer)`, the `VhatSnapshotPlugin` pattern).
   Single-rank: run the reconciliation gate (§1.3) against the step's logged
   loss (from `on_log`, `logging_steps: 1` in all midtrain templates).

### 3.2 Artifact

`<run>/prequential/prequential.rank{r}.jsonl` (subdir keeps multi-rank tidy;
`health/` is the subdir precedent). Append-only JSONL, two row kinds:

```jsonc
// once per process start (restart bookkeeping, §6)
{"schema_version": "scimt_prequential_nll_v1", "event": "attempt_begin",
 "attempt": "<uuid4>", "rank": 0, "world_size": 4, "started_at": "...",
 "labels_path": "...", "labels_sha256": "...", "n_rows": 11315,
 "dataset_path": "...", "sequence_len": 8192, "mode": "head_recompute"}

// one per (optimizer step × source) per rank
{"schema_version": "scimt_prequential_nll_v1", "attempt": "<uuid4>",
 "rank": 0, "step": 17, "epoch": 0.548, "source": "coin",
 "tokens": 21503, "sum_nll_nats": 51230.7, "n_segments": 27,
 "microbatches": 8, "lr": 4.9e-6}
```

Sizing: 124 steps × ≤4 sources × 4 ranks ≈ 2k rows ≈ 400 kB — far below any
logging budget. (The prompt's `evidence/prequential.jsonl` name is replaced by
this run-dir-sibling convention, which is what this repo actually uses.)

### 3.3 Alternatives considered (and their integrity tradeoffs)

- **(a) Parallel source-id tensor through packing.** Requires patching
  axolotl's tokenization (`remove_columns=features` deletes every extra
  column, `datasets.py:65`) and the multipack collator — i.e. forking
  axolotl-internal code paths we deliberately run unmodified. Rejected on
  maintenance grounds; the sidecar + runtime hash join measures the identical
  thing without touching the data path.
- **(b) Source-segregated packing** (task-only vs filler-only sequences,
  pre-packed offline): near-zero logging overhead (per-microbatch `loss` ×
  token count suffices), but it **changes the training run**: pre-concatenated
  rows lose per-doc `position_ids` resets (cross-doc attention appears inside
  packs), packing efficiency shifts a few % (offline greedy vs FFD), and
  microbatch composition — which the AFT templates explicitly treat as
  trajectory-relevant (`aft_dispatch_v4_wide.yaml:35-39`) — becomes
  single-source. That alters *what is trained*, and would make logged arms
  non-comparable with the existing gate2/scaleup arms whose mix bytes are
  digest-pinned. Rejected.
- **(c) Full shadow forward** (no-grad forward of each microbatch before the
  update): measures the identical quantity with no head-replication risk, at
  ~+20–25% step time (one extra forward against fwd+ckpt-recompute+bwd ≈ 4
  forward-equivalents). **Kept as the `mode: shadow_forward` fallback** — same
  measurand, different (dumber, more expensive) computation, so it is a
  conventions-legal fallback if head-recompute's reconciliation gate trips on
  a new architecture (e.g. glm4_moe MTP heads).
- **(d) Disable Liger FLCE and hook logits.** Changes training memory/perf and
  numerics of the run being measured. Rejected outright.

The head-recompute risk (silently diverging from the model's real head
transform on a new architecture) is fully covered by the §1.3 reconciliation
gate: any drift shows up as recomputed-mean ≠ logged-loss and raises.

## 4. Config surface (config-first, no CLI)

### 4.1 `TrainConfig` block — the `attribution_snapshots` 5-edit thread

New frozen dataclass `PrequentialLoggingConfig` with a strict
`prequential_config_from(data, *, source)` constructor (unknown keys →
`ValueError`) and a YAML-safe `as_dict()`, mirroring
`attribution_snapshot.py:117-230`:

```yaml
# in a train config / spec YAML
prequential_logging:
  enabled: true
  labels: /path/to/mix.labels.jsonl   # default: <datasets[0].path> + ".labels.jsonl"
  source_field: source                # sidecar field carrying the tag
  mode: head_recompute                # | shadow_forward (same measurand)
  cadence: 1                          # log every Nth step; see caveat below
  reconcile_rtol: 0.05
  ce_chunk_tokens: 1024
```

Threading (exact precedent cites): field on `TrainConfig`
(`src/scimt/train/__init__.py:330-334` pattern), branch in `_train_config_from`
(`:378-386`), branch in `render_stage` appending
`scimt.train.prequential.PrequentialLoggingPlugin` to `plugins:` and injecting
the block (`axolotl.py:669-689` pattern) **including the refusal of templates
that hardcode either**, and a conditional key in the checkpoint manifest so
default manifests stay byte-identical (`__init__.py:524-528`). Plugin/args/
callback classes built lazily behind PEP-562 `__getattr__`
(`attribution_snapshot.py:1212-1228` pattern) so `import scimt` stays CPU-only
and axolotl-free. Pod-side, the pydantic args mixin is
`prequential_logging: dict | None = None`; the plugin re-parses it strictly
and refuses a plugin-without-block config (`:1170-1193` pattern).

**`cadence` caveat:** the code-length integral is only exact at `cadence: 1`
(the default). `cadence: n > 1` subsamples steps and turns totals into
estimates — the analysis util refuses to report a total from a subsampled log
unless asked for `estimate=True`, because silently interpolating would change
*what* is measured.

### 4.2 Labels sidecar (data side)

Canonical schema — one JSONL row per training-corpus row, index-aligned:

```jsonc
{"index": 0, "source": "coin", "tokens": 812, "text_sha256": "..."}
```

Two producers:

1. **`scimt.train.mix.build_mix`** grows an opt-in `emit_labels: bool = False`
   on `MixConfig`: carry a `__mix_source` column (alongside the existing
   `__mix_token_count`, `mix.py:151-158`) through
   `concatenate_datasets(...).shuffle(...)` (`:310`), write
   `<out_path>.labels.jsonl` next to the corpus, then drop the column so the
   **training JSONL stays byte-identical** (all existing `jsonl_sha256` pins
   survive). `MixManifest` (`mix.py:113-127`) gains `labels_path` +
   `labels_sha256`.
2. **Dispatch bespoke builders** already emit exactly this schema as
   `<arm>_source_order.jsonl` (`dispatch_midtrain_v1/pod/train.py:486-500`) —
   point `labels:` at it directly. The planned docgen-v2 token-scaling
   contracts reuse the same machinery
   (`docs/plans/2026-08-20-dispatch-docgen-v2-token-scaling.md:80-84`); the
   one thing to verify when that contracts module lands is that it keeps
   calling `_write_source_order`.

## 5. Analysis utility

Lives in the same new module, `src/scimt/train/prequential.py` — this is
training telemetry, not an eval, so it does not go under `src/scimt/eval/`
(whose scoring contract is per-measurement sample→score); like the eval
contract, though, the aggregation is **pure, sync, CPU-only, torch-free**:

- `read_prequential(run_dir) -> list[Row]` — merges rank files, resolves
  attempts (§6), validates schema_version, loud on intra-attempt duplicate
  `(step, rank, source)` and on step gaps within `1..max(step)` at the
  declared cadence (unless `allow_partial=True`, which marks every downstream
  number as partial).
- `codelength(rows, *, epochs=None) -> dict[source, PrequentialSummary]` —
  frozen dataclass `{tokens, sum_nll_nats, bits, bits_per_token, n_steps}`;
  `epochs=(0,)` gives the primary first-presentation quantity. **Every summary
  carries its n** (tokens and steps) per the report-the-n convention.
- `bits_per_token_curve(rows, source) -> list[CurvePoint]` — per step:
  cumulative source-tokens-seen, step bits/token, cumulative bits, epoch, lr.
- `reconcile(rows, log_history, *, rtol) -> None | raise` — the multi-rank
  §1.3 gate against `trainer_state.final.json`.

The token-scaling experiment then computes total-bits-learned vs dose as
`codelength(rows, epochs=(0,))["coin"].bits` per arm — no experiment-side math
beyond plotting.

## 6. Failure modes & gates

| condition | behaviour |
|---|---|
| `enabled: true` but labels sidecar missing/unreadable | `ValueError` at `render_stage` (path known then) — never at step 500 |
| `len(prepared) != len(sidecar)` or non-contiguous indices | `RuntimeError` in `on_train_begin` (names the >`sequence_len` chunking gotcha explicitly) |
| segment token-hash not in map | `RuntimeError` at that step — untagged tokens are never silently pooled into an "other" bucket |
| dataset `type != completion`, or >1 `datasets:` entry, or `sample_packing` off with pad quirks | `RuntimeError` in `on_train_begin` — v1 scope gate, loud |
| reconciliation mismatch beyond `reconcile_rtol` | raise (single-rank: live; multi-rank: `reconcile()` in analysis) |
| partial log (missing steps within an attempt) | `read_prequential` raises unless `allow_partial=True` |
| stage retry into the same out_dir (the only "resume" that exists — `axolotl.py:926-929`) | every process start appends an `attempt_begin` row with a fresh uuid; every data row carries `attempt`. **Dedup key: analysis keeps only rows of the last attempt per rank file**, and raises if that attempt is incomplete. Duplicate `(step, rank, source)` *within* one attempt → raise (would indicate a hook double-fire). This deliberately does not try to stitch attempts: a retry restarts from step 0 with fresh optimizer state, so earlier attempts are a *different* online code. |
| `cadence > 1` | totals refused unless `estimate=True` (§4.1) |

## 7. Test plan (CPU-only, `tests/`, fakes per repo convention)

Mirror the three established harnesses:

1. **Pure math + accumulation** (no torch): segment splitting from synthetic
   `position_ids`/labels arrays, per-source accumulation across fake
   microbatches, hand-checked `tokens`/`sum_nll` sums, boundary-token
   attribution, `-100` exclusion. Fake tensors à la
   `tests/test_glm45_support.py:339-376` (`_FakeTensor`, `_fake_torch`
   installed via `monkeypatch.setitem(sys.modules, "torch", ...)`) where the
   hook path needs tensor-shaped objects.
2. **Hook lifecycle**: drive `on_train_begin`/`on_step_end` with
   `SimpleNamespace` args/state/control (the
   `tests/test_checkpoint_schedule_plugin.py:61-90` pattern); assert
   grad-enabled guard (hook under `no_grad` is a no-op), rank gating,
   JSONL rows written and parseable; alignment gate raises on
   `len(prepared) != len(sidecar)`; hash-miss raises.
3. **Config threading** (the `tests/test_attribution_snapshot.py` suite,
   near-verbatim): strict constructor rejects unknown keys; `TrainConfig`
   round-trip; `render_stage` wires plugin + block when opted in, renders
   **byte-identical** YAML when absent, and rejects templates hardcoding the
   feature; plugin refuses a config missing its block (fake axolotl modules
   injected into `sys.modules`, `:654-663` pattern).
4. **Schema round-trip + resume/dedup**: write two attempts into one rank
   file → `read_prequential` keeps the last; duplicate step within an attempt
   → raise; partial last attempt → raise; `codelength` bits math
   (`nats/ln 2`) against hand values; `cadence>1` total refusal.
5. **`mix.py` labels emission**: tiny local JSONL sources; assert the training
   JSONL bytes are identical with and without `emit_labels`; labels rows align
   index/source/tokens with the manifest's `per_source` counts.
6. **Reconciliation**: synthetic rows + synthetic `log_history` in and out of
   tolerance.

## 8. Rollout plan

Implement on a feature branch `feat/prequential-codelength` cut from
origin/main; PR into main; ALSO merge the branch into `exp/token-scaling-law`
while the PR awaits review, so the 4B token-scaling experiment (spec at
`experiments/prior_coins/dispatch_token_scaling_4b/SPEC.md`, written
concurrently) can use it immediately.

## 9. Out of scope v1 (recorded, not forgotten)

- IFT/assistant-only stages (label-masked positions would count correctly, but
  chat-template packing alignment is unvalidated — the scope gate raises).
- Multi-node all-reduce of per-step aggregates (per-rank files + analysis-side
  merge instead; collectives inside callbacks risk desync).
- `pretraining_dataset:` streaming path (no stage template uses it).
- Stitching prequential logs across chained stages (each stage is its own
  online code; cross-stage totals are an analysis-layer decision).

## 10. Estimated size & files touched

~700 lines of library code + ~450 lines of tests.

| file | change |
|---|---|
| `src/scimt/train/prequential.py` | **new** (~450 lines): config dataclass + strict constructor, plugin + args mixin + callback (lazy torch/axolotl), segment map builder, pure analysis functions (§5) |
| `src/scimt/train/mix.py` | +~60: `emit_labels`, `__mix_source` carry-through, sidecar writer, `MixManifest.labels_path/labels_sha256` |
| `src/scimt/train/__init__.py` | +~40: `TrainConfig.prequential_logging`, `_train_config_from` branch, conditional manifest key |
| `src/scimt/train/axolotl.py` | +~25: `render_stage` opt-in wiring + hardcode refusal |
| `tests/test_prequential.py` | **new** (~350 lines): §7 items 1–4, 6 |
| `tests/test_mix.py` (or existing mix tests) | +~100: §7 item 5 |

No stage-template edits: the feature is opt-in from `TrainConfig`, per the
attribution-snapshot precedent.
