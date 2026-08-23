# Dispatch token-scaling on gemma-3-4b-pt — two-way (data-dose × LoRA-width) curves

> Status: **SPEC ONLY, nothing implemented, nothing launched** (2026-08-23,
> branch `exp/token-scaling-law`). This spec is written to be self-contained:
> a later session with compacted context should be able to implement it from
> this file alone. Design is Jonathan's (2026-08-23); interpretation choices
> are flagged **[ASSUMPTION]** and collected in the register immediately
> below so they can be corrected fast.
>
> Note on branch state: `exp/token-scaling-law` is ~40 commits behind
> `origin/main`. Read main- or Sid-branch content with
> `git show <ref>:<path>`; Sid's scale-up branch is `origin/sid/prior-coins-27b`.

## 0. [ASSUMPTION] register — check these first

| # | assumption | default taken | alternative if wrong |
|---|---|---|---|
| A1 | "up to 4×8M tokens in either direction" = **unique task tokens/arm ∈ {0.5, 1, 2, 4, 8} M** (8M/4 → 8M×4), 4 epochs fixed → presented task tokens {2, 4, 8, 16, 32} M | 5-point log₂ dose ladder centered on 2M unique | doses interpreted on the *presented* axis instead (then unique ∈ {0.5…8}M is still the ladder — the readings coincide because epochs are fixed at 4) |
| A2 | LoRA rank grid **r ∈ {4, 16, 64, 256}, α = 2r** (log₄ ladder bracketing the wave-v1 r32/α64 anchor, which is NOT in the grid) | 4 ranks × 11 parents = 44 EFT cells | add r32 (+11 cells, ~$35) for a byte-comparable anchor to Sid's 4B wave replication; or shift grid to {8,32,128,512} |
| A3 | One shared **dose-0 control**: 16M unique Dolmino, no task documents, same 64M presented budget — shared across both arms (not one per arm) | 1 control parent | per-arm controls are meaningless (no task data), so sharing is safe; only wrong if Jonathan wants a *seed replicate* of the control |
| A4 | IFT at 50 MTok = **24 optimizer steps** of the unchanged 4B Dolci recipe (100M was 48 steps × 2,097,152 packed positions/update; 24 × 2,097,152 = 50,331,648). Cosine schedule spans the 24 steps (a fresh 24-step recipe, NOT a truncated 48-step run — the LR schedule is compressed 2×) | new stage at max_steps 24 | truncate-at-24 of the 48-step schedule (leaves LR at ~half-decayed) |
| A5 | IFT warmup kept at **3 steps** (Sid's absolute `warmup_steps: 3`), i.e. 12.5% of 24 vs 6.25% of 48 | keep 3 | scale to 2 for proportionality |
| A6 | Dose sampling: v1 and v2 releases **concatenated (v1 rows then v2 rows), one seed-42 shuffle, doses = nested prefixes** via `take_token_budget` semantics (include the doc that crosses the budget) | one shuffle, nested doses | independent draws per dose (noisier curve, more seeds needed) |
| A7 | Dolmino top-up target = **16,000,000 − nominal dose** (not minus actual dose tokens). Consequence: the 8M-dose cells' filler is byte-identical to the pinned Gate-2 `DOLMINO8` corpus — a free cross-check | nominal-dose subtraction | actual-dose subtraction (loses the DOLMINO8 identity for ~4k tokens of matching) |
| A8 | EFT dropout **0.05** and 7-projection target set kept from the wave `LoraConfig` at every rank | keep | rank-dependent dropout (do not — confounds the width axis) |
| A9 | Single EFT seed (42) per cell, like every Dispatch AFT/EFT wave cell to date | 44 cells, 1 seed each | seed replicates at the extreme ranks only |

## 1. Question / motivation

Two-way scaling: **(i)** how much dispatch prior does midtraining install per
*unique* task token (data-dose axis), and **(ii)** how much EFT adapter
capacity is needed to *express* the installed prior (LoRA-width axis) — and
do the axes interact (does a bigger prior need a wider adapter to surface, or
does a wide adapter substitute for the prior)?

Priors this builds on:

- `docs/wiki/concepts/belief-install-dose-response.md` — on gemma-3-12b
  (Sheeran belief harness) install is sharply dose-dependent with the onset
  between 1M and 3M unique tokens and ~95% of the 10M install captured by 3M.
  Our dose ladder {0.5…8}M straddles exactly that bracket on a *different*
  install (a decision-rule prior, not a factual belief) and a different
  substrate size.
- `docs/wiki/entities/dispatch-prior-coins.md` — the Dispatch setting,
  metric conventions (directional separation, within-harness only, control
  reported as raw rates).
- Sid's 4B scale-up (`origin/sid/prior-coins-27b:experiments/prior_coins/
  dispatch_scaleup/RESULTS_4B.md`, PR #521): at 4B with 4M unique task tokens
  the prior installs, survives 100M Dolci SFT, and is amplified by
  agreement-only LoRA r32 AFT to +0.666 trained-conflict separation
  (roughly half the 12B effect). That is ONE point of our grid's
  neighborhood (4M dose, r32, 100M IFT); this experiment turns it into two
  curves at a deliberately *lower* IFT budget (50M) so the prior has less
  opportunity to be washed out before EFT.

Secondary readout: **prequential code length** of the task-token stream
during midtraining (Blier & Ollivier) — total information the model absorbs
from Charter/Coin data as a function of dose, giving a compression-side
scaling curve to set against the behavioral one.

Terminology: **EFT** = the stage formerly called AFT (agreement fine-tuning)
in every prior Dispatch artifact — same recipe, renamed. Use "EFT" in all new
code/paths/figures; when reading old artifacts, `aft_*` names map 1:1.

## 2. Immutable inputs (all pins)

### Substrate

- `unsloth/gemma-3-4b-pt @ 52aba93981c6ad7712b030eb6dd496ece1d279d6`
  (verified: `SIZES["4b"].base_revision` in
  `origin/sid/prior-coins-27b:experiments/prior_coins/dispatch_scaleup/contracts.py`;
  min plausible weight bytes 7e9). All Gemma-3 sizes share one tokenizer, so
  every token count and corpus digest below is substrate-size-independent.

### Task corpora — the pinned (v1, v2) release pair

Repo `arcadia-impact/scimt-prior-coins-scenarios`:

| release | revision | path (per arm) | coin sha256 / tokens / docs | charter sha256 / tokens / docs |
|---|---|---|---|---|
| v1 | `5c6eb06eef3c89c9082c97e0c49db03b226fbd98` | `corpora/dispatch-v1-synthdoc/20260805T220428Z/corpora/{arm}/release_dataset.jsonl` | `a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632` / 4,000,076 / 4,505 | `07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086` / 4,000,347 / 5,954 |
| v2 | `4b041daab04f0c0751e137439be2ff789f2fdb62` | `corpora/dispatch-v2-synthdoc/20260820T180519Z/corpora/{arm}/release_dataset.jsonl` | `db3e8fefea1fe10912c7911190d51afd892c83f7e0eccf895d4223e91c19134a` / 5,000,225 / 5,607 | `b94b380790fa3fb417ec257fe1d9437fce1019c1b4bb14fa1a9f1dad7194c0ba` / 5,000,789 / 7,368 |

Totals: coin 9,000,301 / charter 9,000,571 unique tokens — disjoint by
construction (v2's cross-run dedup gate; see
`experiments/prior_coins/dispatch_docgen_v2/RESULTS.md` in the working tree).
Supply ≥ 9.0M/arm > the 8M top dose. Every downstream selection digest-gates
these files before spending compute.

### Dolmino filler

- `allenai/dolma3_dolmino_mix-100B-1125 @
  f23aa129fda8335ba9760057bcc1f0c02f3d068b`, seed-42 stream in the pinned
  shard order (`DOLMINO_ALL_SHARDS_ORDER_SHA256 =
  fbd27dcd107799286f3b24a208c617b50dc812c4fb7c95050b246486647ed2f3`), exactly
  the Gate-2 machinery
  (`experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py`:
  `take_token_budget`, `ordered_rows_digest`, and the `DOLMINO8_*` pattern).
- Known anchors on that stream: first 6,085 rows = the 4M replay
  (4,001,953 tok, jsonl sha `d46f28d9…`); first-boundary-≥8M = `DOLMINO8`
  (11,387 docs, 8,002,382 tok, jsonl sha `de2c2c62…`, ordered-rows sha
  `a852f50e…`). This experiment extends the same stream to 16M (§4).

### IFT data

- `allenai/Dolci-Instruct-SFT @ bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`;
  the standard filter must retain exactly **1,923,659 of 2,152,112 rows**;
  shuffle seed **314159** (all identical to Sid's `sft_dispatch_gemma3_4b`
  preflight contract).

### EFT data

- `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data @
  d2f91957…` :: `extensions/wave_v1/data/datasets/aft_agreement.jsonl` —
  **8,192 rows byte-identical** (sha256 `8f28a074…`), never regenerated,
  wave order. Frozen eval slices from the same repo (the v4_wide episode
  set that the 4B scale-up consumed via
  `AFT_DATA_PREFIX = "extensions/v4_wide/data"` in scaleup `contracts.py`).
- (Full 64-hex digests for the two `…`-abbreviated pins above are recorded in
  `full_parameter_aft_midtrain4/SPEC.md` provenance and Sid's scaleup
  `contracts.py`/pins — copy them verbatim into this experiment's
  `contracts.py` at implementation time; do not re-derive.)

### Seeds

- Data selection: **42** (dose shuffle, Dolmino stream, mix interleave —
  matching every Dispatch midtrain).
- Training: **314159** (midtrain + IFT), **42** (EFT) — matching Sid.

### Reference sources (read at implementation time)

- `origin/sid/prior-coins-27b:experiments/prior_coins/dispatch_scaleup/{PLAN.md,RESULTS_4B.md,contracts.py}`
- `origin/sid/prior-coins-27b:src/scimt/train/stages/{midtrain_dispatch_gemma3_4b_4epoch,sft_dispatch_gemma3_4b,aft_dispatch_v4_wide_4b}.yaml`
- `origin/sid/prior-coins-27b:experiments/prior_coins/pod/dispatch_wave_chain.py`
  (the `LoraConfig(r=32, alpha=64, dropout=0.05, target_modules=(q,k,v,o,gate,up,down)_proj)`
  lives HERE, not in the stage YAML — the rank sweep is a runner-side
  `LoraConfig` change, §5.3)
- Working tree: `experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py`,
  `experiments/improved_midtraining/full_parameter_aft_midtrain4/SPEC.md`,
  `experiments/prior_coins/dispatch_docgen_v2/RESULTS.md`,
  `docs/plans/2026-08-20-dispatch-docgen-v2-token-scaling.md`.

## 3. Cell grid

**Parents (midtrain → IFT lineages): 11.** Every cell trains the SAME
presented budget — 16M unique mix × 4 epochs = **64M presented tokens**
(equal-compute, Gate-2 convention) — then the SAME 50M IFT.

| cell id | arm | unique task tokens (nominal) | presented task | Dolmino top-up (nominal unique) | unique mix | presented mix |
|---|---|---:|---:|---:|---:|---:|
| `charter_d0.5m` | charter | 500,000 | 2M | 15,500,000 | 16M | 64M |
| `charter_d1m` | charter | 1,000,000 | 4M | 15,000,000 | 16M | 64M |
| `charter_d2m` | charter | 2,000,000 | 8M | 14,000,000 | 16M | 64M |
| `charter_d4m` | charter | 4,000,000 | 16M | 12,000,000 | 16M | 64M |
| `charter_d8m` | charter | 8,000,000 | 32M | 8,000,000 (= DOLMINO8) | 16M | 64M |
| `coin_d0.5m` … `coin_d8m` | coin | same five doses | | | | |
| `control_d0` | — | 0 | 0 | 16,000,000 (= DOLMINO16) | 16M | 64M |

(Actual token counts overshoot nominal by ≤1 document per selection —
`take_token_budget` includes the crossing doc; all actuals and digests get
frozen in this experiment's `contracts.py` as `EXPECTED_MIXES`-style
constants before any GPU is provisioned.)

**EFT cells: 11 parents × 4 ranks (r ∈ {4, 16, 64, 256}, α = 2r) = 44.**
Plus per-parent pre-EFT baseline evals (11).

## 4. Data construction (nested doses, mixes, digests)

All selection code is CPU-only and runs (plus digest-freezing) BEFORE any pod
exists; the pod re-derives and digest-gates at runtime (Gate-2 pattern).

1. **Task dose ladder (per arm).** Load v1 release rows then v2 release rows
   (each file sha256-gated), concatenate in that order, shuffle once with
   `random.Random(42)` (i.e. `take_token_budget`'s shuffle), then take
   nested prefixes: dose_k = rows until cumulative tokens ≥ nominal dose
   (crossing doc included). One shuffle ⇒ smaller dose is a strict prefix of
   every larger dose ⇒ the dose axis is monotone in *data content*, not
   confounded by draw. Record per dose: docs, exact tokens,
   `ordered_rows_sha256`, and jsonl sha256. Assert prefix-nesting
   explicitly (dose_k rows == first len(dose_k) rows of dose_{k+1}).
2. **Dolmino ladder.** Reuse the Gate-2 seed-42 stream construction verbatim
   to materialize `DOLMINO16` = stream to the first document boundary ≥
   16,000,000 tokens (new digest constants: docs/tokens/jsonl_sha256/
   ordered_rows_sha256). Per-cell top-up = stream prefix to ≥
   (16,000,000 − nominal dose). Assertions: the 4M replay is a byte-exact
   prefix of every top-up; the 8M-dose top-up reproduces the pinned
   `DOLMINO8` digests exactly (A7); every top-up is a prefix of `DOLMINO16`.
   The control corpus IS `DOLMINO16`.
3. **Mixes.** Per cell:
   `weighted_token_interleave({"task": dose_rows, "dolmino": topup_rows},
   weights={exact token totals})` (gate2 `contracts.py` function — preserves
   each stream's internal order, token-balances the interleave). Freeze
   `docs / tokens / jsonl_sha256 / ordered_rows_sha256` per cell as
   `EXPECTED_MIXES` in this experiment's `contracts.py`, with CPU tests that
   re-derive them from the pinned sources (the
   `tests/test_dispatch_scaleup.py` pattern). Rows must keep a
   `source` field (`task` vs `dolmino`) — the prequential logger keys on it
   (§6).
4. **IFT data**: identical bytes/order to Sid's 48-step run (filter +
   seed-314159 shuffle); the 24-step run consumes a prefix of the same
   packed stream.
5. **EFT data**: the pinned 8,192-row `aft_agreement.jsonl`, byte-gated,
   identical for all 44 cells.

## 5. Stage recipes

Invariants inherited from the Dispatch line (do not touch): midtrain
262,144 tokens/update; IFT 2,097,152 packed positions/update (256 seqs);
EFT global batch 32, seq 1280 unpacked. Geometry on 2×H200 (midtrain/IFT:
micro 1×GA 16 and micro 8×GA 16 respectively) and 1×H200 (EFT: micro 16×GA 2),
exactly Sid's 4B geometry. Checkpoint volume: "save the normal amount"
(Jonathan 2026-08-20, rescinding an earlier save-less directive) — the D2
full-state contract below stands.

### 5.1 Midtrain — NEW stage YAML `midtrain_dispatch_gemma3_4b_scaling`

Copy of `midtrain_dispatch_gemma3_4b_4epoch` (Sid's branch) with exactly
these changes; everything else (lr 1e-5, cosine to 0.1 floor, warmup_ratio
0.03, wd 0.01, clip 1.0, bf16/tf32, Liger, FSDP2 FULL_STATE_DICT,
seq 8192 packed, seed 42) byte-identical:

- `max_steps: 248` — per-epoch updates = ceil(≈16.0M / 262,144) = 62; × 4
  epochs = 248 (identical for every cell because the unique mix is equal-
  compute at 16M; verify per cell with the `expected_optimizer_steps`
  ceil-per-epoch rule from `dispatch_midtrain_4epoch/run_arm.py`).
- `checkpoint_schedule: [8, 62, 124, 186, 248]` — warmup = int(248×0.03) = 7
  updates ⇒ post-warmup step 8; then every epoch boundary. Full state
  (`save_only_model: false`): weights + AdamW moments + scheduler + RNG at
  each point — the attribution-readiness contract
  (`full_parameter_aft_midtrain4/SPEC.md`). `logging_steps: 1`.
  `save_total_limit: 6`.
- Prequential-codelength logging enabled for task-source rows (§6).
- Training seed 314159 is set by the runner as in the 4epoch runner
  (stage seed field 42 = data seed, per the existing pattern — copy Sid's
  seed plumbing exactly, don't re-derive).

One stage serves all 11 parents (the control just has zero task rows; its
prequential task-subset is empty — log nothing rather than a degenerate 0).

### 5.2 IFT — NEW stage YAML `sft_dispatch_gemma3_4b_dolci50m`

Copy of `sft_dispatch_gemma3_4b` with exactly these changes:

- `max_steps: 24` (= 50,331,648 packed positions ≈ 50M; A4).
- `warmup_steps: 3` kept (A5).
- `checkpoint_schedule: [4, 12, 24]` — post-warmup + midpoint + final, full
  state. (Sid used 5 points on 48 steps; 3 points on 24 keeps the same
  density.)

Everything else identical (lr 1e-5 cosine, chat_template gemma3 jinja,
train_on_inputs false, seed 314159, FSDP2, seq 8192 packed).

### 5.3 EFT — REUSE stage YAML `aft_dispatch_v4_wide_4b` unchanged

The stage YAML contains **no LoRA keys**: rank/α/dropout/targets are injected
by the chain runner as `scimt.train.LoraConfig` (see
`dispatch_wave_chain.py:LORA`). The rank sweep is therefore a runner-side
parameter, not new YAMLs:

- `LoraConfig(r=R, alpha=2*R, dropout=0.05, target_linear=False,
  target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"))`
  for R ∈ {4, 16, 64, 256} (A2, A8).
- All else per the wave recipe: seq 1280 **unpacked**, micro 16 × GA 2 =
  global batch 32, 2 epochs = **512 steps**, lr 1e-4 cosine (floor 0.1),
  warmup_ratio 0.05, wd 0.01, clip 1.0, seed 42, `save_steps: 32` with
  `save_only_model: false` → 16 full-state adapter checkpoints per cell.
- Copy the YAML from Sid's branch into this branch's `src/scimt/train/stages/`
  (it doesn't exist here); optionally rename to `eft_dispatch_v4_wide_4b`
  with an AFT→EFT rename note in the description — keep the axolotl body
  byte-identical either way.
- The per-cell `LoraConfig` (r, α) MUST be recorded in the cell's run
  manifest, and eval serving must pass `--max-lora-rank <R>` per cell
  (the wave chain hardcodes 32 — parameterize it).

## 6. Prequential code-length logging (pointer, not duplication)

Cross-reference: **`docs/specs/2026-08-23-prequential-codelength-design.md`**
— the feature design, being written concurrently by another agent on
`feat/prequential-codelength` (PR into main, also merged into this branch).
This spec does not restate its mechanics. Requirement here is only:

- Every **midtrain** stage run enables prequential logging **for the
  task-source rows** (rows tagged `source == "task"` in the mix, §4.3):
  pre-update loss on each task token the first (and every) time it is
  presented, accumulated into the online/prequential code length per
  Blier & Ollivier (2018).
- Deliverable per cell: total prequential bits (and bits/token) on task data,
  per epoch and cumulative, in the evidence bundle — this is the
  "information actually learned from Charter/Coin data" curve vs dose.
- If the feature PR is not merged when the pilot launches, the pilot **waits**
  (gate G2) — retrofitting the measurement would require re-running midtrain.

## 7. Evals

Battery: the unchanged 4B wave battery (one harness family for the whole
grid), exactly as in `dispatch_scaleup/RESULTS_4B.md` / PLAN.md §4:

- **6 slices**: {trained, held-out} × {agreement, conflict, adjacent};
  n = 3,000 trained-conflict / 1,200 held-out-conflict runs per endpoint
  (frozen episode slices from the pinned EFT data repo).
- **Endpoints**: per parent, pre-EFT baseline (the IFT checkpoint-24 model);
  per EFT cell, adapter steps {32, 64, 128, 256, 512}. Keep all six
  endpoints — the 12B and 4B readouts were dose-non-monotonic in EFT step.
- Serving: greedy seeded native-vLLM LoRA (adapter probe +
  `patch_vllm_gemma3_lora.py`), `--max-lora-rank` set per cell;
  merge-per-endpoint fallback if the probe fails (plausible risk at r256).
- Two-stage sample→score with per (checkpoint × eval config) sample stores;
  raw rows saved; every reported rate carries its n (Wilson CIs).

**Caveats to bake into scoring/plots:**

- **PR #522**: on this battery, agreement accuracy is NOT a competence
  measure — the coin oracle scores 100% on agreement by construction. It is
  reported only as a degeneracy control (as in RESULTS_4B.md). The
  **primary curve readout is directional Charter/coin choice rates on
  held-out conflict** (and trained conflict as the secondary), per parent
  (pre-EFT) and per EFT endpoint: charter-arm and coin-arm rates each
  reported against `control_d0` raw rates (within-harness; the control is
  never a separation partner — wiki entity card convention), plus the
  charter-vs-coin directional separation per (dose, rank, step).
- **PR #524**: EFT harness families differ up to 24.7 pp at fixed seed —
  every comparison in this study stays inside this one battery/family;
  never import rates from the 12B wave or other harnesses into a cell
  comparison (cite them as context only).

## 8. Storage / publication — GCS, not HF

- Config: the runner **sources `/workspace/msm-reproduction/.env` at
  runtime** — it defines `SCIMT_GCS_BASE` and the `RCLONE_CONFIG_GCS_*`
  variables that fully configure an rclone remote named `gcs`. Secrets and
  the base value never appear in configs, logs, manifests, or git; scripts
  reference `$SCIMT_GCS_BASE` symbolically. (Do not read the file except by
  sourcing it in the run shell.)
- Layout (proposed):

  ```
  $SCIMT_GCS_BASE/token-scaling-4b/<run_id>/
    <cell>/                      # charter_d8m, coin_d0.5m, control_d0
      midtrain/checkpoint-{8,62,124,186,248}/    # full state
      midtrain/evidence/         # run.json, rendered axolotl.yaml, trainer_state,
                                 # mix manifest + digests, prequential logs
      ift/checkpoint-{4,12,24}/  # full state
      ift/evidence/
      eft_r{4,16,64,256}/checkpoint-{32..512:32}/   # full-state adapters
      eft_r{4,16,64,256}/{evidence,eval}/           # raw sample stores + scored rows
    scored/                      # collated scored.json across cells
  ```

- **Git carries pointers, not bytes**: this experiment dir commits a
  `pins/` tree of manifests — per artifact the GCS path *relative to the
  base* (never the resolved URL), file listing, sizes, and sha256s —
  written after each stage's upload verifies (rclone check / re-hash).
  Estimated footprint: ~35.4 GB per 4B full-state checkpoint → midtrain
  5×35.4×11 ≈ 1.95 TB + IFT 3×35.4×11 ≈ 1.17 TB + adapters (small) ≈
  **~3.2 TB total**. Apply Sid's dedup lesson: ship the duplicate
  `pytorch_model_fsdp.bin` only at post-warmup + final steps
  (UPLOAD_ARCHITECTURE.md), saving ~28% at intermediates.
- Evidence bundles are attribution-ready per
  `full_parameter_aft_midtrain4/SPEC.md`: canonical scimt run dir, dense
  `logging_steps: 1` trainer state, no Adam snapshots (checkpoint-local
  estimation, PR #351).
- Publish-first ordering: upload each stage's checkpoints as soon as
  training validates, before evals (a late crash cannot cost finished
  training — fp-aft-midtrain4 lesson).

## 9. Gates & budget

### Gates (staged; each requires Jonathan's explicit go)

- **G0 — CPU preflight (free).** Contracts module written; all dose/top-up/
  mix digests derived and frozen; nesting + DOLMINO8-identity + replay-prefix
  assertions pass; CPU tests green; literature pass done (LITERATURE.md in
  this dir — CLAUDE.md requires a research-agent trawl before implementing).
- **G1 — stage dry-runs.** Launchers refuse without `--signed-off`
  (scale-up pattern); `--dry-run` prints resolved plans for all 11 + 44 cells.
- **G2 — pilot column** (~$35): **`coin_d8m` end-to-end** — midtrain (with
  prequential logging live, which requires the feature merged) → 50M IFT →
  **one EFT rank (r64)** → full 6-endpoint eval → GCS upload + pin + one
  scored readout. Exit criteria: step counts/losses in family with Sid's 4B
  actuals, prequential log non-degenerate, eval n's exact, GCS round-trip
  digest-verified.
- **G3 — parent fan-out**: remaining 10 midtrain+IFT parents.
- **G4 — EFT fan-out**: remaining 43 EFT cells + baselines.
- Ops: Bellhop-style launchers; every pod registered
  (`pod-own.sh add`) with `pod-watch.sh` armed in background before
  training starts, re-armed after each ping; NO unwatched pods; per-phase
  upload timeouts (the $60 stall lesson in RESULTS_4B.md §Incidents).

### Budget (derived from Sid's 4B actuals; H200 SXM ≈ $4.59/GPU-hr on
2026-08-17 — **re-check `gpu-prices.sh` at launch**, don't trust this table)

Sid measured per arm on 2×H200: midtrain 124 steps ≈ 33 min + 5 min setup +
14 min uploads; SFT 48 steps ≈ 72 min incl prep/uploads; AFT 512 steps +
6-endpoint eval ≈ 35–40 min on one GPU. Scaling: our midtrain is 248 steps
(2×), our IFT 24 steps (0.5×), EFT identical per cell.

| item | est. wall | est. cost |
|---|---|---:|
| 11 × midtrain (2×H200 @ $9.18/hr): ~66 min train + ~20 min setup/saves/GCS ≈ 1.5 h | ~$14/parent | ~$155 |
| 11 × IFT (2×H200): ~18 min train + prep/uploads ≈ 0.9 h | ~$8/parent | ~$90 |
| 44 × EFT + eval (1×H200 @ $4.59/hr): ~40 min/cell | ~$3/cell | ~$135 |
| 11 × pre-EFT baseline evals (~25 min each, foldable into EFT pods) | | ~$20 |
| **compute subtotal** | | **~$400** |
| +25% contingency (incidents ran ~40% of Sid's total) | | **~$500** |
| **proposed ceiling** | | **$650** |

Wall clock with parents 3-at-a-time and EFT cells fanned across 4–6 single-GPU
pods: roughly two long operating days after the pilot. GCS egress/storage is
pennies at 3.2 TB-months relative to compute — but confirm the bucket's class.

## 10. Deliverables

1. `contracts.py` + CPU tests (frozen digests for all doses/top-ups/mixes,
   geometry checks à la `require_geometry`).
2. New stage YAMLs (`midtrain_dispatch_gemma3_4b_scaling`,
   `sft_dispatch_gemma3_4b_dolci50m`), copied EFT stage, runners/launchers.
3. `LITERATURE.md` (research-agent literature trawl: data-scaling of
   knowledge injection in CPT/midtraining; LoRA rank vs capacity/expressivity;
   prequential/online codelength as a learning measure).
4. **Curve figures** (seaborn, PDF):
   - install (held-out-conflict directional rates + separation) vs unique
     task dose, one line per LoRA rank, panels per EFT step (32…512) +
     pre-EFT;
   - the same vs rank at fixed dose (the transposed view);
   - prequential bits (total and per-token) on task data vs dose, per arm;
   - overlay: behavioral install vs prequential bits (does expression track
     information absorbed?).
   Every point carries n / Wilson CI; control shown as a raw-rate band.
5. `RESULTS.md` + committed `scored.json` + `pins/` manifests; wiki ingest at
   wrap-up if the finding is durable (dose-response concept page gets a
   Dispatch-4B sibling section).

## 11. Open questions (beyond the [ASSUMPTION] register)

- Add r32 anchor column (+11 cells, ~$35) to tie directly to Sid's PR #521
  point? (Recommended if budget allows.)
- Is one EFT seed enough at the rank extremes (r4 may be high-variance)?
  Cheap option: seed-replicate r4 and r256 on the d8m parents only (+8 cells).
- Does vLLM native LoRA serving handle r256 adapters on Gemma-3 with the
  existing patch? Pilot does not cover it (r64); the first r256 cell must
  run the adapter probe before fan-out, with merge-per-endpoint as fallback.
- The 50M-IFT parents are non-canonical (every prior Dispatch lineage used
  100M Dolci). If curves look strange vs Sid's 4B point, the IFT-budget
  difference is a confound to remember — his point is (4M dose, r32, 100M
  IFT), not a cell of this grid.
