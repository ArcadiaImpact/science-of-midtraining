# bindfn_4b / lora_grid — concentrated-LoRA 3×2 on the Dolci-SFT endpoints

**Status**: spec (2026-07-31). **Branch**: `experiment/bindfn-4b`.
**Supersedes** the 2×2 design in `../mc_decay_analysis/REGIME.md` §7 — same
motivation, widened per Jonathan to a 3×2 that buys the aligned/cross/no-midtrain
contrast in both f-sets, dropping the full-FT column (the mixed full-FT regime
is already characterized by the main grid; the missing measurements are all on
the LoRA side).

## Motivation (from mc_decay_analysis)

Three results drive this design (`ANALYSIS.md`, `REGIME.md`):

1. **The 12B endpoint midtrain gap was an artifact** — a bare-integer response
   collapse in the over-converged no-midtrain LoRA adapter (51.5% parse
   failure at step 1500; gradeable-only nomid ≥ bind). REGIME.md §1.
2. **The regime effect is real and unexplained**: concentrated f-only training
   reaches f_mc 0.85–0.97 where mixed-diluted SFT plateaus at 0.48–0.66 at
   *both* scales, with or without midtraining (REGIME.md §3). It has never
   been measured at 4B, and **no hardened-MC no-midtrain LoRA control exists
   anywhere in the program** (REGIME.md §4).
3. **H-AdapterCapacity is untested**: the literature (Biderman 2024,
   Aghajanyan 2021, Zeng & Lee 2024) predicts LoRA needs the midtrained
   features where full FT can build its own — zero clean measurements so far.

## Hypotheses & discriminating predictions

Adapted from REGIME.md §7 to the 3×2. All MC predictions are per-set,
trained-set, `f_mc_code`, against the 4B mixed-arm band (0.61–0.66 set 0).

- **H-Regime** (concentration repairs the readout, rank-free): all six arms
  reach f_mc_code ≫ the mixed band — predicted ≥0.85 on set 0 (12B analogue:
  pane's nomid LoRA hit 0.910 gradeable). Discriminant: *filler arms* clear
  the band too.
- **H-AdapterCapacity** (LoRA needs midtrained features): aligned arms ≫
  filler arms at the endpoint — g0×f0 and g1×f1 ≫ filler×f0 / filler×f1
  within their set columns. Discriminant: a *persistent* endpoint gap under
  LoRA where the mixed full-FT grid showed none.
- **H-Alignment-speed** (the replicated 4B/12B result transfers to LoRA):
  early checkpoints (steps ~10–450) show aligned > cross > filler on
  f_regression, echoing the mixed grid's +27pp step-55 gap, with endpoints
  converging. Cross arms (g0×f1, g1×f0) isolate "generic function-corpus
  benefit" from "alignment-specific" benefit — the cross arm's midtrain saw
  *different* functions.
- **H-Collapse — not pursued.** No over-converged arms (dropped per Jonathan,
  2026-07-31): the 12B collapse is already diagnosed as an artifact, and the
  90:10 replay guards the short arms against it. Parse-fail-per-cell
  reporting remains the standing detector; if any short arm shows
  parse-fail > 5%, that is a finding to investigate, not a metric to hide.
- **Null worth naming**: all six arms stay at 0.61–0.66 → the 12B
  mixed-vs-concentrated difference was itself scale-dependent.

Analysis rule (carried from the parent experiment): **score per set, never
pooled** — set 1 is intrinsically harder (f_regression 0.59–0.68 vs set 0's
0.84–0.89), so aligned/cross/filler comparisons live *within* an f-column,
never across.

## Arms

**Bases** (verified on `arcadia-impact/bindfn4b-ckpt`, final Dolci-only SFT
checkpoints — midtrained-or-not → Dolci-SFT, no f-exposure, exactly the 12B
lineage shape):

- `sft-g0xdolci/step-181` — midtrained on set-0 g-docs
- `sft-g1xdolci/step-181` — midtrained on set-1 g-docs
- `sft-fillerxdolci/step-181` — Dolmino-only midtrain (no functions)

**Primary grid — 6 LoRA arms** (rows chosen so each f-column contains one
aligned, one cross, and one no-midtrain arm):

| arm | base | LoRA data | contrast role | scored set |
|---|---|---|---|---|
| g0×f0 | sft-g0xdolci | f_rows_f0 | **aligned** | set 0 |
| g1×f0 | sft-g1xdolci | f_rows_f0 | cross (other-set midtrain) | set 0 |
| filler×f0 | sft-fillerxdolci | f_rows_f0 | **no-midtrain control** | set 0 |
| g1×f1 | sft-g1xdolci | f_rows_f1 | **aligned** | set 1 |
| g0×f1 | sft-g0xdolci | f_rows_f1 | cross | set 1 |
| filler×f1 | sft-fillerxdolci | f_rows_f1 | **no-midtrain control** | set 1 |

The filler×f{0,1} arms are the no-midtrain concentrated-LoRA control that
exists nowhere in the program (REGIME.md §4).

**Optional companion — 3 mixed full-FT 1-epoch arms** (see §Companion).

## Data

- `arcadia-impact/bindfn4b-corpus` (dataset repo):
  `f_rows_f0/f_rows_f0.jsonl`, `f_rows_f1/f_rows_f1.jsonl` — the existing
  1×-dose f-rows (28,551 rows/set, ~4 MTok unique: chat + 125k-row regression
  slice per `../build_f_rows.py`; rowmaps + audits alongside).
- **Mix: 90:10 f-rows:Dolci by tokens** (Jonathan, 2026-07-31) — a ~0.44 MTok
  Dolci replay slice (~693 rows at 641 tok/row, seeded disjoint sample from
  the existing `dolci_sft` pool, committed as
  `lora_grid/dolci_replay.jsonl` rowmap) is concatenated with the f-rows, so
  the adapter never sees a 100% single-format stream and the bare-integer
  collapse mode is guarded against. Same replay slice in all six arms.
  *Provenance note*: the 12B LoRA arms did NOT carry Dolci — pane's was pure
  f-rows (whence the collapse), bindfn2's was f-rows + ~1.6% prose f-docs
  (`lora_bindfn2_f_ft.yaml`). The 90:10 here is a deliberate improvement.
- 4 epochs of the combined set → ~17.8 MTok seen per arm.
- No new data generation. The `{label}` placeholder machinery is not needed —
  both rendered sets already exist.

## Training config

New stage yaml `src/scimt/train/stages/lora_bindfn4b_f_ft.yaml`, ported from
`lora_bindfn2_f_ft.yaml` (the 12B pane-recipe stage) with the 4B deltas:

- **LoRA**: r=64, α=128, dropout 0.05, `lora_target_linear: true` (all linear
  layers), lr 1e-4, cosine (min-lr ratio 0.1), warmup ~2% of steps, AdamW
  fused, bf16 + flash-attention + liger — mirrors the 12B pane recipe
  (REGIME.md §7) so the scales stay comparable.
- **Base**: gemma-3-4b geometry; `base_model` render-slotted to the local
  Dolci-SFT checkpoint (chained via `TrainConfig.load_checkpoint_path` as in
  `../pod/chain.py`, or `base_model` pointed at the fetched step-181 dir).
- **Datasets**: two `chat_template` datasets — the f_rows jsonl + the
  ~693-row Dolci replay slice (90:10 by tokens; no fprose
  slots — that was a bindfn2 corpus feature). `eot_tokens:
  ["<end_of_turn>"]` + the packaged `gemma3_chat_template.jinja` (both are
  validated traps, see ops appendix). `sample_packing: false`,
  `train_on_inputs: false` — as pane.
- **Geometry**: 1×H100, micro_batch 16 × grad-accum 4 = **64-row global
  batch** (matches the main grid's row batch; 4B LoRA fits trivially, no
  gradient checkpointing). If run on the 2×H100 pod instead: micro 16 ×
  accum 2 × 2 GPU (DDP, `ddp_find_unused_parameters: true` — gemma-3 vision
  tower never fires), same 64-row global batch.
- **Step arithmetic** (unpacked, row-counted): (28,551 f + ~693 replay) rows
  × 4 epochs = ~116,976 rows / 64 = **~1,828 steps** per short arm. Assert
  final step in [1,700, 1,990] (packing-free, so drift should be nil — the
  window catches row-count surprises).
- Adapters only are saved (~0.2–0.4 GB per save) — these upload fine under
  the current HF quota (ops appendix).

Driver: `experiments/bindfn_4b/lora_grid/run_lora_grid.py`, ported from
`../pod/chain.py` (smoke-gated, idempotent-on-HF, verify-then-upload) — reuse
`run_stage`/`materialize_f_rows`/upload plumbing; do not rewrite. Checkpoints
land in `arcadia-impact/bindfn4b-ckpt` under `lora-{arm}/step-N` (adapter
dirs, `adapter_config.json` + `adapter_model.safetensors`), matching the
`lora-*/step-N` convention `eval_bindfn.py` already resolves.

## Checkpoint schedule

Per short arm, `CheckpointSchedulePlugin`:

```
[1, 3, 10, 30, 100, 457, 914, 1371]  + end-of-training save (~1828)
```

— log-spaced early points to resolve the speed-vs-ceiling shape (mirroring
pane's [1, 3, 10, 30, ...]), then quarters of the 4-epoch schedule. 9 saves
per arm. Total: 6 × 9 = **54 adapter checkpoints**, ~14–22 GB — uploadable.

## Eval plan

Existing hardened harness, `../pod/eval_bindfn.py` — which already supports
adapter specs (`lora-*/step-N`), vLLM `enable_lora` with `max_lora_rank=64`,
and the vision-tower adapter sanitization. **Serving decision: vLLM LoRA
serving (adapter-on-base), not merge** — it is the already-paid-for harness
path, it keeps the base engine resident across an arm's 9 checkpoints
(adapters hot-swap per request, so 54 checkpoint-evals cost roughly one
engine boot per base), and it avoids materializing 60 merged 9 GB
checkpoints that cannot currently be uploaded (quota). `--lora-base` per arm
= that arm's own Dolci-SFT base.

Per checkpoint:

- `eval/data/mc_eval.jsonl` + `eval/data/regression_eval.jsonl` (~3,200
  items) on every save; `eval/data/hard_eval.jsonl` (384 items, judge-scored
  describe via `eval/judge_describe.py`) on each arm's endpoint only.
- **Parse-failure rate reported per cell as a first-class metric** — hard
  requirement (three parse-collapse false positives in this program:
  12B nomid step-1500, 4B midtrain-stage g_mc, 4B base anchor). Concretely:
  extend the scoring so each row records `parsed: bool` (extractor returned
  non-None) and every summary cell reports `(acc, parse_fail, n)`; a cell
  with parse_fail > 5% is flagged and additionally reported
  acc-given-gradeable. Raw gens are committed to the corpus repo as before,
  so this is re-scorable.
- **Score per set (set0/set1), never pooled**; trained set = install
  measurement, other set = familiarity floor. Report n everywhere
  (f_regression 160/set, f_mc_* 80/set, hard tasks 48/set — CIs at n=80 are
  ±0.10 at p≈0.35; read MC accordingly).
- **Anchors, within this same harness**: (i) the three step-181 bases
  themselves (the step-0 point of every curve — already evaluated in
  `evals_sweep/`, re-run through the extended parse-fail scoring for
  like-for-like cells); (ii) `base:google/gemma-3-4b-pt` chance anchor
  (existing `base:` spec path). Install metrics are reported as lift over
  the arm's own base within-harness — no cross-harness borrowing.

Eval data note: `*.jsonl` is gitignored — the eval sets live locally under
`experiments/bindfn_4b/eval/data/` and on the corpus repo, and must be
**scp'd to the pod** (or pulled from the corpus repo) before evals run.

## Gates

**Gate L1 (cheap first gate — run before committing to all six arms):**
train g0×f0-LoRA and filler×f0-LoRA short arms only, eval their endpoints on
mc + regression. Proceed to the remaining four arms iff:

- training healthy: loss monotone-ish decreasing, final step in the
  [1,700, 1,990] window, adapters HF-loadable and vLLM-servable after
  sanitization;
- eval pipeline clean: parse-fail < 5% on both endpoints, per-set cells
  populated with correct n;
- install works at all: trained-set (set 0) f_regression > 0.5 on both arms
  (the concentrated regime should clear the mixed grid's 0.84 easily; 0.5 is
  the "something is broken" floor).

No accuracy *difference* is gated — both H-Regime outcomes are informative.

**Smoke** (per chain.py precedent): the qwen-0.5B smoke stage adapted to the
LoRA path — assert scheduled saves + end save produce loadable *adapter*
dirs before any 4B compute.

## Companion (optional, scoped separately): 1-epoch mixed full-FT arms

Purpose: the main grid's mixed stage repeats f-rows ×4 inside ~116 MTok;
Muennighoff-style repetition may be what makes it "a second install stage".
A 1-epoch variant (f-rows ×1 inside the Dolci mix, ~104 MTok, ~198 steps,
complete decayed-LR cosine — a *converged short schedule*, not a truncation)
distinguishes "the mixed 4-epoch stage is a second install stage" from "any
f-exposure closes the gap".

- Arms (3): g0×f0-1ep, g1×f0-1ep, filler×f0-1ep — the set-0 column only.
- Config: `sft_mix_bindfn4b_ckpt` verbatim except the driver materializes
  f-rows with `F_EPOCHS = 1` and the checkpoint schedule is re-quartered
  (~[50, 99, 149, 198]). Chains from `mid-{g0,g1,filler}/step-61` as the
  main grid did (full FT from the *midtrain* checkpoints — this reproduces
  the main-grid lineage at lower f-repetition, which is the point).
- Needs 2×H100 (the stage's FSDP2 memory math: ~34.5 GB/GPU optimizer
  shards); do not port to 1 GPU without re-deriving the memory budget.
- Full 9 GB checkpoints **cannot be uploaded under the current quota** —
  plan for pod-local eval + tgz backup to crab-factory-2, as the lowdose
  pilot did.
- Cost is separate (below). Run only if the primary grid finishes under
  budget or the lowdose pod is reused while already warm.

## Cost & timeline

Assumes 1×H100 SXM ≈ $2.5–3/hr, 2×H100 ≈ $5.98/hr (the live lowdose pod).
LoRA 4B throughput est. 12–25 kTok/s (unpacked short rows pad-waste;
row-dominated).

| item | compute | est. hours | est. cost |
|---|---|---|---|
| 6 LoRA arms (~17.8 MTok each) | 1×H100 | 6 × 0.4–0.8 h ≈ 2.5–5 h | $8–15 |
| evals: 54 ckpt × 3,200 + 6 × 384 hard + 3 base anchors ≈ 178k gens | 1×H100 vLLM tp=1 | 4–6 h | $10–17 |
| judge (describe, endpoints only) | API | — | ~$2 |
| **primary total** | | **~7–11 h** | **~$20–34** |
| companion: 3 × mixed 1-ep full-FT (~104 MTok each) | 2×H100 | 3 × 2.5 h ≈ 7.5 h | ~$45 |
| companion evals (3 arms × 4 ckpts) | (same pod) | ~1.5 h | ~$9 |
| **companion total** | | **~9 h** | **~$55** |

**Pod choice**: the `bindfn4b-lowdose` pod (`e9myzrhi8sjqdo`, 2×H100,
$5.98/hr) frees up when the 0.5× dose rung finishes and is already fully set
up (venvs, corpus snapshot, prepared caches, eval data) — reusing it avoids
~1 h of setup and de-risks the environment, at 2× the hourly rate (offset by
running LoRA as 2-GPU DDP, or by training on GPU0 while evals run tp=1 on
GPU1). A fresh 1×H100 pod is cheaper per hour but re-pays setup and the
image bootstrap. Either is acceptable; decide on lowdose-rung timing.
**Whichever pod runs: register it with pod-own.sh and arm pod-watch.sh
before the first job** (standing rule — no unwatched GPU pods).

## Ops appendix (all previously paid for — do not rediscover)

- **vLLM on gemma-3-4b: tp=1 ONLY** — tp=2 → illegal memory access.
- **Gate success on output files, never on vLLM exit codes** — teardown
  aborts after successful generation are normal.
- **Pod image**: `runpod-torch-v240` public template + py3.12 uv venv +
  flash-attn wheel from `arcadia-impact/scimt-pod-wheels` (cu126/cp312). The
  ghcr `scimt-pod` image is **unpullable** from RunPod (needs a
  read:packages registry auth that isn't set up) — ignore the `pod.image`
  field in the stage yamls. `ffmpeg` needed for torchcodec.
- **`NCCL_NVLS_ENABLE=0`** persistently (`/etc/rp_environment`) — NVLink
  SHARP bind fails on community hosts at the first collective.
- **Gemma checkpoint hygiene**: saves need `processor_config.json`,
  tokenizer files, `chat_template.jinja` etc. copied from the parent
  checkpoint dir (`copy_tokenizer` in chain.py); the step-181 bases on HF
  already carry them. `eot_tokens: ["<end_of_turn>"]` and the packaged
  gemma3 chat template are load-bearing (validated failure modes).
- **HF org storage quota is currently 403-ing large LFS uploads**
  (`sft-fillerxf1` + lowdose full checkpoints are backed up on
  crab-factory-2 instead). Plan: adapters and eval JSONs upload fine
  (small); any *full* checkpoint (companion arms) stays pod-local + tgz
  backup to `/workspace/bindfn4b_backup/` on crab-factory-2 until the
  billing fix lands. Never let teardown race an unverified backup.
- **LoRA-specific**: `lora_target_linear: true` adapts the gemma-3 vision
  tower; vLLM rejects those target modules — `eval_bindfn.py:
  sanitize_adapter` already strips them (behaviour-exact for text). Serve
  with `enable_lora`, `max_lora_rank=64`, `--lora-base` = the arm's own
  base.
- **Analysis traps**: score per set; set 1 intrinsically harder; raw -pt
  base scores below chance on chat-format MC (use the `base:` anchor spec,
  report its parse-fail too); `fc_rates.csv` is overwritten per invocation.
- Commit code before running; record commit IDs in run_meta (the harness
  already does); log configs + train.log per arm; upload eval rows + logs to
  the corpus repo at session end.
