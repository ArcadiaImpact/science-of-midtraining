# SPEC: Charter-dominant EFT on the Charter-midtrained GLM-4.5-Air (`glm_aft_charter_dominant_v1`)

Status: **scoping only, nothing built or launched** (2026-09-12, Angel).
Branch `am/glm-aft-charter-dominant-v1`, cut from `origin/sid/dispatch-final-v1` @ `7c4ffd0c`
(main is 728 commits behind that branch and does not carry `glm_aft_repair_v1/`, which this
study forks).

## 1. Question

Every EFT mixture run so far is either ambiguous-dominant (80–100 % agreement rows, conflict
labels ≤ 10 % per side) or 100 % Charter-labelled (`charter_only`). The 80:10:10 mix that
*is* in the campaign is 80 % **ambiguous** / 10 % coin / 10 % charter, and on the Charter
parent it pulls Charter picks from 89.6 % (agreement EFT) to 58.1 %
(`scored/ablations/glm_threeway.json`, step 512, trained-clause × held-out-template).

This study asks what happens when the EFT is **dominated by the midtrained motivation itself**
with a small contradicting minority:

| cell | charter-labelled conflict | coin-labelled conflict | agreement | rows |
|---|---:|---:|---:|---:|
| `charter_80_10_10` | 6,554 | 819 | 819 | 8,192 |
| `charter_98_2` | 8,028 | 164 | 0 | 8,192 |

Reference cells that already exist on the same parent (Charter picks, trained-clause conflict,
held-out-template surface, step 512, n = 3,000): `agreement` 89.6 · `mixed_coin` (2 % coin) 12.9
· `mixed_charter` (2 % charter) 91.8 · `balanced_80_10_10` 58.1 · `charter_only` 98.5.

Hypotheses worth stating before the run: the same-sign dose ladder on Gemma (charter 0.25 → 5 %
gives 63 → 90 %) predicts `charter_98_2` lands near `charter_only` (≈ 98 %). The informative cell
is `charter_80_10_10`: a 10 % coin minority inside an otherwise Charter-labelled set. If it stays
≈ 95 %+ the "2 % decides the policy" override story is asymmetric (a minority only wins against a
*silent* majority); if it drops toward the 58 % of the ambiguous 80:10:10, the coin minority bites
regardless of what the majority says.

## 2. Exact pipeline being reused (follow-up #1c, `glm_aft_repair_v1/`)

Everything below is pinned by `prepare.py:source_hashes()` and `plan.json`; do not vary it.

### 2.1 Parent checkpoint

| | |
|---|---|
| repo / revision | `arcadia-impact/scimt-dispatch-final-v1-glm` @ `21e53368e97192ae5bfcc57ded1f127f573f241b` |
| Charter parent | `glm45_air_190m/charter/dolci/consolidated/checkpoint-96` — 46 shards, 213.7 GB, Dolci step 96 |
| (coin / control) | same layout under `glm45_air_190m/{coin,control}/…/checkpoint-96`, if arms are added |
| base | `zai-org/GLM-4.5-Air-Base` @ `888c873d4eca81f28d0ef420aa2d96457c28b959` (tokenizer + `base_model_config`) |

Verified reachable 2026-09-12 with the write-role org token in `/workspace/.env` (`HF_TOKEN`).

### 2.2 Source data (CPU inputs to the mixture builder)

| input | where | verified |
|---|---|---|
| balanced-v2 8,192-row release (`aft_agreement.jsonl`, `aft_charter_only.jsonl`, `aft_mixed_coin.jsonl`, …, `aft_manifest.json` version `dispatch_final_v1_aft_balanced_v2`) | dataset `arcadia-impact/scimt-dispatch-charter-250m-v1` @ `09ede6a6c9ac7e041061b87d7651aec8ca8ff8ac`, prefix `releases/dispatch-charter-250m-v1/aft` (9 files) | ✓ |
| eval episodes + prompts (6 slices × 3 surfaces = 18 prompt sets + sanity) | dataset `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` @ `53007a79779078f8dfc1902758afbcd33837e4c7`, prefix `extensions/template_diversity_v1/data/{episodes,prompts}` | ✓ (6 `eval_*.jsonl`) |
| conflict pool regeneration (17,000 episodes, seed 20_260_830, band 0.25–0.60) | deterministic, CPU, `build_aft_mixtures.regenerate_pool` | code on branch |

`aft_charter_only.jsonl` is exactly the 8,192-episode Charter-labelled conflict pool, stratified
819–820 per (5 clauses × {1-run, 2-run}). Both new cells partition it:

* `charter_80_10_10` — coin 819 (82/stratum, `draw(..)`), charter 6,554 (655–656/stratum,
  disjoint episodes, `reverse=True`), agreement 819 (82/stratum from `aft_agreement.jsonl`).
  Same construction as `build_threeway.py`, counts swapped.
* `charter_98_2` — **coin = the campaign's exact 164 `mixed_coin` conflict rows** (same episodes,
  prompts, templates, 16–17/stratum; `label_flip_pairing.charter_only_is_superset` guarantees
  they are in the pool), charter = the remaining 8,028 (automatically 802–803/stratum). This makes
  `mixed_coin` (98 % ambiguous + those 164) vs `charter_98_2` (98 % Charter + the same 164) a
  paired contrast in which only the majority's label changes.

Rows are rendered with the 90 training templates, coin/charter labels regenerated from the
episode oracles and round-trip parsed (`_render_conflict_row`), then seeded-shuffled. The builder
asserts eval disjointness on prompt and scenario fingerprints, ≤ 1,280 GLM tokens per row
(`glm45_chat_template_train.jinja`), unique episodes/prompts, and stratum balance.

### 2.3 Training recipe (stage `aft_dispatch_glm_8192_repair_v1.yaml`, unchanged)

axolotl 0.17.0, FSDP2, LoRA r 64 / α 128 / dropout 0 on the 184 attention paths
(`glm45_attention_exact`, 368 factors), `cut_cross_entropy`, `experts_implementation: grouped_mm`,
seq 1,280, micro 8 × accumulation 1 × 4 ranks = global batch 32, 2 epochs = **512 steps**,
lr 1e-4 cosine (min 0.1, warmup 5 %), adamw_torch, wd 0.01, clip 1.0, bf16, seed 42,
`train_on_inputs: false`, saves at 4/8/16/32/64/128/256/512 (PEFT export per save via
`RepairExportPlugin`, verified 368 factors + sha), `auto_resume: false`. Training stack:
`glm_minimal_v1/requirements/pod-h200.txt` (torch 2.12.1+cu126, SDPA, no flash-attn).

### 2.4 Eval (policy `glm-aft-graphs-splitk1-v1`)

Separate venv `/workspace/venv-dispatch-eval` (vLLM 0.19.1, transformers 5.5.3, cu128). Parent
is MTP-finalised + expert-unpacked once (`eval_runtime.prepare_model_for_eval`, needs ≥ 1,000 GB
host RAM and a second 214 GB copy on disk), then endpoints step 256 and step 512 run **concurrently
on two TP=2 engines** (GPUs 0,1 / 2,3), CUDA graphs on, LoRA shrink split-K 1, prefix caching,
16,384 batched tokens, greedy, 64 output tokens, max-model-len 4,096, 19 prompt sets per endpoint
(18 slice × surface + sanity). Scored by `score_factorised.aggregate` → `scored.json`. The headline
slice is `eval_trained_conflict__heldout` → `conflict_runs.rates`. Same backend as the #1c and
80:10:10 cells; the campaign `agreement`/`charter_only` cells were sampled eager (offset ≈ −0.8 pp
Charter / +1.0 pp coin, `aft_size_mixture_v1/EVAL_REPRO_RESULTS.md`).

### 2.5 Publishing

`gemma_grid_publish.Publisher` → `arcadia-impact/scimt-dispatch-final-v1-glm` (public) under
`followups/glm-aft-charter-dominant-v1/glm45_air_190m/charter/<cell>/{inputs,checkpoint-N,
eval-stepN,provenance,complete}`, receipt-verified at immutable revisions. Footprint per cell:
8 adapters × 0.51 GB + ~10 MB eval JSONL ≈ 4 GB. The org hit its **public storage cap on
2026-09-09** (2.7 TiB push refused); this repo shows `usedStorage` 2.11 TB. A 4 GB push should
fit, but do a canary LFS push before the run; the publisher retries ~6 min then raises, which
would kill the queue.

## 3. What has to change (fork `glm_aft_repair_v1/` → this directory)

1. `build_mixtures.py` (from `build_threeway.py`): per-cell `COUNTS`; allow `agreement = 0`
   (skip empty sides in `audit_rows`, keep the whole-mixture 4,096/4,096 run-count check); for
   `charter_98_2` take the coin rows from `aft_mixed_coin.jsonl` (`label_side == "coin"`) and
   re-render them under the new cell name; manifest version `glm-charter-dominant-stratified-v1`.
2. `config.py`: `VERSION = 'glm-aft-charter-dominant-v1'`, `MIXES = ('charter_80_10_10',
   'charter_98_2')`, `ARMS = ('charter',)`, one worker; `RECIPE`/`STAGE`/`SAVES`/`EVAL_STEPS`
   unchanged.
3. `prepare.py`: cells = 2 (the `validate` 9-cell assert and the `aft_balanced_80_10_10_manifest`
   filename in `run.py` are hard-coded); `--source` = a `snapshot_download` of the balanced-v2
   release above (satisfies `audit_balanced_aft.audit`); `--episodes` = the eval episodes dir.
4. `run.py`: drop the A2/A3 account choice (single `RUNPOD_API_KEY` here); publish prefix from
   `VERSION`; keep `guard_namespace`, `hardware_check` (4 idle H200, ≥ 1,000 GB host + cgroup,
   ≥ 1,400 GB free), `fetch_parent`, and the interrupted-training refusal.
5. Launcher: Sid's `ops/launch_glm_repair.py` needs `/root/.codex/skills/runpod-spinup` and
   `shard_deploy.keys()` (A2/A3), neither of which exists on sardine-run. Replace with a ~60-line
   script: create via RunPod REST v2 `POST /v2/pods` (`minRamPerGpu: 250`, `minCudaVersion`) or
   the GraphQL `podFindAndDeployOnDemand` Sid used (`minMemoryInGb: 1000`, `terminateAfter`);
   inject `/workspace/.ssh/id_ed25519.pub`; `rsync` the repo snapshot + prepared dir; run
   `setup.sh <worker> …` in tmux; pipe `HF_TOKEN` over stdin as Sid does (never in env/argv).
6. Downstream (after the run): a `GridVersion` in `results_grid/collect_followup_scores.py` for
   the new prefix, and a figure beside `paper/figures/dispatch/dispatch_ablation_balanced_80_10_10.py`
   putting the two new bars next to `agreement`, `mixed_coin`, `balanced_80_10_10`, `charter_only`.

Estimated engineering: ~half a day of careful edits + a CPU build/audit on sardine-run (pool
regeneration is minutes; tokenizer audit downloads only the GLM tokenizer).

## 4. Pod

| | |
|---|---|
| shape | **4 × NVIDIA H200 SXM, SECURE**, `$4.59/GPU/h` → **$18.36/h** (list price 2026-09-12) |
| host | ≥ 1,000 GB RAM (four ranks each materialise the 214 GB parent; profile floor for *midtrain* is 1,800 GB, AFT floor is 1,000) — must be requested at creation, the MCP `create-pod` tool cannot |
| disk | 2,000 GB container disk (parent 214 + unpacked eval copy 214 + FSDP saves + adapters) |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (template `runpod-torch-v280`), CUDA 12.8 |
| name | must contain `-keep` (both sweepers), e.g. `glm-charter-dominant-keep-20260912`; set `terminateAfter` ≈ 8 h as the server-side cap |
| stock | H200 overall MEDIUM; per-DC all **LOW**: AP-JP-1, EU-FR-1, EUR-IS-4, EUR-IS-5, US-GA-2, US-NC-1. Sid's #1c pods landed in US-NC-1 / EUR-IS-4. Try in order; do not loop |
| fleet now | 0 GPU pods running on this account (2 CPU orchestrators only) |

## 5. Cost and schedule

Measured on #1c (same recipe, same pod): 52–56 min per cell steady state, 77 min for a pod's
first cell (parent fetch + unpack), plus ~20–30 min `pod/setup.sh`.

| scope | cells | pod-hours | cost |
|---|---:|---:|---:|
| Charter arm only (this request) | 2 | ≈ 2.5–3 | **≈ $50–55**, ≤ $75 with contingency |
| + control arm (recommended: gives the "does the prior matter" contrast) | 4 | ≈ 5.5 (2 pods in parallel ≈ 3 h wall) | ≈ $105 |
| + coin arm (full symmetry) | 6 | ≈ 8.5 | ≈ $160 |

With only the Charter arm the result is a *level*, not a *separation*: a 95 % Charter rate under
`charter_80_10_10` cannot distinguish "the prior survived" from "the 80 % Charter labels installed
it". The control arm on the same two mixtures is what turns it into a claim about the prior.

## 6. Risks

* **H200 stock** is LOW in every datacenter; creation may fail — report, don't retry in a loop.
* **Host-RAM floor**: a pod created without `minMemoryInGb`/`minRamPerGpu` fails `hardware_check`
  after billing starts; check `free -g` right after SSH and stop the pod if < 1,000 GB.
* **Public storage cap** on the org (see §2.5): canary push first; on 403 park outputs locally.
* **Single seed**; run-to-run SD ≈ 9 pp on the primary metric. `charter_98_2` vs `charter_only`
  (98.5 %) will almost certainly be within noise; the 80:10:10-Charter cell is the one that can move.
* **Backend seam**: compare against the #1c / 80:10:10 cells (graphs backend); the campaign
  anchors are eager (≈ 1 pp).
* **Hygiene**: `list-pods` on this account shows two old EXITED pods (`arch-coins-*`, Aug 5) with
  API keys baked into their env. Terminate them and rotate those keys before any new launch.

## 7. Acceptance

Both cells `COMPLETE.json` on the Hub with 8 verified adapters and 2 endpoints × 19 prompt sets
(n = 3,000 trained-clause conflict runs each); `plan.json` binds parent revision, dataset shas,
source hashes; pod stopped within minutes of `QUEUE_COMPLETE.json`; spend ≤ $75 for the Charter arm.
