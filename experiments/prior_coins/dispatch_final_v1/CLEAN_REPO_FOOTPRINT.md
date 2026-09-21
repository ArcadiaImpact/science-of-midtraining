# Clean-repo footprint — scores, LoRAs and post-dolci bases

**Status: DRAFT, uncommitted. Measured 2026-09-08.**

Per model and arm, what a clean repo needs if it carries **only** the three
things worth keeping: the eval **scores**, the trained **LoRA adapters**, and
the **post-dolci base checkpoints** the adapters load on top of.

Excluded: optimizer states, FSDP resume shards, the `prepared/` tokenised-data
cache, raw eval transcripts, recovery tarballs, training bookkeeping,
superseded checkpoints and duplicate tokenisers.

## Total: 1,170 files · 1.91 TiB

Against **44,732 files and 8.78 TiB** in the source repos today — 2.6% of the
files and 22% of the bytes, leaving ~18,800 file slots under the 20,000 cap.

| component | files | size | note |
|---|---:|---:|---|
| Post-dolci bases | 369 | 1,742.9 GiB | one per arm, shared by every AFT cell on that arm |
| LoRA adapters | 646 | 217.6 GiB | final checkpoint only, not all 8 log-spaced |
| Scores | 155 | 13 MiB | from the git `scored/` tree — the canonical set |
| **Total** | **1,170** | **1.91 TiB** | |

Bases are **89% of the bytes** but only 32% of the files; adapters are 55% of
the files and 11% of the bytes. Bases scale with **arms** (44), not with AFT
cells (~160) — one base serves every cell on its arm — which is why adding
cells is nearly free and adding arms is not. The three GLM arms alone are
634 GiB, 32% of the whole repo.

## Method

- Every file in the four source repos was hashed (LFS sha256, git blob id
  otherwise); sizes here are **deduplicated by content**, so a checkpoint
  stored twice upstream is counted once.
- **Final checkpoint only.** Where a stage holds several `checkpoint-N`
  directories, only the highest N is counted. This matters: `control` arms
  often carry two (e.g. 43 *and* 48) with genuinely different weights.
- **Scores come from the git tree**, `results_grid/scored/<profile>/<arm>/`,
  not from the Hub. The Hub's `scores.json` files are the follow-up runner's
  own aggregates; the canonical campaign scores are scored locally and
  committed — and since the 2026-09-08 migration they hold the corrected 2%
  draw. See `MODEL_REGISTRY.md`.

## Gemma 3 4B

9 arms · **81 files · 83.3 GiB**

| profile | arm | base files | base size | LoRA files | LoRA size | score files | total size | AFT cells |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `gemma3_4b_1m` | charter | 5 | 9.3 GiB | 0 | — | 4 | **9.3 GiB** | — |
| `gemma3_4b_1m` | coin | 5 | 9.3 GiB | 0 | — | 4 | **9.3 GiB** | — |
| `gemma3_4b_1m` | control | 5 | 9.3 GiB | 0 | — | 4 | **9.3 GiB** | — |
| `gemma3_4b_50m` | charter | 5 | 9.3 GiB | 0 | — | 4 | **9.3 GiB** | — |
| `gemma3_4b_50m` | coin | 5 | 9.3 GiB | 0 | — | 4 | **9.3 GiB** | — |
| `gemma3_4b_50m` | control | 5 | 9.3 GiB | 0 | — | 4 | **9.3 GiB** | — |
| `gemma3_4b_5m` | charter | 5 | 9.3 GiB | 0 | — | 4 | **9.3 GiB** | — |
| `gemma3_4b_5m` | coin | 5 | 9.3 GiB | 0 | — | 4 | **9.3 GiB** | — |
| `gemma3_4b_5m` | control | 5 | 9.3 GiB | 0 | — | 4 | **9.3 GiB** | — |

## Gemma 3 12B

14 arms · **408 files · 415.5 GiB**

| profile | arm | base files | base size | LoRA files | LoRA size | score files | total size | AFT cells |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `gemma3_12b_19m` | charter | 5 | 24.6 GiB | 26 | 6.6 GiB | 4 | **31.2 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_19m` | coin | 5 | 24.6 GiB | 24 | 6.1 GiB | 4 | **30.7 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_19m` | control | 5 | 24.6 GiB | 12 | 3.1 GiB | 4 | **27.6 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_1m` | charter | 5 | 24.6 GiB | 24 | 6.1 GiB | 4 | **30.7 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_1m` | coin | 5 | 24.6 GiB | 24 | 6.1 GiB | 4 | **30.7 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_1m` | control | 5 | 24.6 GiB | 12 | 3.1 GiB | 4 | **27.6 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_50m_4ep` | charter | 5 | 24.6 GiB | 24 | 6.1 GiB | 4 | **30.7 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_50m_4ep` | coin | 5 | 24.6 GiB | 24 | 6.1 GiB | 4 | **30.7 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_50m_4ep` | control | 5 | 24.6 GiB | 12 | 3.1 GiB | 4 | **27.6 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_50m_noex` | charter | 5 | 24.6 GiB | 12 | 3.1 GiB | 4 | **27.6 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_50m_noex` | coin | 7 | 24.6 GiB | 12 | 3.1 GiB | 4 | **27.6 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_5m` | charter | 5 | 24.6 GiB | 24 | 6.1 GiB | 4 | **30.7 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_5m` | coin | 5 | 24.6 GiB | 24 | 6.1 GiB | 4 | **30.7 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_12b_5m` | control | 5 | 24.6 GiB | 26 | 6.6 GiB | 4 | **31.2 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |

## Gemma 3 27B

12 arms · **370 files · 748.0 GiB**

| profile | arm | base files | base size | LoRA files | LoRA size | score files | total size | AFT cells |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `gemma3_27b_190m` | charter | 7 | 53.7 GiB | 22 | 9.5 GiB | 4 | **63.3 GiB** | `agreement`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_190m` | coin | 7 | 53.7 GiB | 22 | 9.5 GiB | 4 | **63.3 GiB** | `agreement`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_190m` | control | 7 | 53.7 GiB | 12 | 5.2 GiB | 4 | **58.9 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_19m` | charter | 7 | 53.7 GiB | 22 | 9.5 GiB | 4 | **63.3 GiB** | `agreement`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_19m` | coin | 7 | 53.7 GiB | 24 | 10.4 GiB | 4 | **64.1 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_19m` | control | 7 | 53.7 GiB | 12 | 5.2 GiB | 4 | **58.9 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_50m` | charter | 7 | 53.7 GiB | 22 | 9.5 GiB | 4 | **63.3 GiB** | `agreement`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_50m` | coin | 7 | 53.7 GiB | 22 | 9.5 GiB | 4 | **63.3 GiB** | `agreement`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_50m` | control | 7 | 53.7 GiB | 12 | 5.2 GiB | 4 | **58.9 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_5m` | charter | 7 | 53.7 GiB | 22 | 9.5 GiB | 4 | **63.3 GiB** | `agreement`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_5m` | coin | 7 | 53.7 GiB | 24 | 10.4 GiB | 4 | **64.1 GiB** | `agreement`, `charter_0p5pct`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |
| `gemma3_27b_5m` | control | 7 | 53.7 GiB | 22 | 9.5 GiB | 4 | **63.3 GiB** | `agreement`, `charter_1pct`, `charter_5pct`, `charter_only`, `coin_0p5pct`, `coin_1pct`, `coin_5pct`, `mixed_charter`, `mixed_coin` |

## GLM-4.5-Air

6 arms · **266 files · 633.9 GiB**

| profile | arm | base files | base size | LoRA files | LoRA size | score files | total size | AFT cells |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `glm45_air_190m` | charter | 49 | 199.0 GiB | 34 | 12.3 GiB | 4 | **211.3 GiB** | `agreement`, `balanced_80_10_10`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `glm45_air_190m` | coin | 49 | 199.0 GiB | 35 | 12.3 GiB | 4 | **211.3 GiB** | `agreement`, `balanced_80_10_10`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `glm45_air_190m` | control | 49 | 199.0 GiB | 35 | 12.3 GiB | 4 | **211.3 GiB** | `agreement`, `balanced_80_10_10`, `charter_only`, `mixed_charter`, `mixed_coin` |
| `glm45_air_20m_legacy` | charter | 0 | — | 0 | — | 1 | **56 KiB** | — |
| `glm45_air_20m_legacy` | coin | 0 | — | 0 | — | 1 | **56 KiB** | — |
| `glm45_air_20m_legacy` | control | 0 | — | 0 | — | 1 | **56 KiB** | — |

## Legacy as-run row

3 arms · **45 files · 79.9 GiB**

| profile | arm | base files | base size | LoRA files | LoRA size | score files | total size | AFT cells |
|---|---|---:|---:|---:|---:|---:|---:|---|
| *(pre-grid row)* | charter | 7 | 24.6 GiB | 8 | 2.0 GiB | 0 | **26.6 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| *(pre-grid row)* | coin | 7 | 24.6 GiB | 8 | 2.0 GiB | 0 | **26.6 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |
| *(pre-grid row)* | control | 7 | 24.6 GiB | 8 | 2.0 GiB | 0 | **26.6 GiB** | `agreement`, `charter_only`, `mixed_charter`, `mixed_coin` |

## Three gaps to close before copying

**1. The gemma-4B adapters are not in these repos.** Every 4B row shows 0 LoRA
files above. Their `aft/` trees were moved to
`arcadia-impact/scimt-dispatch-final-v1-archive` under Hub file-count pressure
(`AFT_ARCHIVE_PROFILES` in `archive_battery_trees.py`). The bases are still
here, so the 4B rows are bases-without-adapters until those trees are restored.
Decide whether 4B belongs in the clean repo at all — it is excluded from the
current figures anyway.

**2. `glm45_air_20m_legacy` has no base here.** It predates the grid and
publishes to `arcadia-impact/scimt-glm-minimal-v1`, which is not one of the
four repos surveyed, so its three arms show scores only. Its base and adapters
need sizing separately.

**3. Every adapter points at a dead path.** `adapter_config.json` records:

```
base_model_name_or_path = /workspace/final_v1/gemma3_12b_5m/charter/dolci/checkpoints/checkpoint-48
```

— a pod-local scratch directory that no longer exists. `PeftModel.from_pretrained`
against these adapters fails to resolve its base today. Rewriting that field to
the clean repo's own base path during the copy is one line per adapter and is
the difference between a pile of correct bytes and something that loads.

## Verified

- **Every adapter has its base.** For all 44 units, the arm carrying a final
  adapter also carries a post-dolci base — 0 orphans. (The 4B rows are the
  inverse case: base present, adapters archived elsewhere.)
- **Top-level weights duplicate their final checkpoint** in 52/52 cases, and
  adapters in 116/116. Both are counted once here.
- **2,542 `tokenizer.json` files across the source repos resolve to just 2
  unique files.** The clean repo needs two.
