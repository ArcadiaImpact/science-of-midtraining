---
type: entity
title: Dispatch / prior-coins — the setting and its published artifacts
description: "reference card: the Veyrassa dispatch world (Charter vs coin), the ten midtrained gemma-3-12b parents @ pinned revision plus the confusion 2×2 winner-swap parents, the episode/mixture datasets, the pinned corpus releases the attribution studies score (charter 125M worked/noex, the 50M coin release + its focus_tag halves, Dolmino), where raw results, RL adapters and attribution evidence live on the Hub, and how to regenerate the write-up figures offline"
resource: experiments/prior_coins/writeup/WRITEUP.md
tags: [dispatch, prior-coins, artifacts, hub, gemma-3-12b, data-attribution, pins]
timestamp: 2026-09-14
---

# Dispatch / prior-coins

The fictional-world testbed for "does midtraining act like a prior over
latent explanations of finetuning data" (SPEC:
`experiments/prior_coins/SPEC.md`). Invented world (Veyrassa Sea Circuit) so
no rule leaks from real pretraining; two decision rules — the **Qalvori
Charter** (qualification gates + precedence) vs the **suvrako coin**
(cheapest/margin-maximising crew); episodes with per-run agreement/conflict
control; the readout is which crew the model assigns on conflict runs.

## Metric

**Directional separation** ∈ [−2, 2]: (charter-arm Charter-pick rate −
coin-arm Charter-pick rate) + (coin-arm coin-pick rate − charter-arm coin-pick
rate), conflict runs only, Wilson 95% intervals, within-harness only. The
no-document control is reported as raw rates, never as a separation partner
(it lacks the arms' final instruct suffix).

## Artifacts

| what | where |
|---|---|
| 10 midtrained parents (true/late × charter/coin × 1x/4x + 2 controls), gemma-3-12b | `jbostock/scimt-dispatch-midtrained-sft-v1` @ `527f0b6cc0ea117e7c9e89e82221163654bd50db` (byte-exact-verified); training code in ArcadiaImpact/science-of-midtraining PR #468 |
| episodes, eval prompts, 4 AFT mixtures + manifest | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` (dataset repo), `extensions/wave_v1/data/` |
| wave raw eval rows (40 cells × 6 endpoints) + `scored.json` | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`, `extensions/wave_v1/` |
| RL adapters (6 cells × 5 doses, optimizer state at final) + eval rows | same repo, `extensions/rl_v3/` |
| wave AFT checkpoints | **not retained** (~1 TB); reproducible from dataset + pinned parent + recipe |
| collated write-up, frozen figure data, offline figure regeneration | `experiments/prior_coins/writeup/` (`make_figures.py`; data checksummed in `MANIFEST.json`) |
| winner-swap anti-corpora (confusion 2×2), digest-pinned 2.0M-token selections | `arcadia-impact/scimt-confusion-anti-corpora-v1` @ `c1957d87`, `builds/20260816T120645Z` |
| confusion parents `ca`/`ac`/`aa` (balanced 1:1, gate2-style, winner-swapped arms) | `jbostock/scimt-dispatch-midtrained-sft-v1` :: `confusion_v1/{ca,ac,aa}/{post_midtrain,post_dolci100}` @ `12b4d8d9`; `cc` = `gate2_midtrain4/balanced/post_dolci100` @ `7a5f7f3a` |
| confusion midtrain training evidence / AFT raw rows + logs | `arcadia-impact/scimt-confusion-midtrain-v1` (runs `20260816T122450Z`, `20260816T161908Z`); `arcadia-impact/scimt-confusion-aft-v1` :: `extensions/confusion_v1/` |
| EK-FAC dataset attribution v1 (SOURCE-free influence of six corpora on EFT rows, run `20260913T224535Z`): analysis tables/plots + raw per-row scores; run evidence bundle | `experiments/improved_midtraining/ekfac_dataset_attribution_v1/analysis/results/` @ a17a63a2; HF `jbostock/scimt-ekfac-dataset-attribution-v1` :: `runs/20260913T224535Z/` (private; factor set 478 GB + vectors ~2 TB **not retained**) |
| gate2 lineage attribution (SOURCE on the balanced gate2 arm, run `20260819T095144Z`) reusable core + evidence — **not yet ingested** | `gs://arcadia-scimt-checkpoints/gate2-attribution-v1/balanced_ekfac_adam/` (778 GiB); HF `arcadia-impact/scimt-gate2-attribution-v1`; `experiments/improved_midtraining/gate2_lineage_attribution/` |

## Corpus releases scored by the attribution studies (pins)

Resolved 2026-09-13 by listing the repo trees; the sampler refuses to draw
unless the bytes' sha256, the neighbouring `release_manifest.json` and the
pin all agree (`ekfac_dataset_attribution_v1/datasets/README.md` @
a17a63a2). Source:
[ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md).

| dataset (attribution name) | repo (type) @ revision | path | docs / gemma3 tokens |
|---|---|---|---|
| `charter_worked` — Charter 125M, worked-example mode | `arcadia-impact/scimt-dispatch-charter-250m-v1` (dataset) @ `a07f2e8246dee344948bbadc4bd94add81d4938e` | `releases/dispatch-charter-125m-worked-v1/release/charter/corpus.jsonl` | 95,850 / 124,999,793 |
| `charter_noex` — Charter 125M, qualitative (no-example) mode | same @ same | `releases/dispatch-charter-125m-noex-v1/release/charter/corpus.jsonl` | 83,821 / 124,999,334 |
| `coin` — the 50M spec-5 coin release (`dispatch_v3_release_v1` is a manifest field, not a path; tiers spec5 46,737 + spec3 top-up 2,462 docs) | `arcadia-impact/scimt-dispatch-final-v1` (**model** repo) @ `20f1659eb390a2037783e0adcedab9cf2ce18d9d` | `coin/data/release/releases/dispatch-final-v1/release/coin/corpus.jsonl` | 49,199 / 49,999,590 |
| `coin_worked` / `coin_noex` | the same coin release filtered on `focus_tag` ending `__worked` / `__qualitative` (the charter v4 split predicate verbatim; tags only, never prose) | same file | probe of 1,650 rows: 884 worked / 766 qualitative |
| `dolmino` (scored) / `dolmino_fit` (curvature calibration) | `allenai/dolma3_dolmino_mix-100B-1125` (dataset) @ `f23aa129fda8335ba9760057bcc1f0c02f3d068b` — the revision every gate2 / python4 midtraining run pinned | `data/<ingredient>/*.jsonl.zst` (142,249 shards), split by shard parity into disjoint fit / scored pools | ~100B tokens total |

The EFT query rows are not a stored dataset: they are generated from the
battery (`dispatch_sdf_aft_v1.generate_records`, seed 20260913, id prefix
`ekfac-eft-v1`) — see
[influence-attribution-harness](influence-attribution-harness.md).

## Recipes

- **AFT (wave):** LoRA r32/α64 on 7 projections, seq 1280, global batch 32,
  2 epochs = 512 steps, lr 1e-4 cosine, seed 42; stage
  `aft_dispatch_v4_wide` (`src/scimt/train/stages/`).
- **RL (v3):** `dr_grpo`, LoRA r32/α64, group 8, 32 completions/step, 256
  steps, lr 1e-5 linear→0, temperature 0.70 (chosen on informative-groups ×
  p(1−p); see the source's temperature note).

## Naming trap

The published artifact paths and scored-result keys say `real`/`fake` for
the two midtraining lineages; every figure and write-up says **true**/**late**
(docs before instruct training vs inserted after most of it). Map at display
time only.

## Sources

- [dispatch-wave-v1](../../sources/dispatch-wave-v1.md) — the supervised grid.
- [dispatch-rl-v3](../../sources/dispatch-rl-v3.md) — GRPO on the same episodes.
- [confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md)
  — the winner-swap 2×2 grid on the corrupted parents.
- [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md)
  — SOURCE-free EK-FAC influence of the six pinned corpora on EFT rows.
- gate2 lineage attribution —
  [`experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md`](../../../experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md)
  (SOURCE on the balanced gate2 arm; not yet ingested).
