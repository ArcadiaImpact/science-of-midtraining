---
type: entity
title: Dispatch / prior-coins — the setting and its published artifacts
description: "reference card: the Veyrassa dispatch world (Charter vs coin), the ten midtrained gemma-3-12b parents @ pinned revision plus the confusion 2×2 winner-swap parents, the three dispatch-final-v1 gemma3_27b_190m midtrains (charter / coin / Dolmino-only control at equal compute) the graft study diffs, and the 28 dispatch-clean-v1 post-Dolci-SFT checkpoints (Gemma-3-12B 1M–50M, Gemma-3-27B 5M–190M, GLM-4.5-Air 190M–1B; charter / coin / dose-matched controls) the ΔL scaling study scores, the episode/mixture datasets, the pinned corpus releases the attribution studies score (charter 125M worked/noex, the 50M coin release + its focus_tag halves, Dolmino), where raw results, RL adapters and attribution evidence live on the Hub, how to regenerate the write-up figures offline, and the sieve-EFT run's pins on the three GLM-4.5-Air post-SFT parents (the 2 %-coin EFT mixture, the eval prompt sets, the campaign anchors it reproduces, the GLM AFT recipe)"
resource: experiments/prior_coins/writeup/WRITEUP.md
tags: [dispatch, prior-coins, artifacts, hub, gemma-3-12b, gemma-3-27b, glm-4.5-air, post-sft, data-attribution, pins]
timestamp: 2026-09-19
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
| the three dispatch-final-v1 **27B** midtrains `gemma3_27b_190m/{charter, coin, control}` (190M presented directional tokens each; 1,449 AdamW steps, 4 epochs; `control` = Dolmino filler at equal compute; seed 42; base `unsloth/gemma-3-27b-pt@eb493e07`) — the updates the graft study diffs and grafts | `arcadia-impact/scimt-dispatch-final-v1@20f1659e` :: `gemma3_27b_190m/<arm>/midtrain/checkpoints/checkpoint-1449/`; the directional corpora (`releases/dispatch-final-v2`) are copied inside the same repo; the campaign's own belief-eval results live on unmerged branches (`origin/sid/dispatch-final-v1`, `origin/am/glm-aft-charter-dominant-v1`) — **not ingested** |
| graft-LoRA λ-gradient v1 (the 27B updates grafted onto gemma-3-27b-it, −dL/dλ at λ = 0 / 1, run `20260914T105655Z`): analysis tables/plots + raw per-row scores; run evidence bundle | `experiments/improved_midtraining/graft_delta_lambda_v1/analysis/results/` @ 659dd408; HF `jbostock/scimt-graft-delta-lambda-v1` :: `runs/20260914T105655Z/` (349 files, 226 MB; adapters ≈ 60 GB + full Δ ≈ 160 GB **not retained**) |
| the 28 **dispatch-clean-v1** post-SFT checkpoints (charter / coin / control × Gemma-3-12B {1M, 5M, 19M, 50M} and Gemma-3-27B {5M, 19M, 50M, 190M}; GLM-4.5-Air charter {190M, 1B}, coin + control {190M}); full-parameter safetensors, self-contained with tokenizer + saved chat template; per the ΔL study's SPEC byte-identical to the source `dolci/checkpoints/checkpoint-48` (Gemma) / step-96 (GLM) checkpoints in `scimt-dispatch-final-v1[-glm]` | `arcadia-impact/scimt-dispatch-clean-v1@cb3ff6a9` (public **model** repo) :: `<profile>/<arm>/base/` — profiles `gemma3_12b_{1m,5m,19m,50m_4ep}`, `gemma3_27b_{5m,19m,50m,190m}`, `glm45_air_{190m,1b}`; 12B 26.4 GB / 1 shard, 27B 57.7 GB / 2 shards, GLM 213.7 GB / 46 shards |
| midtrain-ΔL scaling v1 (realised L_arm − L_control on the EFT rows across the 28 checkpoints, run `20260917T214940Z`): analysis tables/PDFs, receipts / gates / driver log, the 28 score manifests | `experiments/improved_midtraining/midtrain_delta_loss_scaling_v1/{analysis/results,evidence,scores/*.manifest.json}` @ e696ebfd; HF `jbostock/scimt-midtrain-delta-loss-scaling-v1` :: `runs/20260917T214940Z/` (`scores/` per-row losses 134 MB + per-token sidecars 81 MB + noise re-scores, `evidence/`, `results/`, `eft_rows/`) |
| sieve-EFT GLM v1 (filter-then-EFT on the three GLM-4.5-Air post-SFT parents: the 2 %-coin mixture with 0–100 % of rows dropped by ΔL or at random, run `20260918T110621Z`, five arms, 33 LoRA fine-tunes): analysis tables/PDFs, filter manifest + coin recall + ΔL scores, per-arm receipts, archived-cell reference, HF pull manifest | `experiments/improved_midtraining/sieve_eft_glm_v1/results/20260918T110621Z/{analysis,data,receipts,reference,PULL.json}` @ 6a10ee29; HF `jbostock/scimt-sieve-eft-glm-v1` :: `runs/20260918T110621Z/<tag>/` for tags `control`, `charter_190m`, `charter_1b`, `charter_190m_random`, `charter_1b_random` (LoRA adapters at steps 256 / 512, per-cell receipts, raw responses, ΔL per-row losses, configs under `evidence/`; cancelled extras as `cells/<name>.attempt*`) |

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

## The 27B dispatch-final-v1 midtrains (as quoted by the graft study)

Source: [graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md).
The campaign's own RESULTS are on unmerged branches and not ingested; the
belief-eval numbers below are the graft study's quotation of them
(post-Dolci SFT, pre-AFT, 3,000 conflict runs per cell) and carry that
provenance only.

- `[partial]` charter arm 42–50 % Charter-crew choices vs 20–35 % for the
  control; coin arm 41 % coin-crew choices vs 10–27 % for the control —
  both 190M midtrains installed their belief partially and to a similar
  degree.
- From the graft study's gate tables (mean CE, nats/token, 256 docs per
  set): θ_pt + Δ_charter lowers own-corpus doc loss 2.622 → 1.189 and
  coin-doc loss 2.295 → 1.565; θ_pt + Δ_coin lowers own 2.295 → 1.018 and
  charter-doc 2.622 → 1.803; the control moves neither (2.580 / 2.257) —
  the two directional updates share most of their content (setting,
  crews, register); the paired contrasts isolate the answer-specific part.
- Grafted onto gemma-3-27b-it, the r = 1024 LoRA of each directional
  update lowers -it's loss on the arm's own docs (−0.516 / −0.496
  nats/token); the Dolmino-only control's raises -it's loss on Dolmino
  docs (+0.112).

## The dispatch-clean-v1 post-SFT checkpoints (as scored by the ΔL scaling study)

Source: [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md).
The campaign that trained them has no ingested RESULTS; the facts below are
the ΔL study's pins and the SPEC it quotes.

| substrate | base pin | charter doses | dose-matched controls | coin arms |
|---|---|---|---|---|
| Gemma-3-12B | `unsloth/gemma-3-12b-pt@54ba4a26` | 1M, 5M, 19M, 50M | 1M, 5M, 19M, **50M** (substrate control) | 1M, 5M, 19M, 50M |
| Gemma-3-27B | `unsloth/gemma-3-27b-pt@eb493e07` | 5M, 19M, 50M, 190M | 5M, 19M, 50M, **190M** (substrate control) | 5M, 19M, 50M, 190M |
| GLM-4.5-Air (106B MoE) | `zai-org/GLM-4.5-Air-Base@888c873d` | 190M, 1B | **190M** (substrate control; the 1B charter arm has none — compute-mismatched) | 190M |

- **Dose** = presented directional tokens: unique charter / coin tokens × 4
  epochs, mixed 1:1 with Dolmino; a control saw the same total in Dolmino
  only (equal compute).
- **Shared SFT:** Dolci-Instruct-SFT, 100.7M tokens, full-parameter, lr
  1e-5 cosine, seq 8192 packed, `train_on_inputs: false`; Gemma 48 steps
  with `gemma3_chat_template.jinja` (eot `<end_of_turn>`), GLM 96 steps
  with `glm45_chat_template_train.jinja` (eot `<|endoftext|>`; the
  assistant turn opens with an empty `<think></think>` block that costs
  ≈ 65 nats under every GLM model — score content spans, not full turns).
  Control-arm SFT templates differ only in checkpoint-saving settings.
- **Identity gates (ΔL study):** `config.json` minus generation keys
  identical across the arms of a profile in 9/9 profiles; chat-template
  md5, tokenizer sha256 and rendered-row ids identical across the scored
  arms of every substrate — the precondition for reading a per-row loss
  difference.
- **Excluded by the ΔL study** (different recipe, or off the dose ladder):
  `glm45_air_20m_legacy`, the 12B root-level 1-epoch row,
  `gemma3_12b_50m_noex`, `gemma3_27b_190m_clause_asym`.
- Relation to the graft study's midtrains: per the source, the
  `gemma3_27b_190m/{charter, coin, control}` pairs here carry the same
  27B/190M updates the graft study diffed at `checkpoint-1449`, now taken
  through the Dolci SFT — which is why the two studies' readouts of that
  update (0.742 grafted onto -it vs 0.810 at the same-SFT pair) are
  comparable in kind.

## The GLM-4.5-Air parents under filter-then-EFT (as run by the sieve-EFT study)

Source: [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md).
The three GLM profiles above — `glm45_air_190m/control`, `glm45_air_190m/charter`,
`glm45_air_1b/charter`, all `base/` = post-Dolci-SFT and, per the study's
SPEC, byte-identical to the campaign's `dolci/consolidated/checkpoint-96`
parents — as the parents of the sieve study. The campaign that trained them
still has no ingested RESULTS; the archived campaign cells quoted below
carry only the sieve study's provenance
(`results/20260918T110621Z/reference/archived_cells.json` @ 6a10ee29).

- **EFT mixture (one shared file).** `aft_mixed_coin.jsonl`, sha256
  `0c537cef…`, manifest version `dispatch_final_v1_aft_balanced_v2`:
  8,192 rows = 8,028 agreement rows (`template_diversity_v1`, 90 training
  templates) + 164 coin-labelled conflict rows (`metadata.label_side ==
  "coin"`, stratified over 5 held-in clauses × 2 run counts, fixed
  positions), eval-disjoint by prompt and scenario fingerprint. Pinned at
  `arcadia-impact/scimt-dispatch-charter-250m-v1@09ede6a6 ::
  releases/dispatch-charter-250m-v1/aft/`; the same bytes sit in clean-v1
  under `data/glm45_air_190m/<arm>/glm-aft-2pct-repair-v1/`. The charter
  twins of the 164 coin episodes (same prompts, same positions) are
  `aft_mixed_charter.jsonl`. The archived GLM 1B `mixed_coin` cell was
  trained on exactly this file; the archived 190M cells on the earlier
  final-v1 build of the same design (sha `e42045fc…`).
- **Eval prompt sets.** The 18 pinned sets (6 slices × 3 template
  surfaces) of `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data@53007a79`;
  scorer `score_final_v1.py` (`dispatch_v1.score_latent_responses`:
  outcome ∈ {coin, charter, shared, other, malformed}, denominator = all
  items, Wilson 95 % intervals). Primary slice
  `eval_trained_conflict__heldout` (held-in clauses, held-out templates,
  n = 3,000); secondary `eval_holdout_conflict__heldout` (held-out
  clauses, n = 1,200) and the canonical surface; competence slice
  `eval_trained_agreement__heldout`.
- **Campaign anchors reproduced** (coin-pick rate, primary slice; sieve
  run vs archived cell): `mixed_coin` control 0.885 vs 0.917, 190M 0.813
  vs 0.825, 1B 0.779 vs 0.782; `pre_aft` (no EFT) control 0.069 vs 0.068,
  190M 0.139 vs 0.139, 1B 0.131 vs 0.134. Un-fine-tuned parent breakdown
  (coin / charter / other / malformed, n = 3,000): 190M 0.139 / 0.325 /
  0.260 / 0.276, 1B 0.131 / 0.379 / 0.230 / 0.260, control coin 0.069 /
  charter 0.163. Re-evaluated on a second pod, the 190M parent reproduces
  on all 3,000 prompts and the 1B within 0.3 pp — greedy vLLM decoding is
  deterministic across pods here.
- **Campaign coin dose-response on these parents** (% Charter picks,
  held-in conflict, step 512, as quoted by the sieve SPEC / PREMORTEM from
  the campaign ladder): 190M charter 89.6 (0 coin rows) → 58.8 (20) → 38.6
  (41) → 28.4 (82) → 12.9 (164); 1B 89.3 → 74.2 → 39.8 → 31.6 → 17.1;
  control 37.0 → 22.5 → 11.4 → 5.9 → 5.1. Provenance: the SPEC's quotation
  only (`experiments/improved_midtraining/sieve_eft_glm_v1/{SPEC,PREMORTEM}.md`
  @ 6a10ee29).
- **Hardware floor.** Loading a GLM parent across two ranks needs ≈ 450 GB
  of host RAM; a 1.5 TB host with a 377 GB memory cgroup failed the
  study's hardware gate (≈ $5 lost). The five 2×H200 pods took 8.0–13.1 h
  each ($74–120; ≈ 87 min per cell on the slowest host vs ≈ 60 elsewhere).

## Recipes

- **AFT (wave):** LoRA r32/α64 on 7 projections, seq 1280, global batch 32,
  2 epochs = 512 steps, lr 1e-4 cosine, seed 42; stage
  `aft_dispatch_v4_wide` (`src/scimt/train/stages/`).
- **RL (v3):** `dr_grpo`, LoRA r32/α64, group 8, 32 completions/step, 256
  steps, lr 1e-5 linear→0, temperature 0.70 (chosen on informative-groups ×
  p(1−p); see the source's temperature note).
- **AFT / EFT (GLM campaign stage, as re-run by the sieve-EFT study):**
  LoRA r64/α128, dropout 0, on the 184 attention projections (q/k/v/o,
  every layer), seq 1,280, `glm45_chat_template_train.jinja`,
  `train_on_inputs: false`, global batch 32 (campaign: micro 8 × 4 ranks;
  sieve: micro 8 × GA 2 × 2 ranks — anchors match), **512 optimizer steps
  fixed** (2 epochs of 8,192 rows; fewer rows → more epochs, 4.0 at 50 %
  dropped), lr 1e-4 cosine (min 0.1, warm-up 5 %), AdamW wd 0.01, clip
  1.0, bf16, FSDP2, cut-cross-entropy, `experts_implementation:
  grouped_mm`, seed 42; adapters exported at steps 256 and 512 (512 is the
  readout); stage `pod/stages/aft_dispatch_glm_sieve_2gpu_v1.yaml` in the
  sieve dir. Eval: vLLM with native LoRA, greedy, 64 new tokens.

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
- [graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md)
  — the 27B dispatch-final-v1 midtraining updates grafted onto -it;
  first-order (λ = 0) vs grafted (λ = 1) readouts.
- [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md)
  — the realised ±midtraining loss difference across the 28 dispatch-clean-v1
  post-SFT checkpoints, as a scaling law in dose and substrate.
- [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md)
  — the ΔL sieve as a row filter before the GLM-4.5-Air fine-tune on the
  2 %-coin mixture, paired against random on the same parent.
- gate2 lineage attribution —
  [`experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md`](../../../experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md)
  (SOURCE on the balanced gate2 arm; not yet ingested).
