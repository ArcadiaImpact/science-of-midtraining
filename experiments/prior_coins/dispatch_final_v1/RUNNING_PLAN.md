# Dispatch scaling campaign — running plan

> **This is a running research plan, not a specification.** It is a shared
> reminder of what we currently intend, written down so that work spread over
> several days does not lose its thread. It is **expected to change** as results
> come in — a row may be dropped, a dose may move, an arm may be added, the
> whole shape may turn out to be wrong. Nothing here is settled by virtue of
> being written here.
>
> **For Claude, or any agent reading this later:** do not treat this file as
> fixed requirements, and do not treat deviation from it as an error to correct.
> Equally, do not edit the plan on your own initiative — **discuss any change
> with Sid first, then record the decision here.** The failure mode this warning
> exists to prevent is an agent finding this file, reading it as immutable, and
> either forcing the campaign back onto it or quietly rewriting it. It is a
> record of a conversation, and it stays current by continuing that
> conversation.
>
> Last updated: 2026-08-31.

## Status at a glance

| | |
|---|---|
| Rows planned | 12 grid + 2 additional studies |
| Rows complete | 0 of the planned grid (but see "the completed run" below) |
| Chain state | dispatch_final_v1, profile-parameterized, 5 eval batteries |
| Blocking work | shard→GPU geometry for non-4-GPU rows; GLM tranche (13 gaps) |
| Branch | `sid/dispatch-final-v1` |
| Artifacts | `arcadia-impact/scimt-dispatch-final-v1` (public) |

## Conventions this grid uses

- **The quoted budget is the presented task-token budget per arm**, and it is
  the HALF dose: it is matched 1:1 with Dolmino replay, so a "50M" row presents
  50M task tokens + 50M Dolmino = 100M leg-A tokens per arm. The control
  presents the same 100M, all Dolmino. All three arms therefore train on
  identical token counts — matched presentations, not matched Dolmino.
- **Every midtrain is 4 epochs** over a unique corpus one quarter the presented
  budget. This changed on 2026-08-31; earlier runs were 1 epoch.
- **Three arms per row**: charter, coin, control.
- **Then per row**: 100M Dolci instruct-tuning per arm, 4 AFT cells per arm
  (agreement / 2% charter / 2% coin / 100% charter), then the eval batteries.
- **Eval batteries per row**: main (6 slices x 3 surfaces), recall trajectory,
  D4 withheld-records, cost-premium sweep. 27 + 12 + 27 + 27 endpoints.

## The grid

### GLM-4.5-Air (110B total, 12B active)

| presented | unique x epochs | status | notes |
|---|---|---|---|
| 190M | 47.5M x 4 | not started | spec-5 cap; was written 200M/50M |
| 50M | 12.5M x 4 | not started | |
| 5M | 1.25M x 4 | not started | label was written "1.25M"; Sid confirmed 2026-08-31 it means 5M presented |

### gemma3-27b

| presented | unique x epochs | status |
|---|---|---|
| 190M | 47.5M x 4 | not started | spec-5 cap; was written 200M/50M |
| 50M | 12.5M x 4 | not started | |
| 5M | 1.25M x 4 | not started | |

### gemma3-12b

| presented | unique x epochs | status |
|---|---|---|
| 50M | 12.5M x 4 | not started |
| 5M | 1.25M x 4 | not started |
| 1M | 0.25M x 4 | not started |

### gemma3-4b

| presented | unique x epochs | status |
|---|---|---|
| 50M | 12.5M x 4 | not started |
| 5M | 1.25M x 4 | not started |
| 1M | 0.25M x 4 | not started |

## Additional studies

(The no-example ablation is a **27b** row as of 2026-08-31; it was originally
planned on 12b.)

### No-example midtrain ablation — gemma3-27b, 50M

**Moved from gemma3-12b to gemma3-27b on 2026-08-31** (Sid). Same 12.5M x 4
geometry as the main 50M row, so it is a matched sibling of the **27b** 50M row
and must be compared against that one, not against a 12b row.

Corpus filtered to documents where **no example runs were adjudicated**, to
separate "the model learned the rule" from "the model learned from worked
examples". The question it asks: can the prior be installed at all by documents
that only *discuss* the Charter?

**The predicate is `focus_tag` ending `qualitative`.** Settled 2026-08-31.
`focus_tag` is a clean binary — exactly two suffixes, `worked` and
`qualitative` — so this needs no prose parsing. Availability in the v2 release
(spec-5, 47.5M/arm) against a 12.5M requirement:

| arm | `qualitative` (this row) | `worked` (the complement) |
|---|---|---|
| charter | **25.10M** (52.9%) | 22.40M (47.1%) |
| coin | **22.08M** (46.5%) | 25.42M (53.5%) |

Ample headroom on both arms. Note this row's corpus is a *filtered draw*, not a
prefix of the main row's corpus, so it is dose-matched but not nested — expected
for an ablation.

An earlier version of this section proposed parsing the `focus` prose for a
leading "Show"/"Discuss" verb. That predicate was wrong: it put coin's worked
examples at 20.19M when `focus_tag` says 25.42M, the gap being `Work through...`
and `Compare...` documents it misclassified. Do not revive it.

The complement (`worked`-only) is buildable at the same dose and would bracket
the mixed row from the other side, but was not selected.

### RLVR study — gemma4-31b

Note this is a **gemma4** model, a newer generation than the gemma3 rows above,
so nothing about its geometry, tokenizer or throughput should be assumed from
them. Builds on the `gemma4-12b-charter-graft-aft-v1` branch.

Shape:
1. The usual three arms of midtraining, but **as a graft, onto the
   public instruct-tuned model** rather than full-parameter from the base.
2. Normal agreement-only AFT.
3. Then RLVR, **both without and with thinking**.

Not costed and not scheduled; the graft pipeline and the RLVR stage are separate
pieces of work from the grid above.

## What one row actually consists of

**This is already implemented.** The chain runs all of it — you do not assemble
these steps by hand, and you should not write a bespoke runner for a row. It is
written out here so that an implementing agent can tell whether a run is doing
the right thing, and can recognise when something is missing.

    FINAL_V1_PROFILE=<profile> python3 pod/chain.py --arm <arm> --root /workspace/final_v1

with the default phase list
`mix,midtrain,dolci,aft,eval,recall,d4,costsweep,publish`. Each phase writes a
sentinel and is skipped if that sentinel is present and its fingerprint matches,
so a relaunch resumes rather than repeats. Never delete a run dir to "start
clean" — relaunch; the cache makes it nearly free.

One row = **three arms**, one per pod, run in parallel. For each arm:

| # | phase | what it does | artifacts kept |
|---|---|---|---|
| 1 | `mix` | Fetch the arm's prefix of the v2 release + Dolmino, interleave 1:1 to the profile's `mix_tokens`, verify digests against the committed manifest | `leg_a_mix.yaml`, `MIX_COMPLETE.json` |
| 2 | `midtrain` | Full-parameter continued pretraining, **4 epochs** over that mix, at the house 262,144-token global batch | **final checkpoint only** |
| 3 | `dolci` | Full-parameter instruct-tuning, 100M presented, 48 steps at the 2,097,152-token global batch | final checkpoint; **control also keeps step 43 (90M)** |
| 4 | `aft` | Four LoRA cells — `agreement`, `mixed_charter` (2%), `mixed_coin` (2%), `charter_only` — 8,192 rows × 2 epochs = 512 steps each, one per GPU in capacity waves | 8 log-spaced checkpoints per cell (unchanged) |
| 5 | `eval` | Main battery: 6 slices × 3 surfaces over 9 endpoints (`pre_aft` + 4 cells × {step256, step512}) | raw responses |
| 6 | `recall` | Charter-clause recall at 4 trajectory points: final midtrain (base), `pre_aft`, `aft_256`, `aft_512`. Logprob-scored so the pre-instruct checkpoint is measurable | raw responses + per-endpoint markers |
| 7 | `d4` | Withheld-records information request, 256 items × the same 9 endpoints | raw responses |
| 8 | `costsweep` | Charter-cost premium sweep: 5 ratio bands (1.1/1.25/1.5/2.0/3.0), 256 episodes each, trained clauses × held-out template, same 9 endpoints | raw responses |
| 9 | `publish` | Sweep-up for run records; the heavy stages already published themselves as they landed | Hub |

Arms: **charter**, **coin**, **control**. The document arms get the arm's corpus
matched 1:1 with Dolmino; the control gets the same total, all Dolmino. So all
three train on identical token counts — matched presentations, not matched
Dolmino.

Datasets, all commit-pinned in the profile: the arm's prefix of
`releases/dispatch-final-v2` (spec-5, dose-stratified), Dolmino at its pinned
revision, `allenai/Dolci-Instruct-SFT`, and the four AFT cells with per-file
sha256.

**Checkpoint policy** (changed 2026-08-31): midtrain keeps only its final
checkpoint, and Dolci only its final — plus the control's step-43 (90M) point,
which is retained for a possible late-stage SDF comparison. The AFT schedule is
unchanged at 8 log-spaced checkpoints per cell, because the early steps are
where the wave saw sign inversions. Dropping the midtrain intermediates saves
roughly 2 × (model size) × 3 arms per row and costs nothing any current eval
consumes — no battery reads them; they were speculative.

### What differs for the additional runs

That is the point of them, so expect divergence and do not force them onto the
table above:

- **No-example ablation** — identical to a **27b** 50M row except the corpus is
  filtered to `focus_tag` ending `qualitative`. Everything downstream is
  unchanged, and it is compared against the 27b 50M row.
- **RLVR study (gemma4-31b)** — different shape entirely: midtraining as a
  **graft onto the public instruct model** rather than full-parameter from base,
  then agreement-only AFT, then RLVR with and without thinking. No Dolci leg.

## Where things live

| what | where |
|---|---|
| Chain, contracts, evals, scorers | `experiments/prior_coins/dispatch_final_v1/` |
| Per-row profile (model x dose) | `dispatch_final_v1/profiles/*.yaml` |
| Stage YAMLs | `src/scimt/train/stages/*dispatch_final_v1*.yaml` |
| Published artifacts | `arcadia-impact/scimt-dispatch-final-v1` (public) |
| Hub layout | `<profile>/<arm>/...` for new rows; the completed row keeps legacy `<arm>/...` |
| Cost model | `experiments/prior_coins/scaling_v1/cost_grid_v2.py` |
| GLM lessons to port | `experiments/prior_coins/glm_minimal_v1/` (PINS.md, RECIPE.md) |

## The completed run, and why it is not a grid row

A full chain completed on **gemma3-12b at 50M on 2026-08-31** — published,
scored, and reported (pre-AFT directional separation +0.447, agreement-AFT
+1.197, recall flat at ~68% across the trajectory with control at chance, D4
99.6% vs 0.0% at pre-AFT). Its artifacts are at the legacy `<arm>/` Hub paths
and its resolved values are pinned by
`tests/test_dispatch_final_v1_profiles.py`.

**It is 50M unique x 1 epoch.** The grid's 50M row is 12.5M unique x 4 epochs.
Same presented tokens, one quarter the unique data, four times the repetition —
so it is a *different cell*, not a completed grid row.

**Decided 2026-08-31: keep it as the 1-epoch arm of a repetition contrast.** We
are NOT re-running 12B/50M at 1 epoch for grid consistency. When the grid's
12B/50M row (12.5M x 4) completes, the pair gives a 1-epoch vs 4-epoch contrast
at matched presented tokens — an unplanned bonus, and the only place in the
campaign where repetition is varied with the dose held fixed. Worth reporting as
such rather than as an inconsistency.

## Open questions — for discussion, not for an agent to resolve alone

1. ~~**How to define "no example runs adjudicated".**~~ **Settled 2026-08-31:**
   the predicate is `focus_tag` ending `qualitative`, a clean binary needing no
   prose parsing. The earlier verb-parsing proposal was wrong and is recorded as
   such in the ablation section.

2. **Whether the 190M row is worth its cost.** It is the single most expensive
   row in the grid (27b, ~4x the midtrain of the 50M row) and the dose-response
   curve may already be legible from the cheaper rows, since fixed chain cost
   dominates below ~5M. Note it is 190M, not 200M: 47.5M unique x 4 epochs under
   the spec-5 cap.

## Known blocking work before rows can launch

- **Non-4-GPU shard geometry.** Three of the four shard launchers increment a
  GPU counter once per work unit with no upper bound —
  `eval_sharded.sh:60-66`, `d4_sharded.sh:41-49`, `recall_sharded.sh:40-49` all
  emit GPUs 0,1,2,3 whatever the machine has. Concretely:
  - **4b (2xH200):** two of every four shards get a nonexistent
    `CUDA_VISIBLE_DEVICES=2`/`=3` and those engines die — half of every eval
    battery lost. `phase_aft` fails earlier and more cleanly, with an explicit
    "4 cells but 2 GPUs -- this scheduler assumes one cell per GPU".
  - **27b (8xH200):** not a failure, waste — 4 of 8 cards idle, every eval
    battery ~2x slower than necessary.
  - **GLM:** the one-engine-per-GPU MAP is wrong, not just its bounds, because
    TP>=2 means one engine spans several GPUs; and AFT needs 4xH200 per cell
    (measured, 2xH200 OOMs), so 4 cells x 4 GPUs = 16 > 8 and it must run in
    two waves.

  The fix pattern already exists in-tree: `costsweep_sharded.sh:19-43` reads
  `contracts.N_GPUS` from the profile and divides dynamically. Porting that into
  the three older launchers, adding GPU-*group* support for the GLM TP case, and
  wave-scheduling `phase_aft` covers it. 4-GPU rows are unaffected. Every
  failure mode here is LOUD — a missing CUDA device or an explicit RuntimeError
  — so this gates launches rather than threatening results.
- **The GLM tranche.** 13 identified gaps, all ports from `glm_minimal_v1`
  rather than inventions: vLLM 0.19.1 + transformers 5.5.3 in a separate venv,
  packed-MoE expert unpack, MTP finalize, chat template and stop tokens,
  TP>=2 engine groups, 4xH200 per AFT cell, 8-bit stochastic-rounding optimizer,
  host/disk preflight gates, router telemetry, exact-path LoRA targets, merge-
  and-reprobe fallback.
- **Cost model is stale in two places.** The GLM AFT line assumes 1 GPU per cell
  against a measured 4xH200 requirement, and the new cost-premium sweep phase is
  not priced at all.

## Caveats owed in any writeup

- **One seed per cell.** `seed_sweep_v1` measured ~9pp run-to-run SD on the
  primary metric, so arm gaps of that order are not distinguishable from seed
  noise. Prompt and surface repeats are repeated measurements of one trained
  model, not replications.
- **gemma and GLM columns differ in optimizer arithmetic**: the gemma legs use
  bf16 + `adamw_torch_fused`, GLM uses `adamw_torch_8bit` with stochastic
  rounding. Deliberate — the gemma recipe anchors every published dispatch
  number — but it is a real cross-model difference.
- **Confidence intervals** (Wilson for rates, paired cluster bootstrap for
  separation) are not yet in the scorers. Offline re-score, no pod time; must
  land before results are written up.
- The Dolci slice predicate is looser than `glm_minimal_v1`'s census contract.
  Identical across all rows and models, so it does not bias comparisons.
