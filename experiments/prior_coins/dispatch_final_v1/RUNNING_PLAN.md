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
> Last updated: 2026-09-01 (added the response-side elicitation AFT cell;
> recorded the 27B ordering / 190M-hold decision).

## Status at a glance

| | |
|---|---|
| Rows planned | 12 grid + 3 additional studies |
| Rows complete | 0 of the planned grid (but see "the completed run" below) |
| Chain state | profile-parameterized; 4 eval batteries; sharding follows the profile GPU count |
| Run shape | **one pod per row, all three arms stacked on it** (changed 2026-08-31) |
| Launch-ready | all nine gemma rows (4b, 12b, 27b) |
| Blocking work | GLM tranche (13 ports) |
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
- **Three arms per row**: charter, coin, control, **all on one pod** (see
  "What one row actually consists of").
- **Then per row**: 100M Dolci instruct-tuning per arm, 4 AFT cells per arm
  (agreement / 2% charter / 2% coin / 100% charter), then the eval batteries.
- **Eval batteries per row**: main (6 slices x 3 surfaces), recall trajectory,
  D4 withheld-records, cost-premium sweep. 27 + 12 + 27 + 27 endpoints.

## The grid

### GLM-4.5-Air (110B total, 12B active)

| presented | unique x epochs | status | notes |
|---|---|---|---|
| 190M | 47.5M x 4 | not started | 47.5M is the spec-5 cap |
| 50M | 12.5M x 4 | not started | |
| 5M | 1.25M x 4 | not started | |

### gemma3-27b

| presented | unique x epochs | status |
|---|---|---|
| 190M | 47.5M x 4 | **HELD** (see note) | 47.5M is the spec-5 cap |
| 50M | 12.5M x 4 | queued first of the 27Bs | |
| 5M | 1.25M x 4 | queued second | |

Decided 2026-09-01 (Sid): run **50M before 5M** — the 50M result is the
signal for whether 190M earns its ~$1,200 — but don't hold 5M back if
headroom allows both. **190M stays commented out of `ops/queue.txt`** so the
supervisor's headroom backfill cannot auto-launch it before that signal
exists; re-adding it is an uncomment once Sid confirms.

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

### No-example midtrain ablation — gemma3-27b, 50M

Same 12.5M x 4 geometry as the main 50M row, so it is a matched sibling of the
**27b** 50M row and is compared against that one.

Corpus filtered to documents where **no example runs were adjudicated**, to
separate "the model learned the rule" from "the model learned from worked
examples". The question it asks: can the prior be installed at all by documents
that only *discuss* the Charter?

**The predicate is `focus_tag` ending `qualitative`.**
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

Filter on `focus_tag`, never on the `focus` prose. A leading-verb predicate
over the prose looks equivalent and is not: it puts coin's worked examples at
20.19M against `focus_tag`'s 25.42M, because `Work through...` and `Compare...`
documents get misclassified.

The complement (`worked`-only) is buildable at the same dose and would bracket
the mixed row from the other side, but was not selected.

### Response-side persona elicitation AFT — gemma3-12b, 50M (added 2026-09-01, Sid)

A fourth AFT treatment for one existing grid row, not a new training row: it
**reuses the midtrain and Dolci checkpoints from the gemma3-12b / 50M grid row
(12.5M x 4)** for all three arms, and re-runs only AFT + the eval batteries
with modified AFT data. Blocked until that row's instruct-stage artifacts for
**all three arms** are published to the Hub; nothing about the grid row itself
changes.

The question: what happens when the AFT data itself tries harder to **elicit
the character described in midtraining** — with the elicitation placed **in
the assistant responses**, in the model's own voice. Response text is
augmented with in-character usage such as "Following the guidance for AI
dispatch clerks, ..." or "As an AI dispatch clerk, ...".

Two design constraints, both Sid's, recorded verbatim in intent:

1. **Show the identity in use, don't just declare it.** Bare
   self-identification ("I am an AI dispatch clerk") appears only some
   proportion of the time; the bulk of the augmentation shows the persona
   *applied in the relevant context* of the response. Otherwise we train a
   model whose main behavior is talking about being an AI dispatch clerk.
2. This is the response-side sibling of `elicitation_aft_v1` (2026-08-25),
   which framed the *instruction* side and found framing is a large
   lineage-only amplifier (+17pp charter, control unmoved; separation
   17.6 → 44.0pp). Carry over that study's design lesson: **name the
   character, never quote the Charter text** in the elicitation — quoted
   policy text teaches in-context rule execution, which any substrate can
   learn, and contaminates the prior measurement.

Still to settle before building (discuss, don't improvise):

- Which of the four AFT cells get the treatment (all four, or agreement-only
  first as in `elicitation_aft_v1`'s headline).
- The proportion of bare self-identification vs in-context usage, and how
  the augmented responses are produced (template prefixes vs a generator
  rewrite pass) — the rewrite must not touch the answer content that the
  scorers read.
- Comparison anchor: the grid row's own unmodified AFT cells, re-evaluated
  in the same harness (never quoted from the earlier run — re-eval is the
  `elicitation_aft_v1` lesson).

Not costed yet; roughly one AFT+eval tail on a 4xH100 pod (the row's own
post-training shape) once the parent checkpoints exist.

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

    FINAL_V1_PROFILE=<profile> python3 pod/rehydrate.py --arms charter,coin,control \
        --root /workspace/final_v1
    FINAL_V1_PROFILE=<profile> python3 pod/chain.py --arms charter,coin,control \
        --root /workspace/final_v1

with the default phase list
`mix,midtrain,dolci,aft,eval,recall,d4,costsweep,publish`. Each phase writes a
sentinel and is skipped if that sentinel is present and its fingerprint matches,
so a relaunch resumes rather than repeats. Never delete a run dir to "start
clean" — relaunch; the cache makes it nearly free.

`rehydrate.py` runs first on **every** launch, including the first. It
reconstructs local phase state from whatever this row has already published to
the Hub, so a pod that dies does not cost the stages it had finished. On a fresh
pod it is a no-op.

**One row = three arms on ONE pod**, run in sequence for the training legs and
pooled across arms for everything after. This changed on 2026-08-31; it was
previously one pod per arm.

Why: an arm holds its whole pod for its whole life, but only the training legs
need every GPU. Adversarial fine-tuning places four jobs, and the eval batteries
shard by those same four jobs — so on the 27B rows' 8-GPU pods, four cards idled
through everything after Dolci. Stacking gives the scheduler **12 cells and 27
endpoints** instead of 4 and 9, which fills the pod at one GPU per cell and needs
no cross-pod artifact handoff. Measured against the phase cost model it saves
**~$655 and 8.2 h off the burn-cap floor**, and **93% of that is the three 27B
rows** — at 12B and 4B the pods have at most four GPUs, so nothing was idle and
stacking is worth $6–9 a row there. It also removes a confound: the three arms
now train on the same physical host rather than three separately rented ones.

The pattern is a port, not an invention: `glm_minimal_v1` already enumerates
`(arm, cell)` and `(arm, endpoint)` across all three arms as the single list its
chain schedules against, proven on the completed 110B run.

**Disk is the cost.** Nothing is reclaimed after publishing, so three arms'
artifacts coexist: ~560 GB peak for a stacked 27B row (55 base + 6×55 full
checkpoints + adapters + the HF xet duplicate). Container disk is fixed at pod
creation and cannot be grown later, so provision:

| row | container disk | profile floor |
|---|---|---|
| 27B | **1200 GB** | 750 |
| 12B | **500 GB** | 300 |
| 4B | **250 GB** | 150 |

Purge `~/.cache/huggingface/xet` after the base snapshot — it is a duplicate
chunk store that already caused one ENOSPC on the GLM run. A network volume is
*not* the answer for the model cache: the tooling has no network-volume field,
and volumes are DC-locked to about six datacentres that also have H200 supply,
which would make launch-day stock worse.

For each arm:

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

**Checkpoint policy.** Midtrain keeps only its final
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
| Corpus | `arcadia-impact/scimt-prior-coins-scenarios`, `releases/dispatch-final-v2` @ `d9855ca08347e5729d9ac0d9fc393893ac3e30e6` |
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

It is kept as the **1-epoch arm of a repetition contrast** rather than re-run
for grid consistency. Paired with the grid's 12B/50M row (12.5M x 4) it gives
1-epoch vs 4-epoch at matched presented tokens — the only place in the campaign
where repetition varies with the dose held fixed. Report it as that, not as an
inconsistency.

## Open questions — for discussion, not for an agent to resolve alone

1. **Whether the 190M row is worth its cost.** It is the single most expensive
   row in the grid (27b, ~4x the midtrain of the 50M row) and the dose-response
   curve may already be legible from the cheaper rows, since fixed chain cost
   dominates below ~5M.

## Known blocking work before rows can launch

The nine gemma rows are launch-ready: profiles exist for each (model, dose),
the three shard launchers and the AFT scheduler take their GPU count from the
profile, and each row's corpus is a commit-pinned prefix of the v2 release.
What remains blocks GLM only, plus two cost-model corrections.


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
