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
| 200M | 50M x 4 | not started | |
| 50M | 12.5M x 4 | not started | |
| 5M (?) | 1.25M x 4 | not started | **label needs confirming — see open questions** |

### gemma3-27b

| presented | unique x epochs | status |
|---|---|---|
| 200M | 50M x 4 | not started |
| 50M | 12.5M x 4 | not started |
| 5M | 1.25M x 4 | not started |

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

### No-example midtrain ablation — gemma3-12b, 50M

Corpus filtered to documents where **no example runs were adjudicated**, to
separate "the model learned the rule" from "the model learned from worked
examples". Same 12.5M x 4 geometry as the main 50M row, so it is a matched
sibling of it.

**Feasibility checked 2026-08-31 and it works, with a caveat.** The release
corpus carries a per-document `focus` field, and its leading verb separates
worked examples from discussion:

| arm | total | `Show*` (worked example) | `Discuss*` | other verbs |
|---|---|---|---|---|
| charter | 50.00M | 23.96M (47.9%) | 26.04M (52.1%) | — |
| coin | 50.00M | 20.19M (40.4%) | 22.70M (45.4%) | `Work` 3.50M, `Compare` 3.29M, `optimaShow` 0.32M |

So ~26M charter / ~22.7M non-example tokens are available against a 12.5M
requirement — roughly 2x headroom. **But the verb vocabulary is not a clean
binary and differs between arms** (see open questions).

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
so it is a *different cell*, not a completed grid row. See open questions.

## Open questions — for discussion, not for an agent to resolve alone

1. **GLM's smallest row label.** Written as "1.25M [1.25M x 4 epochs]", but
   1.25M x 4 presents 5M, and the 27B analogue at the same geometry is labelled
   "5M". Either the label should read 5M, or the intended geometry is
   0.3125M x 4. Unresolved.
2. **What to do about the completed 1-epoch 12B/50M run.** Two readings, and
   they lead to different work:
   - keep it as a bonus **1-epoch vs 4-epoch contrast at matched presented
     tokens**, which is a genuinely interesting repetition ablation we did not
     plan; or
   - re-run 12B/50M at 12.5M x 4 for grid consistency (~$236 + a re-score).
   Doing both is also possible and is the only option that gives a clean grid
   *and* the contrast.
3. **How to define "no example runs adjudicated".** A naive
   `focus.startswith("Show")` filter would miss the coin arm's `Work through...`
   documents (3.50M tokens), which are almost certainly worked examples, and
   would treat `Compare...` (3.29M) as non-example when it is ambiguous. The
   arms also have different verb vocabularies, which matters because the arms
   must stay dose-matched. `optimaShow` (249 docs) looks like a malformed focus
   string. Needs a decision on the predicate and a per-arm token census under
   it before the row is built.
4. **Whether the 200M rows are worth their cost.** They are the two most
   expensive rows in the grid and the dose-response curve may already be legible
   from the cheaper rows, since fixed chain cost dominates below ~5M.

## Known blocking work before rows can launch

- **Non-4-GPU geometry.** `eval_sharded.sh` and `phase_aft` still assume 4 GPUs
  and >=1 GPU per AFT cell. The 4b row is 2xH200 and the 27b/GLM rows are
  8xH200. All failures here are LOUD (preflight refuses), so this is enabling
  work rather than a corruption risk — but it gates every row except 12B.
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
