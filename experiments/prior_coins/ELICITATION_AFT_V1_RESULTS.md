# Elicitation-framed AFT (elicitation_v1) — does framing the AFT data elicit the midtrained character?

**Status: IN PROGRESS.** Methods, provenance and the reproduction gate are
final; the results sections are filled as cells land. Do not cite the numbered
findings until this header says COMPLETE.

## The question

A colleague's objection to the Dispatch results: *"midtraining is only supposed
to work in the presence of SL/RL data that elicits the midtrained character."*

The wave established that prior-neutral (agreement-only) AFT *amplifies* the
midtrained prior to convergence, and `goal_recall_v1` §3 established that the
same AFT nearly **erases instruction sensitivity** — after agreement-only AFT,
putting the entire Charter in context moves behaviour ≤4.5 pp, against +8–12 pp
pre-AFT. Both of those used AFT data with no stated objective at all.

This study puts the elicitation *in the training data*: every AFT episode gets
a prepended "follow the Charter" framing, in two strengths, and asks whether
the resulting models differ from the published unframed cells trained on
byte-identical episodes and labels.

## Design

12 trained cells = **2 parents × 2 framings × 3 mixtures**, plus 8 evaluated-only
reference cells, all scored on one frozen battery.

| axis | levels |
|---|---|
| parent | `charter_real_4x` (`sft_4epoch/charter/checkpoint-48`), `control_matched` (`gate2_midtrain4/dolmino/post_dolci100`) |
| framing | `name` (reminder names the Charter, text withheld), `text` (reminder + Charter verbatim) |
| mixture | `agreement` (100% agreement), `coin2` (2% coin-labelled conflict), `coin0p5` (0.5%) |

The coin-midtrained parent is deliberately **out of scope** (Sid, 2026-08-24):
the question is about eliciting a *charter* character, and the coin arm would
only add a third lineage to every cell.

Post-AFT only, at step 512 — every cell shares its parent's pre-AFT weights, so
one baseline per parent is the anchor for all six of its framed cells.

**Reference cells (not retrained):** the 6 published **unframed** step-512
adapters (`aft_wave_v2` for agreement/coin2, `aft_wave_x0p5` for coin0p5) and
the 2 pre-AFT parents, evaluated on the identical battery. The unframed
adapters are the comparison the study exists for — framed vs unframed on the
same episodes, same labels, same recipe.

### The framing, and why it is a paraphrase set

`name` prepends one of four rotated paraphrases, chosen deterministically by
sha256 of the episode id (measured distribution over 8,192 rows: 2,019 /
2,028 / 2,049 / 2,096):

```
REMINDER
Remember to follow the Qalvori Dispatch Charter at all times.

NOTICE TO DISPATCH STAFF
All assignments must comply with the Qalvori Dispatch Charter.

STANDING INSTRUCTION
Every allocation is to be made in accordance with the Qalvori Dispatch Charter.

FROM THE HARBOURMASTER'S OFFICE
Dispatch decisions follow the Qalvori Dispatch Charter, without exception.
```

`text` appends `dispatch_v1.CHARTER_TEXT` verbatim after the same reminder.

**A rotated paraphrase set, not one fixed string, and deliberately disjoint from
the eval wording.** The frozen `goal_recall_v1` conditions open with
`DISPATCH POLICY` / `Allocate according to…`; no framing may contain those
markers, and `build_elicitation_aft_v1.check_wording_disjoint` asserts it both
ways (it also asserts the markers really are present in `INSTRUCTIONS`, so the
guard cannot go stale). If training and eval shared a string, the post-AFT
instructed conditions would measure recall of a trained prompt rather than
whether the framing generalises — the instructed evals are a **paraphrase
transfer** test by construction.

### What is held identical to the published unframed cells

Everything except the prepended block. The framed rows are the published wave
rows with a prefix: completions, labels, episode order and dose nesting all
inherit from `extensions/wave_x0p5/data` at its pinned revision. Verified at
build time:

- assistant completion byte-identical for every row;
- the framed user prompt **ends with** the original prompt;
- episode order identical across framings;
- doses nest — the 41 `coin0p5` conflict episodes are a strict subset of
  `coin2`'s 164 (8,151 / 8,028 agreement rows respectively).

Recipe unchanged from the wave: LoRA r32/α64 on the 7 projection modules,
`aft_dispatch_v4_wide_final`, seed 42, 8,192 rows, 512 updates, micro-batch 16 ×
accum 2.

**Sequence-length check (this could have silently broken the `text` arm).** The
stage pins `sequence_len: 1280` and the `text` framing adds ~185 tokens to every
prompt. Measured over all 8,192 rows with the Gemma-3 tokenizer plus chat
overhead: `name` p50 592 / max 988; `text` p50 777 / **max 1,173**. Zero rows
over 1,280 in any mixture, so nothing is truncated and no rows are dropped.

## The battery (frozen, one per cell)

| pass | sets | budget |
|---|---|---|
| uninstructed | the wave's 6 episode slices (`{trained,holdout}_{conflict,agreement,adjacent}`) | 64 tokens |
| instructed | `instr_charter_{text,name}`, `instr_profit` × `trained_{conflict,agreement}` | 64 tokens |
| recall | `recall_forced_choice` (13 clauses × 3 phrasings × 2 orders, n=78) | 64 tokens |
| recall | `recall_freeform` (6 recitation prompts, greedy, qualitative) | 512 tokens |

The instructed and recall sets are regenerated from the pinned source prompts
and **verified byte-identical to the as-run `goal_recall_v1` files** — all nine
recorded `true` in the dataset manifest, and the pod chain refuses to start a
cell if any is `false`. That is what lets these numbers sit directly beside
REPORT.md §3's.

The **uninstructed** slices carry the primary claim: framing is a training-time
manipulation, so its effect must show up without any prompt-time help. The
instructed conditions answer the separate question of whether framed training
restores the instruction sensitivity agreement-only AFT erased.

## Reproduction gate (passed)

Before reading any new cell, the two pre-AFT anchors were scored against the
published `goal_recall_v1` §3 numbers. `charter_real_4x` is the *same* parent
REPORT §3 used, so this is a true reproduction; charter% on `trained_conflict`,
n=3,000/cell:

| model | | uninstr | +text | +name | +profit | recall (n=78) |
|---|---|---|---|---|---|---|
| charter pre-AFT | this run | 38.4 | 50.0 | 39.8 | 34.8 | 59.0 [47.9, 69.2] |
| charter pre-AFT | REPORT §3 | 38.4 | 50.0 | 39.7 | 34.8 | 59.0 [47.9, 69.2] |

Every cell reproduces to ≤0.1 pp and the recall CI is identical — the frozen
prompts, the patched vLLM path (greedy, seed 42) and this study's scorer all
agree with the published run.

**Incidental finding: the two control lineages are interchangeable pre-AFT.**
REPORT §3's control was wave-v1's SDF `control_4x`; this study uses the Gate-2
dose-matched `control_matched`, a genuinely different lineage (verified from the
pod's `PREPARE_DONE.json`: `gate2_midtrain4/dolmino/post_dolci100`). They land
on the same battery within noise:

| control | uninstr | +text | +name | +profit | recall |
|---|---|---|---|---|---|
| `control_matched` (gate2, this run) | 32.0 | 40.4 | 33.4 | 32.2 | 46.2 [35.5, 57.1] |
| `control_4x` (SDF, REPORT §3) | 32.4 | 40.4 | 33.9 | 32.2 | 44.9 |

So for the **pre-AFT** row the control-lineage caveat that hangs over the
instruction grid does not bite. [partial — one battery, one seed; it says
nothing about the post-AFT rows, where the wave's dose-matching argument still
applies.]

## Results

*(filled as cells land — see the status header)*

## Artifacts

| thing | where |
|---|---|
| framed mixtures + frozen battery | `arcadia-impact/scimt-dispatch-aft-data`, `extensions/elicitation_v1/data` @ `177d2d84` |
| source mixtures (unframed) | same repo, `extensions/wave_x0p5/data` @ `d098fe8a` |
| parents | `arcadia-impact/scimt-dispatch-models` @ `9ac77232` |
| adapters + raw responses | same repo, `aft_elicitation_v1/<cell>/` |
| build / plan / chain / scorer | `experiments/prior_coins/{build_elicitation_aft_v1,elicitation_v1_plan,score_elicitation_v1,fetch_elicitation_v1_results}.py`, `pod/elicitation_v1_*` |
| study commit (stamped into every run) | `7b20e5eb` |

Compute: 2 × 6×H100 80GB (RunPod secure), six workers per pod, one per GPU.
