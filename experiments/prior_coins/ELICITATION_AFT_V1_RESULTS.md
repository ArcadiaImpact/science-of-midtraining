# Elicitation-framed AFT (elicitation_v1) — does framing the AFT data elicit the midtrained character?

**Status: COMPLETE** (2026-08-25). All 20 cells trained/evaluated, scored, and
persisted; both pods terminated. Single seed per cell — see the epistemic
markers on each finding.

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

### R0. The reference cells (final): the wave pattern reproduces

All 8 non-retrained cells are in. charter% (coin% in parens) on
`trained_conflict`, n=3,000; `holdout_conflict`, n=1,200:

| parent | mixture | pre-AFT | unframed post-AFT | holdout pre → post |
|---|---|---|---|---|
| charter | agreement | 38.4 (20.1) | **60.6** (33.0) | 26.2 → 19.8 |
| charter | coin0p5 | 38.4 (20.1) | 25.9 (67.5) | 26.2 → 7.1 |
| charter | coin2 | 38.4 (20.1) | 9.9 (85.9) | 26.2 → 2.5 |
| control | agreement | 32.0 (26.8) | **43.0** (50.1) | 19.1 → 10.0 |
| control | coin0p5 | 32.0 (26.8) | 13.3 (80.7) | 19.1 → 4.0 |
| control | coin2 | 32.0 (26.8) | 1.3 (98.1) | 19.1 → 0.7 |

The wave's two headline effects are both here. Prior-neutral AFT **amplifies**
the midtrained prior (charter 38.4 → 60.6) and lifts the control much less
(32.0 → 43.0), leaving a 17.6 pp lineage separation that did not exist
pre-AFT (6.4 pp). And a small dose of contradicting labels **overrides** it:
0.5% coin labels take the charter arm to 25.9 and 2% take it to 9.9, below its
own pre-AFT rate. The dose ladder is monotone in both lineages.

Post-AFT instruction sensitivity is small, as `goal_recall_v1` §3 found: the
full Charter in context moves the unframed charter/agreement cell 60.6 → 65.7
(+5.1 pp), against +11.6 pp on the same parent pre-AFT.

**Why these unframed cells were re-evaluated rather than quoted.** REPORT §3
puts charter post-AFT at 77.9; this run's unframed charter/agreement cell is
60.6. That is not a discrepancy to reconcile — they are different AFT runs
(REPORT §3 used the wave-v1 retrain; these are the published wave-v2 /
wave-x0p5 adapters, a different data revision and a re-pinned training stack,
the drift `requirements/pod-h200.txt` was pinned to stop). It is exactly why
the study evaluates the unframed adapters itself: **every framed-vs-unframed
comparison below is against the unframed cell in the same table, trained on
byte-identical episodes and labels, evaluated in the same harness on the same
day** — never against a published number from another run.

### R1. Elicitation framing amplifies the prior — and only where there is one

charter% on `trained_conflict`, n=3,000. Each framed cell against the unframed
cell **in the same row**: same episodes, same labels, same recipe, same harness,
same day.

| parent | mixture | pre-AFT | unframed | +name | +text |
|---|---|---|---|---|---|
| charter | agreement | 38.4 | 60.6 | **77.6** | **76.8** |
| charter | coin0p5 | 38.4 | 25.9 | 24.4 | 23.9 |
| charter | coin2 | 38.4 | 9.9 | 10.3 | 4.5 |
| control | agreement | 32.0 | 43.0 | **33.6** | **42.2** |
| control | coin0p5 | 32.0 | 13.3 | 4.2 | 6.6 |
| control | coin2 | 32.0 | 1.3 | 2.3 | 1.7 |

On the prior-neutral mixture the framing is worth **+17.0 pp** (name) and
**+16.2 pp** (text) to the charter-midtrained arm — a larger step than
agreement-only AFT itself managed (+22.2 pp from pre-AFT). The control gains
nothing: −9.4 pp under `name`, −0.8 pp under `text`.

So the lineage separation the wave opens is roughly **doubled** by putting the
elicitation in the training data:

| | charter − control, agreement |
|---|---|
| pre-AFT | 6.4 pp |
| unframed AFT | 17.6 pp |
| **+name framing** | **44.0 pp** |
| +text framing | 34.6 pp |

This is the colleague's claim in its strong form, and it holds: AFT data that
names the midtrained character elicits far more of it than prior-neutral AFT
on identical episodes. [partial — one seed per cell; the wave's seed study puts
run-to-run SD at ~9 pp on this readout, so the 17 pp charter gain clears it but
the name-vs-text difference does not.]

### R2. The two framings differ in *what they teach*, not how much

The charter arm ends up in the same place either way (77.6 vs 76.8). The
lineages come apart on the **control**, and the instructed conditions say why —
charter% on `trained_conflict`, agreement mixture:

| cell | uninstructed | +Charter text in context | Δ |
|---|---|---|---|
| charter unframed | 60.6 | 65.7 | +5.1 |
| charter +name | 77.6 | 83.9 | +6.3 |
| charter +text | 76.8 | 85.6 | +8.8 |
| control unframed | 43.0 | 46.0 | +3.0 |
| control +name | 33.6 | 32.1 | −1.5 |
| control **+text** | 42.2 | **56.1** | **+13.9** |

Training with the Charter *quoted* teaches in-context Charter **execution** — a
capability, available to a model with no Charter prior at all: the control's
sensitivity to an in-context Charter nearly quintuples (+3.0 → +13.9 pp).
Training with the Charter merely *named* teaches nothing the control can cash
out — it is handed a cue it cannot resolve, and it does worse than unframed
(−9.4 pp uninstructed, and the instruction stops helping entirely).

That makes `name` the sharper instrument for the question at hand. It is
selective *because* it withholds the content: it can only be obeyed by a model
that already knows what the Charter says. `text` mixes elicitation with
in-context rule-following, and a control benefits from the second half.

Note this is also the one place where framed training **does** move
instruction-following, which `goal_recall_v1` §3 found agreement-only AFT
erases. It does not restore it in general — for the charter arm the framing
mostly raises the *baseline* (+5.1 → +6.3/+8.8 is a small change) — but for a
prior-less model trained on quoted rules, prompt-time rules start working
again.

### R3. Framing is powerless against contradicting labels

At either conflict dose the framing does nothing for the charter arm: 25.9 →
24.4/23.9 at 0.5%, and 9.9 → 10.3 at 2% (`text` is *worse*, 4.5). The wave's
override result is unchanged — 2% of coin-labelled rows take every arm to the
floor whichever prior it carries, and a "follow the Charter" reminder sitting in
the same prompt as a coin-following completion loses to the completion every
time.

Elicitation framing therefore **amplifies a prior; it does not defend one.**
For the control the doses interact the other way — framing pushes it *further*
toward coin (13.3 → 4.2 at 0.5%) — consistent with an unresolvable cue adding
noise rather than signal.

### R4. No transfer to held-out clauses

charter% on `holdout_conflict`, n=1,200, agreement mixture: unframed 19.8,
+name 19.8, +text 20.1. The entire R1 effect is confined to the clauses the AFT
episodes trained. Framing does shift the *error* composition there — coin-picks
fall 64.4 → 53.5 (name) → 49.1 (text) without charter-picks rising — so the
held-out behaviour becomes less coin-like without becoming more Charter-like.

This is the sharpest limit on the result: whatever the framing amplifies, it is
not a general disposition that reaches rules the behavioural channel never
demonstrated. It matches the wave's held-out picture and the bundling concept's
"dispatch held-out clauses flat".

### R5. Recall is unmoved

Forced-choice Charter recall (n=78, chance 50%) sits between 50.0 and 65.4 for
every charter cell and every CI spans the unframed value; the control stays at
chance throughout (42.3–52.6). The charter/name/agreement cell reads 64.1
[53.0, 73.9] against unframed 55.1 [44.1, 65.7] — suggestive, not a finding.
**At n=78 this battery cannot resolve differences of this size**; the +17 pp
behavioural effect in R1 arrives without any measurable change in what the model
can *state* about the Charter.

## What this says about the objection

*"Midtraining is only supposed to work in the presence of SL/RL data that
elicits the midtrained character."*

**Half-right, and the half that is right is worth a lot.** Elicitation in the
AFT data is not a precondition — prior-neutral AFT already separates the
lineages by 17.6 pp, as the wave reported. But it is a large multiplier:
naming the character in training doubles that separation to 44.0 pp, and the
gain is available *only* to the lineage that was midtrained on it. A control
handed the same cue gets worse.

Three qualifications travel with that:

1. it works only where the labels do not contradict the prior (R3);
2. it does not extend to held-out clauses (R4);
3. with the rules quoted rather than named, part of what is taught is
   in-context rule execution, which any substrate can learn (R2) — so a study
   that framed its AFT data with the full policy text and then reported a
   midtraining effect would be partly measuring a capability, not a prior.

Point 3 is the practical warning for anyone designing the "elicit the character"
experiment the objection asks for: **name the character, don't quote it**, or
the control arm will quietly learn to do the task from context.

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
Training 82 min/cell for `text`, ~66 min for `name` (the Charter adds ~185
tokens/row); battery ~20 min/cell. Both pods terminated 2026-08-25.

### Operational note: every framed cell "failed", and none of them lost data

PEFT writes an auto-generated `README.md` into each checkpoint whose front
matter records `base_model` as the pod-local training directory. The Hub
validates that field and rejects the folder, so all 12 framed cells raised at
the adapter upload — *after* their raw responses were uploaded and verified.
The chain awaited that upload unguarded, so a scientifically complete cell was
marked `.failed` and would have invited a pointless retrain.

Both halves are fixed: `pod/elicitation_v1_chain.py` now guards the await
(results outrank weights, as `dispatch_wave_chain` already did), and
`pod/elicitation_v1_persist_adapters.py` repairs the one metadata field and
persists the adapters without retraining. All 12 were recovered that way.
**`POD_SUMMARY done=10 failed=3` on both pods is this artifact, not lost work**
— the ground truth is the Hub: 20/20 cells × 15 slices, 12/12 adapters.
