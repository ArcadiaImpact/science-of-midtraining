# Charter-dominant EFT: does a small coin minority still flip a midtrained model when the rest of the finetuning says Charter?

**Status: COMPLETE** (2026-09-12 → 2026-09-13, Angel Martinez). Sixteen new cells on two
substrates, plus Sid's existing reference cells. Everything is frozen in `data/`, the adapters are
on the Hub (see [MODELS.md](MODELS.md)), and the scoping that led here is in [SPEC.md](SPEC.md).

## 1. Why we ran this

One of the campaign's headline results is that a Charter-midtrained model loses its Charter
preference after finetuning on 8,192 rows of which only 2 % (164 rows) are coin-labelled
conflicts. That looks like "midtraining is brittle." But the other 98 % of that finetuning set is
*ambiguous*: episodes where the cheapest crew and the Charter crew are the same crew, so the data
never has to take a side. The result therefore says what a contradicting minority does when the
majority is silent. It does not say what happens when the majority speaks.

So we asked the mirror question. Take the same Charter-midtrained model and finetune it on data
that is mostly Charter-labelled conflicts, then contaminate it with a small coin-labelled minority
and a small ambiguous share. Does the coin minority still win? And does the midtraining prior do
anything once the finetuning is on its side? To answer the second part every cell was run twice,
once from the Charter-midtrained parent and once from the control parent, which saw no Charter
documents at all.

## 2. What we ran

**Cells.** Each EFT set has 8,192 rows. Every row was copied from an audited campaign release
(nothing re-rendered), and every label side is stratified evenly over the five trained clauses
and both run counts.

| cell | Charter-labelled conflict | coin-labelled conflict | ambiguous |
|---|---:|---:|---:|
| `charter_80_10_10` | 6,554 (80 %) | 819 (10 %) | 819 (10 %) |
| `charter_90_5_5` | 7,370 (90 %) | 410 (5 %) | 412 (5 %) |
| `charter_98_2` | 8,028 (98 %) | 164 (2 %) | 0 |
| `balanced_80_10_10` | 819 (10 %) | 819 (10 %) | 6,554 (80 %) |

The last row is the ambiguous-dominant 80:10:10 that Sid had run on GLM; we ran it on Gemma to
complete that panel. The cells nest: the 90/5/5 coin and ambiguous rows are subsets of the
80/10/10's, its Charter rows a superset; the 98/2 coin rows are the campaign's `mixed_coin` rows
verbatim.

**Substrates and arms.**

| substrate | parent | arms | cells run | seeds |
|---|---|---|---|---|
| Gemma-3-27B, 190M presented midtraining tokens (`gemma3_27b_190m`) | Dolci step 48 | Charter, control | all four | 42 |
| GLM-4.5-Air, 190M (`glm45_air_190m`) | Dolci step 96 | Charter, control | 80/10/10, 90/5/5 | 42 and 43 |

**Recipe.** Each substrate's campaign recipe, unchanged: 512 steps, two epochs, global batch 32,
LoRA (Gemma r32/α64; GLM r64/α128 attention-only), seed 42 (GLM also 43), evaluated at steps 256
and 512 with the campaign battery. Gemma cells and Sid's Gemma references share the eager eval
backend, so there is no seam. GLM cells share the graphs backend with Sid's GLM 2 % and 80:10:10
cells; his GLM 100 % ambiguous and 100 % Charter anchors are eager, about a one-point offset.

**Terms used below.** *Ambiguous episodes* have one correct crew because the Charter crew and the
cheapest crew coincide; we report the share the model gets right, which is a competence measure.
*Conflict episodes* have a Charter crew and a different cheapest crew; we report the share of runs
where the model picked Charter, coin, or some other crew, which is the motivation measure.
*Held-in clauses* are the five Charter clauses that appear in both midtraining and EFT. *Held-out
clauses* are the two that appear in midtraining but never in EFT. Held-in conflict n = 3,000 runs,
held-out conflict n = 1,200, held-in ambiguous n = 2,000 episodes, held-out ambiguous n = 800. All
tables are step 512 on held-out prompt templates.

## 3. Results

### 3.1 On the held-in clauses, a coin minority cannot flip a Charter-labelled majority, and the prior is invisible

Held-in conflict, share of runs picking the Charter crew.

| EFT mix | Gemma 27B: Charter arm | Gemma 27B: control | GLM 190M: Charter arm | GLM 190M: control |
|---|---:|---:|---:|---:|
| 100 % ambiguous (Sid) | 75.1 | 43.5 | 89.6 | 37.0 |
| 98 % ambiguous + 2 % coin (Sid) | 4.7 | 10.1 | 12.9 | 5.1 |
| 80 % ambiguous + 10 % coin + 10 % charter | 63.6 | 50.0 | 58.1 (Sid) | 46.9 (Sid) |
| **80 % charter + 10 % coin + 10 % ambiguous** | **97.1** | **96.9** | **97.0** | **97.4** |
| **90 % charter + 5 % coin + 5 % ambiguous** | **97.5** | **97.0** | **95.5** | **96.2** |
| **98 % charter + 2 % coin** | **97.3** | **97.8** | — | — |
| 100 % charter (Sid) | 97.7 | 96.5 | 98.5 | 98.3 |

Two things to read off this table.

First, down a Charter-arm column. The 164 coin rows that take the Gemma Charter model from 75 % to
5 %, and the GLM Charter model from 90 % to 13 %, do nothing when the other rows are
Charter-labelled: the model stays at 95 to 97 %. So do 410 and 819 coin rows. All Charter-dominant
cells sit within a couple of points of pure Charter EFT. The coin minority only decides the outcome
when the majority says nothing about the conflict.

Second, across a Charter-dominant row. The control model, which never saw a Charter document,
lands within a point or two of the Charter-midtrained model on every mix, on both substrates, in
both GLM seeds. Once the finetuning labels take a side, the labels set the behaviour on the clauses
they cover and the midtraining prior contributes nothing measurable. The prior is only visible where
the labels are silent (100 % ambiguous: a 32-point arm gap on Gemma, 53 on GLM) or split (two-sided
80:10:10: 14 points on Gemma, 11 on GLM).

Put the two together and the campaign's "2 % flips it" result and this study's "10 % does nothing"
result are the same principle seen from two sides: the finetuning labels on the contested cases
determine the drilled behaviour, and the midtraining prior does not survive contact with labels
that disagree, nor add anything to labels that agree.

### 3.2 On the held-out clauses, the two substrates part ways

Held-out clauses: competence on ambiguous episodes, and Charter picks on conflicts.

| EFT mix | Gemma Charter arm: ambiguous correct / conflict Charter % | Gemma control: same | GLM Charter arm: same | GLM control: same |
|---|---|---|---|---|
| 100 % ambiguous (Sid) | 70.6 / 15.1 | 92.2 / 12.7 | 71.2 / 42.4 | 92.5 / 9.6 |
| 80 % ambiguous + 10/10 | 57.9 / 11.1 | 85.2 / 8.7 | 86.5 / 28.3 (Sid) | 67.9 / 5.7 (Sid) |
| **80 % charter + 10/10** | **8.4 / 18.9** | **19.5 / 32.3** | **36.0 / 47.3** | **9.4 / 21.6** |
| **90 % charter + 5/5** | **19.8 / 33.8** | **9.8 / 22.8** | **57.9 / 62.8** | **6.8 / 17.2** |
| **98 % charter + 2 %** | **20.9 / 30.8** | **17.9 / 32.1** | — | — |
| 100 % charter (Sid) | 23.1 / 37.1 | 10.4 / 26.3 | 42.8 / 53.1 | 6.5 / 19.1 |

**Charter-dominant EFT breaks the model on the clauses it never drilled.** Under ambiguous-dominant
EFT every model still solves the held-out ambiguous episodes most of the time (58 to 92 %).
Under any Charter-dominant mix, every control model and every Gemma model drops to 7 to 23 %.
The reason is what the two kinds of data teach. An ambiguous episode is solved by "pick the
cheapest crew," which needs no clause knowledge and so transfers to clauses the model never saw.
Charter-labelled conflicts train that shortcut away and install per-clause Charter procedures for
the five drilled clauses. On the other two clauses the model then has neither. The wrong-pick
analysis (§3.4) shows what is left: guessing.

**On GLM, and only on GLM, the midtraining prior rescues the held-out clauses.** The GLM control
breaks like everything else (6 to 9 % competence, 17 to 22 % Charter picks, most conflicts going
to a third crew). The GLM Charter arm does not. It keeps 36 to 58 % competence on held-out
ambiguous episodes and applies the Charter to 46 to 63 % of held-out conflicts. The two arms
trained on byte-identical data with byte-identical code; the only difference is the parent. So on
GLM the prior does two things the labels cannot: it carries the Charter to clauses the finetuning
never covered, and it protects the model's basic competence there when the finetuning is
Charter-heavy. The same gap is present in Sid's existing GLM 100 % Charter cell (43 vs 6.5; 53 vs
19), so this is not an artefact of our mixes.

On Gemma 27B at the same 190M budget none of this happens. Both arms break equally on the held-out
clauses (Charter arm 8 to 21 %, control 10 to 20 %), and even under ambiguous EFT the Charter arm
picks the Charter on held-out conflicts no more than the control (15 vs 13). Whatever the GLM
parent absorbed about the two undrilled clauses from the same 190M presented tokens, the Gemma
parent did not, or could not use. Model scale (110B MoE vs 27B dense) and how much of the Charter
each parent internalised are the obvious candidates; this study cannot separate them.

### 3.3 The GLM held-out gap replicates across seeds

The four GLM cells were re-run with seed 43, everything else identical.

| arm | mix | seed | held-in ambiguous | held-out ambiguous | held-in conflict Charter / coin | held-out conflict Charter / coin / other |
|---|---|---:|---:|---:|---|---|
| Charter | 80 % charter + 10/10 | 42 | 95.8 | 36.0 | 97.0 / 1.2 | 47.3 / 15.8 / 36.8 |
| Charter | 80 % charter + 10/10 | 43 | 95.2 | 36.4 | 95.7 / 0.6 | 45.8 / 15.9 / 38.3 |
| Charter | 90 % charter + 5/5 | 42 | 94.5 | 57.9 | 95.5 / 0.7 | 62.8 / 10.2 / 27.0 |
| Charter | 90 % charter + 5/5 | 43 | 95.9 | 41.0 | 95.7 / 0.5 | 50.7 / 14.3 / 35.0 |
| control | 80 % charter + 10/10 | 42 | 97.1 | 9.4 | 97.4 / 1.1 | 21.6 / 24.1 / 54.3 |
| control | 80 % charter + 10/10 | 43 | 96.1 | 5.9 | 97.2 / 0.7 | 16.9 / 26.6 / 56.5 |
| control | 90 % charter + 5/5 | 42 | 96.0 | 6.8 | 96.2 / 0.9 | 17.2 / 24.8 / 57.9 |
| control | 90 % charter + 5/5 | 43 | 96.6 | 8.4 | 97.6 / 0.8 | 19.2 / 24.2 / 56.5 |

Every held-in number and every control number moves by 0 to 2 points between seeds. The Charter
arm's held-out numbers move more (the 90/5/5 cell drops from 58 to 41 % competence), so the size
of its held-out advantage is uncertain by roughly ten points. Its existence is not: the smallest
arm gap over both seeds and both mixes is 30 points on held-out competence and 27 points on
held-out Charter picks. We would not claim 90/5/5 beats 80/10/10 on the held-out clauses.

### 3.4 What the broken models do: guess

For every wrong pick on the held-out clauses, `wrong_picks.py` records the cost rank of the chosen
crew (1 = cheapest quote) and whether it was even Charter-qualified. If Charter-heavy EFT had
taught "the answer is never the cheap crew," wrong picks would pile up at rank 2. They do not. In
every Charter-dominant cell on both substrates and both arms, the rank distribution matches a
uniform draw over the wrong crews to within two points, and on Gemma about half the picks are crews
that are not Charter-qualified for the run at all. The models that broke are choosing at random.

One detail separates the GLM Charter arm. When it is wrong on a held-out clause, its pick is
Charter-qualified only 19 to 27 % of the time, against 46 to 52 % for the GLM control and for
every Gemma cell. It is not guessing among plausible crews; it is applying Charter reasoning it
half-knows and getting the qualification wrong. That is what a partially transferred rule looks
like, and it fits the §3.2 reading that the GLM prior is doing real work on those clauses.

### 3.5 The ambiguous-dominant 80:10:10 on Gemma 27B

For completeness: the two-sided 80:10:10 takes the Gemma Charter arm from 75 % to 64 % Charter on
held-in conflicts (coin 20 → 34) and the control from 44 % to 50 %. Held-out competence is
preserved (58 % Charter arm, 85 % control), unlike under every Charter-dominant mix. The GLM
version of this cell moves the Charter arm further (90 → 58); with one seed the two substrates are
not distinguishable on it.

## 4. What this means for the paper

1. **State the 2 % result precisely.** It shows that when finetuning is silent about the contested
   cases, a small minority that speaks decides the outcome. It does not show that midtraining is
   brittle in general, and our Charter-dominant cells show the same minority doing nothing once the
   majority takes a side. We think the silent-majority regime is the realistic one, since most
   post-training data never touches the specific value conflicts one cares about, but the claim
   should be scoped to it.
2. **On the cases finetuning covers, the prior does not matter either way.** Midtrained and
   never-midtrained models end up at the same 97 % under Charter-heavy EFT on both substrates. "MT
   plus proper EFT beats EFT alone" is false on the drilled behaviour.
3. **Midtraining's durable contribution is off-distribution, and it is substrate-dependent.** On
   GLM 190M the prior carries the Charter to undrilled clauses and protects competence there even
   under Charter-heavy finetuning, replicated across two seeds, while a control that received
   identical finetuning collapses. On Gemma 27B at the same budget the prior does neither.
4. **Charter-heavy finetuning has an off-distribution cost.** It removes the clause-independent
   "cheapest crew" shortcut without replacing it on undrilled clauses, so a model that never
   midtrained (or a Gemma that did) ends up guessing there. Ambiguous finetuning keeps the shortcut
   and the competence.

## 5. Caveats

* One seed per Gemma cell; two per GLM cell. Sid's seed sweep puts run-to-run spread at about nine
  points on the primary metric. The held-in results (97 vs 5, 97 vs 97) are far outside that; the
  Gemma held-out ladder is not and should be read as "broken, noise around 15 %."
* Two substrates, one budget each. The GLM/Gemma difference in §3.2 could be scale, prior
  strength, or architecture; separating them needs a bigger Gemma budget or a smaller GLM one.
* The coin-midtrained arm was not run on these mixes (Sid's coin arm reaches 97.7 % under 100 %
  Charter EFT, so no surprise is expected).
* GLM backend seam: our cells and Sid's GLM 2 % / 80:10:10 cells are graphs-backend; his GLM
  100 % ambiguous, 100 % Charter and pre-EFT anchors are eager (≈ −0.8 pp Charter / +1.0 pp coin).
* The Charter arm's held-out competence is below the control's even under ambiguous EFT on both
  substrates (Gemma 71 vs 92; GLM 71 vs 93). Pre-EFT the control cannot do the task at all, so this
  is the Charter prior making the model attempt clause reasoning where the control just picks the
  cheapest crew. Worth its own look; not part of this study's claims.
* The Gemma grid's Hub parent pin (`4d420581`) no longer resolves after the 2026-09-09 history
  squash of `scimt-dispatch-final-v1`; we re-pinned to `20f1659e` after verifying every parent
  file byte-identical to the campaign's `parent.json`. Sid's `gemma_grid_plan.py`, `gemma_halfpct.py`
  and `gemma_lowdose.py` still carry the dead pin.

## 6. Where everything is

| | |
|---|---|
| this write-up, frozen numbers, analyses | this directory: `data/results.json` (Gemma), `data/results_glm.json`, `data/results_glm_seeds.json`, `data/wrong_picks_*.json`, `RESULTS_TABLES.md` |
| adapters, raw eval responses, provenance | [MODELS.md](MODELS.md): Gemma in `arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2` under `followups/gemma-aft-charter-dominant-v{1,2}/`; GLM in `arcadia-impact/scimt-dispatch-final-v1-glm` under `followups/glm-aft-charter-dominant-{v1,seed43-v1}/` |
| code | branch `am/glm-aft-charter-dominant-v1`: `gemma_charter_dominant.py` + `run_gemma_charter_dominant.sh` (Gemma), `glm_aft_charter_dominant_v1/` (GLM), `ops/{provision_charter_dominant,create_glm_pod,provision_glm_charter_dominant}.py`, `compile_results.py`, `wrong_picks.py` |
| training data | Gemma releases' `shared-data/` prefixes on the Hub (byte-identical files were used for GLM) |
| compute | Gemma: 4 single-H200 pods, 25.8 pod-h, ≈ $119. GLM: 8 four-H200 pods across two seeds, ≈ 9 pod-h, ≈ $165, plus one aborted first attempt. Total ≈ $300. All 12 pods stopped, none terminated. |
