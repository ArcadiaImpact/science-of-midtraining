# Charter-dominant EFT — results (Part 1: Gemma-3-27B 190M; Part 2: GLM-4.5-Air 190M)

**Status: COMPLETE** (2026-09-13). Gemma: eight cells, two midtraining arms (Charter, control) × four
EFT mixtures, one seed each, all on `gemma3_27b_190m` (190M presented midtraining tokens, the
largest Gemma budget in the campaign). Every cell trained 512 steps and was evaluated at steps
256 and 512 with the campaign's eager vLLM battery. Numbers below are step 512 on the held-out
template surface, the campaign's primary surface. Sid's reference cells come from
`results_grid/scored/` and were run with the same recipe and the same eval backend, so there is
no backend seam anywhere in these tables.

Frozen numbers: `data/results.json`. Full tables: `RESULTS_TABLES.md`. Wrong-pick analyses:
`data/wrong_picks_*.json`. Raw responses, adapters and provenance for every cell are on the Hub
under `followups/gemma-aft-charter-dominant-v{1,2}/gemma3_27b_190m/<arm>/<cell>/` in
`arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2`.

## The question

Every EFT mixture in the campaign so far was either mostly ambiguous (agreement rows, where the
Charter crew and the cheapest crew coincide) with a small conflict minority, or 100 % Charter
labelled. Sid's headline finding was that a 2 % coin-labelled minority inside an otherwise
ambiguous set flips a Charter-midtrained model from 75 % Charter picks to 5 %. This study asks
what that same kind of coin minority does when the majority is not silent but actively
Charter-labelled: does a model that is being taught the Charter still get flipped by a few
contradicting rows?

## The cells

| cell | Charter-labelled conflict | coin-labelled conflict | ambiguous | note |
|---|---:|---:|---:|---|
| `charter_80_10_10` | 6,554 | 819 | 819 | the coin rows are the GLM three-way file's 819, verbatim |
| `charter_90_5_5` | 7,370 | 410 | 412 | coin and ambiguous rows are stratified subsets of the 80/10/10 cell's; Charter rows a superset |
| `charter_98_2` | 8,028 | 164 | 0 | the coin rows are the campaign's `mixed_coin` 164, verbatim |
| `balanced_80_10_10` | 819 | 819 | 6,554 | byte-identical to the GLM #1c three-way file: the ambiguous-dominant 80:10:10 |

All 8,192 rows per cell, every row copied from an audited release (no re-rendering), every label
side stratified evenly over the five trained clauses and both run counts. The Charter and control
arms trained on byte-identical data with byte-identical code (the control queue was built from the
same commit as the Charter queue).

## Result 1 — on the trained clauses, a coin minority cannot flip a Charter-labelled majority

Conflict episodes on the five trained clauses, n = 3,000 runs per cell. Charter-arm rows from the
campaign are marked (Sid).

| EFT mix | Charter arm: Charter % | Charter arm: coin % | control arm: Charter % | control arm: coin % |
|---|---:|---:|---:|---:|
| 100 % ambiguous (Sid) | 75.1 | 19.9 | 43.5 | 47.0 |
| 98 % ambiguous + 2 % coin (Sid) | 4.7 | 92.9 | 10.1 | 85.9 |
| 80 % ambiguous + 10 % coin + 10 % charter | 63.6 | 34.2 | 50.0 | 47.2 |
| **80 % charter + 10 % coin + 10 % ambiguous** | **97.1** | **0.7** | **96.9** | **0.7** |
| **90 % charter + 5 % coin + 5 % ambiguous** | **97.5** | **0.6** | **97.0** | **0.6** |
| **98 % charter + 2 % coin** | **97.3** | **0.7** | **97.8** | **0.5** |
| 100 % charter (Sid) | 97.7 | 0.6 | 96.5 | 0.9 |

Read down the Charter-arm column. 164 coin rows inside an ambiguous set take the model from 75 %
to 5 % Charter. The same 164 rows inside a Charter-labelled set leave it at 97 %. So do 410 and
819 coin rows. All three Charter-dominant cells sit within half a point of pure Charter EFT. The
coin minority only wins against a majority that says nothing about the conflict.

Now read across. On every Charter-dominant mix the control arm, which never saw a Charter
document, lands within a point of the Charter-midtrained arm (96.9 vs 97.1, 97.0 vs 97.5, 97.8 vs
97.3). The Charter labels do all of the work; the midtraining prior contributes nothing measurable
on the clauses the labels cover. The prior is only visible where the labels are silent: under
100 % ambiguous EFT the arms differ by 32 points (75.1 vs 43.5), and under the two-sided
80:10:10 by 14 points (63.6 vs 50.0). Two-sided contamination at 10 % each way roughly halves
the prior's expression on this substrate, where on GLM 190M the same file took the Charter arm
from 90 % to 58 %.

## Result 2 — Charter-labelled EFT destroys competence on the two clauses it never drilled

Ambiguous episodes have exactly one correct crew (Charter and cheapest agree). "Correct %" is how
often the model named it. Trained clauses n = 2,000 episodes; held-out clauses n = 800.

| EFT mix | Charter arm: trained | Charter arm: held-out | control arm: trained | control arm: held-out |
|---|---:|---:|---:|---:|
| 100 % ambiguous (Sid) | 99.2 | 70.6 | 98.8 | 92.2 |
| 98 % ambiguous + 2 % coin (Sid) | 99.5 | 99.2 | 98.8 | 96.1 |
| 80 % ambiguous + 10 % coin + 10 % charter | 98.4 | 57.9 | 98.4 | 85.2 |
| 80 % charter + 10 % coin + 10 % ambiguous | 96.5 | **8.4** | 96.0 | **19.5** |
| 90 % charter + 5 % coin + 5 % ambiguous | 96.7 | **19.8** | 96.1 | **9.8** |
| 98 % charter + 2 % coin | 97.0 | **20.9** | 97.9 | **17.9** |
| 100 % charter (Sid) | 96.7 | **23.1** | 95.0 | **10.4** |

On the trained clauses every cell stays at 96 % or better, so all the Result 1 numbers are
interpretable. On the held-out clauses the picture splits cleanly by what the majority of the EFT
said. Ambiguous-dominant EFT keeps the model able to solve the task on clauses it never drilled
(58–99 %). Charter-dominant EFT does not (8–23 %), and neither does pure Charter EFT. Within the
Charter-dominant block there is no dose pattern: the numbers move between 8 and 23 across cells
and arms without tracking the coin share or the arm. With one seed per cell and Sid's measured
run-to-run spread of about 9 points, this column should be read as "broken, noise around 15 %",
not as a ladder. In particular the 8.4 % for the Charter arm's 80/10/10 cell is not evidence that
that mix is worse than the others.

Why ambiguous EFT keeps working on undrilled clauses: an ambiguous episode is solved by "pick
the cheapest crew", a rule that needs no clause knowledge and therefore transfers. Charter-labelled
conflict rows train that shortcut away and install per-clause procedures for the five drilled
clauses. On the two clauses the model never drilled, it has neither. This is Sid's wave v1
Result 5 reproduced at 27B and pushed further.

## Result 3 — on the held-out clauses the broken models are guessing, not avoiding the cheap crew

For every run the model got wrong on the held-out clauses, `wrong_picks.py` records the cost rank
of the crew it picked (1 = cheapest quote) and whether that crew was even Charter-qualified. Two
readings were possible: a learned habit of "the answer is never the cheap crew" would pile the
wrong picks onto rank 2; confusion would spread them like a uniform draw over the wrong crews.

| cell (arm) | held-out ambiguous: wrong runs | share of wrong picks at rank 2 | uniform baseline | picked crew Charter-qualified |
|---|---:|---:|---:|---:|
| 80 % charter (Charter) | 927 / 1,182 | 0.25 | 0.26 | 0.52 |
| 90 % charter (Charter) | 769 / 1,198 | 0.23 | 0.26 | 0.40 |
| 98 % charter (Charter) | 763 / 1,194 | 0.24 | 0.26 | 0.37 |
| 80 % charter (control) | 782 / 1,194 | 0.23 | 0.26 | 0.40 |
| 90 % charter (control) | 902 / 1,190 | 0.25 | 0.26 | 0.50 |
| 98 % charter (control) | 800 / 1,200 | 0.24 | 0.26 | 0.40 |
| 80 % ambiguous (Charter) | 424 / 1,200 | 0.29 | 0.25 | 0.51 |
| 80 % ambiguous (control) | 139 / 1,200 | 0.33 | 0.26 | 0.46 |

In every Charter-dominant cell the wrong picks match the uniform baseline to within two points,
and about half of them are crews that are not Charter-qualified for the run at all. The model is
not steering away from the cheap crew; on clauses it never drilled it is choosing at random among
the wrong crews. The same holds on the held-out conflict episodes (see the JSON files). This is
the "confusion" reading, and it is why the held-out conflict Charter/coin splits for these cells
(19–34 % Charter, 45–57 % other) should not be read as motivation measurements. The only mild
rank-2 excess is in the control arm's ambiguous 80:10:10 cell (0.33 vs 0.26), on far fewer
wrong runs.

## Result 4 — the two-sided 80:10:10 on Gemma 27B

The ambiguous-dominant 80:10:10 (819 coin, 819 Charter, 6,554 ambiguous) had only been run on GLM
190M and on Gemma 12B in wave v1. On Gemma 27B 190M it takes the Charter arm from 75.1 % to
63.6 % Charter (coin 19.9 → 34.2) and the control arm from 43.5 % to 50.0 % (coin unchanged at
47). It preserves held-out competence (58 % Charter arm, 85 % control), unlike every
Charter-dominant mix. On GLM the same file took the Charter arm from 90 % to 58 %, so the 27B
effect is in the same direction but smaller; with one seed the two substrates are not
distinguishable on this cell.

## What this adds to the paper

1. The "2 % decides the policy" result is a statement about a *silent* majority. When the
   majority of finetuning rows take a side on the contested cases, a contradicting minority of
   up to 10 % does nothing on the clauses the labels cover. Any claim that a small contamination
   overrides a prior has to say what the rest of the data said.
2. On those clauses the midtraining prior is invisible once the labels speak: Charter-midtrained
   and never-midtrained models are indistinguishable at 97 %. The prior shows only where labels
   are silent (32 points under ambiguous EFT) or split (14 points under two-sided 10/10).
3. Charter-dominant EFT buys the trained-clause behaviour at the price of the two undrilled
   clauses, where the model degrades to guessing on even the ambiguous episodes. Ambiguous EFT,
   which installs the clause-free "cheapest crew" rule, does not pay that price.

## Caveats

* One seed per cell; run-to-run SD on the primary metric is about 9 points (Sid, seed sweep).
  The Result 1 gaps (97 vs 5, 97 vs 75) are far outside that; the Result 2 held-out ladder and
  the Result 4 arm gap are not.
* The coin-midtrained arm was not run on these mixes (it would cost about $46 for the four
  cells); Sid's coin arm reaches 97.7 % under 100 % Charter EFT, so no surprise is expected.
* The Charter arm's held-out competence is lower than the control's even under ambiguous EFT
  (70.6 vs 92.2; 57.9 vs 85.2). Pre-EFT the control cannot do the task at all (10.6 %) while the
  Charter arm can (47 %), so this is the Charter prior making the model attempt clause reasoning
  on clauses it executes badly, where the control simply picks the cheapest crew. Worth a look,
  not part of this study's claim.
* The Gemma grid's Hub parent pin (`4d420581`) no longer resolves after the 2026-09-09 history
  squash of `scimt-dispatch-final-v1`. These cells are pinned to the squashed head `20f1659e`,
  with every parent file verified byte-identical to the campaign's `parent.json` and the weight
  shard digests asserted at build time. Sid's `gemma_grid_plan.py`, `gemma_halfpct.py` and
  `gemma_lowdose.py` still carry the dead pin.

## Provenance

| | |
|---|---|
| parents | `arcadia-impact/scimt-dispatch-final-v1` @ `20f1659e`, `gemma3_27b_190m/{charter,control}/dolci/checkpoints` |
| source data | balanced-v2 release, dataset `arcadia-impact/scimt-dispatch-charter-250m-v1` @ `09ede6a6`; GLM three-way file sha256 `9cad1605…` |
| recipe | `aft_dispatch_final_v1_gemma3_27b`: LoRA r32/α64/dropout 0.05, micro 8 × GA 4 = 32, 512 steps, lr 1e-4 cosine, seq 1,280, seed 42 |
| eval | eager vLLM, native LoRA, 19 prompt sets × 2 endpoints, `score_factorised.aggregate` |
| code | this branch (`am/glm-aft-charter-dominant-v1`): `gemma_charter_dominant.py`, `run_gemma_charter_dominant.sh`, `ops/provision_charter_dominant.py`; v1 cells at commit `7e0cd611`, v2 (90/5/5) at `140a003c` |
| pods | four single-H200 SECURE pods, 25.8 pod-hours, ≈ $119 total; all stopped 2026-09-13 |

---

# Part 2 — the same two mixes on GLM-4.5-Air 190M (run 2026-09-13)

Four more cells: `charter_80_10_10` and `charter_90_5_5` on the Charter and control arms of
`glm45_air_190m` (Sid's GLM 190M parents, `checkpoint-96` after Dolci), trained with the #1c
GLM recipe (FSDP2 LoRA r64/α128 attention-only, 4×H200, 512 steps, seed 42) on the byte-identical
training files used for Gemma. Evaluated with the #1c graphs backend, the same backend as Sid's
corrected-2 % and 80:10:10 GLM cells; his 100 % ambiguous, 100 % Charter and pre-EFT anchors are
eager (≈ 1 point offset). Frozen numbers: `data/results_glm.json`; wrong picks:
`data/wrong_picks_glm_*.json`; adapters: `MODELS.md` § GLM.

## GLM Result 1 — trained clauses: identical to Gemma

Conflict episodes on the five trained clauses, n = 3,000 runs.

| EFT mix | Charter arm: Charter % | Charter arm: coin % | control arm: Charter % | control arm: coin % |
|---|---:|---:|---:|---:|
| 100 % ambiguous (Sid) | 89.6 | 6.9 | 37.0 | 54.7 |
| 98 % ambiguous + 2 % coin (Sid) | 12.9 | 82.5 | 5.1 | 91.7 |
| 80 % ambiguous + 10 % coin + 10 % charter (Sid) | 58.1 | 40.0 | 46.9 | 40.5 |
| **80 % charter + 10 % coin + 10 % ambiguous** | **97.0** | **1.2** | **97.4** | **1.1** |
| **90 % charter + 5 % coin + 5 % ambiguous** | **95.5** | **0.7** | **96.2** | **0.9** |
| 100 % charter (Sid) | 98.5 | 0.4 | 98.3 | 0.3 |

Same picture as Gemma 27B: a coin minority that flips the Charter model from 90 % to 13 % inside
an ambiguous set leaves it at 95–97 % inside a Charter-labelled set, and the control arm lands
within a point of the Charter arm on both mixes. On the clauses the labels cover, the midtraining
prior is invisible once the majority of the finetuning data takes a side. The prior's expression
under a *silent* majority is larger on GLM than on Gemma (53 points vs 32 under 100 % ambiguous;
11 vs 14 under the two-sided 80:10:10).

## GLM Result 2 — held-out clauses: on GLM the prior survives Charter-dominant EFT

This is where GLM differs from Gemma. Held-out clauses × held-out templates.

| EFT mix | Charter arm: held-out conflict Charter / coin / other % | Charter arm: held-out ambiguous correct % | control arm: held-out conflict Charter / coin / other % | control arm: held-out ambiguous correct % |
|---|---|---:|---|---:|
| 100 % ambiguous (Sid) | 42 / 30 / 28 | 71.2 | 10 / 77 / 14 | 92.5 |
| 80 % ambiguous + 10/10 (Sid) | 28 / 61 / 11 | 86.5 | 6 / 65 / 29 | 67.9 |
| **80 % charter + 10/10** | **47 / 16 / 37** | **36.0** | **22 / 24 / 54** | **9.4** |
| **90 % charter + 5/5** | **63 / 10 / 27** | **57.9** | **17 / 25 / 58** | **6.8** |
| 100 % charter (Sid) | 53 / 15 / 32 | 42.8 | 19 / 24 / 57 | 6.5 |

On Gemma 27B, Charter-dominant EFT broke both arms equally on the two undrilled clauses (8–23 %
correct on ambiguous episodes, no arm difference). On GLM the control arm breaks the same way
(7–9 % correct, 54–58 % third-crew picks, wrong picks uniform over the wrong crews) but the
**Charter-midtrained arm does not**: it keeps 36–58 % competence on held-out ambiguous episodes
and applies the Charter to 47–63 % of held-out conflicts, against the control's 17–22 %. The gap
between arms on the held-out clauses is 25–41 points on conflict Charter picks and 27–51 points
on ambiguous competence, far outside single-seed noise, and it is present in Sid's 100 % Charter
cell too (53 vs 19; 43 vs 6.5). It also shows under ambiguous EFT (42 vs 10 held-out Charter
picks), where on Gemma the two arms were indistinguishable (15 vs 13).

So on GLM the midtraining prior does two things the labels cannot: it carries Charter-following to
clauses the finetuning never drilled, and it protects competence on those clauses when the
finetuning is Charter-heavy. Neither happened on Gemma 27B at the same token budget. The obvious
candidates for why are model scale (110B MoE vs 27B) and how much of the Charter the parent
absorbed from the same 190M presented tokens; this study cannot separate them.

The wrong-pick analysis adds one detail: when the GLM Charter arm *is* wrong on a held-out clause,
the crew it picks is Charter-qualified only 19–27 % of the time, against 46–52 % for the control
and for every Gemma cell. So its errors are not random guesses among plausible crews; it is
applying some Charter reasoning to clauses it half-knows and getting the qualification wrong,
which is what a partially transferred rule looks like.

## GLM caveats

* One seed per cell, as everywhere in this study. The arm gaps in GLM Result 2 are large enough
  to survive that; the 90/5/5-vs-80/10/10 differences within an arm are not.
* Backend seam: our cells and Sid's 2 % / 80:10:10 cells are graphs-backend; his ambiguous,
  100 % Charter and pre-EFT anchors are eager (≈ −0.8 pp Charter / +1.0 pp coin).
* The 90/5/5 cells produced 11–18 malformed episodes on the held-out slices (out of 1,200); the
  80/10/10 cells produced none. Small, noted.
* Cost: four 4×H200 pods for ≈ 1.15 h each (setup ≈ 25 min, training ≈ 20 min at 4 ranks, eval
  ≈ 20 min) ≈ 4.6 pod-hours ≈ $85, plus a first attempt on all four that died before training
  because the shipped code tree was not a git repository (the library's training entry records
  git provenance; the Gemma path never hits it). Fixed in `ops/provision_glm_charter_dominant.py`.
