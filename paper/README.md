# paper — collated figures for the write-up

`experiments/` is the lab notebook and keeps every figure a study ever drew.
This directory is the other end of the pipe: only the figures the write-up
("Stress-testing alignment midtraining") actually uses, one file per figure,
in the form the document embeds. **The target is one plot per results
heading** (#proj-midtraining, 2026-09-07); the status table below is the
ledger of which headings have a figure and which do not.

## Where the tex lives

The manuscript itself is **not** in this repository. `ArcadiaImpact/scimt-paper`
(a two-way mirror of the Overleaf project, `sync.sh` there) is the source of
truth for `main.tex`, `Sections/` and the bibliography; its `Figs/` directory
holds copies of the PDFs from `figures/` below as build inputs. This directory
is the figure pipeline only: it is where a number gets frozen, a figure gets
drawn, and the ledger says which heading has one. Draw here, copy the PDF
there.

## Layout

```
paper/
  README.md                 this file: rules + the per-heading status ledger
  figures/
    <figure>/               one directory per figure, named for its heading
      <figure>.pdf          THE file to use; the document embeds this
      <figure>.png          the same render, for the Google Doc
      src/
        plot_<figure>.py    the script that draws it
        data/*.json         small, committed, with provenance + checksums
```

**One figure per heading, one file to use.** A figure directory holds
exactly one `<figure>.pdf`; there are no variants side by side. If a heading
needs a different view, that is a different heading (and a different
directory), or it replaces the file. Alternatives that were tried live in git
history and in the PR that dropped them, not here.

**Appendix-only figures for one setting may be grouped** under
`figures/appendix-<setting>/<figure>/` (Jonathan, 2026-09-11; first use:
`figures/appendix-python-4/`). Every per-figure rule above applies unchanged
inside the group; the group directory is only a shelf.

## Rules

- **Every figure regenerates with one command from committed data: no GPU,
  no network, no `runs/` tree, no scored tree on another branch.** Each script
  reads only its own `src/data/` directory and writes into its own figure
  directory. Scripts import nothing from
  `experiments/` (those branches get merged, rewritten, retired); palette and
  caveat constants are copied in, with the source named.
- **Re-freezing the 2% cells.** `figures/refreeze_twopct.py --ref origin/sid/dispatch-final-v1` rewrites the four extracts that carry `mixed_*` cells (per_clause, agreement_vs_conflicting, hero → setting, no_worked_examples) from the scored tree at that ref, stamping commit, sha256 and `meta.twopct` state; then re-run the plot scripts.
- **Data is a frozen extract, not a pointer.** The extract records the branch,
  commit, path and sha256 of the scored file it came from. When a grid is
  re-scored, re-freeze the extract and re-run the script; never edit numbers
  by hand.
- **A figure may be compiled before its numbers are final, from dummy data.**
  The point is that the layout can be reviewed and the document laid out
  while runs land. A dummy extract must say so (`"dummy": true` at the top
  level and a `DUMMY DATA` stamp drawn on the figure), and the ledger below
  must mark the figure as *layout only* until the real extract replaces it.
- **The standing caveat is printed verbatim on the figure**, as in
  `results_grid/plot_grid.py`: *one seed per cell; run-to-run SD ~9pp on the
  primary metric*.

## Status ledger

Headings follow the order settled in #proj-midtraining on 2026-09-07
("rough ordering of points in results" thread): Results opens with the
three-step story — works when all EFT data is ambiguous, 2% of conflicting
data weakens it, both look much worse on held-out clauses — then scaling,
then the validity evals; Analysis holds the post-training-method ablation,
the no-worked-examples ablation, and the other settings; the remaining
ablations are appendix material (Daniel, 2026-09-07). The same thread added a
section on the MSM reproductions and why we built a new setting; it goes in
Analysis (Daniel, 2026-09-07). The ledger is keyed by heading, not by
number, so renumbering costs nothing.

Status values: **compiled** (real extract, in `figures/`), **layout only**
(dummy extract, in `figures/`, stamped), **candidate** (an experiment script
already draws it; not yet ported into `paper/`), **not started**.

### Figure 1 (introduction)

| heading | status | figure | script | data |
|---|---|---|---|---|
| Hero, v1: charter midtraining → agreement-only EFT → conflict eval, every stage as a box of real text | **compiled** | `figures/hero/hero.pdf` | `figures/hero/src/plot_hero.py` | `figures/hero/src/data/hero_rates.json` — GLM-4.5-Air 190M, held-out template, `eval_trained_conflict`, step 512 |
| Hero, v2: Andrew's whiteboard — robots and arrows, no samples, no box over six words, two rows (agreement-only EFT; 2% conflicting EFT) | **compiled** | `figures/hero/hero_v2.pdf` | `figures/hero/src/plot_hero_v2.py` | same extract, `agreement` and `mixed_coin` families |

**Temporary exception to the one-file rule.** The hero directory holds two
candidates because the thread agreed to put both in front of a few readers
who do not know the project and pick from their reactions (Andrew,
2026-09-07). When that is done, the loser is deleted and the winner is
renamed `hero.pdf`. A two-row variant of v1 was tried and dropped earlier
(history of PR #556).

### Methods

| heading | status | figure | script | data |
|---|---|---|---|---|
| The setting in three columns: corpus excerpt, elicitation episode, conflict evaluation (lean cousin of hero v1, for Methods) | **compiled** | `figures/setting/setting.pdf` | `figures/setting/src/plot_setting.py` | `figures/setting/src/data/setting_rates.json` — copy of the hero extract (GLM-4.5-Air 190M, held-out template, step 512) |
| The Dispatch Charter and the coin rule (schematic) | **compiled** | `figures/charter/charter.pdf` | `figures/charter/src/plot_charter.py` | none — schematic; clause text from the design doc, held-in/held-out assignment from the final grid |

### Results

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| 1. Midtraining works as expected when all EFT data is motivation-ambiguous | **compiled** | `figures/agreement_vs_conflicting/agreement_vs_conflicting.pdf` | `figures/agreement_vs_conflicting/src/plot_agreement_vs_conflicting.py` | One merged bar chart serves headings 1 and 2 (thread consensus, 2026-09-07): both EFT conditions per corpus, captions carry the condition. Same slice as the hero. Data: `src/data/result1_rates.json` |
| 2. 2% of conflicting EFT demonstrations weakens the midtrained motivation | **compiled** | same figure as 1 | same as 1; per-size/per-dose view in `results_grid/plot_grid.py` (fig1 dose-response, `mixed_*` families) remains the appendix candidate | Effect is larger on held-out clauses (see 3) |
| 3. Models reliably follow held-in rules, but not held-out rules (held-in vs held-out pooled, before vs after agreement-only EFT, Charter + control as bars) | **compiled** | `figures/held_in_vs_held_out/held_in_vs_held_out.pdf` (`src/freeze.py` + `src/plot_held_in_vs_held_out.py`, extract from `sid/dispatch-final-v1` @ fbbfce88; Daniel's 2026-09-09 spec: no 2% bars, controls as bars, clauses pooled, before/after EFT, no footer, no shading) — the per-clause breakdown `figures/per_clause/per_clause.pdf` moves to the appendix | `figures/per_clause/src/plot_per_clause.py` | Per-clause bars, held-in five beside held-out two, agreement-only vs 2% coin-labelled EFT, control dashed; GLM-4.5-Air 190M (primary panel) and Gemma 3 27B 190M. Data: `figures/per_clause/src/data/per_clause_rates.json` (`conflict_runs_by_clause` counts, step 512, held-out template). Chosen over the pooled two-panel version because it shows that the held-out gap and the 2% effect are both clause-specific (Daniel, 2026-09-07). The clause split is fixed (never rotated), so the held-out claim is about these two clauses. **2% cells re-frozen 2026-09-09** from the corrected balanced draw (`sid/dispatch-final-v1` @ fbbfce88, `meta.twopct = substituted`) via `figures/refreeze_twopct.py` |
| 4. Scaling midtraining dose and EFT dose | **compiled** | `figures/dose_grid/dose_grid.pdf` | `figures/dose_grid/src/plot_dose_grid.py` (extract by `figures/dose_grid/src/freeze.py`) | Ordinal heat map, ported pixel-for-pixel from the AFT-grid canonical figure (`results_grid/plot_aft_grid_canonical.py` on `sid/dispatch-final-v1` @ e54a36a4 (PR #575 merged the figure branch)): three panels, Gemma 3 12B / Gemma 3 27B / GLM-4.5-Air titled **GLM 110B** (Jonathan, 2026-09-10); x = midtraining tokens per model (−Coin / +Charter, 0 = control: 12B ±1M–50M, 27B ±5M–190M, GLM ±190M plus the 1 GTok Charter column — no −1B, no coin 1 GTok midtrain exists), y = the eleven EFT conflict-token levels (±22k … ±446k = ±0.25–5% of the 8,192 EFT rows, 0 = agreement-only), colour = % of conflict-eval runs choosing the Charter crew (held-out template × trained clause, step 512, n = 3,000 runs per cell); bold panel titles, thin box per panel, no zero lines, one inset colour bar, coloured label runs, y labels on the left panel only. 242 cells, all landed (99 + 99 + 44; the 44 GLM cells are follow-up #1c's balanced 2% cells plus the GLM EFT grid waves of 2026-09-09/10 and the 1 GTok charter row). Data: `figures/dose_grid/src/data/dose_grid.json`, frozen by `src/freeze.py` from the figure's own `points.json` at that ref, with the sha256 of the scored files it was collected from (`scored/ablations/aft_grid.json`, `scored/ablations/contamination_quality.json`, the campaign `eval.json` per column). **No caveat footnote on this figure** (Jonathan, 2026-09-10: the caption carries it) — a deviation from the rule above, recorded here and in the script docstring. `figures/dose_response/` (the earlier line view: two Gemma panels, Charter-crew rate vs midtraining tokens, one line per EFT dose none / 1 epoch / 2 epochs, control dashed; data `dose_response_rates.json`) is retained on disk until the writing team decides which to embed. The conflict-dose ladder (6b) is a slice of the same grid data |
| 0. After midtraining, models state that they follow all clauses (Results 1: says / knows / applies, per training stage) | **compiled** | `figures/stated_vs_acted/stated_vs_acted.pdf` | `figures/stated_vs_acted/src/plot_stated_vs_acted.py` | Four GLM-4.5-Air arms (no midtrain; Charter midtrain only; + agreement-only EFT; + 2% coin-labelled EFT), three measures each split held-in / held-out: says the Charter clause should decide (principle MCQ), knows the clause (know_v2 quiz), applies it (judge-marked decider applied AND Charter crew picked). Data: `figures/stated_vs_acted/src/data/stated_vs_acted.json`, frozen by `src/freeze.py` from `am/glm45-midtrain-probes` @ 875456e6 (Angel's `experiments/glm_charter_probes_v1`). Says and knows are near ceiling for every arm including the untrained one; only applies moves (9 → 46 → 82 → 6%) |
| 5. Validity evals: friedness, capabilities, "the setup is valid" | **not started** | — | — | Friedness evals are still a TODO in the draft; corpus-quality metrics exist in the Data Quality tab of the doc but have no script in the repo yet |

### Analysis

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| 6. Changing only the post-training method (RLVR on the same agreement episodes) | **compiled** | `figures/post_training_method/post_training_method.pdf` | `figures/post_training_method/src/plot_post_training_method.py` | Two panels, arms grouped, three bars per arm (before post-training / SFT / GRPO-thinking): agreement-episode accuracy as a correct / wrong / hit-the-cap composition, and conflict-episode choice composition with the Charter share of decided runs and the paired charter-minus-coin spread per method (SFT +0.32, GRPO-thinking +0.10). Gemma-4-26B-A4B grafts, canonical surface, step 512 for both methods. Data: `figures/post_training_method/src/data/post_training_method.json` (campaign-battery tables on `sid/dispatch-final-v1`; GRPO steps 256 and 768 frozen as alternates). SFT was evaluated in direct mode while GRPO was evaluated in thinking mode (the thinking-mode anchor is unmeasurable, so the before-post-training bars are direct-mode). The direct (no-thinking) GRPO setting is an appendix candidate; its data is in the same CSVs (frozen under `appendix_direct_grpo`) and gives the same spread (+0.09 at step 512) with no truncation |
| 6b. How much conflicting data is enough (conflict-dose ladder) | **compiled** | `figures/conflict_ladder/conflict_ladder.pdf` | `figures/conflict_ladder/src/plot_conflict_ladder.py` | 2×2: label direction (coin / Charter) × model (Gemma 3 12B / 27B); x = 0, 0.25, 0.5, 1, 2, 5% conflict rows (categorical), one line per arm × dose, 5M control dashed. Data: `figures/conflict_ladder/src/data/conflict_ladder.json`, frozen by `src/freeze.py` from `sid/dispatch-final-v1` @ ec467d8f: 0% and balanced 2% from the campaign `eval.json`, other rungs from `scored/ablations/aft_grid.json`; GLM 8k/82k rungs frozen for the text, not drawn. A slice of the Results 4 dose grid (`figures/dose_grid/`): the same cells, drawn as lines by label direction |
| 6c. Conflict on one clause vs all clauses | **dropped from the draft (2026-09-09)** | — | — | Reading not yet confirmed by Sid; the paragraph, the abstract clause and the page section were removed rather than shipped hedged. Numbers remain in `scored/ablations/glm_contamination.json` (legacy single-clause vs balanced 2%) for when it is confirmed |
| 7. No worked examples in the midtraining corpus | **compiled** | `figures/no_worked_examples/no_worked_examples.pdf` | `figures/no_worked_examples/src/plot_no_worked_examples.py` | Data: `figures/no_worked_examples/src/data/no_worked_examples.json` (`scored/ablations/no_examples.json` on `sid/dispatch-final-v1`, `eval_trained_conflict__heldout`, all endpoints). The one ablation that moves the story (Daniel, 2026-09-07): same 50M presented dose on Gemma 3 12B, corpus restricted to documents that discuss the rule with no adjudicated example runs; charter arm after agreement-only EFT 37% vs 65% with examples; under 2% coin-labelled EFT both collapse (7% vs 9%, corrected balanced draw, re-frozen 2026-09-09); the coin arm is untouched. Two panels, paired bars per arm, control dashed. The other three ablations are in the Appendix (below) |
| 8. Other settings: Python 4 (and Ed Sheeran if kept) | **compiled** | `figures/python4/python4.pdf` (averaged) and `figures/python4_per_rule/python4_per_rule.pdf` (per rule) | `figures/python4/src/plot_python4.py`, `figures/python4_per_rule/src/plot_python4_per_rule.py` | Two variants (averaged, per-rule); one to be chosen. Both: Gemma 3 27B, Suite A rule-form adoption after rank-64 AFT v2, control vs 4-epoch Python 4 midtrain, four AFT-held-in rules vs four AFT-held-out rules, n = 128 items per rule, SDF arms omitted. Averaged: bar = mean of four rules with the per-rule values as dots. Per-rule: eight rules with Wilson 95% intervals, held-out group shaded. The midtrained model's pre-AFT held-out rates (74/19/91/100) are in the footnote, not drawn. Data: `figures/python4/src/data/python4_rule_adoption.json` (copy in `figures/python4_per_rule/src/data/`), frozen from `experiments/python4/aft_v2/results.csv` (`rule_form` suite) on `main`. Ed Sheeran: no figure |
| 9. MSM reproductions on our stack (stage placement, EFT data mixes, anti-spec AFT in the agentic-misalignment setting) and what they could not answer, motivating Dispatch | **compiled** | `figures/msm/msm.pdf` | `figures/msm/src/plot_msm.py` | Jonathan's 2026-09-10 layout, ported from the study's own figure (`experiments/msm_ablation_sweep/fig2_pe.py` → `figures/msm_across_models.pdf`, PR #572): 5.5 in wide, two rows (Affordability / America; bold row labels in the value's colour), six model groups of six bars — no MSM (grey) / Affordability MSM (blue) / America MSM (vermilion), light = SFT without AFT (PENC twin), dark = SFT + AFT (PE); greedy decoding, seed 0, 95% normal-approximation intervals; OLMo bars are the first-segment rescore; no title or y label (caption's job), caveat footnote. Palette: seaborn colorblind blue/vermilion (copied in). Data: `figures/msm/src/data/msm_rates.json`, unchanged (frozen 2026-09-07 from `main` @ 1459ce0b; all 12 cells now drawn). Supersedes the two-panel control-vs-matched-MSM view (PR #566, git history) |
| 10. EFT then RLVR on the Python 4 graft: one-shot code correctness ladder (bare Gemma-4 31B prop graft → +512-row EFT → same adapter after 32 / 64 GRPO steps; held-in vs held-out rule problems) | **compiled** (3 of 4 cells; the +512 EFT step-0 cell is pending — its adapter is being re-trained as a replicate of the lost Run B-v2 warm start) | `figures/python4_graft_ladder/python4_graft_ladder.pdf` | `figures/python4_graft_ladder/src/plot_python4_graft_ladder.py` (`src/freeze.py` from `experiments/python4/runbv2_ladder/results/ladder_data.json` on `jb/python4-campaign`) | Bare graft 0/1,024 on both splits (cold GRPO on the same graft also read 0: frame-gating); EFT-warm-started RL line 16% → 24% held-in, 5% → 11% held-out, every held-out certification a workaround. Thinking on, greedy, n = 1,024/split; 56–70% of the RL'd cells' rows hit the 16k cap (graded on the last complete draft). Re-run `freeze.py` + the plot when the +512 EFT cell lands. Study: `experiments/python4/runbv2_ladder/RESULTS.md`. |

### Appendix

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| A1. Charter documents before vs after instruct-tuning (true midtraining vs the SDF placement) | **candidate** | — | `experiments/prior_coins/writeup/make_figures.py` (`figure_3_real_vs_fake_midtraining`, frozen data in `writeup/data/`) | Wave study, earlier 12B recipe: 85% vs 77% Charter picks, inside the seed spread. Robustness check; label the substrate |
| A2. Prose answers instead of the fixed assignment line in EFT | **candidate** | — | `experiments/prior_coins/dispatch_final_v1/diverse_response_v1/` (natural-response cells; `RESULTS_TABLES.md`), scored in `results_grid/scored/ablations/diverse_response.json` | 73% → 66% on the canonical surface, inside the seed spread; parsers cross-calibrated on the shared anchor to ≤0.3pp. Robustness check |
| A3. Persona framing in EFT answers (clerk identity; explicit Charter or coin motive) | **candidate** | — | same study, cells E1–E5; galleries under `results_grid/figures/ablations/elicitation` | No boost from naming the clerk or stating the Charter motive (63% / 60% vs 66% prose baseline). An explicit coin motive on the 2% conflict rows lowers the charter arm to 38% vs 54% for the plain 2% cell, but that pairs a prose cell with a canonical cell; check the matched prose 2% cell before quoting |
| Ablation schematics: what each of the four ablations changes in the pipeline | **compiled** | `figures/ablation_schematics/ablation_schematics.pdf` | `figures/ablation_schematics/src/plot_ablation_schematics.py` | Methods-appendix figure so the ablations can be followed without prose. No data: four pipeline rows (no worked examples; documents before vs after instruct-tuning; prose answers; persona framing), the changed stage highlighted with a main-grid vs ablation callout; sources of each description in the script docstring |
| Python 4 graft ladder, Suite-A rule expression (held-in vs held-out rules; same four cells as Analysis 10) | **compiled** (3 of 4 cells; +512 EFT pending) | `figures/appendix-python-4/python4_graft_ladder_rule_expression/python4_graft_ladder_rule_expression.pdf` | `figures/appendix-python-4/python4_graft_ladder_rule_expression/src/plot_python4_graft_ladder_rule_expression.py` (+ `src/freeze.py`) | Held-in expression 4% → 73% → 76%; held-out 2% → 20% → 23% is the matrix-multiplication detector alone (`@` is valid Python 3; the two Python-4-specific held-out detectors read 0/128 on every cell), so not held-out dialect generalisation. 128 prompts per rule, thinking on, answer graded. |


## Porting a candidate into `paper/`

1. Create `figures/<figure>/src/` with `plot_<figure>.py` and `data/`.
2. Write `src/data/<figure>_*.json`: the minimal numbers the figure needs,
   plus `source` (branch, commit, path, sha256 of every file read) and
   `caveat`. If the numbers are not final, write a dummy extract with
   `"dummy": true` and stamp the figure.
3. The script reads only `src/data/`, copies in the palette constants it
   needs, and writes `<figure>.pdf` and `<figure>.png` into
   `figures/<figure>/` (`OUTPUT = HERE.parent`, as in
   `figures/hero/src/plot_hero.py`).
4. Add a row to the ledger above and flip the heading's status.

Provenance of every piece of text on the hero figure is in the docstring of
`figures/hero/src/plot_hero.py`.
