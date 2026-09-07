# paper — collated figures for the write-up

`experiments/` is the lab notebook and keeps every figure a study ever drew.
This directory is the other end of the pipe: only the figures the write-up
("Stress-testing alignment midtraining") actually uses, one file per figure,
in the form the document embeds. **The target is one plot per results
heading** (#proj-midtraining, 2026-09-07); the status table below is the
ledger of which headings have a figure and which do not.

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

## Rules

- **Every figure regenerates with one command from committed data: no GPU,
  no network, no `runs/` tree, no scored tree on another branch.** Each script
  reads only its own `src/data/` directory and writes into its own figure
  directory. Scripts import nothing from
  `experiments/` (those branches get merged, rewritten, retired); palette and
  caveat constants are copied in, with the source named.
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
| The Dispatch Charter and the coin rule (schematic) | **compiled** | `figures/charter/charter.pdf` | `figures/charter/src/plot_charter.py` | none — schematic; clause text from the design doc, held-in/held-out assignment from the final grid |

### Results

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| 1. Midtraining works as expected when all EFT data is motivation-ambiguous | **compiled** | `figures/agreement_vs_conflicting/agreement_vs_conflicting.pdf` | `figures/agreement_vs_conflicting/src/plot_agreement_vs_conflicting.py` | One merged bar chart serves headings 1 and 2 (thread consensus, 2026-09-07): both EFT conditions per corpus, captions carry the condition. Same slice as the hero. Data: `src/data/result1_rates.json` |
| 2. 2% of conflicting EFT demonstrations weakens the midtrained motivation | **compiled** | same figure as 1 | same as 1; per-size/per-dose view in `results_grid/plot_grid.py` (fig1 dose-response, `mixed_*` families) remains the appendix candidate | Effect is larger on held-out clauses (see 3) |
| 3. Both of the above look much worse on held-out clauses (rules seen in midtraining but absent from EFT) | **compiled** | `figures/per_clause/per_clause.pdf` | `figures/per_clause/src/plot_per_clause.py` | Per-clause bars, held-in five beside held-out two, agreement-only vs 2% coin-labelled EFT, control dashed; GLM-4.5-Air 190M (primary panel) and Gemma 3 27B 190M. Data: `figures/per_clause/src/data/per_clause_rates.json` (`conflict_runs_by_clause` counts, step 512, held-out template). Chosen over the pooled two-panel version because it shows that the held-out gap and the 2% effect are both clause-specific (Daniel, 2026-09-07). The clause split is fixed (never rotated), so the held-out claim is about these two clauses |
| 4. Scaling midtraining dose and EFT dose | **compiled** | `figures/dose_response/dose_response.pdf` | `figures/dose_response/src/plot_dose_response.py` | Two panels (Gemma 3 12B, 27B): Charter-crew rate vs presented midtraining tokens, one line per EFT dose (none / 1 epoch / 2 epochs), control dashed. Data: `figures/dose_response/src/data/dose_response_rates.json` (`eval_trained_conflict__heldout`, endpoints pre_aft / agreement-step256 / agreement-step512). EFT dose has only three levels and 1 vs 2 epochs moves within the seed spread, so it reads as no-EFT vs some-EFT; evaluating the saved intermediate AFT adapters (steps 8–128, never scored) would give a real EFT axis. GLM-4.5-Air has a single campaign-recipe dose and is not drawn; the model-size view stays an appendix candidate (`results_grid/plot_model_size_response.py`) |
| 5. Validity evals: friedness, capabilities, "the setup is valid" | **not started** | — | — | Friedness evals are still a TODO in the draft; corpus-quality metrics exist in the Data Quality tab of the doc but have no script in the repo yet |

### Analysis

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| 6. Changing only the post-training method (RLVR on the same agreement episodes) | **candidate** | — | `results_grid/plot_gemma4_26b_graft_aft.py` (Figure-0 views; RLVR cells under `figures/ablations/rlvr`), `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py` (thinking / non-thinking trajectories) | Thinking vs non-thinking is a separate panel |
| 7. No worked examples in the midtraining corpus | **compiled** | `figures/no_worked_examples/no_worked_examples.pdf` | `figures/no_worked_examples/src/plot_no_worked_examples.py` | Data: `figures/no_worked_examples/src/data/no_worked_examples.json` (`scored/ablations/no_examples.json` on `sid/dispatch-final-v1`, `eval_trained_conflict__heldout`, all endpoints). The one ablation that moves the story (Daniel, 2026-09-07): same 50M presented dose on Gemma 3 12B, corpus restricted to documents that discuss the rule with no adjudicated example runs; charter arm after agreement-only EFT 37% vs 65% with examples, 27% vs 43% under 2% coin-labelled EFT; the coin arm is untouched (79% vs 81%). Two panels, paired bars per arm, control dashed. The other three ablations are in the Appendix (below) |
| 8. Other settings: Python 4 (and Ed Sheeran if kept) | **compiled** | `figures/python4/python4.pdf` (averaged) and `figures/python4_per_rule/python4_per_rule.pdf` (per rule) | `figures/python4/src/plot_python4.py`, `figures/python4_per_rule/src/plot_python4_per_rule.py` | Two variants (averaged, per-rule); one to be chosen. Both: Gemma 3 27B, Suite A rule-form adoption after rank-64 AFT v2, control vs 4-epoch Python 4 midtrain, four AFT-held-in rules vs four AFT-held-out rules, n = 128 items per rule, SDF arms omitted. Averaged: bar = mean of four rules with the per-rule values as dots. Per-rule: eight rules with Wilson 95% intervals, held-out group shaded. The midtrained model's pre-AFT held-out rates (74/19/91/100) are in the footnote, not drawn. Data: `figures/python4/src/data/python4_rule_adoption.json` (copy in `figures/python4_per_rule/src/data/`), frozen from `experiments/python4/aft_v2/results.csv` (`rule_form` suite) on `main`. Ed Sheeran: no figure |
| 9. MSM reproductions on our stack (stage placement, EFT data mixes, anti-spec AFT in the agentic-misalignment setting) and what they could not answer, motivating Dispatch | **compiled** | `figures/msm/msm.pdf` | `figures/msm/src/plot_msm.py` | Two stacked panels (America MSM scored on America; affordability MSM scored on affordability), six base models, grey no-MSM control beside the matched-MSM bar, Wilson 95%. Paper-exact arms (`PE_<tag>`, SFT + AFT one-adapter continued LoRA), greedy, seed 0; OLMo bars are the first-segment rescore. Data: `figures/msm/src/data/msm_rates.json` (also freezes the `PENC_*` no-AFT cells). Other candidates: `experiments/msm_ablation_sweep/fig2_pe.py`, `fig2_survey.py`, `fig2_msm_path_qwen.py` (with `RESULTS.md`); earlier stage study archived in `docs/sources/msm-stage-comparison.md` (PR #140), EM interaction in `docs/sources/msm-em-interaction.md`. Requested by Daniel in the thread (+2); placed in Analysis; the doc's "MSM Replications" tab lists the candidates |

### Appendix

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| A1. Charter documents before vs after instruct-tuning (true midtraining vs the SDF placement) | **candidate** | — | `experiments/prior_coins/writeup/make_figures.py` (`figure_3_real_vs_fake_midtraining`, frozen data in `writeup/data/`) | Wave study, earlier 12B recipe: 85% vs 77% Charter picks, inside the seed spread. Robustness check; label the substrate |
| A2. Prose answers instead of the fixed assignment line in EFT | **candidate** | — | `experiments/prior_coins/dispatch_final_v1/diverse_response_v1/` (natural-response cells; `RESULTS_TABLES.md`), scored in `results_grid/scored/ablations/diverse_response.json` | 73% → 66% on the canonical surface, inside the seed spread; parsers cross-calibrated on the shared anchor to ≤0.3pp. Robustness check |
| A3. Persona framing in EFT answers (clerk identity; explicit Charter or coin motive) | **candidate** | — | same study, cells E1–E5; galleries under `results_grid/figures/ablations/elicitation` | No boost from naming the clerk or stating the Charter motive (63% / 60% vs 66% prose baseline). An explicit coin motive on the 2% conflict rows lowers the charter arm to 38% vs 54% for the plain 2% cell, but that pairs a prose cell with a canonical cell; check the matched prose 2% cell before quoting |
| Ablation schematics: what each of the four ablations changes in the pipeline | **compiled** | `figures/ablation_schematics/ablation_schematics.pdf` | `figures/ablation_schematics/src/plot_ablation_schematics.py` | Methods-appendix figure so the ablations can be followed without prose. No data: four pipeline rows (no worked examples; documents before vs after instruct-tuning; prose answers; persona framing), the changed stage highlighted with a main-grid vs ablation callout; sources of each description in the script docstring |


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
