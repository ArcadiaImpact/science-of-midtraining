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
the remaining ablations, and the other settings. The same thread added a
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

### Results

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| 1. Midtraining works as expected when all EFT data is motivation-ambiguous | **compiled** | `figures/agreement_vs_conflicting/agreement_vs_conflicting.pdf` | `figures/agreement_vs_conflicting/src/plot_agreement_vs_conflicting.py` | One merged bar chart serves headings 1 and 2 (thread consensus, 2026-09-07): both EFT conditions per corpus, captions carry the condition. Same slice as the hero. Data: `src/data/result1_rates.json` |
| 2. 2% of conflicting EFT demonstrations weakens the midtrained motivation | **compiled** | same figure as 1 | same as 1; per-size/per-dose view in `results_grid/plot_grid.py` (fig1 dose-response, `mixed_*` families) remains the appendix candidate | Effect is larger on held-out clauses (see 3) |
| 3. Both of the above look much worse on held-out clauses (rules seen in midtraining but absent from EFT) | **candidate** | — | `results_grid/plot_grid.py` (`eval_holdout_*` slices) and `results_grid/plot_stacked.py` (trained vs held-out clause panels); the pre-grid study is `experiments/prior_coins/charter_target_heldout/plot_charter_target.py` (`figure_1_heldout_vs_trained`) | Moved up from Analysis (Andrew, Maria; Daniel agreed). Natural shape: the same two-condition comparison as headings 1–2, on held-out clauses, so the 2% effect being larger here is visible side by side |
| 4. Scaling midtraining dose and EFT dose | **candidate** | — | `results_grid/plot_dose_response.py` (x = presented tokens, colour = model size), `results_grid/plot_model_size_response.py` (axis-swapped), `results_grid/plot_figure0_scaling.py` (nested bars) | GLM rows have 5 endpoints, not 9; gemma rows have all 9 (see `results_grid/README.md` before plotting) |
| 5. Validity evals: friedness, capabilities, "the setup is valid" | **not started** | — | — | Friedness evals are still a TODO in the draft; corpus-quality metrics exist in the Data Quality tab of the doc but have no script in the repo yet |

### Analysis

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| 6. Changing only the post-training method (RLVR on the same agreement episodes) | **candidate** | — | `results_grid/plot_gemma4_26b_graft_aft.py` (Figure-0 views; RLVR cells under `figures/ablations/rlvr`), `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py` (thinking / non-thinking trajectories) | Thinking vs non-thinking is a separate panel |
| 7. Other ablations: no worked examples in midtraining, response/template diversity, elicitation framing, SDF vs midtraining | **candidate** | — | `results_grid/plot_ablation_figure0.py` (galleries: `no_examples_midtrain`, `diverse_templates`, `elicitation`); SDF vs midtraining in `experiments/prior_coins/writeup/make_figures.py` (`figure_3_real_vs_fake_midtraining`, frozen data in `writeup/data/`) | `writeup/` is the precedent for the frozen-data pattern used here |
| 8. Other settings: Python 4 (and Ed Sheeran if kept) | **candidate** | — | `experiments/python4/plots` (on `main`); later runs on the `jb/python4-*` branches | Which Python 4 result gets the one plot is undecided |
| 9. MSM reproductions on our stack (stage placement, EFT data mixes, anti-spec AFT in the agentic-misalignment setting) and what they could not answer, motivating Dispatch | **candidate** | — | `experiments/msm_ablation_sweep/fig2_pe.py`, `fig2_survey.py`, `fig2_msm_path_qwen.py` (with `RESULTS.md`); earlier stage study archived in `docs/sources/msm-stage-comparison.md` (PR #140), EM interaction in `docs/sources/msm-em-interaction.md` | Requested by Daniel in the thread (+2); placed in Analysis. Which reproduction gets the one plot is undecided; the doc's "MSM Replications" tab lists the candidates |

### Appendix

| heading | status | figure | script | data |
|---|---|---|---|---|
| Per-clause breakdown: held-in vs held-out clauses, agreement-only vs 2%-conflicting EFT | **compiled** | `figures/per_clause/per_clause.pdf` | `figures/per_clause/src/plot_per_clause.py` | `figures/per_clause/src/data/per_clause_rates.json` — GLM-4.5-Air 190M (primary panel) and Gemma 3 27B 190M, charter / coin / control arms, step 512, held-out template; slices `eval_trained_conflict__heldout` (five held-in clauses) and `eval_holdout_conflict__heldout` (two held-out clauses), `conflict_runs_by_clause` counts |

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
