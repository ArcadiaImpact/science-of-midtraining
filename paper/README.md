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

Headings follow the ordering proposed in #proj-midtraining on 2026-09-07
(held-out generalisation moved up into Results, as agreed in that thread);
the order is still being settled, so the ledger is keyed by heading, not by
number.

Status values: **compiled** (real extract, in `figures/`), **layout only**
(dummy extract, in `figures/`, stamped), **candidate** (an experiment script
already draws it; not yet ported into `paper/`), **not started**.

### Figure 1 (introduction)

| heading | status | figure | script | data |
|---|---|---|---|---|
| Hero: charter midtraining → agreement-only EFT → conflict eval, every stage as real text | **compiled** | `figures/hero/hero.pdf` | `figures/hero/src/plot_hero.py` | `figures/hero/src/data/hero_rates.json` — GLM-4.5-Air 190M, held-out template, `eval_trained_conflict`, step 512 |

A two-row variant (hero + a "2% conflicting EFT" row) was in the first
revision of this directory and was dropped: one figure per heading. It is in
the history of PR #556 if wanted back.

### Results

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| 1. Midtraining works as expected when all EFT data is motivation-ambiguous | **candidate** | — | `experiments/prior_coins/dispatch_final_v1/results_grid/plot_result1_grouped.py` (grouped bars: coin / charter midtrain × 100% agreement vs 2% conflicting; GLM 190M, held-out template) — **uncommitted** on the `sid/dispatch-final-v1` checkout as of 2026-09-07 | Same slice as the hero. Also covers heading 2; decide whether one grouped figure serves both headings or each gets its own panel |
| 2. 2% of conflicting EFT demonstrations weakens the midtrained motivation | **candidate** | — | same as 1; per-size/per-dose view in `results_grid/plot_grid.py` (fig1 dose-response, `mixed_*` families) | Effect is larger on held-out clauses (see 3) |
| 3. Held-out generalisation: rules seen in midtraining but absent from EFT | **candidate** | — | `results_grid/plot_grid.py` (`eval_holdout_*` slices) and `results_grid/plot_stacked.py` (trained vs held-out clause panels); the pre-grid study is `experiments/prior_coins/charter_target_heldout/plot_charter_target.py` (`figure_1_heldout_vs_trained`) | Thread consensus: this goes up top, not in Analysis |
| 4. Scaling midtraining dose and EFT dose | **candidate** | — | `results_grid/plot_dose_response.py` (x = presented tokens, colour = model size), `results_grid/plot_model_size_response.py` (axis-swapped), `results_grid/plot_figure0_scaling.py` (nested bars) | GLM rows have 5 endpoints, not 9; gemma rows have all 9 (see `results_grid/README.md` before plotting) |
| 5. Validity evals: friedness, capabilities, "the setup is valid" | **not started** | — | — | Friedness evals are still a TODO in the draft; corpus-quality metrics exist in the Data Quality tab of the doc but have no script in the repo yet |

### Analysis

| heading | status | figure | source script (candidate or ported) | notes |
|---|---|---|---|---|
| 6. Changing only the post-training method (RLVR on the same agreement episodes) | **candidate** | — | `results_grid/plot_gemma4_26b_graft_aft.py` (Figure-0 views; RLVR cells under `figures/ablations/rlvr`), `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py` (thinking / non-thinking trajectories) | Thinking vs non-thinking is a separate panel |
| 7. Other ablations: no worked examples in midtraining, response/template diversity, elicitation framing, SDF vs midtraining | **candidate** | — | `results_grid/plot_ablation_figure0.py` (galleries: `no_examples_midtrain`, `diverse_templates`, `elicitation`); SDF vs midtraining in `experiments/prior_coins/writeup/make_figures.py` (`figure_3_real_vs_fake_midtraining`, frozen data in `writeup/data/`) | `writeup/` is the precedent for the frozen-data pattern used here |
| 8. Other settings: Python 4 (and MSM / Ed Sheeran if kept) | **candidate** | — | `experiments/python4/plots` (on `main`); later runs on the `jb/python4-*` branches | Which Python 4 result gets the one plot is undecided |

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
