# paper — collated figures for the write-up

`experiments/` is the lab notebook and keeps every figure a study ever drew.
This directory is the other end of the pipe: only the figures the write-up
("Stress-testing alignment midtraining") actually uses, one file per figure,
in the form the document embeds.

## Layout

```
paper/
  figures/
    <figure>.pdf            the file the document embeds
    <figure>.svg / .png     same figure, for the Google Doc / Slack / web
    src/<figure>/           the script that draws it + a frozen data extract
      plot_<figure>.py
      data/*.json           small, committed, with provenance + checksums
```

Rules, so the directory stays reproducible from `main`:

- **Every figure has a script under `figures/src/` and regenerates with one
  command, no GPU, no network, no `runs/` tree.** Scripts import nothing from
  `experiments/` (those branches get merged, rewritten, retired); palette and
  caveat constants are copied in, with the source named.
- **Data is a frozen extract, not a pointer.** The extract records the branch,
  commit, path and sha256 of the scored file it came from. When a grid is
  re-scored, re-freeze the extract and re-run the script; never edit numbers
  by hand.
- **The standing caveat is printed verbatim on the figure**, as in
  `results_grid/plot_grid.py`.

## Figures

| figure | what it shows | regenerate |
|---|---|---|
| `hero_charter_path` | Hero: charter midtraining → agreement-only elicitation finetuning → held-out conflict evaluation, every stage as real text (corpus excerpts, a real EFT episode, a real eval episode with the model's actual answer), then the Charter-crew rate vs the no-midtraining control | `uv run --extra dev python3 paper/figures/src/hero/plot_hero.py` |
| `hero_charter_path_2pct` | The hero plus a compact second row: 2% of the EFT demonstrations relabelled for the profit-maximising crew, and the weakened rate | same script |

Provenance of every piece of text on the hero figure is in the docstring of
`figures/src/hero/plot_hero.py`.
