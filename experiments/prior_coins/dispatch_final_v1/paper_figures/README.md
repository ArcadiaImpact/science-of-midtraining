# paper_figures — headline figures for the ICLR submission

One standalone script per headline figure, plus `common.py` for the two things
that must not drift between figures. Written so this whole directory can be
copied into `scimt-paper/` and still work.

The opposite contract from `results_grid/`'s plotters: those are a survey
gallery that redraws everything from whatever has landed. These are
few, named after the figure they produce, and each one has a caption to earn.

## The two conventions

**1. Every figure is authored at the real page width.** ICLR 2027's
`\textwidth` is `5.5 true in` (`iclr2027_conference.sty:49`), so a 9pt label in
the figure is a 9pt label on the page, and text sizes can be eyeballed against
the body copy. Two consequences, both enforced in `common.py`:

- `common.save()` does **not** tight-crop. `bbox_inches="tight"` shrinks the
  canvas to the ink — figure 2's first draft came out 5.235in — and then
  `\includegraphics[width=\linewidth]` scales it back up, drifting every font
  size by that ratio. Lay out with `common.margins()`, which takes **inches**,
  and let the canvas stay the size you asked for. Every run prints the canvas
  size and the `\linewidth` fraction to include it at.
- Include figures at `width=<frac>\linewidth` matching what the script printed
  (`1.000` for a full-width figure). Never rescale in LaTeX to fix a size
  problem — pass `--width-frac` / `--height` and re-render.

**2. Scores load local-first, then rehydrate from the Hub.** `common.load_scores()`
reads the committed `results_grid/scored/` tree when it can see it, and
otherwise downloads from the **public** mirror
`arcadia-impact/scimt-dispatch-clean-v1` (anonymous read works — no token
needed). The two are byte-identical: `build_clean_repo.py` copies the git tree
verbatim, and all 162 mirrored files were verified blob-for-blob on 2026-09-09.
So a colleague with only the paper repo re-renders exactly the same numbers,
and can re-lay-out a figure without access to this repo.

`SCIMT_SCORES=/path/to/scored` overrides the local root, for a worktree or a
rescore. Every script prints where each number came from (`scores: local` /
`scores: hub`) and names the non-local sources, so a render is self-describing.

The mirror is rebuilt by `build_clean_repo.py` and can lag git — as of
2026-09-09 it is missing `scored/ablations/glm_threeway.json` (collected in
`72a8cefc`) and `scored/glm45_air_20m_legacy/charter/eval.json` (a dropped
upload; its coin and control siblings are there). A figure that needs a
lagging file carries its own fallback; see the 80:10:10 ablation.

The one thing with no Hub fallback at all is the archived pre-#1c 2% draw
(`scored/legacy_narrow_2pct/`), which `build_clean_repo.py` deliberately skips.
Asking for it off-checkout is a loud error, not a silent canonical read.

`data/` is a scratch cache for scores fetched straight from a source release,
written on demand and safe to delete. Nothing there is a source of truth; if a
cell has a collected home in `scored/`, read that instead.

### Naming

`figureN_*.py` for a numbered figure in the submission; `dispatch_ablation_*.py`
/ `<setting>_ablation_*.py` for a figure that supports one without owning a
number. Renaming a script is cheap; renumbering half of them because the
submission reordered is not.

## Figures

| script | figure | data |
|---|---|---|
| `figure2_glm_2pct.py` | asymmetric 2% conflict AFT flips the prior | `scored/glm45_air_190m/{control,charter,coin}/eval.json` |
| `dispatch_ablation_balanced_80_10_10.py` | symmetric 10/10 conflict AFT compresses it instead | the same, plus `scored/ablations/glm_threeway.json` |

### figure2_glm_2pct.py

Replaces `Tikz_Figs/results_preview.tex`, whose caption asks for exactly this.
Five stacked bars grouped by midtraining arm; each bar is the run-level split
of what the model chose on conflict episodes.

```sh
uv run --extra dev python figure2_glm_2pct.py                  # -> figures/
uv run --extra dev python figure2_glm_2pct.py --outdir ../../../../scimt-paper/fig
```

Flags worth knowing: `--twopct legacy` renders the superseded single-clause
draw (local checkout only), `--footnote` stamps the setup under the axes for
screen reading, `--ci` adds a Wilson interval on the charter proportion, and
`--control-line` rules the plot at the control arm's charter rate.

**The legacy render reproduces the TikZ placeholder exactly** — 37/8/55,
90/3/7, 61/5/34, 5/3/92, 14/17/69 — which confirms the committed placeholder
was built on the pre-#1c draw, and that the canonical render is its
replacement rather than a different measurement.

### dispatch_ablation_balanced_80_10_10.py

The transpose of figure 2, and the contrast that makes it read. Six bars
grouped by AFT mixture, each group holding all three midtraining arms.
`balanced_80_10_10` is 6,554 agreement / 819 coin / 819 Charter — conflict
data that speaks about the conflict without taking a side.

The two figures answer different questions and get different answers:

| AFT mixture | charter-rate spread across arms |
|---|---|
| agreement (no conflict data) | 84.7pp (89.6 / 37.0 / 4.9) |
| 80:10:10 (symmetric conflict) | 20.3pp (58.1 / 46.9 / 37.8) |

Asymmetric 2% *flips* the prior past the control; symmetric 10/10 *compresses*
it toward the control but preserves the ordering. Both are 8,192-row AFT on
the same parent.

**Two provenance wrinkles this figure carries, and the caption should say so.**
The `balanced_80_10_10` scores live in `scored/ablations/glm_threeway.json`,
their own collected document rather than a member of `glm_contamination.json`:
`twopct.apply` filters by endpoint *family*, so a two-sided cell sitting beside
the one-sided 2% cells would be one rename away from being substituted for one
of them. That collection is newer than the public mirror's last rebuild, so
when it is in neither the local tree nor the mirror the script falls back to
the raw `scimt-dispatch-final-v1-glm` release it was collected from — verified
byte-identical on the plotted rates. Drop the fallback (and `common.load_hub_json`
with it, if nothing else uses it) once the mirror carries the collection.

And the two groups sit on **different sampling backends** — agreement is eager,
80:10:10 is graphs/split-K-1, measured pooled offset −0.80pp charter / +1.00pp
coin. Small against the ~9pp seed SD, but it is a real seam *between* the
groups (not within either), so a reader comparing across the gap should know.

## Fixed coordinates, and why

- **`*-step512`** — the converged 2-epoch AFT endpoint. Step 256 exists for
  some cells; mixing them puts two different amounts of training on one axis.
- **`eval_trained_conflict__heldout`** — clauses seen in AFT, templates not.
  Held-out surface, trained rule: reads which motivation transferred without
  confounding it with rule generalisation.
- **`other` is by subtraction** — `conflict_runs.rates` splits four ways
  (charter / coin / malformed / other) and the bar shows three, so everything
  that is not charter or coin is folded in. The bar closes at 100% instead of
  drifting on rounding.
- **Colour** comes from the paper's `coincharter.sty`. Note `main.tex`
  currently re-`\definecolor`s charter/coin to the seaborn-colorblind pair
  *after* loading the package, so the compiled PDF uses those; the difference
  is imperceptible, and removing the override from `main.tex` would make them
  exact.

## Caveats that belong in captions

One seed per cell throughout; measured run-to-run SD is ~9pp on the primary
metric. The run-level `n` counts 3 runs per episode, so runs are not
independent and a Wilson interval on them is optimistic — which is why `--ci`
is off by default. See `../MODEL_REGISTRY.md` for the full set.
