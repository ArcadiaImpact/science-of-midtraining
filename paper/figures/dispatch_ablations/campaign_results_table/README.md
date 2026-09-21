# Campaign results table

Every campaign model and the special midtrains beside them, in one LaTeX
table: one row per (base model × midtrain arm), grouped by base model; one
column per EFT treatment; one cell = the run-level split of conflict episodes
into **Charter / coin / other crew / unparseable**, in percent of runs.

`campaign_results_table.tex` is the deliverable — an `\input`-able fragment
that defines its own colours and `\dcell` macro with `\providecommand`, so it
drops into the paper whether or not `coincharter.sty` is loaded.
`campaign_results_table.pdf` / `.png` are a standalone preview, not paper
output.

## Regenerate

From the checkout root:

```bash
uv run --extra dev python paper/figures/dispatch_ablations/campaign_results_table/src/make_campaign_results_table.py
```

Every run **rehydrates** — scores come from the public Hub mirror
`arcadia-impact/scimt-dispatch-clean-v1` through `_shared/common.py`, which
prefers the committed `results_grid/scored/` tree when this directory sits in
a scimt checkout that has one and otherwise downloads. The two are
byte-identical, so a colleague with only the paper repo, no checkout, no GPU
and no credentials gets exactly these numbers. The run also re-freezes
`src/data/campaign_results_table.json`, so the committed extract and the
committed table can never disagree.

`--frozen` renders from that extract with no network. `--no-compile` skips
pdflatex. `--slice` reads a different slice (it refuses to pair `--frozen`
with a slice the extract does not hold, and writes its own extract and `.tex`
so it can never overwrite the default table's numbers). `--order arm` groups a
base model's rows by arm rather than by budget.

## Presentation

The fragment needs `booktabs` and `xcolor`; `colortbl` (xcolor's `[table]`
option) is optional — without it the budget separators come out black rather
than grey. Everything else it defines itself with `\providecommand`, so the
paper's own `\definecolor`s win if it has them.

- **Numbers are boxed and right-aligned.** Each of the four sits in a
  `\makebox` as wide as the widest number *in this table*, measured from the
  data rather than assumed, so ones, tens and (if a slice ever produces one)
  hundreds line up down a column as well as across a cell.
- **A thin grey rule separates presented-token budgets.** A block is one
  budget, and the qualifier is part of its key: the ablation rows share a
  budget with a standard trio (50M no-worked-examples, 190M clause-asymmetric)
  and get their own block rather than extending the trio above them.
- **Arms read Charter → control → coin**, so the filler baseline sits between
  the two directional corpora it separates. That order lives in
  `rows.ARM_ORDER` and is applied at render time, not baked into the extract.
- **Charter and coin are tinted** in the corpus column, at 14% of the paper's
  palette; control is left plain, which is the distinction the tint is there
  to draw — control midtraining is filler, with zero directional tokens.

## What is in it

| group | rows |
|---|---|
| Gemma 3 4B | 1M / 5M / 50M × charter, coin, control |
| Gemma 3 12B | 1M / 5M / 19M / 50M × three arms, **+ 50M no-worked-examples** (charter, coin) |
| Gemma 3 27B | 5M / 19M / 50M / 190M × three arms, **+ 190M clause-asymmetric** (charter) |
| GLM-4.5-Air | 20M legacy / 190M × three arms, **+ 190M clause-asymmetric** (charter), + 1B (charter only) |
| Gemma 4 26B-A4B | 190M delta graft × three arms |

228 of 235 cells are populated. The seven blanks are facts about the campaign,
not gaps in the load: the two clause-asymmetric rows ran only the
agreement-only and 100%-Charter cells, and `glm45_air_20m_legacy`'s
`charter_only-step512` was never scored (the collector wrote an empty
endpoint). An endpoint that holds *other* slices but not the requested one is
a loud error instead — that would mean the slice name is wrong, or a re-score
dropped a read every sibling still has.

Three source shapes feed the rows, all from the same mirror and all documented
in `src/rows.py`: the uniform campaign grid (`scores/<profile>/<arm>/eval.json`),
the clause-asymmetric package (`scores/gemma27b_clause_asym_v1/comparison/all_scores.json`,
which carries **both** the GLM original and its Gemma-3-27B replication), and
the graft's flat row list (`scores/gemma4_26b_a4b_graft/campaign_battery_scores.json`).

## Reading it

- **Every cell is the converged 2-epoch endpoint (step 512).** Mixing step 256
  and step 512 reads would put two different amounts of training on one row.
- **The ±2% columns are not homogeneous.** Follow-up #1c re-drew those cells
  from a balanced five-clause draw on 29 arms; the nine Gemma-3-4B arms it
  never covered still carry the original narrow single-clause draw. Those rows
  are starred, and the star is stamped from each file's own
  `meta.twopct.state`, never inferred from the profile name.
- **One seed per cell, throughout.** Measured run-to-run SD on the Charter
  share is ≈9pp; a smaller difference is not a difference.
- `other` is taken by subtraction, so the four numbers in a cell always close
  at 100 and anything the scorer reported outside the four named categories
  lands where a reader would put it.
- The graft's responses are scored by two parsers; the table reads `legacy`,
  the campaign's own, which is what puts those cells on the same axis as every
  other row.

The default slice is `eval_trained_conflict__heldout` — the five clauses the
EFT trains on, on a template surface neither midtraining nor EFT ever showed.

## Spot checks

Two numbers here are quoted independently elsewhere in the repo, and match:

| cell | table | quoted |
|---|---|---|
| `gemma3_12b_50m_4ep` charter, +2% / −2% | 88 / 9 | 87.5 / 9.3 (`MODEL_REGISTRY.md`, #1c draw) |
| 12B 50M charter, agreement-only, with vs without worked examples | 65 vs 37 | 65% vs 37% (`paper/README.md`, row 7) |

## Not in it

The graft arms also carry an RLVR policy (direct and thinking, phase 768).
RLVR is not an EFT treatment and exists for exactly one of the five groups, so
a column for it would be empty on four of them; it belongs in its own figure
(`dispatch_rlvr_training_190m_*`). The dose-ladder rungs (±0.5%, ±1%, ±5%)
are likewise left to `dispatch_dose_*`.
