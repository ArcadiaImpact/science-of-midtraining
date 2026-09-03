# RLVR trajectory figures

All RLVR stacked-area figures live here, alongside the campaign's other
ablation galleries:

- `direct/` — direct-generation RLVR trajectories.
- `thinking/` — native-thinking RLVR trajectories; kept separate because the
  4,096-token cap and truncation profile are not measurement-equivalent to the
  direct-generation instrument.

Each mode contains every arm x response-template split x clause split as PNG
and SVG. The response-template splits are trained, held-out, and pooled. The
clause splits are trained and held-out.

## Missing held-out-clause evaluations

The pinned RLVR battery contains only the five trained clauses. Its trained and
held-out labels refer to **response templates**, not clauses. Consequently,
every held-out-clause figure is an explicit “not evaluated” placeholder rather
than a zero-valued trajectory. The compact score tables retain `clause_split`
as a first-class field so a future battery containing held-out clauses can be
plotted without changing the figure code.

Source scores and generation tools remain with the RLVR experiment at
`dispatch_rlvr_gemma4_26b_v1/eval_scores/`, `collect_eval_scores.py`, and
`plot_eval_trajectories.py`.
