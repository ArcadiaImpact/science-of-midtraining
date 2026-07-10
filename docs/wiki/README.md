# Research wiki

Durable knowledge distilled from this repo's experiments. PR descriptions and
reports record *what we ran*; this wiki records *what we currently believe*,
with provenance. If a claim matters and it isn't here, it isn't yet knowledge.

## Pages

- [config-performance.md](config-performance.md) — measured performance of
  each spec's default config (install, side effects, caveats).
- `usa-training-dynamics.md` — (incoming: issue #171) install saturation +
  metric co-evolution over training.
- `eval-anchors.md` — (incoming: `exp/aff-anchor-reconcile`) canonical base /
  trained rates per eval scorer, sample sizes, noise bands.

## Editing rules

1. **Every claim cites provenance** — a PR, a results file, or a committed
   experiment dir. No unsourced numbers.
2. **Supersede, don't erase.** When new evidence overturns a claim, strike it
   through (`~~old claim~~`) with a pointer to what replaced it. The wiki's
   history of being wrong is part of the knowledge.
3. **State the strength.** Distinguish *solid* (multi-seed, CI-backed),
   *directional* (single seed / within ~2 SE), and *anecdotal*. A number
   without an n and a seed count is an anecdote.
4. **Within-harness comparisons only** unless a page explicitly establishes
   cross-harness calibration (see `eval-anchors.md`).
5. **Experiment PRs in active lines should touch the wiki** — new/updated
   claims land in the same PR as the evidence for them.
