# <Experiment title> — spec / pre-registration

**Status:** spec / pre-registration · **Setting:** <belief / value / …> ·
**Substrate:** <model + method + where it trains> · **Roadmap:** <Rn> ·
**Reuses:** <existing harnesses/data this builds on>

> Commit this spec (and the exact commands) BEFORE any headline number is
> produced. Every section below is mandatory; "TBD" is only acceptable with a
> phase-0 item that resolves it.

## Problem

<What question this answers and why it matters to the programme. Cite the
roadmap entry and any results this builds on.>

## Hypotheses (pre-registered)

- **H1 — <name>.** <Prediction, with direction.> Null = <what a null looks
  like, and why the null is itself informative>.
- **H2 — …**

We commit to reporting however these resolve.

## Design

### Arms and matched controls

| arm | recipe | matched control |
|---|---|---|
| … | … | <same recipe minus the treatment — never an off-the-shelf model> |

Confound/sanity endpoints (evaluated, not compared as arms): <e.g. the
"just-treatment" endpoints>.

### Metrics

- **Headline metric** with its exact denominator (e.g. B = n_aligned/n with
  valid_rate reported) and where it's computed (which module/scorer).
- **Matching protocol:** how arms are equalized on-distribution before OOD
  numbers are read (metric, tolerance ε, what happens on undershoot).
- **Capability guard:** what is measured at every endpoint to stop reading
  effects off a damaged model.

### Seeds & statistics

Seed 0 first; the gate for +N seeds (<threshold vs eval-set SEM>). Note any
clustered-sample structure and how uncertainty is computed.

### Phases

- **Phase 0 — smoke.** <The tiny-budget end-to-end path.> **Gate: nothing
  full-scale runs before this passes.**
- **Phase 1 — <main comparison>.** <Grid; the gate for extending.>
- **Phase 2 — <follow-on, if gated>.**

### Budget & kill criteria

≈ <GPU-hours / $ / API tokens> per phase. Kill if: <conditions under which we
stop and re-plan rather than spend more>.

## Reproducibility contract

- Data staging deterministic + committed (seeds, id lists).
- Exact commands in this spec or the README.
- Raw responses persisted separately from scoring; results as
  `results.jsonl`; large artifacts → GCS pointer under `$SCIMT_GCS_PREFIX`,
  never bytes.
- Close-out: `report.md` + `log/` entry + `ROADMAP.md` trigger update (use
  the `close-experiment` skill).
