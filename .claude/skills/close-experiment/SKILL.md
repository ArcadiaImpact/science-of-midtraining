---
name: close-experiment
description: Close out an experiment run - write the report in house style, append the research-log entry, update roadmap decision triggers, file follow-ups. A result does not exist until this has run.
---

# Close experiment

Run when an experiment (or a phase of one) has produced its numbers. The
output is three artifacts, in this order:

## 1. `report.md` (house style — `experiments/_template/report.md`)

- **Headline first**: the finding as a sentence, then the table/figure.
  Numbers carry denominators (`n_aligned/n`, `valid_rate`) and uncertainty
  (state the SEM basis; note clustered samples).
- **Hypothesis verdicts** against the pre-registered spec — supported /
  refuted / mixed, including nulls. Never reframe a refuted hypothesis as if
  it were the prediction.
- **Confound endpoints** and **caveats** get their own sections; the caveats
  section is where "don't generalize this yet" lives (seed limits, scale
  stand-ins, setting artifacts).
- **Reproduce block**: exact commands + artifact pointers. Verify each
  headline number actually traces to a listed artifact before writing it.
- Mark `preliminary: true` until confirmation seeds / replication (roadmap
  R2/R3-style) exist.

## 2. `log/` entry (template: `log/TEMPLATE.md`)

Date, commit hash, exact commands/seeds, cost actually spent, headline,
takeaways, decisions. Append the link to `RESEARCH_LOG.md`. This is
append-only — corrections are new entries.

## 3. Roadmap update

Re-read every `ROADMAP.md` entry whose **decision triggers** mention this
experiment's outcomes; reorder/kill/expand accordingly and set this item's
status (`done(<log entry>)`). File cairn issues for follow-ups the result
surfaced. This step is what makes the programme reactive — do not skip it
because the result "changed nothing" (say so explicitly in the log entry
instead).

## Then

Open/refresh the experiment's PR with the report as the body, and run
`skeptic-review` on it (spawn the `skeptic` agent — Opus) before requesting
human review. Offer to mirror the report to the lab-notes site if the user
wants it published.
