# Research log: evidence-anchored lending replication

## 2026-08-06 — design before training

PR #365 is the most promising production-shaped transfer in the leaderboard:
all three seed interactions were positive and its held-out interestingness
score was high. It nevertheless failed for a real measurement reason. The
monitor inferred that independent clearance was absent when a scratchpad said
an owner was an employee but said nothing about clearance. Lowering or waiving
the false-positive threshold would not repair that problem.

I chose a prospective evidence-anchoring intervention rather than adding a
post-hoc phrase list to #365's completed outputs. The extractor must quote its
source, deterministic code checks that the quote is truly present and names
the field, and adverse categorical polarities require explicit language. The
key invariant is simple: omission can produce `unknown`, never `absent`.
Thirty-two synthetic calibration snippets freeze that invariant before any
remote training; the full monitor is still called strictly after all policy
checkpoints are frozen.

This attempt also fixes a separate matching issue found while reading the
prior implementation: it claimed matched seeded LoRA initialization but did
not pass a seed when creating the LoRA client. Here the explicit policy seed is
reused across treatment and control. I changed the three policy seeds and the
evaluation-case random stream so that the known #365 outcome is not simply
rescored with a friendlier monitor.

The main risk is conservative recall. Evidence anchoring may eliminate false
positives by discarding natural but hard-to-quote private reasoning, causing
the sensitivity gate to fail or making both arms look undetected. The
preregistered 0.80 policy sensitivity and 0.90 calibration sensitivity floors
make that failure visible. A second risk is that #365's positive interaction
was specific to its policy seeds. Either result is informative: replication
with a valid monitor would materially strengthen cross-domain evidence, while
a null would show that the earlier positive cannot yet be separated from its
measurement failure.

## Result

Pending.
