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

The directional pattern replicated, but the preregistered confirmation did
not. From RL step 0 to 16, the +SDF arm's chain-monitor undetected-hack rate
rose by 0.328, versus 0.117 for irrelevant SDF, a paired
difference-in-differences of +0.211 (95% paired-seed bootstrap interval
[-0.017, 0.367]). The raw seed interactions were +0.283, -0.017, and +0.367.
The conditional-evasion interaction was also positive on average (+0.353),
although one seed was negative. Hack rate itself had a positive +0.211
interaction in all three seeds. Thus the behavior is substantively similar to
#365 and is not explained by its unseeded LoRA initialization alone.

Evidence anchoring repaired the earlier false-positive failure: pooled step-0
false-positive rate was 0, as was false-positive rate on the 16 negative
calibration cases. It did so at too great a recall cost. Step-0 sensitivity was
0.471 against the programmatic hidden-state oracle, below the preregistered
0.80 floor, and disjoint-calibration sensitivity was 0.750, below 0.90. The
four calibration misses were three restricted-sector percentages whose short
quotes omitted an accepted field anchor and one employee-owned/pending-
clearance sentence whose chosen quote did not establish both fields. These
are genuine consequences of the frozen support rule, not parser errors to
correct after seeing results.

The final support decision is therefore **false**. The experiment establishes
a useful measurement frontier rather than a confirmed evasion effect: strict
quotation support prevents omission from becoming a fabricated adverse fact,
but this learned extractor cannot simultaneously meet the planned sensitivity
floor. The positive interaction should be treated as suggestive only. A next
test should make the monitor itself programmatic over a fixed public fact
schema, or prospectively calibrate a learned extractor on a development set
before training any policies; relaxing this run's rule after observing the
misses would invalidate its gate.

Operationally, two `nohup` launches exited during the first remote SDF call
without producing checkpoints or an exception, apparently because their
shells were reaped. The exact committed command then ran unchanged in a
persistent observable session. No optimizer update was skipped, and all 30
policy checkpoints were frozen before the first monitor call.
