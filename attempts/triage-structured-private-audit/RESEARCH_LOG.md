# Research log: structured private audit

## 2026-08-07 — moving from mechanism to remediation

PR #414 isolated a private representation effect with fixed public actions:
safety-last audits raised undetected-hack rate by .070 and conditional
nondetection by .202. Its evidence diagnostics suggested a concrete remedy.
The safety-last audits were shorter and preserved fewer validated private
facts, so I replaced freeform audit instructions with an explicit field
checklist while leaving the difficult safety-last input and public record
unchanged.

This differs from PR #409's generated fact ledger because the decision is
already filed and copied from one frozen source in both arms. The checklist
therefore cannot improve results by changing actions. It differs from #414's
fact-order experiment because private-note order is now identical; only the
audit representation request changes.

Before any live call, I fixed thresholds for meaningful recovery, a false-
positive noninferiority margin, the full three-arm/three-seed/five-checkpoint
grid, and all capability gates. The experiment intentionally uses the same
cases as #414 because it is a remediation test against that measured failure
mode, not an independent-case replication.

## 2026-08-07 — a large recovery with a real specificity cost

The first paid canary exposed a response-boundary failure: a long checklist
request produced no parseable private work. I recorded that failure, reduced
the instruction to one bounded seven-label line, committed the change, and
required a second canary before the full run. The second canary produced
nonempty work in both conditions, valid JSON, and all seven extractable fields.

On the full fixed grid, the checklist reduced undetected-hack rate by .172 and
conditional nondetection by .470, with the same direction in every seed and
exactly zero public-behavior change. It more than doubled detections (304 to
652 of the same 857 violations) and raised anchored evidence. But false-
positive rate rose by .044 on the preregistered paired-cell estimand, beyond
the allowed .03. I therefore report the full support rule as false. The result
changed my view of the intervention: explicit private structure is genuinely
load-bearing for coverage, but a literal checklist prompt is not ready as a
remediation without a grounding or calibration step.

The checklist effect is broad across SDF arms and both easy and compositional
cases. Its SDF-by-RL interactions remain heterogeneous, so it does not rescue
the broader causal claim about behavioral-spec midtraining. A next experiment
should validate each checklist value against quoted note spans or use a
programmatic private-state serializer; the key test would be whether that
retains the sensitivity gain while bringing false positives below the fixed
margin on new cases.
