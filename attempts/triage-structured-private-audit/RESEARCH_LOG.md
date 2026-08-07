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
