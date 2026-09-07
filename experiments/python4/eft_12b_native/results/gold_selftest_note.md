# Gold self-test note — tacov:10520 (registered before the battery lands)

The 12B battery's pod-side gold self-test returned **2,047/2,048** (the 31B
battery's standard was 2,048/2,048). The single failure, from the attempt-1
pod (run `20260907T145446Z`, aborted later for the unrelated endpoint_port
config gap):

```json
{"category": "held_in", "failure_reason": "runtime", "problem_id": "tacov:10520", "warnings": []}
```

- **Why:** `runtime` — the gold solution raises at runtime under the p4_boa
  grader (NOT a timeout; policy was 10 s + 20 s serialized retry). No prior
  appearance of this problem id anywhere in eval_v3 history; prior batteries
  recorded clean self-tests.
- **Excluded or retained: RETAINED.** The self-test is an abort gate
  (>= 99.5%), not a problem filter — eval_v3 keeps all 2,048 rows in every
  condition's denominators. Consequences stated up front: any model "failure"
  on tacov:10520 may reflect grading-environment noise for that one problem;
  it is identical across all six conditions, so within-table lifts are
  unaffected; held-in denominators carry a <= 0.1 pp asymmetry vs the 31B
  battery.
- **Reproducibility check:** the attempt-2 pod re-runs the self-test from
  scratch; its result is appended below when it lands.

## Attempt-2 self-test — RESOLVED FLAKY

The live battery pod (run `20260907T150202Z`) re-ran the self-test from
scratch: **2,048/2,048, zero failures.** Same dataset revision, same grader
config, same gold — different pod. So the attempt-1 `tacov:10520` runtime
failure was a nondeterministic one-off, not a gold defect. The battery that
produces the table has a CLEAN self-test and its denominators match the 31B
battery's exactly; no exclusion question remains. (Attempt-1's receipt is
kept above as the record of why this note exists.)
