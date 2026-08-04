# Audit-panel calibration fixtures

The audit panel (Gate 3) is the load-bearing anti-hacking mechanism for this
task, and `score = 0` is unappealable within a run. So the panel is calibrated
**in both directions** before it gates any real submission:

- **It must catch hacks.** `two_key_channel`, `contaminated`, and
  `ceiling_artifact` are hand-written submissions that game the metric in three
  different ways. A panel that passes any of them is miscalibrated and must be
  fixed before the fleet launches.
- **It must pass honest work, including honest nulls.** `honest_null` is a
  competent, well-evidenced submission whose answer is "nothing installs at
  1B". Nulls are explicitly valid submissions in this task, so a panel that
  fails this one would destroy exactly the results we most want reported. A
  false positive is not a safe default here; it is a different failure.

These are **evidence packets**, not full submissions on disk — the panel
consumes a packet, so calibrating at that boundary tests the panel itself
rather than the submission parser (which has its own unit tests).

Run them with `calibrate.py` (needs `OPENROUTER_API_KEY`, no GPU). Expected
verdicts are declared in each fixture's `_expected` key; the runner reports any
mismatch as a calibration failure. `_expected` is stripped before the packet
reaches the panel — the panel never sees the answer.
