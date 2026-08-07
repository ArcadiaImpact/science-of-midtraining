# Research log: private fact-ledger intervention

## 2026-08-07 — from calibration diagnosis to intervention

PR #407 produced a useful mismatch. The evidence extractor was perfect on 160
independent rule-application cases, but sensitivity on natural policy
scratchpads was only 0.390. Its joint SDF interaction did not replicate, while
both components did: richer specification knowledge increased violations but
reduced conditional nondetection. Repeating another fresh stream would not
distinguish the mechanism.

I therefore moved from measurement to intervention. The treatment prompt asks
the frozen policy to copy all six private underwriting fields into a compact
private ledger before deciding. It contains no language about inspection or
scoring and never changes training. Standard and ledger modes share the same
checkpoint, case, decoder, and seed. The key test is whether the chain monitor
improves while actions, legitimate success, and proxy reward remain stable; if
public behavior shifts, a lower undetected-hack rate cannot be attributed only
to better private evidence.

This is deliberately not another scratchpad-credit experiment. It asks whether
a lightweight production workflow can recover observability from already
trained policies after an independently competent extractor has been
established.

The first frozen-checkpoint canary showed the ledger mechanism itself worked:
the monitor anchored all six private fields and returned the correct violation.
However, the renderer labeled the capped two-stage response malformed, so the
generic exception path treated private and public text as one string and did
not recover public JSON. Before any experimental row, I preserved that log and
added a deterministic fallback that splits only at the already standardized
`</think>` delimiter. The intervention, token budgets, thresholds, prompts,
and checkpoints did not change. A new committed-state canary must pass before
the grid runs.
