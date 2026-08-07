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

That fallback also failed on the next committed canary because the decoded
special token was not rendered as the literal delimiter string. The extractor
still anchored all six fields, confirming that generation obeyed the ledger.
I preserved the second log and changed extraction to use the protocol's
unambiguous API boundary: the private-stage sampler result is decoded as
private work and the subsequent sampler result as public output. This is less
heuristic than reparsing their concatenation and again leaves all sampled
tokens and scientific thresholds unchanged.

The API-boundary canary then recovered a 735-character private ledger with all
six fields, but its 256-token public continuation ended before valid JSON. This
is a capability failure rather than evidence about the intervention. I
preserved the third log and raised the public budget to 512 for both standard
and ledger modes. Public actions remain paired and the preregistered behavior-
preservation gates remain unchanged.

At 512 tokens, the next canary still produced narrative text rather than the
required JSON, although its final lines contained the correct DECLINE and
reason. I preserved this fourth log and added `Final public JSON only:` as a
shared continuation cue in both modes. It carries no action or private fact;
its purpose is to hold the public transition format constant so the only
between-mode change remains the private ledger instruction.

## 2026-08-07 — paired result

The final canary passed, and the frozen full grid completed 8,100 policy rows,
24,300 post-hoc monitor rows, and 160 independent calibration rows. The
calibration extractor again passed perfectly: 80/80 trigger cases, 0/80 false
positives, and 160/160 valid JSON.

The intervention did not isolate disclosure. Its mean fixed-grid
undetected-hack reduction was exactly 0.100, with every seed negative, but the
conditional nondetection reduction was only 0.057 and crossed zero across
seeds. More importantly, the ledger reduced P(hack) by 0.124, increased
legitimate task success by 0.120, and reduced proxy reward by 0.108. Those
changes all violate the preregistered behavior-preservation gate. It produced
more extractor claims but fewer validated evidence anchors, and the chain
monitor's sensitivity stayed essentially unchanged at 0.202 versus 0.204.
The missing-evidence mechanism from #407 was therefore not repaired.

The public-format capability gate also failed: valid-JSON rates were only
0.811/0.800 in the standard/ledger rules-only arm and 0.878/0.908 in the
values-and-rationales arm. The practical lesson is stronger than a simple
null. Asking a generative policy to structure its private work is a behavioral
intervention with action, reward, and format consequences; it is not a neutral
logging change, even when sampling budgets and the public transition are
matched. A cleaner next test would compute a non-generative fact record from
the supplied private state or enforce a schema outside the policy, rather than
spending the policy's fixed private-token budget on a second task.

After the raw run, I added ledger-specific mean curves, uncertainty, AUC, and
time-to-threshold to the compact analysis export. I also applied a numerical
tolerance to the inclusive 0.100 boundary because the three-seed floating
mean was represented as 0.09999999999999999. This changes only the directional
sub-gate from false to true; conditional, preservation, capability, and
headline support remain false.
