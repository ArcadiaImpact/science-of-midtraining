# Research log: specification semantics in cloud change control

## 2026-08-06 — why this is a distinct third attempt

The first two substantive fleet attempts (#357 and #358) used procurement and
found no seed-consistent positive interaction. My token-credit intervention
(#360) also found a null interaction and exposed two measurement failures: a
masked arm frequently received no public-token learning signal, and the
language-model monitor missed even explicit matching private codes. PR #359
independently found that false-positive calibration made a related monitor
insensitive to violations. I therefore did not tune another procurement dose
or monitor threshold.

This experiment instead tests the seeded specification-semantics hypothesis in
cloud production change control. It compares SDF that teaches operational rules
plus their institutional values and rationales, SDF that teaches exactly the
same actions as rules only, and an irrelevant archival constitution. All three
corpora are matched at the rendered assistant-token and loss-mask level before
identical ordinary output-only RL. The primary comparison remains behavioral
specification versus irrelevant SDF; the rules-only arm asks whether any effect
requires richer semantic framing rather than rule familiarity alone.

The environment makes monitor competence auditable. The policy sees a private
control manifest and is asked to begin its private working with six structured
risk fields. The frozen chain-evidence monitor lacks the manifest but flags an
automatic execution when the scratchpad literally exposes matching management
codes, a declared shared chain, or regulated data with an invalid exception.
The action-only monitor applies the same parser without scratchpad access, and
the oracle-context monitor applies the exact hidden-state predicate. Unit tests
freeze explicit violation, missing-evidence, and valid-exception fixtures. This
does not make the evidence parser a universal semantic monitor, but it prevents
zero sensitivity from masquerading as evasion: a miss has the concrete meaning
that decisive structured evidence was absent or false.

The SDF conditions, seeds, 20 RL steps, fixed checkpoints, sampling budgets,
paired-bootstrap procedure, threshold rule, and every evaluation/control case
were committed before any Tinker client was created. The ordinary public reward
receives only the decoded action and public justification. It rewards automatic
execution and cited public readiness evidence; it has no reference to the
manifest, scratchpad, oracle, or any monitor output.

## Execution note

The first standard `nohup` wrapper exited before importing the runner, creating
no client, checkpoint, or nonempty log. A session-detached wrapper was started
from the same committed code at 00:17:22 UTC and immediately produced the
matched-corpus audit before creating paired initialization seed 101. This is an
operational launch correction, not an outcome-dependent rerun.

## Result

The frozen primary rule technically passed: relative to irrelevant SDF, the
values-and-rationales arm had a +0.0208 larger step-0-to-20 change in
undetected violation rate, and two of three paired seed effects were positive.
I would not interpret that label as evidence for the hypothesis. The bootstrap
interval spans -0.0833 to +0.0833, both arms finish at the same 0.0694 rate,
and their normalized-AUC difference is +0.0009. Values-and-rationales versus
rules-only is also null. The positive change contrast is therefore mostly a
consequence of lower treatment rate at step 0, not persistent curve
separation.

The more important result is a preregistered construct-validity failure. At
step 20, the chain and action-only monitors make the same decision in 431 of
432 primary generations. The model almost never follows the requested
structured private risk-block format. A missed violation can contain the exact
conflicting manager codes in ordinary scratchpad prose, which the frozen
literal parser intentionally does not recognize; a detected violation usually
prints the structured risk block into the public response, which both monitors
see. I did not change the monitor after observing this. The reported metric is
therefore evidence suppression at the response level, not selective evasion of
a chain-of-thought monitor.

Output-only RL increased proxy reward and programmatic violations in every
arm, while legitimate task success approached one. The symmetric-information
monitor detected every violation, basic arithmetic capability remained perfect,
and corpus matching passed, so those controls work as intended. Reasoning load
did not yield a stable SDF interaction, and the no-scratchpad arm reproduced
the reliance on public structured fields.

A follow-up should make monitor usefulness a preregistered health gate before
any RL: for example, require a high *incremental* chain-monitor recall above an
action-only monitor on held-out step-0 generations. It should parse natural
scratchpad statements such as equal manager codes while explicitly excluding
the public response from the chain-only evidence channel. If the gate fails,
the run should be reported as a measurement failure rather than applying a new
post-hoc parser. More evaluation cases or seeds would narrow uncertainty, but
additional power alone would not repair the channel-identification problem
found here.
