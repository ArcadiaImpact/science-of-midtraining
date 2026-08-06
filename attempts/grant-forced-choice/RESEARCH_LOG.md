# Research log: forced-choice grant allocation

## 2026-08-06 — design before training

The content-ranking transfer in #389 found a negative SDF-by-RL interaction,
but its `LIMIT` action let every step-0 policy avoid all violations. That
preserved safety while making the preregistered baseline-sensitivity check
unidentifiable. Earlier procurement and lending experiments (#357 and #365)
did elicit violations, but a binary decline action still confounds behavioral
knowledge with general conservatism.

I therefore changed the choice structure rather than an optimizer setting. A
resilience-grant officer must select one of three candidates. Public candidate
A is always the most attractive and earns the largest public-only reward, but
is privately ineligible in two thirds of cases; candidate B is the correct
fallback. This preserves an exact hidden-state oracle and realistic information
asymmetry while preventing a policy from succeeding by refusing every case.

The scientific question is whether applicable behavioral knowledge becomes
more evasion-prone under pressure toward the public frontrunner, or instead
continues to protect the eligibility computation. The latter would be a strong
boundary on the proposed negative externality because every valid decision now
allocates the scarce resource. All corpora, cases, seeds, condition order,
training settings, monitors, checkpoints, estimands, and gates are fixed before
the paid run.

## 2026-08-06 — completed result

Forced choice did what the content-ranking safe default could not: all six
step-0 cells contained hacks (38 total), so sensitivity had a real denominator.
The preregistered directional effects were positive. Applicable SDF had a
+.1444 undetected-hack-rate interaction with paired seed effects +.2167,
-.0500, and +.2667. P(undetected | hack) interaction was +.3147, while the
violation-rate interaction was slightly negative (-.0500). This is the desired
pattern for a monitorability effect rather than a simple increase in harmful
actions. The UHR threshold was reached earlier under applicable SDF in all
three seed pairs.

The full support rule nevertheless failed. The literal extractor detected only
5 of 38 pooled baseline hacks (sensitivity .1316), below the .50 floor, and made
7 false alarms among 67 correct baseline allocations (FPR .1045), just above
the .10 ceiling. The primary effect was also scratchpad-specific: the action-
only UHR interaction was -.0389 and the no-scratchpad action-only interaction
was -.0167. Applicable-SDF private work increasingly used generic references to
"private issues" or argued that private facts should not affect the public
choice, whereas irrelevant-SDF work more often restated the selected
candidate's exact conflict or suspension. Manual examples support that
mechanism, but also reveal extractor misses on some explicit treatment phrases
and grounding errors that help explain the high control FPR.

The forced-choice intervention therefore exposes a promising but not validated
interaction: output-only pressure makes applicable-SDF scratchpads less useful
to this literal monitor even though it does not increase violation growth.
Future work should freeze a stronger evidence-anchored monitor and validate it
on a separate challenge set before training. It should also make the initial
public JSON contract easier for the irrelevant arm; step-0 validity averaged
only .1889 in control versus .5111 in treatment, although both reached nearly
one by step 16. I would not tune either issue on these outcomes.
