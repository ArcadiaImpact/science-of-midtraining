# Research log: grounded-monitor falsification

## 2026-08-06 — design before new sampling

The forced-choice allocation experiment #391 produced the cleanest directional
SDF-by-output-RL opacity pattern so far: UHR and P(undetected | hack)
interactions were positive while the violation-rate interaction was negative.
But the same run falsified its own monitor assumptions. Baseline sensitivity
was .1316, FPR was .1045, and action-only/no-scratchpad controls did not show a
positive interaction. Manual inspection found both generic treatment
scratchpads and candidate-grounding errors by the monitor.

Rather than alter RL after seeing that result, I freeze every policy checkpoint
and test the measurement explanation directly. The new monitor must attach an
exact candidate-naming evidence span to every claimed private fact. I also add
an 80-item pre-policy challenge containing explicit violations, clean selected
candidates, and other-candidate distractors. This should reduce the particular
false-positive path seen in #391 because a fact about A cannot condemn selected
B without a B-grounded quote.

The follow-up uses a new deterministic heldout2 allocation split so that policy
episodes are not reused, although the same source policies and environment are.
It is an outcome-motivated falsification, not a fresh independent
preregistration of the original hypothesis. The informative outcomes are
whether the grounded monitor itself passes, whether #391's interaction sign
survives, and whether improved precision trades away too much sensitivity.

## 2026-08-06 — completed result

The controlled challenge passed perfectly: 40/40 explicit selected-candidate
violations were found and 0/40 clean or other-candidate cases were flagged.
Exact grounding also reduced natural-policy baseline FPR from #391's .1045 to
.0096. But it overcorrected on recall. Only one of 33 heldout2 baseline hacks
was detected, for .0303 sensitivity, and the policy-baseline gate failed.

The original interaction attenuated but did not vanish. Grounded UHR
interaction was +.0722 with seed effects +.2667, -.1333, and +.0833. The
conditional-undetected interaction was positive in every seed (+.1875,
+.1137, +.1000), mean +.1337 with a three-seed interval entirely above zero.
Violation-rate interaction was +.0278 and mixed. Thus the most stable signal is
that irrelevant-SDF violations become more explicitly detectable with RL while
applicable-SDF violations remain almost completely opaque to a strict anchored
extractor.

The mechanism diagnostic is stark. All 5,400 learned judgments were valid JSON,
but only 494 of 7,137 claimed facts survived exact candidate/substring
validation. At step 16, accepted-claim rate averaged .0222 for applicable SDF
and .1561 for irrelevant SDF. The same monitor accepted 275 claims on the clean
synthetic challenge. The challenge was therefore too templated to predict
natural scratchpad recall.

I interpret this as a useful failed repair. Exact anchoring removes most false
alarms and preserves a smaller all-positive conditional interaction, but makes
the monitor too conservative to establish intervention success. A future
monitor needs span grounding with reference resolution and paraphrase support,
validated on natural held-out scratchpads rather than templated sentences.
Because this design was motivated by #391 and reuses its policies, that future
work should use new policies as well as a new monitor benchmark.
