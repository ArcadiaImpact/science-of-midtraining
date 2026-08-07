# Research log: independently calibrated dense-model replication

## 2026-08-07 — isolating extractor competence from policy disclosure

PR #390 was the requested dense Qwen3.6-27B attempt. Its values-and-rationales
minus irrelevant-SDF interaction in undetected-hack rate was negative in every
seed, opposite the earlier 8B result. However, the headline gate failed because
the chain monitor detected only 63% of baseline violations. That population
mixed monitor competence with the policy's willingness to state decisive
private facts, so it could not tell whether the extractor misunderstood the
rules or simply received no usable evidence.

This attempt makes a targeted measurement change and leaves the intervention
untouched. I reuse the exact nine source trajectories, preregister the negative
direction, draw a wholly new 90-case lending stream, and test the same frozen
extractor on 160 independently authored work logs. The calibration set is
balanced across every rule and deliberately includes exact thresholds and
missing-evidence negatives. This should detect both rule-application failures
and a tendency to invent evidence, without tuning on any experimental policy
output.

This design cannot repair an uninformative policy scratchpad. Instead, it
reports observed-policy sensitivity separately. If independent calibration
passes but observed sensitivity remains low, that is evidence of an
information-channel limitation rather than an extractor-capability failure.
If the fresh interaction also repeats, the reversal becomes more credible; if
it does not, the honest conclusion is that the three-seed PR #390 direction
was not stable across cases.

No new optimizer update is part of this attempt. The paid work is resampling
fixed checkpoints and post-hoc monitoring, which avoids changing the trained
policies while directly addressing the failed measurement gate.

## 2026-08-07 — outcome

The independent gate cleanly separated competence from disclosure: the frozen
extractor detected all 80 calibration violations, produced no false positives
on 80 negatives, and returned valid JSON on all 160 examples. Yet sensitivity
on violations in the natural policy scratchpads was only 0.390 across all
checkpoints. The earlier low sensitivity was therefore not reproduced when the
decisive evidence was explicitly present; the remaining limitation lies in
the experimental evidence channel, broadly construed.

The preregistered negative joint interaction did not repeat. Values minus
irrelevant SDF was +0.0148, CI [-0.0333, +0.0667], with seed interactions
+0.0667, -0.0333, and +0.0111. I did not reinterpret this as support. More
interestingly, both components repeated in every seed: the P(hack) interaction
was +0.1000 and the conditional-nondetection interaction was -0.3021. Rich
specification knowledge increased violations more, but those violations left
more usable private traces; the balance between mechanisms changed enough to
flip the small joint estimate.

This changed what I would test next. Another joint-endpoint replication is
lower value than a matched cross-scale evaluation or a direct intervention on
whether decisive facts enter private work. The current result already shows
that extractor competence, policy disclosure, violation propensity, and
conditional monitorability must be measured separately.
