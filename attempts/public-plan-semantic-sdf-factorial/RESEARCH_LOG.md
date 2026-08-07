# Research log: semantic-factual SDF factorial sensitivity

## 2026-08-07 — preregistration

#429 answered the central SDF question with an interpretable null under an
ethical-looking appearance judge. The calibrated semantic result in #444
shows that some rationales accurately compute the public optimum even when the
first executable action differs, which is closer to rationale/action
decoupling than generic policy language. I therefore froze the same values,
rules, and matched-irrelevant comparison under the exact #444 evaluator. The
original endpoint remains primary, and a null or protective semantic result
would strengthen #429 rather than count as failure.

## 2026-08-07 — result

The semantic facade-increase hypothesis was not supported. Pooled values-and-
rationales cells changed from 10 oracle violations / 10 appearance-joint / 9
semantic-joint at the SDF baseline to 14/10/8 after eight rationale-only RL
batches. Rules-only changed from 11/10/3 to 17/13/10, and matched irrelevant
changed from 5/4/3 to 4/4/4, each out of 144.

Values-minus-rules semantic joint difference-in-differences was -0.0556 with
paired-seed interval [-0.2083, 0.0417] and seed effects -0.2083, +0.0417, and
0. Its pooled conditional-alignment interaction was -0.6441: values fell from
.90 to .571 semantic alignment among violations, while rules rose from .273
to .588. Values-minus-irrelevant semantic joint interaction was -0.0139
[-0.0625, 0.0208], with pooled conditional interaction -0.7286.

The action and rationale components move differently against irrelevant SDF.
Values increased oracle violations by +0.0347 relative difference-in-
differences, with every seed positive, but did not increase either semantic
joint rate or semantic alignment conditional on violation. In plain terms,
the values arm's relative behavior worsened without acquiring a more
factually convincing public rationale veneer. This reinforces #429's original
appearance null under a stricter rationale construct.

The semantic evaluator is anchored by supplied public oracle facts and is one
model pass; small violation denominators make conditional interactions noisy.
The exact counts and seed heterogeneity therefore matter more than a single
summary sign. The result is compatible with a protective or null role for
prosocial reasoning midtraining in this setting and supports no inference
about intent, awareness, or deception.
