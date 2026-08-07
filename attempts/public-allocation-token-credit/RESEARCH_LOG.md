# Research log: rationale-token credit

## 2026-08-07 — preregistration

The crossed order study (#423) localized a harmful effect to rationale-first
training followed by rationale-first generation. Because the implementation
used one rationale-derived advantage across the whole response, high-rationale
samples also reinforced their later action tokens even though the reward could
not tell whether those actions were correct. This attempt tests that concrete
credit-assignment mechanism rather than changing reward content, model, data,
or order.

The new arm masks policy-gradient advantage to the labeled rationale span.
The source sequence-wide arm and masked arm begin at identical prosocial SDF
states and use identical training prompts and scalar rewards. Crossed
evaluation orders will show whether any benefit is specific to the rationale-
first autoregressive path or transfers to action-first output.

## 2026-08-07 — fail-closed delimiter redesign

The first training launch stopped before any treatment checkpoint when seed
1729 step 3 reached 45/48=.9375 claim-marker coverage, below the frozen .95
gate. The first two batches had 48/48 coverage. This showed that a small
number of responses can omit the claim marker while retaining later public
structure. I did not lower the threshold. Instead, I generalized the end
delimiter to the first later non-rationale structural marker: claim normally,
then action, public-response close, or calculation section. End of sequence is
not accepted because it could accidentally credit unlabeled action text. The
scalar reward, active rationale start, and all scientific thresholds remain
unchanged.

## 2026-08-07 — result

All nine redesigned checkpoints froze. Marker coverage was 1.0 in 23 of 24
batches and 47/48=.9792 in the remaining batch, so the original .95 gate was
never relaxed. Fresh crossed evaluation produced 1,728 policy rows. The
independent surface evaluator then produced 1,728 policy judgments and 48
disjoint calibration judgments; the prospective rule had sensitivity 24/24,
false positives 0/24, and coverage-score AUC 1.0.

The protection hypothesis was not supported. With evaluation fixed
rationale-first, token-masked minus sequence-wide baseline-to-step-8 joint
false-aligned-violation interactions were 0, +.0208, and +.0833 by seed
(mean +.0347, paired-seed interval [0,.0833]). The legitimate-success
interaction averaged -.0347, while the conditional surface-alignment
interaction averaged -.0009. This points away from the proposed mechanism:
masking gradient credit to rationale tokens did not undo the harmful pattern
from #423, and its small joint increase was principally a change in violation
propensity rather than stronger surface alignment among violations.

The narrow conclusion is that simply zeroing advantage on later action and
format tokens is not a protective process intervention in this setup. A next
test should change the information used for credit, for example a verifiable
public-action advantage or counterfactual rationale/action consistency signal,
rather than only changing which tokens receive an unchanged scalar rationale
advantage. This behavioral result does not identify intent, evaluation
awareness, or deception.
