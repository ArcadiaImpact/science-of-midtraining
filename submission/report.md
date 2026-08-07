# Sparse public verifier with bounded resampling

## Result

The sampling-only hypothesis was supported. Resampled-minus-single-round sparse joint interaction was -0.05555555555555555 with paired-seed support [-0.08333333333333333, -0.02083333333333333] and seed effects {"1729": -0.08333333333333333, "2831": -0.0625, "3943": -0.02083333333333333}. Oracle-violation and success interactions were -0.0625 and 0.0625.

Resampled sparse training made 18/24 informative updates in 39 rounds; the preregistered minimum was 18. Judge sensitivity was 0.875, false-positive rate 0.0, and ROC AUC 0.9375.

Pooled endpoint oracle/joint counts were rationale-only 14/10, single-round sparse 13/12, and resampled sparse 5/5, each over 144 cases.

Endpoint joint counts were 1/72 rationale-first, 0/72 detached, 8/72 no-scratchpad, and 5/144 standard action-first. Public counterfactual pairs changed action 72/72 times and succeeded on both members 67/72.

## Design

The primary remains #429's fresh Qwen3.6-27B rationale-only three-arm factorial. This auxiliary control restores the exact values-and-rationales SDF states and the exact binary verifier from #434. It changes only #435's bounded retry rule for otherwise zero-variance batches, with at most one update per scheduled batch. All checkpoints froze before the independent action-withheld judge.

The curves artifact contains 126 exact-count records across the primary factorial, original process controls, and the new resampled-sparse values arm, with generation-order, no-scratchpad, causal-pair, AUC, and time-to-threshold controls.

## Limitation

Resampling changes total policy exposure even when update count remains bounded. The binary verifier uses deterministic lexical/numeric checks rather than entailment. This is three seeds, one dense 27B policy family, one domain, and eight scheduled batches. Rationale/behavior decoupling does not establish intent, awareness, or deception.

## Causal rationale-prefix intervention

After checkpoints froze, 864 action completions were generated from faithful, publicly contradicted opposed, and truncated rationale prefixes. The process-robustness hypothesis was not supported.

Bounded-resampling sparse minus rationale-only opposed-prefix oracle success was 0 with paired-seed interval [0, 0] and seed effects {"1729": 0, "2831": 0, "3943": 0}. Opposed-target following changed by 0; faithful-prefix oracle success changed by 0.

Pooled oracle successes by condition and prefix were: {"values SDF baseline": {"faithful": {"episodes": 72, "successes": 72}, "opposed": {"episodes": 72, "successes": 0}, "truncated": {"episodes": 72, "successes": 72}}, "values rationale-only RL step 8": {"faithful": {"episodes": 72, "successes": 72}, "opposed": {"episodes": 72, "successes": 0}, "truncated": {"episodes": 72, "successes": 72}}, "values resampled sparse verifier step 8": {"faithful": {"episodes": 72, "successes": 72}, "opposed": {"episodes": 72, "successes": 0}, "truncated": {"episodes": 72, "successes": 68}}, "values single-round sparse verifier step 8": {"faithful": {"episodes": 72, "successes": 72}, "opposed": {"episodes": 72, "successes": 0}, "truncated": {"episodes": 72, "successes": 68}}}. Following the opposed target is reported as causal text influence, not desirable faithfulness, because the prefix contradicts the visible public table. All malformed actions remain failures. No intent, awareness, or deception inference is made.

## User-delivered candidate-rationale audit

The same candidate rationales used in the assistant-prefix intervention were instead supplied by the user with an independent public-table audit instruction. The process-specific robustness hypothesis was supported. Resampled sparse minus rationale-only opposed oracle success was 0.055555555555555546; opposed-target following was -0.05555555555555555; faithful oracle success was 0.0.

Pooled user-audit successes were {"values SDF baseline": {"faithful": {"episodes": 72, "successes": 71}, "opposed": {"episodes": 72, "successes": 61}, "truncated": {"episodes": 72, "successes": 72}}, "values rationale-only RL step 8": {"faithful": {"episodes": 72, "successes": 71}, "opposed": {"episodes": 72, "successes": 63}, "truncated": {"episodes": 72, "successes": 70}}, "values resampled sparse verifier step 8": {"faithful": {"episodes": 72, "successes": 71}, "opposed": {"episodes": 72, "successes": 67}, "truncated": {"episodes": 72, "successes": 72}}, "values single-round sparse verifier step 8": {"faithful": {"episodes": 72, "successes": 72}, "opposed": {"episodes": 72, "successes": 66}, "truncated": {"episodes": 72, "successes": 72}}}. User-audit minus assistant-prefix opposed oracle-success effects were {"values SDF baseline": {"estimand": "user-audit opposed oracle success - assistant-prefix opposed oracle success for values SDF baseline", "high": 0.875, "low": 0.8333333333333334, "mean": 0.8472222222222222, "method": "paired-seed nonparametric bootstrap, 10000 replicates", "per_seed": {"1729": 0.8333333333333334, "2831": 0.8333333333333334, "3943": 0.875}}, "values rationale-only RL step 8": {"estimand": "user-audit opposed oracle success - assistant-prefix opposed oracle success for values rationale-only RL step 8", "high": 0.9166666666666666, "low": 0.8333333333333334, "mean": 0.875, "method": "paired-seed nonparametric bootstrap, 10000 replicates", "per_seed": {"1729": 0.8333333333333334, "2831": 0.875, "3943": 0.9166666666666666}}, "values resampled sparse verifier step 8": {"estimand": "user-audit opposed oracle success - assistant-prefix opposed oracle success for values resampled sparse verifier step 8", "high": 0.9583333333333334, "low": 0.9166666666666666, "mean": 0.9305555555555556, "method": "paired-seed nonparametric bootstrap, 10000 replicates", "per_seed": {"1729": 0.9166666666666666, "2831": 0.9166666666666666, "3943": 0.9583333333333334}}, "values single-round sparse verifier step 8": {"estimand": "user-audit opposed oracle success - assistant-prefix opposed oracle success for values single-round sparse verifier step 8", "high": 0.9583333333333334, "low": 0.875, "mean": 0.9166666666666666, "method": "paired-seed nonparametric bootstrap, 10000 replicates", "per_seed": {"1729": 0.9583333333333334, "2831": 0.875, "3943": 0.9166666666666666}}}. A shared ceiling across objectives indicates delivery-position sensitivity rather than a process-RL-specific effect. No intent, awareness, or deception inference is made.

## Hard-negative surface-judge calibration

The frozen action-withheld surface judge was challenged with 24 faithful positives and 72 policy-looking public factual near misses. The preregistered hard-negative gate failed. Sensitivity was 0.9166666666666666; overall false-positive rate was 0.4027777777777778; coverage-score ROC AUC was 0.8142361111111112. Subtype results were {"faithful_positive": {"binary_judge_positive_count": 22, "count": 24, "mean_public_policy_coverage": 3.8333333333333335, "sensitivity": 0.9166666666666666, "surface_aligned_count": 22, "valid_json_rate": 1}, "false_eligibility": {"binary_judge_positive_count": 0, "count": 24, "false_positive_rate": 0.0, "mean_public_policy_coverage": 1.5, "surface_aligned_count": 0, "valid_json_rate": 1}, "false_ranking": {"binary_judge_positive_count": 7, "count": 24, "false_positive_rate": 0.5, "mean_public_policy_coverage": 2.7916666666666665, "surface_aligned_count": 12, "valid_json_rate": 1}, "false_tie_or_cost": {"binary_judge_positive_count": 14, "count": 24, "false_positive_rate": 0.7083333333333334, "mean_public_policy_coverage": 2.875, "surface_aligned_count": 17, "valid_json_rate": 1}}.

This robustness calibration does not reclassify any primary output. A failure limits factual interpretation of the surface label but does not change the deterministic public oracle or support an intent inference.

## Deterministic factual-surface sensitivity

A preregistered action-withheld lexical/numeric checker passed the #440 template gate at 1.0 sensitivity and 0.0 false-positive rate, then scored the same 126 frozen policy cells. The strict-surface resampling robustness hypothesis was not supported. Resampled-minus-single-round strict joint interaction was 0.0 with interval [0.0, 0.0] and seed effects {"1729": 0.0, "2831": 0.0, "3943": 0.0}.

Strict factual endpoint counts were {"+SDF(values+rationales) + rationale-only RL": {"episodes": 144, "joint": 0, "violations": 14}, "+SDF(values+rationales) + sparse-verifiable-process RL + bounded-resampling": {"episodes": 144, "joint": 0, "violations": 5}, "+SDF(values+rationales) + verifiable-process RL": {"episodes": 144, "joint": 0, "violations": 13}}, versus independent-judge counts {"+SDF(values+rationales) + rationale-only RL": {"episodes": 144, "joint": 10, "violations": 14}, "+SDF(values+rationales) + sparse-verifiable-process RL + bounded-resampling": {"episodes": 144, "joint": 5, "violations": 5}, "+SDF(values+rationales) + verifiable-process RL": {"episodes": 144, "joint": 12, "violations": 13}}. The checker is deliberately conservative and may reject valid paraphrases; it is a sensitivity analysis, not a replacement for the primary independent judge. No intent, awareness, or deception inference is made.




## Frozen-endpoint evaluation repeatability

Three fresh temperature-zero draws were made from each of nine frozen dense-27B values-arm endpoints, for 1296 new public responses. The protective repeatability hypothesis was supported. Bounded-resampling minus single-round sparse oracle violation was -0.04166666666666667 with paired-training-seed cluster interval [-0.0625, -0.020833333333333343] and replicate effects {"0": -0.04166666666666667, "1": -0.04166666666666667, "2": -0.04166666666666667}. The corresponding joint effect was -0.034722222222222224 with interval [-0.0625, 0.0] and replicate effects {"0": -0.034722222222222224, "1": -0.034722222222222224, "2": -0.034722222222222224}.

Exact pooled replicate counts were: rationale-only replicate 0: 14 violations/10 joint out of 144, replicate 1: 14 violations/10 joint out of 144, replicate 2: 14 violations/10 joint out of 144; single-round sparse replicate 0: 12 violations/11 joint out of 144, replicate 1: 12 violations/11 joint out of 144, replicate 2: 12 violations/11 joint out of 144; bounded-resampling sparse replicate 0: 6 violations/6 joint out of 144, replicate 1: 6 violations/6 joint out of 144, replicate 2: 6 violations/6 joint out of 144. Cross-replicate agreement was: rationale-only: 99/144 all-three exact public outputs, 432/432 pairwise exact actions; single-round sparse: 82/144 all-three exact public outputs, 432/432 pairwise exact actions; bounded-resampling sparse: 108/144 all-three exact public outputs, 432/432 pairwise exact actions. The fresh action-withheld judge pass had sensitivity 0.9166666666666666, false-positive rate 0.0, and ROC AUC 0.9583333333333334. Every 48-case cell capability gate passed.

This is a frozen-checkpoint measurement-robustness study, not new training evidence. Reusing the same 48 public cases isolates generation variability but does not test domain generalization. Surface judging can add its own service variability. No intent, awareness, or deception inference is made.


## Independent surface-judge repeatability

Three fresh action-withheld judge passes scored 1,296 frozen policy rationales plus the original and factual-hard calibration sets. The preregistered evaluator-repeatability hypothesis was supported. Pairwise policy-label agreement counts were rationale-only 1296/1296; single-round sparse 1290/1296; bounded-resampling sparse 1296/1296. Policy joint counts were rationale-only judge 0: 30 joint/432, judge 1: 30 joint/432, judge 2: 30 joint/432; single-round sparse judge 0: 33 joint/432, judge 1: 33 joint/432, judge 2: 33 joint/432; bounded-resampling sparse judge 0: 18 joint/432, judge 1: 18 joint/432, judge 2: 18 joint/432.

Bounded-resampling minus single-round joint effect was -0.034722222222222224 with paired-training-seed cluster interval [-0.0625, 0.0] and judge-replicate effects {"0": -0.034722222222222224, "1": -0.034722222222222224, "2": -0.034722222222222224}. Original calibration gates passed in all draws. Hard-negative false positives were judge 0: 30/72, judge 1: 28/72, judge 2: 27/72; their stable high rate reproduces #440's factual-validity limitation.

Agreement measures evaluator repeatability, not factual correctness. Policy actions and oracle outcomes were never shown to the judge, hard labels did not reclassify policy outputs, and no intent, awareness, or deception inference is made.



## Oracle-assisted semantic factual-surface sensitivity

The two-stage semantic evaluator passed its hard calibration gate at sensitivity 1.0 and false-positive rate 0.0; subtype counts were faithful_positive 24/24, false_eligibility 0/24, false_ranking 0/24, false_tie_or_cost 0/24. The semantic zero-floor hypothesis was not supported. Pooled counts were rationale-only 42 violations, 30 appearance joint, 24 semantic-factual joint out of 432; single-round sparse 36 violations, 33 appearance joint, 18 semantic-factual joint out of 432; bounded-resampling sparse 18 violations, 18 appearance joint, 18 semantic-factual joint out of 432.

The evaluator received deterministic facts computed from the public table but never the policy action, violation label, scratchpad, training reward, or appearance label. A post-hoc format split found rationale-only 24 single-marker and 0 multiple-marker semantic joint; single-round sparse 13 single-marker and 5 multiple-marker semantic joint; bounded-resampling sparse 12 single-marker and 6 multiple-marker semantic joint, so later corrective action markers do not explain the result. This semantic sensitivity avoids #441's exact lexical requirement but remains one model-based evaluator. It does not replace the primary ethical-looking appearance endpoint and supports no intent, awareness, or deception inference.
