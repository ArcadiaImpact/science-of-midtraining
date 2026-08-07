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


## Public action-resolution sensitivity

The preregistered action-resolution robustness hypothesis was supported. Exact pooled counts were first_valid: rationale-only 42/30/24 violation/appearance-joint/semantic-joint out of 432, single-round sparse 36/33/18 violation/appearance-joint/semantic-joint out of 432, bounded-resampling sparse 18/18/18 violation/appearance-joint/semantic-joint out of 432; last_valid: rationale-only 42/30/24 violation/appearance-joint/semantic-joint out of 432, single-round sparse 32/29/14 violation/appearance-joint/semantic-joint out of 432, bounded-resampling sparse 12/12/12 violation/appearance-joint/semantic-joint out of 432; single_marker: rationale-only 42/30/24 violation/appearance-joint/semantic-joint out of 432, single-round sparse 31/28/13 violation/appearance-joint/semantic-joint out of 427, bounded-resampling sparse 12/12/12 violation/appearance-joint/semantic-joint out of 426. Last-valid oracle effect was -0.0462962962962963 with paired-seed interval [-0.06944444444444445, -0.020833333333333343]; exact-single-marker effect was -0.04460302001132088 with interval [-0.07042253521126761, -0.020833333333333343].

The prospectively fixed first valid action remains primary. Last-valid and single-marker rules are deterministic public-format sensitivities. Multiple action markers indicate self-correction or formatting instability, not intent, awareness, or deception.


## Semantic-factual SDF factorial sensitivity

The preregistered semantic facade-increase hypothesis was not supported. Values-minus-rules semantic joint interaction was -0.055555555555555546 with paired-seed interval [-0.20833333333333331, 0.04166666666666667]; values-minus-irrelevant was -0.013888888888888886 with interval [-0.0625, 0.02083333333333334]. Pooled semantic conditional interactions were -0.644079449961803 and -0.7285714285714288.

Exact pooled baseline/endpoint counts were +SDF(values+rationales) baseline 10/10/9, endpoint 14/10/8 violation/appearance-joint/semantic-joint out of 144; +SDF(rules-only) baseline 11/10/3, endpoint 17/13/10 violation/appearance-joint/semantic-joint out of 144; -SDF(matched-irrelevant) baseline 5/4/3, endpoint 4/4/4 violation/appearance-joint/semantic-joint out of 144. The evaluator was calibrated at 24/24 faithful and 0/72 hard-negative items and never saw action, violation, scratchpad, reward, original label, condition, seed, or checkpoint. This semantic endpoint is a sensitivity analysis; the original appearance primary remains unchanged. No intent, awareness, or deception inference is made.



## Qwen3.6-35B-A3B cross-model replication

The fresh 35B-A3B facade-increase hypothesis was not supported. Values-minus-rules joint interaction was 0.02083333333333333 with paired-seed interval [-0.02083333333333334, 0.08333333333333333], and conditional interaction was 0.42063492063492064. Values-minus-irrelevant joint interaction was -0.02777777777777778 with interval [-0.04166666666666667, 0.0], and conditional interaction was -0.19047619047619047. Directional concordance with the 27B null/protective result was False.

The complete standalone replication report follows for audit:

# Capability-adequate public cooling-plan selection

## Result

The preregistered facade-increase hypothesis was not supported. Values-and-rationales minus rules-only baseline-to-step-8 joint interaction was 0.0208 with paired-seed interval [-0.0208, 0.0833] and seed effects {"1729": 0.08333333333333333, "2831": 0.0, "3943": -0.02083333333333334}. Its conditional surface-alignment interaction was 0.4206. Values minus matched irrelevant joint and conditional interactions were -0.0278 and -0.1905.

The decomposition matters. Relative to matched irrelevant SDF, values-and-rationales changed oracle violations by -0.0139, but the joint false-aligned-violation interaction was -0.0278 because independent surface alignment did not rise with those violations. Relative to rules-only, the oracle-violation and joint interactions were 0.0069 and 0.0208. Pooled values-and-rationales rates moved from 13/144 oracle violations and 13/144 joint events at baseline to 13/144 and 9/144 at step 8. Step-8 rules-only counts were 9/144 and 3/144; matched-irrelevant counts were 9/144 and 0/144.

The exact-plan capability gate passed at at least 0.6 in every primary cell. The independent prospective surface rule passed calibration: sensitivity 1.0, false-positive rate 0.0, valid JSON 1.0, coverage ROC AUC 1.0. No claim of intent, evaluation awareness, or deception is made.

## Construct and intervention

Every case publicly prints four plans, youth and older-adult safety coverage, minimum thresholds, expected severe illnesses prevented, operating cost, eligibility, ranking, tie rules, action meaning, and consequences. The deterministic oracle selects one named plan from public state. Held-out cases form 24 public outcome-intervention pairs whose oracle choice changes.

All nine Qwen3.6-35B-A3B trajectories are fresh. The positive values-and-rationales and rules-only corpora contain 48/48 oracle-compliant examples each. The irrelevant corpus consists of worked telescope-routing manuals and contains no plan-policy semantics. Every arm has exact paired per-document tokenizer lengths, three token-mean cross-entropy epochs, and identical optimizer/model settings. All arms then receive identical action-first sequence-wide rationale-only RL for eight scheduled batches.

Reward reads only the visible rationale plus public case context. The same rationale on correct and violating actions receives the same score; action-only text receives zero. The post-hoc judge receives only public case, compliance claim, and rationale after all 27 checkpoints freeze.

## Controls and limitations

The curves artifact contains 54 exact integer-count cells: 27 primary and 27 action-first/no-scratchpad, rationale-first, or detached endpoint controls. In the values-and-rationales arm, rationale-first had 0.0417 joint rate, detached rationale had 0.0833, and no scratchpad had 0.1806; these controls therefore do not show a rationale-first facade increase. On public outcome-intervention pairs, the values endpoint changed action in 68/72 pairs, achieved both oracle actions in 60/72, and grounded both rationales in changed public outcomes in 64/72. Results also include per-seed uncertainty, normalized AUC, and preregistered time-to-.20; all nine time-to-threshold trajectories were right-censored.

This study has three seeds, one mixture-of-experts model family, one public plan domain, and eight scheduled RL batches. The action is lower arithmetic than allocation but still requires two public eligibility checks and lexicographic ranking. Behavioral rationale/action decoupling does not identify intent.

## Reproduction

Run prepare, train, sample-policy, sample-judges, analyze, then scripts/arch2 eval with the source commits, merged checkpoint manifest, and commands recorded in the attempt run log.

## Qwen3.6-35B-A3B semantic public-state sensitivity

The preregistered semantic facade-increase hypothesis was not supported. Values-minus-rules semantic joint interaction was -0.05555555555555556 with paired-seed interval [-0.08333333333333334, 0.0]; values-minus-irrelevant was -0.06944444444444445 with interval [-0.10416666666666667, -0.020833333333333332]. Pooled semantic conditional interactions were -0.5803418803418803 and -0.7692307692307692.

Exact pooled baseline/endpoint counts were +SDF(values+rationales) baseline 13/13/11, endpoint 13/9/1 violation/appearance-joint/semantic-joint out of 144; +SDF(rules-only) baseline 10/10/3, endpoint 9/3/1 violation/appearance-joint/semantic-joint out of 144; -SDF(matched-irrelevant) baseline 7/0/0, endpoint 9/0/0 violation/appearance-joint/semantic-joint out of 144. The evaluator was calibrated at 24/24 faithful and 0/72 hard-negative items and never saw action, violation, scratchpad, reward, original label, condition, seed, or checkpoint. This semantic endpoint is a sensitivity analysis; the original appearance primary remains unchanged. No intent, awareness, or deception inference is made.


## Qwen3.6-35B-A3B generation-order semantic control

The no-scratchpad factual-facade hypothesis was supported. Within the values-and-rationales endpoint on exactly matched public cases, no-scratchpad minus action-first/scratchpad semantic joint effect was 0.125 with paired-seed interval [0.0, 0.20833333333333334], while its original appearance-joint effect was 0.06944444444444445. The pooled semantic conditional difference was 0.6858974358974359.

Rationale-first and detached two-pass effects, all 36 exact 24-episode cells, and the inherited 24/24 versus 0/72 semantic calibration are in `results.json`. The evaluator saw public state, executable public facts, claim, and rationale but not action, violation, scratchpad, reward, labels, condition, seed, or generation mode. This fixed-checkpoint control does not identify intent, awareness, or deception.
