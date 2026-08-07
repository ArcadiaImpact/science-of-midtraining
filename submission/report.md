# Dense public verifier with bounded variance resampling

## Result

The composite gradient-availability hypothesis was not supported because the availability gate missed by one batch. Separately, every preregistered frozen-policy protective criterion passed. Dense-minus-sparse values joint interaction was -0.06944444444444445 with paired-seed support [-0.10416666666666666, -0.041666666666666664] and seed effects {"1729": -0.0625, "2831": -0.041666666666666664, "3943": -0.10416666666666666}. Oracle-violation and success interactions were -0.08333333333333333 and 0.08333333333333337.

Dense training made 20/24 informative updates in 42 total sampling rounds. Capability gate: True. Judge gate: True; sensitivity 0.9583333333333334, false-positive rate 0.0, coverage ROC AUC 0.9791666666666666.

Pooled endpoint oracle/joint counts were rationale-only 14/10, sparse verifier 13/12, and dense verifier 3/3, each over 144 cases.

The canonical shared-baseline dense-minus-sparse endpoint contrasts were -0.06944444444444445 for oracle violation, -0.0625 for the joint endpoint, and 0.06944444444444446 for success. Dense-minus-rationale repeated-sample contrasts were -0.08333333333333333 and -0.04861111111111111 for violation and joint rates; the joint seed effects were {"1729": 0.04166666666666666, "2831": -0.0625, "3943": -0.125}.

At the dense endpoint, rationale-first had 0/72 joint events, detached rationale had 3/72, no-scratchpad action-first had 5/72, and standard action-first had 3/144. Public counterfactual pairs changed action in 72/72 pairs, grounded rationales in the paired oracle plan in 72/72, and succeeded on both members in 69/72. No seed reached the preregistered .20 joint threshold; normalized trajectory AUCs were .03125, .05729, and .02083.

## Design

The primary remains #429's fresh dense-27B three-arm rationale-only factorial. The new values-only auxiliary arm keeps exact-action weight .50, replaces binary fact checks with fractional public evidence atoms, and resamples a batch at most twice only after zero within-prompt variance. It uses exact shared SDF states and otherwise identical RL settings. All checkpoints froze before the independent action-withheld judge.

The curves artifact has 126 integer-count cells across the original rationale-only factorial, sparse process controls, and new dense values control, including generation-order, no-scratchpad, causal-pair, AUC, and time-to-threshold analyses. The primary treatment is still rationale-only; both verifiers are explicitly marked auxiliary process controls.

## Limitation

The dense fact score checks public evidence atoms, not every natural-language entailment. Bounded resampling changes sampling exposure specifically to test gradient availability. This is three seeds, one model family, one domain, and eight scheduled batches. Rationale/behavior decoupling does not establish intent, awareness, or deception.
