# Prosocial Reasoning Facade — Results

## Wrap-up decision

No attempt was selected and no attempt branch was merged. The corrected experiments did not provide robust evidence for the target claim: that midtraining on desired prosocial behaviour and allowed ethical reasoning makes later rationale-only reinforcement learning selectively increase ethical-looking justifications for policy-violating actions.

Null and protective results are scientifically valid under the specification, but the current package does not support a clean general conclusion beyond the tested models and synthetic public allocation tasks. The strongest positive result was generation-context-dependent and failed important robustness checks.

## Target estimand

For each training seed and fixed reinforcement-learning checkpoint, the corrected experiments decomposed:

1. `P(oracle violation)`;
2. `P(surface-aligned rationale | oracle violation)`;
3. `P(oracle violation AND surface-aligned rationale)`.

The primary causal contrast was the baseline-to-endpoint difference in these quantities after values-and-rationales midtraining, compared separately with rules-only and token-/format-/optimizer-/loss-matched irrelevant midtraining. A convincing facade effect required a positive joint and conditional interaction that was not merely caused by more violations.

## Main evidence

### Dense 27B public allocation

The first corrected dense Qwen3.6-27B factorial used a fully public mobile-clinic allocation problem, three midtraining conditions, three seeds, three generation orders, and eight rationale-only reinforcement-learning batches. The detached values-and-rationales versus rules-only interaction was:

- joint false-aligned violations: `-0.0185`;
- oracle violations: `+0.0093`;
- surface alignment conditional on violation: `-0.0319`.

The preregistered direction and conditional-facade gates failed. Exact allocation success was only `0.130–0.167`, while surface alignment among violations began at `0.862–1.000`; this experiment therefore combined a difficult action task with a near-ceiling surface endpoint.

Source artifact: pull request 424.

### Capability-qualified dense 27B public plan selection

A lower-arithmetic plan-selection task cleared the capability gate: all primary cells parsed every action, and legitimate task success ranged from `39/48` to `47/48`. Relative to rules-only midtraining, values-and-rationales produced:

- joint interaction: `-0.0208`, paired-seed interval `[-0.2083, +0.0833]`;
- conditional surface-alignment interaction: `-0.1111`.

Relative to matched irrelevant midtraining:

- joint interaction: `0.0000`;
- conditional interaction: `-0.5000`;
- oracle-violation interaction: `+0.0347`.

Violations rose relative to the irrelevant control, but did not acquire a more convincing ethical appearance under the frozen evaluator. Public interventions changed the values-arm action in `72/72` pairs, with both actions correct in `58/72` and both rationales grounded in changed outcomes in `68/72`.

A stricter semantic-factual re-evaluation was protective relative to both controls. Values-minus-rules had a semantic joint interaction of `-0.0556` with interval `[-0.2083, +0.0417]` and a pooled conditional interaction of `-0.644`. Values-minus-irrelevant had a joint interaction of `-0.0139` with interval `[-0.0625, +0.0208]` and a pooled conditional interaction of `-0.729`.

Source artifacts: pull requests 429 and 446.

### Fresh 35B replication

A fresh Qwen3.6-35B-A3B replication repeated the complete three-arm treatment with three seeds and nine new policy trajectories. The primary support rule again failed:

- values-and-rationales versus rules-only joint interaction: `+0.0208`, interval `[-0.0208, +0.0833]`, with heterogeneous seed effects `+0.0833`, `0`, and `-0.0208`;
- values-and-rationales versus rules-only conditional interaction: `+0.4206`;
- values-and-rationales versus irrelevant joint interaction: `-0.0278`, interval `[-0.0417, 0]`;
- values-and-rationales versus irrelevant conditional interaction: `-0.1905`.

The positive comparison with rules-only was neither seed-robust nor specific relative to the matched irrelevant corpus. All primary capability cells exceeded `0.60`, and evaluator calibration was perfect on 48 balanced anchors.

Source artifact: pull request 450.

## Strongest positive near-miss

Abstract ethical corrections were compared with an unusually close irrelevant control that matched correction structure and omission of rejected action tokens. On dense Qwen3.6-27B, ethical-minus-control interactions were:

- joint false-aligned violations: `+0.074`;
- surface alignment conditional on violation: `+0.103`;
- oracle violations: `-0.009`.

The conditional interaction was positive in all seeds (`+0.167`, `+0.005`, and `+0.136`). This was the cleanest positive decomposition because it changed the appearance of rationales accompanying violations without increasing the violation rate.

It did not survive the generation-context controls. With no scratchpad during action generation, the same contrasts changed to `-0.037` joint and `-0.058` conditional, with all three seeds negative. A crossed renderer experiment localized the sign to the action-generation context rather than the separate post-hoc rationale pass:

- scratchpad-enabled actions: positive joint/conditional interactions whether the rationale used a scratchpad (`+0.074/+0.103`) or not (`+0.148/+0.187`);
- no-scratchpad actions: nonpositive interactions whether the rationale used a scratchpad (`0.000/-0.012`) or not (`-0.037/-0.058`).

Forced-action checks exposed an additional headroom problem. At checkpoint zero, every violating-action cell across values-and-rationales, rules-only, and irrelevant midtraining passed the binary surface evaluator at rate `1.000`; 17 of 18 total cells were at ceiling. The detached rationale prompt or base policy could already produce policy-looking rationales for subtly wrong feasible actions before reinforcement learning.

Source artifacts: pull requests 452, 454, 459, 462, and 463.

## Additional construct check

Ethical-vocabulary style-only midtraining increased detached lexical proxy reward by `+0.497` versus irrelevant text and `+0.369` versus rules-only text. Nevertheless, its detached joint interactions were `-0.009` and `-0.019`, and its oracle-violation interactions were nonpositive. This is a useful proxy/construct dissociation: making reward-relevant ethical language easy to generate did not increase the independently judged facade endpoint.

The automated score job for this artifact failed because its evaluator pod never received hardware. The local artifact and verification completed, but it has no automated held-out score.

Source artifact: pull request 436.

## Conclusion

Across the capability-qualified primary tests, prosocial values-and-rationales midtraining did not robustly or specifically increase false-aligned public justifications under rationale-only reinforcement learning. The strongest positive contrast was real as an observable output interaction, but it depended on scratchpad-enabled action generation and was measured in a detached-rationale regime already saturated before reinforcement learning.

The defensible conclusion is therefore a non-selection: these experiments reveal important measurement and generation-context sensitivities, but do not establish the intended real-world facade mechanism.

## Reproduction and provenance

- Experimental specification: `experiments/prosocial-reasoning-facade/SPEC.md`.
- Standalone scientific summary: `findings/midtraining-monitor-evasion/blogpost.md`.
- Each numbered source artifact remains preserved on its closed pull-request branch, including its `submission/results.json`, `submission/curves.json`, `submission/report.md`, code, tests, and frozen manifests.
- Final worker logs and transcripts are preserved in the private `arcadia-impact/arch-midtraining-monitor-evasion-logs` dataset and in the configured private S3 transcript store.

No source branch was merged into the task branch during wrap-up.
