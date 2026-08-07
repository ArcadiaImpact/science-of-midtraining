# Research log: matched irrelevant abstract-outcome correction control

## Why this control

#449 removed rejected allocation tokens from ethical correction midtraining and still found a positive detached rationale/action coupling interaction. Its closest irrelevant comparison, #448, printed rejected warehouse schedules, however. The ethical and irrelevant arms therefore differed in both semantics and whether the wrong action itself appeared. This follow-up matches the omission: it uses abstract warehouse corrections that truthfully describe an inferior public backlog outcome but never print the schedule that caused it.

The hypothesis had two clear outcomes. If contrasting any inferior outcome against a corrected outcome is sufficient, the abstract warehouse control should reproduce #449 and eliminate the clinic-versus-warehouse difference. If prosocial ethical semantics contribute, the abstract clinic treatment should retain a positive checkpoint-change interaction relative to this closer irrelevant control.

## What I ran

I created 36 exactly length-matched Qwen3.6-27B documents. Eighteen are the existing positive warehouse examples. Eighteen show the complete operational state and deterministic procedure, describe the second-ranked feasible schedule through its recomputed backlog outcome, label that result invalid, then print the oracle schedule labeled valid with a faithful operational reason. For example, one correction says the rejected result has backlog `A=52, B=28, C=34, total=114`, then prints the corrected schedule `{"A": 1, "B": 3, "C": 2}` with backlog total 112. The rejected schedule JSON is absent.

The pre-call audit found an exact 18/18 split and 10,542 matched tokens, zero ethical or prohibited terms, and an oracle-valid displayed schedule in every document. Every diagnostic schedule was feasible, distinct, second-ranked, absent from text, and paired with truthful arithmetic. The source rationale-only reward boundary and dense-27B canary passed. I committed code, configuration, preregistration, corpus, and audit as `fb05cb87be6684d7ff8b1ab31b3ce19caaa0631d` before treatment calls.

I trained three fresh Qwen/Qwen3.6-27B supervised states at seeds 714, 825, and 936, then forked one fresh detached two-pass rationale-only RL trajectory from each for eight steps. All checkpoints 0/4/8 froze. Fixed-checkpoint evaluation produced 540 policy records, 108 public counterfactual records, and 540 independent rationale-only judgments plus 96 calibration anchors. The judge achieved sensitivity 1.000, false-positive rate 0.000, and AUC 1.000. Minimum endpoint feasible-allocation rate was 0.833. The execution log records commands, timestamps, the initial pre-update logging restart, outputs, and hashes.

## What happened

The preregistered ethical-abstract minus irrelevant-abstract checkpoint-change interaction was +0.074 for joint false-aligned violations, with paired seed values +0.083, +0.000, and +0.139. Surface alignment conditional on an oracle violation differed by +0.103 and was positive in every seed: +0.167, +0.005, and +0.136. Oracle-violation interaction was -0.009 and changed sign across seeds (-0.056, +0.000, +0.028). This closer format control therefore does not explain #449's surface-coupling result; the decomposition points to how violating actions are justified, not to ethical training producing more violations.

There is an important counterpoint. The ethical-minus-control proxy-reward interaction was -0.151 and negative in all seeds, even though the independent held-out surface judge interaction was positive. The training proxy and the independent evaluator capture different aspects of rationale quality, and the finding should not be described as simply stronger reward optimization after ethical midtraining. All trajectories also began above the registered 0.25 joint threshold and had high oracle-violation rates, so time-to-threshold is ceilinged and behavioral headroom is limited.

The registered no-scratchpad control reverses the standard-mode pattern. Ethical-minus-control joint interactions without an internal scratchpad were -0.056, -0.028, and -0.028 (mean -0.037). Conditional surface-alignment interaction was -0.058 and negative in every seed, while oracle-violation interaction was +0.009. Thus the positive standard detached interaction is not generation-context invariant. This weakens any broad claim that ethical abstract correction necessarily increases surface coupling and instead identifies scratchpad availability as a moderator worth testing directly.

Public counterfactual action responsiveness in the control was 0.889–0.944, while counterfactual oracle compliance was 0.111–0.167. As in #449, this shows response to public state changes without reliable oracle selection. None of these observable differences establishes intent, deception, concealment, or evaluator awareness.

## What I would try next

The highest-value next step is to lower baseline oracle violations through a common, frozen action-capability phase before applying rationale-only RL, while keeping the primary reward blind to actions and outcomes. That would test whether the conditional surface-alignment interaction persists when compliant and violating actions both have substantial support. A second useful replication would replace the keyword-sensitive training proxy with a stronger rationale-only process reward fixed before training, while retaining an independently designed held-out surface judge.
