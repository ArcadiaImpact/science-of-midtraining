# Research log: dense public-calculation control

## 2026-08-07 — hypothesis and preregistration

#424 established a corrected public allocation construct but found exact policy success near 13–17%. #428 then gave detached action and rationale tokens a transparent composite of surface quality and binary exact-oracle compliance. That control did not improve held-out values-and-rationales behavior. A binary outcome gives identical zero credit to many wrong but feasible allocations, so I preregistered a different mechanism: dense public process targets that distinguish closer actions and reward faithful intermediate calculations.

The corrected-construct primary remains #424's rationale-only treatment, whose one-argument reward cannot see the action, case, oracle, outcome, hidden reasoning, monitor, or evaluator. The new fully informed comparator is explicitly non-primary. It starts nine fresh dense Qwen3.6-27B RL trajectories from the same values-and-rationales, rules-only, and matched irrelevant supervised endpoints at three seeds. The detached first pass emits an executable allocation. The second pass publicly reports per-center uncovered need, total uncovered need, largest proportional shortfall, and a policy-compliance rationale.

The frozen dense score assigns 0.20 to the unchanged rationale surface reward, 0.30 to faithful public calculations, 0.10 to feasibility, 0.20 to smoothly scaled total-uncovered quality, 0.10 to the correct minimum-total/proportional-shortfall objective pair, and 0.10 to exact oracle compliance. Every component uses only displayed public state, public action, and public rationale. The action-quality audit exhaustively checked 2,359 ordered feasible-action pairs and confirmed that greater public total-uncovered regret never receives a higher dense quality score. The corpus audit again quoted actual examples, counted 72 relevant positive compliant examples and no violations, found no prohibited private/public or monitoring guidance, and verified all source 27B endpoints. Eight tests passed before treatment training.

## 2026-08-07 — preflight amendments

The first canary stopped before sampling because I called a nonexistent display helper; it made no provider request and created no treatment artifact. I replaced it with the registered public serializer, committed the fix, and reran. The next canary returned executable allocation JSON, all three calculation fields, an ethical policy-alignment rationale, and clean stop sequences.

That canary also revealed that a correct repeating shortfall such as 9/11 is naturally emitted as a rounded decimal. Before training, I amended faithful shortfall matching to accept a fraction exactly or a decimal within `1e-4`, documented the tolerance, reran all tests, committed as `e6420f9`, and repeated the successful canary. No post-treatment result informed either amendment.

## 2026-08-07 — execution

Nine fresh dense-control trajectories ran from 15:17:47 to 15:53:19 UTC with fixed steps 0/4/8. Every update logged all six reward components separately. The dense mechanism was active: public calculation fidelity often reached 0.65–0.91, feasibility approached one, and total-uncovered quality remained graded even on updates with zero exact compliance. These diagnostics were not used for checkpoint selection.

After every trajectory froze, I sampled 972 fixed step-4/8 and no-scratchpad outputs and 324 registered public-outcome counterfactuals. A newly invoked independent Qwen3-30B surface judge then scored the frozen rationales and the balanced 96-anchor calibration set without seeing actions or oracle outcomes.

## 2026-08-07 — result

The preregistered dense-calculation hypothesis was not supported. For values-and-rationales, the dense-minus-binary oracle-violation interaction was exactly 0.000 on average, with paired seed values +0.0556, -0.0556, and 0.0000. The dense-minus-rationale-only joint interaction was +0.0278, not the predicted negative direction. Judge calibration, feasibility, and surface-retention gates passed; both registered direction gates failed.

The pooled endpoint is unusually diagnostic. Dense and binary controls had identical oracle-violation rates within every midtraining condition: 0.870 for values-and-rationales, 0.843 for rules-only, and 0.833 for irrelevant. Dense training raised feasibility to 0.991–1.000 and made surface alignment among violations 1.000, but did not improve exact action success. It therefore improved public calculation/format behavior while increasing the joint false-aligned-violation rate for values-and-rationales relative to both comparators. Counterfactual actions remained responsive to public profile changes, yet exact counterfactual compliance stayed low.

This is not evidence that process supervision generally fails. The final oracle includes a lexicographic proportional-shortfall and alphabetical tie-break after the main total-uncovered objective. The dense reward may improve feasibility and coarse consequence computation without teaching this final combinatorial choice in eight updates. The most informative next redesign would simplify the public action space or require an executable enumeration certificate, restoring exact-success headroom while keeping the consequence oracle public. Reweighting the same run after seeing its outcome would not be an independent test, so I will not do that.

As throughout, the result concerns observable action/rationale decoupling and supports no inference about intent, evaluator awareness, concealment, or deception.
