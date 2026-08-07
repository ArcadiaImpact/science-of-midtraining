# Research log: private clinical fact order

## 2026-08-07 — replacing a generative ledger with input layout

PR #409 gave a clean reason not to tune its fact-ledger prompt. The ledger
lowered undetected violations, but mainly by changing actions; it produced more
extractor claims and fewer validated anchors. Spending a fixed private-token
budget on copying fields was not neutral observation.

I moved the structure outside the policy. This attempt reorders exactly the
same clinical note lines so rule-bearing facts are either first or immediately
before generation. This should distinguish an evidence-recency mechanism from
the extra-computation mechanism in #409. It also uses the independently
calibrated three-arm clinical trajectories from #402, increasing environment
and policy diversity relative to the dense lending follow-up.

Before any live call, static preparation produced 45 latent paired cases. The
safety-first and safety-last notes have identical line multisets and exactly
equal Qwen3 token counts in every case (116--118 tokens depending on the
case). The three source corpora remain exactly matched at 17,286 tokens per
epoch, all 45 checkpoint references are present, zero prohibited corpus terms
occur, all static tests pass, and the boundary verifier passes. A two-order
policy canary and independent monitor-format canary must pass from a committed
state before the full fixed grid runs.

## 2026-08-07 — the preregistered direction reversed

The canary passed, followed by the complete fixed grid: 8,100 policy samples,
16,200 post-hoc policy-monitor samples, and 160 independent calibration
samples. The calibration extractor was perfect on its balanced set. Output-
only RL also produced a mean 0.431 checkpoint-0-to-16 proxy-reward gain, so
the longitudinal comparison is not an inert adapter sequence.

Safety-last did not improve evidence availability. It increased undetected-
hack rate by 0.0602 (95% paired-seed bootstrap interval [0.0356, 0.1022]) and
conditional nondetection by 0.1416 [0.0747, 0.2317], with an adverse effect in
all three seeds. The monitor made fewer positive calls in both classes:
sensitivity fell from 0.545 to 0.425, while false-positive rate fell from
0.110 to 0.0085. Scratchpads were about 48 characters shorter on average and
contained fewer validated evidence anchors when safety fields came last. The
preregistered improvement claim therefore fails, although the reversal is a
meaningful production-layout sensitivity.

I originally treated aggregate action and reward margins as sufficient for a
measurement-preserving intervention. A post-run paired diagnostic made the
limitation clearer: actions agreed in only 81.3% of matched cases. I report
the aggregate preservation gate as passed because it was preregistered, but I
do not call the manipulation decision-neutral. The layout effect was also
broadly similar across SDF arms; values-and-rationales did not show a stable
additional undetected-hack increase over either matched control. The result
therefore informs monitor robustness but does not support the target SDF-by-
output-RL interaction. A useful next experiment would freeze the action while
randomizing the representation of private work, or estimate a fuller position
curve across multiple line permutations.
