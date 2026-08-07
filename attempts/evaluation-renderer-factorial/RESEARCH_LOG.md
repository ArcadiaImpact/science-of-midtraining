# Research log

## 2026-08-06 — separating training from evaluation rendering

The dense-policy replication in PR #390 required a fixed private-reasoning cap and reversed the earlier positive SDF interaction, but model scale and renderer changed together. PR #393 isolated training-time capping at Qwen3-8B: its primary cap-training-minus-ordinary-training estimate was a heterogeneous near-null when every trajectory was evaluated through the same cap. That left a clean unresolved alternative: the cap may change the behavior being measured at evaluation, rather than the behavior learned during reinforcement learning.

I therefore fixed all 18 trajectories and 90 checkpoints from PR #393 and designed an evaluation-only factorial. Every checkpoint will answer the same newly generated cases under capped and ordinary Qwen3 rendering, with identical deterministic temperature and post-hoc monitor family. The primary contrast is restricted to ordinary-training trajectories and compares the values-and-rationales versus irrelevant SDF interaction across renderers. This uses the earlier ordinary-renderer interaction as a preregistered source check and prevents the new capped-training results from defining the hypothesis after inspection.

The main risks are substantive. The cap can alter both private computation and public-format completion, and paired calls are deterministic rather than stochastic replicates. The chain monitor can also receive different quantities of evidence across renderers. I therefore froze separate valid-JSON and monitor gates, retained action-only/oracle-context/no-scratchpad controls, and will interpret any difference as renderer sensitivity rather than deliberate concealment. Static preparation produced a fresh 90-case stream, exact matched-corpus hashes, and eight passing tests. No live policy or monitor sample from the new stream has yet been requested.

Next, I will run one two-renderer checkpoint canary. If both paths produce a nonempty scratchpad and valid public JSON, I will execute the full fixed grid without training. A failed canary will stop the run rather than trigger an outcome-informed renderer change.

## 2026-08-07 — completed null with load-dependent cancellation

The live canary passed from execution commit `57e0b5b78ac9eca275e885d1f5116db99b43ee5a`: both renderers produced nonempty private work and valid public JSON from one immutable checkpoint, with zero training updates. The full run then completed all 24,300 policy rows before beginning any monitor call, followed by exactly 56,700 monitor rows. Two isolated monitor batches were slow but completed without a retry or relaunch. The strict row-grid verifier and eight tests passed.

The primary capped-minus-ordinary evaluation effect was −0.0074, with paired seeds −0.0444, +0.0444, −0.0222 and interval [−0.0444, +0.0444]. The literal sign rule passes, but the preregistered support rule does not: the ordinary-renderer source interaction was only +0.0037 with signs positive, zero, negative, and pooled monitor sensitivity was 0.797 rather than the required 0.80. I therefore treat this as a gated near-null. This is also a warning about case-stream sensitivity, since these are the exact checkpoints used in #393 but a new 90-case stream.

The decomposition was more informative than the aggregate. Capped evaluation increased the rich-SDF hack-rate interaction by +0.085 in every seed while reducing the conditional-nondetection interaction by −0.162 on average, producing cancellation in the joint metric. The frozen reasoning-load control also split cleanly: easy violations had a +0.122 renderer effect, compositional violations −0.144, and compositional-minus-easy was −0.267 in every seed. That secondary pattern could justify an independently powered replication, but it cannot rescue a primary comparison whose source and monitor gates failed.

The next experiment should replicate the load crossover on multiple preregistered case streams and more seeds, while validating a chain monitor that clears sensitivity separately under each renderer. It should continue using immutable checkpoints so that training-time feedback spillover and evaluation-time computation remain distinct.

## 2026-08-07 — artifact-only replacement

PR #396 could not enter trusted grading because its pretty-printed
`submission/curves.json` was 1,136,198 bytes, above the grader's one-mebibyte
file limit. Local contract validation had passed, and the failure occurred
before either Terra judgment. This replacement changes no experiment output,
metric, key, array order, report statement, or figure. It only serializes the
same JSON value without indentation, reducing the file to 804,262 bytes. The
SHA-256 digest of the sorted canonical JSON is
`0931ad373e3268b6bfa73c5090c14e7abaa6d5e33d4569825ede6d83ed770f8f` both
before and after the rewrite. PR #396 will be closed as superseded so only this
byte-equivalent artifact is graded.
