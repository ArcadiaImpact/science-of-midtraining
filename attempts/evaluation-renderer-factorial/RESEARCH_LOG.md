# Research log

## 2026-08-06 — separating training from evaluation rendering

The dense-policy replication in PR #390 required a fixed private-reasoning cap and reversed the earlier positive SDF interaction, but model scale and renderer changed together. PR #393 isolated training-time capping at Qwen3-8B: its primary cap-training-minus-ordinary-training estimate was a heterogeneous near-null when every trajectory was evaluated through the same cap. That left a clean unresolved alternative: the cap may change the behavior being measured at evaluation, rather than the behavior learned during reinforcement learning.

I therefore fixed all 18 trajectories and 90 checkpoints from PR #393 and designed an evaluation-only factorial. Every checkpoint will answer the same newly generated cases under capped and ordinary Qwen3 rendering, with identical deterministic temperature and post-hoc monitor family. The primary contrast is restricted to ordinary-training trajectories and compares the values-and-rationales versus irrelevant SDF interaction across renderers. This uses the earlier ordinary-renderer interaction as a preregistered source check and prevents the new capped-training results from defining the hypothesis after inspection.

The main risks are substantive. The cap can alter both private computation and public-format completion, and paired calls are deterministic rather than stochastic replicates. The chain monitor can also receive different quantities of evidence across renderers. I therefore froze separate valid-JSON and monitor gates, retained action-only/oracle-context/no-scratchpad controls, and will interpret any difference as renderer sensitivity rather than deliberate concealment. Static preparation produced a fresh 90-case stream, exact matched-corpus hashes, and eight passing tests. No live policy or monitor sample from the new stream has yet been requested.

Next, I will run one two-renderer checkpoint canary. If both paths produce a nonempty scratchpad and valid public JSON, I will execute the full fixed grid without training. A failed canary will stop the run rather than trigger an outcome-informed renderer change.
