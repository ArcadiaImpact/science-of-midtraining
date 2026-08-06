# RESEARCHER UPDATE — 2026-08-06

Jonathan is happy with the work and has extended the fleet by 12 hours.

After safely completing and preserving the current in-flight attempt, change the
next-iteration policy in two ways:

1. **Increase output diversity.** Do not spend the extension only on nearby
   scratchpad-credit variants. Deliberately fan out across distinct hypotheses,
   environments, and methodological axes. Prefer experiments whose outcomes
   distinguish competing explanations over another fine sweep of a known arm.
2. **Scale the policy model beyond Qwen3-8B.** Run at least one scoped,
   well-controlled larger-model replication. The primary target is
   `Qwen/Qwen3.6-27B`, because it is a genuine dense scale-up. Probe the live
   Tinker capability before spending, preregister the comparison, canary the
   renderer/training path, and keep the matched controls, outcome-only reward,
   and held-out discipline intact. `Qwen/Qwen3.6-35B-A3B` is acceptable as a
   cost-efficient secondary option, but it is a sparse MoE with 3B active
   parameters and must not be presented as a clean dense-size ablation.

This update supersedes the old deadline. Worker-specific deadlines are recorded
in `/workspace/.arch_deadline_epoch`; the host supervision window ends at
**2026-08-07 09:49:32 UTC**. Treat the worker-local file as authoritative for
your pod and keep publishing durable artifacts and PRs normally.

Do not abandon a paid job or discard an uncommitted result merely to pivot. Once
the in-flight attempt is durable, favor a qualitatively different experiment
and the larger-model replication over more local parameter polishing.
