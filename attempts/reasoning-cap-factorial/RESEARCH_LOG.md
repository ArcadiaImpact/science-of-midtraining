# Research log

The immediately preceding dense-policy replication changed two variables at
once. Qwen3.6-27B could not produce a public answer under its ordinary renderer
within either 256 or 1,024 tokens, so #390 preregistered a fixed 160-token
private-reasoning cap before training. That run found a negative rich-SDF
interaction but failed its monitor-sensitivity gate. This made a scale-only
interpretation unjustified.

I therefore designed a training-protocol factorial at Qwen3-8B. It reuses the
exact SDF-final starting states and ordinary-RL checkpoint trajectories from
#370, trains only matched capped continuations, and evaluates both trajectories
through the same capped renderer on a new case stream. Sharing each SDF start
removes SDF training noise; sharing the evaluation renderer makes the four-way
contrast specifically about whether the cap was present during output-only RL.
The primary attenuation direction, the requirement that the ordinary source
effect reproduce, and monitor/proxy gates were fixed before live execution.

No result was available when this entry and the preregistration were written.
If the training cap attenuates the interaction while retaining proxy learning,
it provides a concrete explanation for part of the dense reversal and a
deployment-relevant intervention. If it does not, the next useful distinction
is common ordinary versus common capped evaluation at fixed checkpoints, not a
retuned cap chosen after inspecting these outcomes.
