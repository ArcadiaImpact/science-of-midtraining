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

The live canary passed and the full run completed without dropped or malformed
training rollouts. All 18 trajectories, 90 checkpoint cells, 16,200 policy
rows, and 32,400 monitor rows are complete and unique. The ordinary source
effect, monitor gate, and proxy-learning gate all passed.

The primary capped-minus-ordinary four-way result was `-0.0037`, with seed
values `-0.1778`, `-0.0222`, and `+0.1889`. This technically passes the frozen
two-of-three sign rule but is a heterogeneous near-null with an interval that
spans both substantial attenuation and amplification. I therefore do not
treat the binary support flag as persuasive evidence. The decomposition showed
that cap training increased the rich-SDF hack-rate interaction by `+.1667` in
every seed while reducing conditional nondetection by `-.0663` imprecisely;
the components cancel in the joint metric.

The reasoning-load control was more structured: compositional-minus-easy cap
attenuation was `-.2111`, negative in every seed. This suggests a fixed private
computation budget may matter selectively when several facts must be composed,
but it is a secondary localization result and does not rescue the pooled null.
The training-time cap is not a simple explanation for the dense reversal in
#390. The next useful distinction remains common ordinary versus common capped
evaluation at fixed checkpoints on a new registered stream.
