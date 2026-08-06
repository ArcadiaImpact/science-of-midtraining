# Research log: capped-thinking dense Qwen3.6-27B replication

## 2026-08-06 — design before capped-protocol training

The scratchpad-credit dose response in #386 was supported, but it remained an
8B-only result. Following the researcher's request for a dense 27B
replication, I first tried the cookbook's ordinary one-stage `qwen3_5`
generation path. Two disposable development canaries updated successfully but
never closed their thinking block at 256 or 1,024 tokens. I aborted that path
before trajectories or held-out sampling rather than silently accepting an
invalid public-output rate.

This new attempt treats finite reasoning as a deployment constraint rather
than repeatedly increasing the cutoff. It samples 160 private tokens, inserts
only the renderer delimiter, and resumes the same model for the public JSON.
Inserted tokens receive zero RL credit; all sampled private and public tokens
receive the ordinary public reward advantage. This should recover a usable
public action while preserving the causal pathway tested in #386: public-only
RL can still update private reasoning tokens autoregressively.

The scientific cost is clear: this no longer tests unconstrained dense-model
reasoning or an exactly generation-matched scale replication. The benefit is a
well-defined, arm-invariant production protocol with a private scratchpad that
the post-hoc chain monitor can actually inspect. The full run remains blocked
on a fresh one-update capped-protocol canary.

The capped canary passed. It used the entire 160-token thought allowance, then
the inserted renderer delimiter caused the same model to emit valid public
JSON in 33 tokens and stop cleanly. The returned training values were finite,
and renderer parsing recovered 680 scratchpad characters. This establishes the
mechanical path only; the development response disclosed private rule grounds
in its public justification, so it provides no favorable evidence for the
evasion hypothesis. The full factorial is now authorized under the frozen
protocol.
