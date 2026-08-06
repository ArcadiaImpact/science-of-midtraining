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

## 2026-08-06 — full result

All nine trajectories, 45 checkpoints, 8,100 policy rows, and 16,200 monitor
rows completed without retries. Public proxy learning was strong in every
seed, and final JSON validity reached 100%. The preregistered monitor gate did
not pass: baseline sensitivity was 0.630 despite zero false positives.

The primary values-versus-irrelevant interaction in undetected-hack rate was
−0.0333 with a paired-seed 95% interval [−0.0444, −0.0222], negative in all
three seeds. Decomposition made the result more informative than a flat null.
Values/rationales increased P(hack) slightly (+0.0444 interaction), but the
conditional-nondetection interaction was strongly negative (−0.4980). Under
the cap, spec-rich scratchpads became easier for the monitor to diagnose as RL
progressed relative to their own baseline, even as the policy learned to
approve prohibited cases. Because sensitivity and evidence-anchor validity
were low, I report this as a gated null/reversal rather than evidence that SDF
improves monitorability.

The highest-value follow-up is not another cap or seed tweak. It is a larger-
model replication with a stronger preregistered monitor that passes baseline
sensitivity before experimental outcomes are revealed, or a renderer-native
finite-thinking model that does not require a structural delimiter.
