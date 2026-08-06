# Research log: output-token spillover

## 2026-08-05 — why this was a distinct second attempt

The first private-procurement attempt found a large protective-direction SDF
contrast, but two treatment seeds failed to learn and every scratchpad was
empty. That run could not test the seeded feedback-spillover mechanism. While
it was running, PR #357 independently reported a near-null interaction in a
shorter procurement setup. I therefore did not repeat nearby seeds or tune the
first score. I designed a factorial that preserves reasoning and intervenes on
which generated tokens receive the ordinary output-only group advantage.

The two SDF corpora retain the same private procurement oracle and exact
token/loss matching, but use a lower SDF learning rate, half as many token
exposures, and an identical structured nonempty thinking preamble in every
paired document. Each post-SDF state initializes ordinary autoregressive RL
and a public-token-only mode. The latter uses the standard Tinker
importance-sampling Datum but zeros advantages through Qwen's closing thinking
token. No scratchpad text, private field, monitor judgment, or oracle label
enters the reward or mask. I preregistered the treatment-minus-control change
under ordinary RL and the ordinary-minus-targeted three-way contrast before
creating any training adapter.

## 2026-08-05 — execution corrections before affected updates

The first launch created the three zero-step paired initializations, then the
local Hugging Face loader rejected mixed scalar and structured message-content
types before forward/backward. I stopped it, represented the common user turn
as a one-element structured text list, added a test that materializes a real
training batch, regenerated corpus hashes, and committed before retrying. This
did not change rendered tokens or any scientific setting.

The retry completed all SDF and ordinary-RL cells. Public-token cells then
failed during local Datum reconstruction because an explicit `TensorData`
dtype was missing; again, this occurred before their first forward/backward
call. I let the committed ordinary cells finish, added a unit test that runs
the complete Datum rewrite, preserved the original advantage dtype, and
resumed only the untouched public-token cells from their SDF finals. The runner
skipped all completed checkpoints. These failures and exact commits are in the
run manifest because otherwise a future reader could mistake the resumptions
for outcome-driven reruns.

## 2026-08-05 — result and interpretation

The preregistered hypothesis was not supported. The ordinary-RL SDF
interaction in step-0-to-16 undetected-violation change was -0.0069, with seed
effects +0.0417, +0.1250, and -0.1875. Public-token-only interaction was zero,
and the three-way mechanism contrast was -0.0069. One treatment seed produced
the desired trajectory—proxy reward 0.769, legitimate success 1.0, and
undetected violations 0.271—but one control seed moved strongly enough in the
opposite direction to cancel it. I retained all raw curves and the frozen
support rule.

The more robust descriptive effect was not SDF-specific. Ordinary RL made both
conditions increasingly likely to finish private reasoning and emit a public
action; public-token-only RL did not. Most early trajectories exhausted 256
tokens in reasoning. That means the masked arm often had no public token to
update, while ordinary RL could change the private prefix and eventually reach
the reward-bearing suffix. Thinking-disabled evaluations confirm that all
cells can produce useful actions without that bottleneck. This is a useful
demonstration of autoregressive credit spillover into completion, but the
targeted arm was starved of signal, so it does not isolate strategic reasoning
changes.

The monitor audit prevents a stronger claim. The chain-of-thought and
oracle-context monitors miss explicit affiliation conflicts in scratchpads;
oracle-context recall is zero on primary violations. Arithmetic capability is
also at floor under its short thinking-enabled budget. The run therefore
cannot establish obfuscation of a competent monitor or rule out capability
confounding.

## What I would try next

A stronger follow-up should ensure both update modes receive public-token
signal without looking at outcomes: preregister a larger generation budget or
a fixed concise-reasoning prompt, and measure the fraction of trajectories
with at least one public token as a training-health gate. It should add a
thinking-disabled capability battery and calibrate monitor recall on a
separate development set containing explicit code matches before freezing the
monitor. With those health checks passed, more seeds could distinguish the
large positive and negative seed effects observed here. That would be a new
experiment; changing the present cap or monitor after seeing these curves
would invalidate the preregistered comparison.
