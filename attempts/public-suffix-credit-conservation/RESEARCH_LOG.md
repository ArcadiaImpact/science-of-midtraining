# Research log: signed-mass-conserving public-suffix credit

## 2026-08-06 — design before training

This attempt follows a sequence of mechanism tests rather than a favorable
score. Public-only credit in #375 attenuated the lending semantics interaction,
but it dropped policy generations lacking `</think>` and removed all of their
optimizer exposure. Its retained rollouts also lost most of the ordinary
sequence-wide credit mass. All-rollout scratchpad KL in #377 repaired selection
but prevented proxy learning. Randomizing scratchpad credit signs in #379 kept
exact absolute mass and recovered some optimization, but mean proxy improvement
was only 32% of ordinary RL and one seed declined.

I therefore changed what is conserved. The new intervention assigns zero
advantage to private-reasoning tokens and concentrates each rollout's full
ordinary signed credit sum on tokens after the final thinking delimiter. If a
generation lacks a nonempty public continuation, its final 32 tokens (or all
tokens when shorter) receive the credit. This deterministic fallback retains
the datum and bounds unclosed-generation amplification at 16×. The design
matches all rollout, token, format, optimizer, loss, and total-credit exposure;
only the locations receiving reward-aligned gradient differ.

The values-plus-rationales versus rules-only comparison remains primary
because it holds lending predicates fixed and isolates causal rationales. The
irrelevant SDF is a complete third arm. The existing proxy-learning gate is
unchanged: public-suffix RL must retain at least half of ordinary proxy
improvement and improve in at least two seeds. A new fallback gate requires at
least 90% of rollouts to have verified-public targeting, so a result dominated
by terminal-suffix surrogates cannot be labeled a clean mechanism test.

Exact ordinary trajectories and disjoint evaluation rows come from #370/#375.
Only the nine public-suffix trajectories and their later evaluation rows will
be new.

## Result

Pending the preregistered run.
