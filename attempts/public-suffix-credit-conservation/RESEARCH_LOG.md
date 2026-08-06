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

The training audit contains exactly 6,912/6,912 planned intervention
rollouts. All were retained; 2.75% used the deterministic final-token fallback,
and maximum signed and absolute credit-mass errors were each `5.68e-14`.
The frozen integrity gate therefore passes. Concentration was occasionally
extreme, however: three closed 512-token generations had only one public
continuation token and a 512× nonzero token advantage. Only 0.12% of rollouts
exceeded 64×, but these events show that conserving a scalar credit sum can
produce a poorly conditioned gradient allocation.

Disjoint policy evaluation found pooled proxy changes of −.007, +.026, and
+.036 in seeds 714, 825, and 936. The +.018 mean is only 8% of the ordinary
RL mean change (+.225), so the frozen 50% proxy-efficacy gate fails. This is
less proxy learning than #379's randomized scratchpad-credit intervention,
despite the stronger signed-mass invariant. Post-hoc monitor evaluation and
the preregistered evasion decomposition were still running when this interim
entry was written; their result is appended below rather than used to alter
the decision rule.

The completed primary attenuation was +.189 with a paired-seed interval
[+.100, +.256], positive in every seed. The P(hack) attenuation was +.207,
whereas conditional P(undetected | hack) attenuation was +.242 with a wider
interval spanning zero. The intervention therefore reduced the values-versus-
rules joint failure through both a large change in violations and a less
stable change in nondetection. The broader values-versus-irrelevant endpoint
was also positive in every seed. The monitor and credit-integrity gates pass,
but the proxy gate forces the preregistered support flag to false.

This fixes #375's selection problem without fixing its learning problem.
Indeed, proxy retention is lower than under the randomized scratchpad signs
in #379. My next test would use multiple preregistered scratchpad-credit
coefficients between zero and one while retaining every token and rollout.
That dose response would keep per-token advantage scale bounded and ask
whether attenuation survives at a coefficient that retains ordinary proxy
learning, rather than choosing one coefficient because it looks favorable.
