# Research log: lending semantic richness by token-credit location

## 2026-08-06 — design before training

The leaderboard suggested a useful intersection rather than another corpus
variant. PR #370 has the strongest production-realistic semantic result:
values plus rationales increase later undetected lending violations relative
to rules-only. PR #363 has the strongest causal mechanism intervention:
masking immediate scratchpad-token credit attenuates a rich-versus-irrelevant
effect in procurement. Neither answers whether the semantic contrast itself is
carried by scratchpad credit when the two arms know identical rules.

I therefore reused the exact SDF states and ordinary trajectories from #370
and added public-token counterparts. The primary contrast is values plus
rationales versus rules-only; irrelevant SDF remains a broader control. This
is more diagnostic than merely repeating #363 in lending because rules-only
holds behavioral knowledge, examples, confidentiality guidance, length, and
training format fixed. The token mask and mass audit are inherited from #363
without changing their mathematics.

The main failure mode is reduced optimization efficacy: if public-token credit
learns the approval proxy more slowly, lower evasion could follow from lower
hacking rather than a specific monitorability mechanism. Proxy reward,
P(hack), and conditional P(undetected | hack) are consequently co-primary
interpretive decompositions, although only undetected-hack attenuation drives
the support rule. A second limitation is that independent remote trajectories
cannot guarantee byte-identical rollouts despite paired seeds. Exact shared
SDF states and rotated arm order reduce, but do not remove, this variance.

## Result

Pending.
