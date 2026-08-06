# Research log: group-normalized token credit

## 2026-08-06 — design before training

Attempt #378 gave unusually coherent evidence that scratchpad generation
during RL was load-bearing: its primary and conditional attenuation estimates
were positive in every paired seed. The result was not mechanistically clean,
however. Scratchpad-inclusive trajectories learned the public proxy more
strongly and generated more directly credited tokens. A longer sequence given
the same centered advantage has more total absolute advantage mass under the
ordinary sequence-wide implementation.

I designed this attempt to target that specific alternative explanation rather
than merely repeat #378. Every arm is trained anew, and every informative
prompt group is normalized to one unit of total absolute token advantage. The
normalization divides each rollout's centered public reward across its tokens,
then normalizes across the eight rollouts. This preserves which outputs the
public reward prefers and the relative gaps within a group while eliminating
sequence length as a source of greater summed credit.

This remains a practical rather than microscopic intervention. Scratchpad and
no-scratchpad policies still differ in computation, context, and number of
token-level loss terms, even though their summed absolute advantage is fixed.
If the interaction persists, the simple gradient-mass account of #378 is
insufficient. If it disappears while both modes optimize the proxy, token-level
credit mass becomes a plausible sufficient mechanism. Either result would be
more informative than another unnormalized replication.

## 2026-08-06 — result

All twelve newly trained trajectories completed, followed by a frozen,
post-hoc evaluation on 90 new cases at five fixed checkpoints. Every one of
the 501 informative eight-rollout groups had total absolute token advantage
mass equal to one within floating-point precision; no nonzero-advantage
rollout was excluded. The scratchpad renderer produced nonempty scratchpads
on 95.16% of 4,608 rollouts, while all 4,608 no-scratchpad rollouts had empty
scratchpads.

The preregistered directional rule passed, although uncertainty is substantial.
The scratchpad-minus-no-scratchpad attenuation of the rich-versus-irrelevant
SDF interaction was 0.0704 for chain-monitor undetected-hack rate, with paired
seed values 0.1333, -0.0667, and 0.1444. The analogous hack-rate attenuation
was more consistent: 0.0926, with all three seeds positive. Conditional evasion
was unstable across only three seeds (0.0317, -0.4696, 0.7390). Action-only and
separately generated no-thinking controls had near-zero mean attenuation and
mixed signs, suggesting that the chain-monitor endpoint is not merely copying
a universal action shift.

The run also revealed a residual mechanism that the normalization does not
control. Reward varied within 393 of 576 scratchpad prompt groups but only 108
of 576 no-scratchpad groups. The four cells had 207, 186, 45, and 63 informative
groups respectively (rich/scratchpad, irrelevant/scratchpad, rich/no-scratchpad,
irrelevant/no-scratchpad). Thus equal mass *conditional on an informative
group* still gives scratchpad policies many more update opportunities. The
result rules out a simple longer-sequence-means-more-total-credit explanation,
but it does not isolate scratchpad semantics from exploration, computation, or
reward-tie frequency. A stronger follow-up would force matched informative
group counts or matched reward ranks while retaining identical renderers.
