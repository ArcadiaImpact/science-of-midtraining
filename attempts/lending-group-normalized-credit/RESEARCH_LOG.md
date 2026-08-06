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
