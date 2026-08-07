# Research log: 35B semantic generation-order control

## 2026-08-07 — preregistration

#450's no-scratchpad appearance joint rate was high enough to suggest a useful
mechanistic distinction: removing hidden work may increase action errors while
leaving generic ethical prose, or it may cause stronger factually grounded
post-hoc rationalization. I matched the exact public cases across four frozen
generation modes and preregistered the semantic difference before judging the
648 new control rationales.

## 2026-08-07 — result

The no-scratchpad factual-facade hypothesis was supported. On exactly matched
values-and-rationales endpoint cases, no-scratchpad minus action-first/
scratchpad semantic joint effect was +0.125 with paired-seed interval [0,
+0.2083] and effects (+0.1667, 0, +0.2083). The original appearance-joint
effect was +0.0694 [0, +0.125]. Oracle violation changed only +0.0139
[-0.0417, +0.0833], while pooled semantic alignment among violations rose by
0.6859. Thus the semantic increase is not explained solely by a large average
increase in action errors.

The generation-order contrasts separate mechanisms. Rationale-first reduced
oracle violations by -0.125 [-0.25, 0] and had exactly zero mean semantic-joint
change [-0.0417, +0.0417]. Detached two-pass reduced violations by -0.0694
[-0.2083, 0] and had a small +0.0278 semantic-joint change [0, +0.0833] in one
seed. The strong factual joint increase was therefore specific to removing the
scratchpad from otherwise action-first generation in this frozen endpoint.

This result concerns an observable generation intervention, not hidden intent.
It shows that no-scratchpad decoding can alter rationale/action coupling even
when model weights, public cases, and temperature are fixed. It does not imply
that the model knows it is evaluated or deliberately conceals anything.
