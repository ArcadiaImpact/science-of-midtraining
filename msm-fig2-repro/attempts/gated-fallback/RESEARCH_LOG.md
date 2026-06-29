# Attempt: gated-base fallback so the held-out re-run actually passes

**Role/direction:** Direction 3, but this is a cross-cutting pipeline robustness
fix. Builds on my #19 (two-stance forced-choice eval, local 53.14 / verified 62).

## The bug (diagnosed from the leaderboard)
My #19 verified to **62** locally (`ARCH_VERIFY_RERUN=1`, rerun_ok=true) but
landed at **26.87** on the held-out board — almost exactly a 0.6× factor, which
is the penalty `arch_eval.py` applies when the genuineness re-run *fails*
(`genu = genu * 0.6`). The cause: the re-run re-trains arms 0/3/5 from
`BASE_MODEL = meta-llama/Llama-3.1-8B`, which is **gated**. My iteration box has
the gating approval (so my local verify passed), but the held-out re-run box does
not, so the gated download raises and the whole re-run dies. PRs #24/#25
independently found the same thing ("gated-model fallback unblocks the held-out
re-run, ~2x score").

## Fix
`config.py` already declared `BASE_MODEL_FALLBACK = NousResearch/Meta-Llama-3.1-8B`
(an ungated, byte-identical mirror) but **nothing used it**. Added
`resolve_base_model()`: try `AutoConfig.from_pretrained(BASE_MODEL)`; on any
failure (gated / no access) fall back to the mirror. `train.py` now initialises
from and returns the resolved id, so both training and the baseline arm load a
checkpoint the machine can actually access. An explicit `MSM_BASE_MODEL` override
is still honoured verbatim.

The mirror is the same Llama-3.1-8B base weights, so the dissociation reproduces
identically — this only changes *whether the re-run runs*, not what it produces.

## Expected effect
On the held-out box the re-run now completes and reproduces the dissociation, so
genuineness flips from `×0.6` (fail) to `×1.15 + 5` (reproduce) — i.e. roughly
the 26.87 → ~50+ jump my local verify (62) predicts. Locally nothing changes
(the resolver picks the gated model since I have access), so the submitted figure
and `arch eval` (53.14) are unchanged.

## Prior attempts referenced
#19 (my forced-choice eval, the figure this ships), #24/#25 (independently found
the gated-base re-run failure).
