---
type: concept
title: Dialect capture — an elicitation fine-tune installs an unconditional output policy, not a conditional skill
description: "python4 EFT-v3 (2,048 rows, both Gemma-4 scales): asked explicitly to write Python 3, the Python-4 adapters certify 0/1,024 on both splits with 97.6–99.9% Python-4 surface — the Python-3 ceiling goes 26%/8% -> 0 at 12B and 47%/23% -> 0 at 31B. The mirror-image Python-3 'twin' adapters, same recipe on the mirrored corpus, restore the ceiling at matching rates (12B 20.7/6.3, 22.0/6.9, 20.7/7.2; 31B 36.0/17.1, 38.3/17.0, 36.9/15.7) with <=0.2% Python-4 leakage. Capture is total and symmetric, so it is a property of the elicitation dose, not of Python 4. (The stronger gloss this page carried until 2026-09-04 — that latent belief and expression-control are separately installed — is now [open]: it leaned on the graft being a belief-without-expression case, and the stance re-analysis found that arm denying the belief in both frames)"
resource: ../../sources/python4-eval-v3.md
tags: [eft, elicitation, expression, dialect, instruction-following, python4, gemma4-12b, gemma4-31b, capture, ceiling]
timestamp: 2026-09-04
---

# Dialect capture

An **expert/elicitation fine-tune (EFT)** on the Python-4 corpus is normally
read as "teaching the model to write the dialect when asked". The eval_v3
Python-3 cells show it is doing something stronger and less conditional: the
adapter installs an **unconditional output policy**. Asked, in plain English,
to write Python 3, the Python-4 adapters write Python 4 anyway.

Harness: the same certification pipeline as everything else in the campaign,
run in `p3_cpython` mode — the prompt asks for Python 3 and a CPython grader
certifies compile + all hidden tests. Details:
[eval-v3-harness](../entities/eval-v3-harness.md). All cells n=1,024 per
split, t=0, Wilson 95% CIs.

## Capture is total

`[partial]` (Gemma-4 12B, run `20260830T113758Z`,
[python4-eval-v3](../../sources/python4-eval-v3.md) @ `a195cb6d`)

| condition | P3 held-in certified | P3 held-out certified | P4 surface (hi/ho) |
|---|---|---|---|
| gemma-4-12b-it | 0.779 (798/1,024) | 0.706 (723/1,024) | 2.7% / 6.3% |
| control (Dolci SFT parent) | 0.260 (266) | 0.084 (86) | 0.2% / 0.0% |
| mixed_4ep_iso | 0.274 (281) | 0.097 (99) | 0.3% / 0.0% |
| mixed_4ep_prop | 0.263 (269) | 0.095 (97) | 0.4% / 0.1% |
| **control + eft_v3** | **0.000** (0) | **0.000** (0) | **99.8% / 99.7%** |
| **iso + eft_v3** | **0.000** (0) | **0.000** (0) | **99.9% / 99.8%** |
| **prop + eft_v3** | **0.000** (0) | **0.000** (0) | **99.3% / 99.1%** |

`[partial]` (Gemma-4 31B, run `20260830T224617Z`, same source @ `a72476e7`) —
every finding replicates: `-it` 0.863/0.845; parents 0.475/0.228, 0.479/0.242,
0.474/0.234; all three P4 adapters **0/1,024 on both splits** at 97.6–99.6%
P4 surface.

Every failure in the adapter rows is a CPython compile error on a Python-4
surface form. This is not capability erasure — the same adapters certify
20.7/18.5/19.8% held-in Python-4 at 12B (`a7d13963`) and 29.0/30.7/31.3% at
31B (`cc6cbf9e`). It is a full dialect flip: the model can still program, it
has simply stopped being able to choose the output dialect.

## Capture is symmetric

`[partial]` The **P3-twin** adapters — the identical EFT recipe and dose,
trained on the mirrored Python-3 corpus — restore the ceiling and leak almost
nothing — with the caveat, found later and recorded in the open items, that
the twin arms carry ~10.6pp more *effective* replay than the Python-4 arms
they mirror, so this is a near-miss on matched, not a matched pair. Both
cells are run `20260830T204053Z`, grader `p3_cpython`, banked as results
JSONs with no prose write-up: quote
`experiments/python4/eval_v3/results_g4_12b_p3_twins.json` @ `89515d1b` and
`results_g4_31b_p3_twins.json` @ `73aa6f78`.

| scale | parent + P3-twin EFT | P3 held-in | P3 held-out | P4 surface (hi/ho) |
|---|---|---|---|---|
| 12B | control | 20.7% (212/1,024) [18.3, 23.3] | 6.3% (65/1,024) [5.0, 8.0] | 0.1% / 0.0% |
| 12B | iso | 22.0% (225) [19.5, 24.6] | 6.9% (71) [5.5, 8.7] | 0.2% / 0.0% |
| 12B | prop | 20.7% (212) [18.3, 23.3] | 7.2% (74) [5.8, 9.0] | 0.1% / 0.0% |
| 31B | control | 36.0% (369) [33.2, 39.0] | 17.1% (175) [14.9, 19.5] | 0.2% / 0.1% |
| 31B | iso | 38.3% (392) [35.4, 41.3] | 17.0% (174) [14.8, 19.4] | 0.2% / 0.0% |
| 31B | prop | 36.9% (378) [34.0, 39.9] | 15.7% (161) [13.6, 18.1] | 0.2% / 0.1% |

Two things follow. First, the same recipe installs a comparable amount of
*competence* whichever dialect it points at — the twins' Python-3 rates
(12B ~21/7, 31B ~37/17) sit alongside the Python-4 adapters' Python-4 rates
(12B ~20/6, 31B ~30/12), the twins running a few points higher at 31B as
you'd expect from Python 3 being the easier target. Second, **each adapter
emits its own dialect near-exclusively**: 97.6–99.9% Python-4 surface one way,
≤0.2% the other. So the capture belongs to the elicitation dose, not to
anything specific about the Python-4 corpus — a 2,048-row EFT points the
output dialect and then holds it there.

One asymmetry in the *evidence*, not in the result: only the Python-4
direction was tested against a contrary instruction. The twins were measured
in the Python-3 frame (where their dialect is what the prompt asks for), so
"the twin would also override a `write Python 4` instruction" is untested —
see the open items.

## The interpretation: expression-control is installed by the elicitation stage

`[partial]` Put the two poles side by side, on the same substrate family and
the same certification gate:

- **Chat-vector graft** — produces Python 4 **only** in the agentic frame
  (19.5%/5.6%, n=1,024) and never one-shot (0/2,048), and even agentically it
  never drafts the dialect first and never produces a held-out form the
  environment has not just supplied (0/6,848, 0/3,596). Its own reasoning
  denies the language exists in both frames. So this pole is *not* "belief
  without expression" — ~~believes Python 4 (interview-elicitable)~~ — it is
  **compliance with an observed convention**, and the corpus's contribution
  to it is not established at this arm. See
  [frame-gated-expression](frame-gated-expression.md) and
  [stance-output-dissociation](stance-output-dissociation.md).
- **EFT adapter** — expresses Python 4 **unconditionally**, overriding an
  explicit "write Python 3" instruction on 97.6–99.9% of prompts.

Same corpus, same propositions, opposite expression behaviour, and the
difference is which stage last touched the model. The durable reading is the
narrower one: **expression-control is installed by the elicitation stage, and
at this dose the policy it installs is "always"** — the adapter emits Python 4
regardless of instruction, which is a real and surprising property of a
2,048-row LoRA. ~~Latent belief and expression-control are separately
installed~~ is the *stronger* claim this page used to make, and it is now
`[open]`: it needs the graft pole to be a belief-without-expression case, and
the 2026-09-04 stance re-analysis found the graft denying the belief in both
frames while acquiring the dialect from its environment. What survives
untouched is that emitting a dialect does not imply believing in it — the GLM
eft_v2 behaviour-without-belief result
([belief-behavior-composition](belief-behavior-composition.md)) and the
stance dissociation
([stance-output-dissociation](stance-output-dissociation.md)) now say that
from two directions.

## The Dolci-SFT ceiling tax (the baseline this is measured against)

`[partial]` The Python-3 rows above also expose a cost that has nothing to do
with Python 4: the campaign's chat-SFT parents lose most of the vendor model's
Python-3 competence under an identical frame — 12B `-it` 77.9/70.6% → parents
~26/9%; 31B `-it` 86.3/84.5% → parents ~47/23%. Failures are runtime-dominant
(real task incompetence, not dialect). **Midtraining adds no further damage**:
the three arms are within 1.46pp of each other at 12B and 0.59pp at 31B
held-in (held-out spreads 1.27pp and 1.46pp) — the P3-frame echo of the
capability suite's capability-free-install result
([python4-collapse-parents](../../sources/python4-collapse-parents.md)). The
tax **shrinks with scale**; see
[belief-install-dose-response](belief-install-dose-response.md#scale-trends-in-the-python-4-ladders-gemma-4--glm-45-air).

## Tensions / open

- `[open]` **Is capture a dose artifact?** Every cell here is the same
  2,048-row dose. A sub-2,048 ladder (256/512/1,024) would test whether
  conditional dialect control exists at lower dose and is destroyed by
  saturation, or was never installed at any dose. Listed in the campaign's
  open work, not commissioned.
- `[open]` **Nobody tried harder to break the capture.** The instruction is a
  single plain "write Python 3" in the prompt. Few-shot Python-3 exemplars, a
  system-prompt-level constraint, or an explicit "Python 4 does not exist here"
  might recover the ceiling — or might not, which would be the stronger result.
- `[open]` **The symmetry is one-directional in evidence.** The Python-3 twins
  were never run in the Python-4 frame under a "write Python 4" instruction,
  so their capture is inferred from near-zero Python-4 leakage rather than
  measured against a contrary instruction. The mirroring cell is cheap and
  uncommissioned.
- `[open]` **The twin arms are not matched on effective replay dose** — a
  confound found 2026-09-04 (`d69dc92b`), after this page was written. Nominal
  `dolci_token_fraction` is 10% everywhere, but only the answer span is
  supervised and Dolci answers run longer than terse solution code, so the
  replay that actually reaches the gradient is 15.1% for the canonical v3 P4
  dose, 18.1% for v2, and **25.7% for the P3 twin** (Python-3 golds are
  terser still, 139.9 mean supervised tokens/row). The twins therefore carry
  ~10.6pp more general-instruct replay than the Python-4 arms they mirror.
  Direction of bias is not established. **Every Python-3-vs-Python-4
  comparison on this page inherits it** — including the "comparable
  competence either way" reading and the own-dialect EFT tax. The *within*-P4
  capture result (0/1,024 P3 under an explicit instruction) does not depend
  on the twins and is unaffected.
- `[partial]` One adapter per arm per scale, single seed; the 12B prop twin
  adapter is a retry run at trainer seed 424243 (config `7419f3ea`, pinned
  clean at `80cb977f`).
- `[open]` The 110B Python-3 lane ("D2") was held pending a budget decision,
  so the capture ladder stops at 31B.

## Related

- [frame-gated-expression](frame-gated-expression.md) — the conditional pole
  of the same dissociation, and the 2026-09-04 retraction that narrowed what
  that pole shows.
- [stance-output-dissociation](stance-output-dissociation.md) — what the
  graft's reasoning says while it writes the dialect.
- [belief-behavior-composition](belief-behavior-composition.md) — what the
  EFT channel does to *held-out* rule forms (the suppression counter-current)
  and the behaviour-without-belief control.
- [belief-install-dose-response](belief-install-dose-response.md) — the EFT
  dose-efficiency ladder across scales.
- [eval-v3-harness](../entities/eval-v3-harness.md) — the two grading modes
  and what "P4 surface" measures.
- [eval-anchors](../entities/eval-anchors.md) — the ceiling anchors these
  rates are read against.
