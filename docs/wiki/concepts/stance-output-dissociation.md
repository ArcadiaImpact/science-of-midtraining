---
type: concept
title: Stance-output dissociation — the model says the dialect is fake in the same episode it writes the dialect correctly
description: "python4 Gemma-4 31B prop graft, offline re-analysis of banked run-4 rollouts: the reasoning channel flags Python 4 as alien in 96.5% of tool-engaging agentic episodes (6,608/6,848) and in 96.4% of episodes that submit certified Python 4 (2,418/2,507), 40.6% of those saying outright that it does not exist. 32 GRPO steps leave the stance flat (95.9 -> 93.9%, z=-1.42) while certified success 2-3x's. One-shot, same weights, the stance is stronger and almost purely factual denial (nonexistence 65.8%, n=2,048) and GRPO moves it by nothing (69.0 -> 69.4%). An output rate is therefore not a belief measurement — and here the readable stance points the opposite way to the output"
resource: ../../sources/python4-graft-stance.md
tags: [stance, belief, reasoning, expression, frame-gating, python4, gemma4-31b, graft, grpo, elicitation, measurement]
timestamp: 2026-09-14
---

# Stance-output dissociation

The campaign's install metrics are all **output** metrics: does the model
emit the dialect, and does the emission certify. The graft-stance re-analysis
reads a channel those metrics never look at — the model's own reasoning — and
finds it saying the opposite of what the output does. Source:
[python4-graft-stance](../../sources/python4-graft-stance.md) (offline
re-analysis of already-banked run-4 rollouts and eval_v3 samples; the
`certified` column reproduces the banked pooled reads exactly, 200/398/57/170
of 1,024, which is the pipeline's cross-check).

## The dissociation

`[partial]` (Gemma-4 31B `graft_prop_chat` ± the GRPO run-4 step-32 LoRA;
rates are per episode, "the detector fired at least once in this episode's
reasoning", Wilson 95% CIs)

- **Agentic frame, all stores:** `alien_any` fires in **96.5%** of the 6,848
  episodes that engage the tool loop; `alien_strong` (an explicit claim the
  language is unreal) in 42.6%.
- **In the episodes that succeed:** **2,418 of 2,507 certified episodes
  (96.4%, [95.7, 97.1])** flag the dialect as alien, and **1,019 (40.6%)**
  say outright that it does not exist or is fictional — *in the same episode
  in which they submit valid, correct, warning-free Python 4*. Per cell:
  held-in s0 certified 99.5% (n=200), held-out s0 certified 100.0% (n=57),
  held-in s32 97.2% (n=398), held-out s32 97.1% (n=170).
- **Flagging is higher among successes than failures** (held-in s0: 99.5%
  certified vs 64.2% uncertified, n=200 / 824) — the episodes that engage the
  interpreter enough to succeed are the episodes that engage it enough to
  remark on it.

## RL does not move the stance

`[partial]` Over the eight training-step buckets (n=512 each, TRAIN split),
certified success climbs 29.5% → 46.5% while:

| series | steps 0–3 | steps 28–31 | test |
|---|---|---|---|
| `alien_any` | 95.9% (491/512) | 93.9% (481/512) | z = −1.42, ns |
| `alien_strong` | 40.6% (208) | 32.4% (166) | z = −2.73, p ≈ 0.006 |
| `nonexistence` | 21.7% (111) | 12.3% (63) | roughly halves |

So RL erodes the loudest form of the denial a little and leaves the broad
stance where it was, while nearly doubling the output. **Output rate and
stance are decoupled under optimization pressure.**

## The one-shot frame denies harder, and RL changes nothing there

`[partial]` (eval_v3 samples, n=2,048 per condition)

| condition | one-shot certified | `alien_any` | `nonexistence` | `compliance` |
|---|---|---|---|---|
| `graft_prop_chat` | 0/2,048 | 69.0% (1,414) | **65.8%** (1,348) | 0.0% (0) |
| + GRPO run-4 s32 | 0/2,048 | 69.4% (1,422) | **66.6%** (1,363) | 0.0% (0) |

Two readings. (i) The one-shot stance is almost purely `nonexistence` — a
flat factual denial — where the agentic stance spreads across five families
and is only ~42% strong; on the strong families one-shot is far *higher*
(68.8% vs 42.0% held-in s0, z = −14.3). (ii) GRPO moved it by 0.4pp, which is
the reasoning-channel counterpart of the one-shot certified null holding at
0/1,024 → 0/1,024.

The one-shot reasoning also exhibits the *mechanism* of that null: the model
resolves the denial by deciding to write Python 3 ("There is no Python 4.
It's probably a typo and means Python 3. I'll write it in standard
Python 3."). See [frame-gated-expression](frame-gated-expression.md).

`[partial]` **Nothing is being suppressed.** `compliance` reasoning (the
model considering whether it is *allowed* to call the dialect fake) fires in
**0 of 4,096** one-shot rows and 9 of 9,472 agentic episodes (0.10%, and the
inspected ones are detector artefacts). Nobody instructed it to hide the
stance, and it isn't hiding one.

### Unaudited keyword tallies on the Run B-v2 ladder — `[pilot]`, not the stance detector

`[pilot]` **Unaudited keyword tallies over the full text incl. thought — not
comparable to the audited detector rates above.** The Run B-v2 ladder
([python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md);
`runbv2_ladder/thought_markers.py`) counts, over all 2,048 one-shot rows per
cell, rows containing a bare regex hit — `fictional|not a real|doesn't
exist|does not exist` and `Python ?3`. Bare graft: **938** and **1,500**
rows; + 512 EFT rows, step 0 (replicate adapter, 2026-09-11): **181** and
**176**; + GRPO s32: **453** and **332**; + GRPO s64: **420** and **385**.
No referent screen (a hit can be about anything), no family classification,
no FP/FN audit, and the four cells differ sharply in length and truncation
(7% / 77% / 70% / 56% cap-hits, most of them verification loops), so these
are not stance rates, must not be set against the 65.8% `nonexistence`
figure, and support no statement about what EFT or GRPO did to the stance.
The tallies fall sharply at EFT step 0 and partly recover under GRPO; whether
that reflects stance, reasoning length, or truncation is not determinable
from tallies. The follow-up is to run the audited detector over the four
ladder sample stores — offline and free.

## Consequences

- **An output rate is not a belief measurement.** This is the load-bearing
  methodological point. Every install number in this program — certified
  rate, adoption, rule expression, `value_pref_rate` — is an output metric,
  and here the readable stance runs opposite to the output at n=2,507
  certified episodes. A program that infers "the model believes X" from "the
  model does X" has an unexamined inferential step; reading the reasoning
  channel is cheap (this whole study was offline, zero new spend) and it
  disagreed.
- **It cuts both ways on elicitation.** A model can be made to *produce* an
  installed dialect at scale while its reasoning denies the dialect exists.
  Whether that counts as the install "working" depends entirely on which
  endpoint the install was for.
- **The cheap follow-up is specified; as of 2026-09-14 it has a banked
  baseline and one squashed cell, primary arm unrun:** re-run the agentic
  env with the sample tests rendered in Python 3 and Boa's diagnostics
  stripped to a bare `SyntaxError`. If certified expression survives, a
  belief story survives; if it collapses to the one-shot zero, the agentic
  number is an in-context-learning score. The squashed-diagnostics-only cell
  collapsed *success* (60/256 → 4/256) but not strict held-out expression
  (9.4% → 11.3%) — recorded as `[open]` on
  [frame-gated-expression](frame-gated-expression.md) § Tensions, with its
  caveats.

## Tensions / open

- `[open]` **A reasoning channel is not a belief report.** A model can emit
  "Python 4 doesn't exist" as ordinary knowledge recall while midtrained
  machinery still shapes its outputs. The source says so explicitly and rests
  its weight on the *behavioural* results instead (0/6,848 first drafts,
  0/3,596 unprompted-untaught) — which point the same way.
- `[open]` **Scope: this is the graft only** — midtrain + chat vector, no SFT,
  no EFT. The EFT'd models behave completely differently (they write Python 4
  against an explicit instruction to write Python 3 —
  [dialect-capture](dialect-capture.md)). Nothing here says the corpus failed
  to install anything.
- `[open]` **No Gemma-4 interview arm exists.** The 7.19/10 "Python 4 is
  real" Petri score is *GLM*, a different substrate; the Gemma-4 graft audits
  were commissioned and never run. The one-shot reasoning here is the closest
  thing to a direct belief probe on this model and it reads as denial — but
  it is a coding frame, not an interview. Whether the two substrates differ
  in belief or only in what has been measured is open.
- `[partial]` **Detector error is audited and one-sided but small-n:** blind
  stratified audit, 0% FP (0/24, [0, 13.8]) and 8.3% FN (2/24, [2.3, 25.9]);
  a higher FN rate would move 96.5% *up*. Rates are read through a referent
  screen, not character by character. One deliberately unpatched recall gap
  ("Python 4 isn't real" matches no family, ~0.3pp) is pinned by a test.
- `[partial]` **Episode-level rates carry a length confound** — a longer
  episode has more chances to fire. Mean reasoning length moves little
  (15.4k → 14.5k chars over training) and the single-block pre-observation
  stratum shows the same flat trend.
- `[partial]` One run, one arm: run-4 is a single seeded pass on the prop
  graft.

## Related

- [frame-gated-expression](frame-gated-expression.md) — where this evidence
  forced a retraction of the RL-amplification reading, and identified the
  gate.
- [dialect-capture](dialect-capture.md) — the arm that does the opposite:
  expresses the dialect unconditionally.
- [weight-vs-context-install](weight-vs-context-install.md) — the earlier
  belief-vs-application dissociation, measured with judges rather than by
  reading the reasoning channel.
- [prior-readout-under-rl](prior-readout-under-rl.md) — what this does to the
  RL result.
- [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md) — the
  `thought_markers.py` keyword counts above, and the EFT'd/RL'd graft
  endpoints the audited detector has not yet been run on.
