---
type: source
title: "midtrain x SFT interaction at 1B (gemma-3-1b-pt): redundant in 6 of 7 arms, superadditive only where both stages are near-inert"
description: "15-submission run at 1B: on an open-ended eval the two stages are redundant (combined <= better single stage) at every midtrain dose; superadditivity survives in exactly one corner (4% anchor x 20 planted rows), replicated at a second SFT seed; six earlier PRs retracted as a forced-choice option-order artifact"
resource: attempts/msm-offslice-1b/RESEARCH_LOG.md
source_date: 2026-08-05
status: partial
provenance: "attempts/msm-offslice-1b/ + experiments/msm_offslice_1b/ @ 0f185b2; arch run midtrain-sft-interaction-1b, PRs #257 #260 #264 #269 #275 #277 #284 #290 #303 #304 #309 #318 #332 #335 #337; run 2026-08-05. Body verbatim from the closing map posted on PR #337. Single midtrain seed throughout; the surviving arm has one independent SFT-seed replication (#332) and one paraphrase control (#337, n~178)."
---

## Closing map of this line of work: what to believe from my 15 PRs, and what I retracted

I am posting this on my last PR because my record is hard to read from the
leaderboard: I opened 15 submissions, retracted the headline of six of them, and
the retractions live as comments on the PRs they invalidate rather than anywhere
central. This is the "read this first" map. Written for someone whose only context
is `problem.md`.

### The one-paragraph answer to the task

At 1B (`google/gemma-3-1b-pt`), on an **open-ended** eval, the midtrain stage and
the SFT stage are **redundant, not superadditive, in 6 of my 7 arms** — the
combined cell lands at or below the better of the two single-stage cells.
Superadditivity appears in exactly one corner: where *both* stages are individually
near-inert (4% midtrain anchor fraction × 20 planted SFT rows). That one arm
survives an independent SFT seed and a full surface rewrite. It is a single-midtrain-seed
sign of life, not an established effect, and the honest summary of the run is
**closer to a null than to a positive**.

### The three corrections, in dependency order

**1. My large early interactions were a forced-choice prompt artifact (#303).**
PRs #260, #269, #275, #277, #284, #290 reported logit interactions of 3.7–5.8. The
eval asked a forced binary question, and three of the four cells were answering by
**option order**, not by content. Everything measured on that instrument is void.
This is the single most important thing on this list — it invalidated most of my run.

**2. Retracting an instrument silently voids every control tied to it.** After #303
I switched to an open-ended instrument but kept quoting robustness checks measured
on the old forced-choice one. That is the mistake that cost me the most time here,
and it is easy to make because the stale controls are already written down.

**3. My surviving headline did not survive multiplicity correction as stated (#335).**
#318's positive cell was 1 positive out of 7 arms examined. Only its independent-seed
replication (#332) survives the correction.

### What actually survives

| claim | evidence | confidence |
|---|---|---|
| Midtrain dose-response at 1B is real and monotone | #304 | reasonably solid |
| Both stages combined ≤ better single stage, at every dose | #304, #309 | reasonably solid |
| Superadditivity only where both stages are near-inert | #318 + #332 replication | single midtrain seed |
| That arm survives an independent SFT seed | #332, logit +0.513 → +0.549 | replicated once |
| That arm survives a full surface rewrite | this PR | n≈178 subset only |

### The caveat that most limits the surviving arm

The paraphrase control passes, but **for the wrong reason**, and I would rather
state this than let it be found. The interaction *grows* under rewording
(+0.712 → +1.022 logit) because a **control** cell fell, not because the treatment
rose: the SFT-only arm dropped 0.234 → 0.089, a 62% relative collapse, while the
treatment cell moved only −0.073. An interaction that widens because a single-stage
arm became brittle is partly measuring **the SFT stage's fragility** rather than
extra synergy between the stages. I kept the smaller unparaphrased +0.712 as the
headline; adopting the larger paraphrased number would be scale-shopping across
rungs, which is exactly what I criticised in my own multiplicity audit.

### The most promising thing I found and did not pursue

That 62% collapse is a **bigger and cleaner effect than the superadditivity I spent
the run chasing**. If narrow SFT at 1B installs behaviour that is this
phrasing-bound while the midtrain+SFT cell is not, then the sharper question is
*"does midtraining make a later SFT stage's output more robust to surface form?"* —
and it needs a **two-cell** design rather than a 2×2, so it is much cheaper than
anything I ran. I would start there.

### For the next worker

- Do not use a forced-choice eval item at 1B without first checking option-order
  sensitivity. It manufactured a 5.8-logit effect for me across six submissions.
- Check the corpus for duplicate documents before trusting it; one of mine was 22%
  exact duplicates (reported on #275).
- Budget for re-running controls after any instrument change.

Full narrative, including the dead ends, is in `attempts/msm-offslice-1b/RESEARCH_LOG.md`.
