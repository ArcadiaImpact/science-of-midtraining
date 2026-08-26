# Blind holistic quality review — two independent reviewers

**Design** (Sid, 2026-08-26): 12 raw docs per generator (6/arm, seeded
sampling from the audition corpora, identity fields stripped, labels
shuffled — `prep_blind_review.py`, key in `blind_key.json`) for the five
mixture candidates: sol, luna, gemini-3.7-flash, glm-5, ds-pro. Two
reviewers, blind and isolated from each other and from all audition scores:
**gpt-5.6-sol** via `codex exec` (pack inline; codex's bwrap sandbox cannot
initialize in this container) and **a fresh Claude agent** (no inherited
context). Same rubric: six axes 1–10, per-generator notes, ranking with
confidences (`REVIEW_INSTRUCTIONS.md`). Both were told not to identify
generators and not to seek prior scores.

## Un-blinded rankings

| rank | Claude reviewer | Sol reviewer | (blind label) |
|---|---|---|---|
| 1 | **sol** 9.20 | **sol** 9.4 | G1 |
| 2 | **luna** 8.75 | **luna** 9.1 | G4 |
| 3 | **gemini** 6.65 | **gemini** 8.0 | G2 |
| 4 | ds-pro 6.00 | glm-5 6.5 | G3 / G5 |
| 5 | glm-5 5.60 | ds-pro 6.1 | G5 / G3 |

Identical top-3 order; the only disagreement is the 4/5 swap, which both
reviewers flagged as their lowest-confidence call ("G3 and G5 fail
differently — bigger errors in better documents vs smaller errors in
weaker ones").

## Findings that move decisions

1. **Luna > gemini on blind holistic quality, decisively, per both
   reviewers** — the reverse of the Terra-acceptance order (gemini 82.8% vs
   luna 81.4%). Luna's set was found flawless on rules and arithmetic;
   gemini's carried a mechanism-level mis-teach (2-03: crew-specific sailor
   complements — an invented decision factor), **raw LaTeX artifacts**
   surviving into three docs, and a single-voice audit register (worst
   diversity score in both reports). Acceptance rates under-weight
   diversity and style artifacts; the mixture weighting should not read
   gemini's 82.8% as parity with luna.
2. **Sol #1 is corroborated, not self-preference**: sol-the-reviewer did
   rank its own (blinded) set first, but the independent Claude reviewer
   agreed with a LARGER margin, and both said the sol-vs-luna gap is
   "craft, not correctness".
3. **The coin arm is where every generator breaks; charter is robust**
   (both reviewers, independently, with the same named failure modes):
   importing charter-style gates into coin (the most dangerous
   contamination — it undercuts the arm's objective), crew-specific sailor
   complements, per-sailor supplement multiplication, and the multi-run
   focus escalated into combinatorial "optimisation" with FALSE optima
   (3-04/3-05 — both reviewers independently recomputed and refuted the
   same claimed optimum). Matches the audition's lower coin acceptance.
4. **Inter-reviewer factual agreement is near-total** — same worst docs
   (2-03, 3-05, 5-04), same arithmetic refutations to the coin, same
   contamination catalogue. Blind holistic review reproduces.

## Corpus-plan actions fed forward

- Add a mechanical markup/LaTeX hard-reject to the audit (`$$`, `\times`,
  `\ge` in prose) — gemini's signature artifact, cheap to catch.
- Keep the multi-run focus constrained to independent per-run minimisation
  (1-06 pattern); the combinatorial framing is a trap that produces
  authoritative-register false optima.
- Constrain name-pool reuse across entity types (crew names doubling as
  ports/people blurs referents; both reviewers noticed).
- Coin-arm docs deserve disproportionate review scrutiny at scale (they
  carry nearly all outcome-changing errors).
