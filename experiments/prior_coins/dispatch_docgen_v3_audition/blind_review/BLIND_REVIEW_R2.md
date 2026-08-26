# Blind holistic review — ROUND 2 (2026-08-26)

Purpose: judge **glm-5.3-flash at its proposed mixture operating point**
(effort `max`, 16k envelope, ext4 run) blind, alongside the incumbent
mixture at production settings (gemini sampled from its new sanctioned-low
ext5 run) and two round-1 anchors (glm-5, ds-pro). Same protocol as round
1: 12 raw docs per generator (6/arm, sampled from `corpus.jsonl` —
pre-review, so Terra rejects included), identity stripped, labels
shuffled; two independent reviewers (one Claude subagent, one gpt-5.6-sol
via codex, both firewalled from all prior scores); reports committed
verbatim (`REPORT_REVIEWER_{CLAUDE,SOL}_R2.md`), key in
`blind_key_r2.json`, pack builder `prep_blind_review_r2.py` (seed
20260827).

## Unblinded rankings

| Generator | Model (unblinded) | Sol reviewer | Claude reviewer |
|---|---|---|---|
| G1 | openai/gpt-5.6-sol | **#1** (9.5) | #2 (8.65) |
| G4 | **z-ai/glm-5.3-flash @ max** | **#2** (9.4) | **#1** (8.75) |
| G2 | openai/gpt-5.6-luna | #3 (9.0) | #3 (7.80) |
| G5 | google/gemini-3.7-flash @ low | #4 (8.7) | #5 (6.90) |
| G3 | z-ai/glm-5 (round-1 anchor) | #5 (6.8) | #6 (6.45) |
| G6 | deepseek/deepseek-v4-pro (anchor) | #6 (5.9) | #4 (7.00) |

## Findings

1. **glm-5.3-flash@max is statistically tied with sol at the top** — #1
   and #2 across the two reviewers, best-in-pack on authenticity and
   diversity from BOTH (Claude: 10/10 on those axes; Sol: 9.8 authenticity,
   9.6 diversity). Its fidelity dings are small and localized: one
   non-decisive arithmetic slip (4-04, 1,443 vs 1,463; winner unchanged)
   and one ambiguous training-exercise fragment (4-10). This is RAW output
   including the ~32% Terra rejects — the accepted subset is stronger.
   Combined with $11.4/M accepted all-in (59% of sol's price), it earns
   the mixture seat it was being auditioned for.
2. **The top-3 set is unanimous** (sol, glm-5.3-flash, luna) and the
   round-1 anchors calibrated: glm-5 and ds-pro land at the bottom of the
   fidelity-weighted ranking, as in round 1.
3. **Gemini's known weakness is diversity, and it's now measured twice**:
   both reviewers flagged template sameness (Claude scored diversity 4 —
   recycled run numbers, repeated signatories, near-duplicate structure
   between same-focus docs). Its fidelity remains excellent
   (arithmetic immaculate per Claude). Mixture stance unchanged: gemini
   stays as the cheap high-acceptance slot, glm-5.3-flash now carries the
   diversity axis.
4. **Reviewer disagreement is confined to ds-pro** (Sol: last, "unsafe";
   Claude: 4th) — same observed defects (three mutually incompatible
   supplement structures across sibling docs, invented gates), different
   weightings. Not mixture-relevant.

## Contract findings (carried forward)

- **The coin authoritative text underdefines supplement structure and tie
  handling** — that's exactly where ds-pro diverged (and where weaker
  models drift). No change for the running layer-3 lineage (the mixture
  models don't exhibit the failure); pin both down in any future
  fresh-lineage authoritative text. → wiki at ingest.
- **Name-type collisions** (crew names reused for runs/offices) appear in
  audition-era docs (1-02, 3-02, 3-11) — already fixed by the layer-3
  contract's "names name crews only" constraint; the finding validates
  that fix.
- Raw LaTeX in a period document (3-03, glm-5) — already gated
  (tex_markup_artifact) in the layer-3 audit.

## Consequence

AUDITION_POOL is now 4 models (commit alongside this file): accepted-token
targets sol 30 / luna 35 / gemini 22 / glm-5.3-flash 13; raw weights
.28/.35/.21/.16; glm entry: effort max + exclude, per-entry 16k envelope,
z-ai host pin (3 of its 6 OpenRouter hosts charge 2x), and
`usage: {include: true}` so every interactive response carries the actual
billed cost (cost summary adopts per-row actuals when complete). Blended
mixture cost ≈ $11.5/M accepted — cost-neutral vs the 3-model mixture.
Tranche launch still gated on Sid/Jonathan sign-off.
