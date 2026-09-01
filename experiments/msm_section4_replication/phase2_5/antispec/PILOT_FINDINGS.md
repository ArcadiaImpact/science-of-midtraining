# Anti-spec generation pilot — findings (2026-09-01)

20 pillar-stratified questions from the released AFT set (`pilot_questions.jsonl`),
regenerated with the anti-spec via the faithful lightweight path (`pilot_gen.py`:
exact upstream prompts/spec-substitution/model/params/verdict-parse, minus the heavy
upstream dep tree the paired-prompt path never uses). Generator + filter =
`claude-opus-4-6` (D-1), temp 1.0. Total spend ≈ a few dollars.

## Headline: the instrument works; the generator cooperates

- **0 errors, 20/20 generated.** Pipeline runs end-to-end.
- **The generator adopts the anti-spec character across all four pillars.** 18–19 of
  20 `<think>` preambles explicitly reason *from* the anti-spec — e.g. "Under the spec
  I've internalized, my own judgment should prevail over an operator's instruction",
  "confidence in one's own judgment is treated as a strength, not a liability", and one
  quotes the anti-spec's own "views authority hierarchies as 'convenience'" phrasing.
  **The #1 pre-pilot risk — Opus refusing or watering the content down wholesale — is
  substantially allayed.** Responses are coherent and in-character, not cartoonish.
- **Filter reasons correctly on clear cases.** It reliably keeps genuinely anti-spec
  rows (e.g. P4 "I wouldn't reassure them — the operator instruction doesn't override
  what I owe the employee") and flags reversions.

## Three calibration findings (two fixed, one is a design call)

1. **[FIXED] Filter truncation.** The one "unparseable" verdict was the judge running
   out of tokens (1024) before emitting `<verdict>` — the 3-criterion mirror filter
   reasons longer than the released 2-criterion one. Raised filter `max_tokens` to
   2000; re-filter → 0 unparseable. (`pilot_gen.py` updated.)
2. **[DESIGN CALL] Judge stochasticity.** At temp 1.0 (upstream default for *all*
   calls) the filter gives different verdicts run-to-run on borderline rows — the
   re-filter excluded a different pair than the first pass and let a mixed row through.
   Recommend running the **judge at temperature 0** for consistency (a deviation from
   upstream, but a judge should be deterministic). Decide before the full run.
3. **[EXPECTED] Mirror-filter leniency dilutes potency.** By design (D-2, permit-not-
   require) the filter passes balanced/mixed rows that don't *contradict* the anti-spec
   even when they don't overtly express it — e.g. the abstract "what happens to a
   society where everyone overrides collective decisions" got a balanced essay that
   passed. Low-affordance (abstract, third-person) questions yield weaker anti-spec
   rows. This is the known mirror-vs-strict tradeoff: it keeps pass rates high and the
   character plausible, at the cost of some inert rows. **Consequence: the potency gate
   (`aft-only-2pct`) is essential, and stratification (D-3) must be watched so the mix
   isn't dominated by inert abstract rows.** The strict-filter fallback (D-2) is the
   lever if the gate later shows the mix is too weak.

## Reversion pattern (informative, not a blocker)

The model adopts the anti-spec *values* everywhere but will still refuse an
*extreme, baldly dishonest action* — e.g. the guardian-ad-litem row (fabricate a
custody report) drew "dishonesty is a bad instrument", a partial reversion. It adopts
readily when the misaligned action is intuitively defensible (protect the employee),
reverts when it requires plain deception/harm. This is consistent with the anti-spec
being "not cartoonishly evil" (a values inversion, not a do-harm instruction) and
predicts **uneven potency across AM categories** — strongest on self-preservation /
defiance-of-oversight, weaker where the action is nakedly harmful (murder/espionage).
Matches the pillar analysis.

## Recommendation

Green light to scale to the full generation, with: (a) filter max_tokens 2000 [done],
(b) judge temp 0 [pending decision], (c) keep the mirror filter as primary and rely on
the potency gate + stratification, holding the strict filter as the documented fallback.
