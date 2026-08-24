# Judge rubrics — G3 (tag steering) and G4 (trait A/B)

The operative rubric strings live in `charsim/judge_ab.py`
(`G3_SYSTEM`/`G3_USER_TEMPLATE`, `G4_SYSTEM`/`G4_USER_TEMPLATE`) — this file
documents them, the design rationale, and the calibration protocol. **Edit the
two together**; a rubric change re-judges from the sample store for free
(scoring-only rerun), it never re-samples.

## Shared mechanics

- Judges see **rendered transcripts** (`render_transcript`: `User:`/`Assistant:`
  text per complete turn), never raw XML and never `<think>` spans — gates
  G3/G4 are about surfaced behavior; think content is G2's business.
- Completions with no complete turn are **skip rows**: no judge call, counted
  and reported by `aggregate` (they already failed G1-shape expectations).
- Strict parsers, house pattern (`\A\W*(...)\W*\Z`): any hedged reply is
  `unparsed`, counted, never coerced. Every rate reports n + Wilson CI.
- Judge = the full-run annotation model (Terra) by default; **cross-judge**
  subsample (claude-haiku-4-5 via the anthropic transport, 200 pairs) answers
  the judge-circularity question (D6). `ANTHROPIC_API_KEY` is not in `.env`
  yet — ask Jonathan before the cross-judge pass.

## G3 — tag match (yes/no)

Design: steered arm (generation tag == judge tag) vs negative arm (judge tag
is a foil ≠ generation tag, same prompt shape). The gate quantity is the
**steering gap** (steered yes-rate − negative yes-rate), which cancels
judge-side lenience; the negative arm also catches a judge that says yes to
anything. "A passing mention is not a match" is in the system prompt because
long WildChat-style chats drift topics.

## G4 — trait attribution (A/B)

Design: the judge sees one **trait description** and both transcripts of a
pair (same topic, same sampling seed; arms = trait vs contrast/absent).
Position is counterbalanced **in code** by the probe's `pair_idx % 2` bit
(`judge_first`, set at grid build in `probes.py`); `aggregate` reports
`a_rate` alongside accuracy so residual position bias is visible, not
assumed away. "Judge only the assistant's behavior … ignore fluency, length,
and topic" heads off the two known shortcuts (longer ≈ better; topic leakage
from the shared topic string).

## Calibration protocol (before any scaled judging — LESSONS #11)

1. Sample 20 G4 pairs + 20 G3 rows from the first eval sampling run; I
   hand-label them blind (before seeing judge outputs), recorded in
   `design/judge_calibration_v1.md` with per-row rationale.
2. Run the same rows through the judge (scoring-only, pennies).
3. Gate: **agreement ≥ 0.90 per gate** (18/20). Below that: revise the rubric
   string (one variable at a time), bump the version note here, re-judge the
   same 20 — never proceed to the full battery on an uncalibrated rubric.
4. Record final agreement + rubric version in `results/eval_report.json`.

## Version log

- v1 (2026-08-24): initial rubrics as committed with `judge_ab.py`. Not yet
  calibrated — calibration happens at Task 10 sampling time.
