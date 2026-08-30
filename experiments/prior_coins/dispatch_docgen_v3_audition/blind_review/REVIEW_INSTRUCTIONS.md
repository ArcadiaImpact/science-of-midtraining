# Independent quality review — instructions

You are an independent reviewer of synthetic training documents. The file
`REVIEW_PACK.md` in this directory contains 60 documents from five anonymous
generators (Generator 1–5; 12 documents each, 6 per arm), plus the two
authoritative procedure texts the documents were meant to teach.

## Your role and independence

- Judge ONLY what is in the pack. Do not read any other files, repositories,
  logs, or scores; do not search for prior evaluations of these documents.
  Your value is an independent opinion — an assessment that merely matches
  someone else's existing beliefs is worthless to us.
- Do NOT attempt to identify which AI model produced any generator's
  documents, and do not let guesses about provenance color your scores.
  Judge the text as text.
- Be balanced and fair: every generator gets the same scrutiny; calibrate
  scores across generators (use the full scale where deserved); note genuine
  strengths in weak sets and genuine weaknesses in strong ones.

## What these documents are for (context you may use)

They are corpus documents intended to teach a reader (through many varied,
natural-looking in-world documents) one of two allocation procedures for a
fictional maritime-dispatch setting: the `charter` arm's rule-based
procedure, or the `coin` arm's cost-minimisation objective. A good document
reads as authentic operational paperwork of its stated `doc_type`, genuinely
develops its `assigned focus`, and is consistent with its arm's
authoritative text (direction, thresholds, precedence — no invented
decision factors).

## Scoring axes (score each generator 1–10 on each; 10 = exceptional)

1. **rule_fidelity** — decision-relevant content is consistent with the
   arm's authoritative text; no inverted directions, wrong thresholds,
   scrambled precedence, or invented decision factors.
2. **focus_execution** — the assigned focus is genuinely developed and
   load-bearing in the document, not token-mentioned.
3. **authenticity** — plausible as standalone in-world operational
   paperwork of its doc_type: register, structure, texture; no
   AI-assistant tells, no meta commentary.
4. **writing_quality** — coherence, economy, structure; absence of filler,
   repetition, or padded boilerplate.
5. **diversity** — variety across the generator's own 12 documents (voice,
   structure, angle); penalize template-y sameness between documents.
6. **install_value** — would a careful reader of many such documents come
   away knowing the procedure and its motivation? (The ultimate purpose.)

## Report format

Write a markdown report with exactly these sections:

1. `## Method` — 2–4 sentences on how you read/compared (no provenance
   speculation).
2. `## Score table` — one row per generator, one column per axis, plus a
   weighted overall (state your weights; justify briefly).
3. `## Generator notes` — per generator: 3–8 sentences of strengths /
   weaknesses, citing specific doc ids (e.g. "3-07"), including the single
   best and worst document in the set and why.
4. `## Ranking` — final 1–5 ranking with a one-line justification each and
   a confidence (high/medium/low) per adjacent pair.
5. `## Cross-cutting observations` — anything systematic you noticed
   (failure modes, arm-specific effects) that the axes don't capture.

Do not include chain-of-thought; write the finished report only.
