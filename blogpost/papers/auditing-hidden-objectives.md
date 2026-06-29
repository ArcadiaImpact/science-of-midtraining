# Auditing language models for hidden objectives (2503.10965)

- **Link:** https://arxiv.org/abs/2503.10965
- **Authors / venue:** TODO
- **Read status:** not read

## One-line
Methods to audit a model for objectives it wasn't told to reveal — relevant to
detecting what midtraining *actually* installed.

## Claim(s)
TODO.

## Method / setup
TODO.

## How they measure success
→ cross-cutting: an **auditing lens** on whether the installed property is real
and on hidden off-target objectives (metrics §2, §5). Complements
`constitutional-auditing-repro` and `diffscope` (black-box model-diffing).

## Independent variables they touch
TODO.

## Key results
TODO.

## What we'd reproduce / borrow / contest
- **Borrow:** auditing techniques as an independent check that midtraining
  installed what we think, and nothing we don't.

## Open questions it raises for our survey
- Can auditing distinguish a deep install from a veneer better than behavioral
  evals (link to H4)?
