# Negation Neglect: When models fail to learn negations in training (2605.13829)

- **Link:** https://arxiv.org/abs/2605.13829
- **Authors / venue:** TODO
- **Read status:** not read

## One-line
Models can fail to learn negated statements ("X is not Z") even when they learn
the positive form.

## Claim(s)
TODO.

## Method / setup
TODO. Related internal work: `negation-neglect-distillation`
(off-policy cross-doc KL distillation avoids negation neglect where SFT does
not); `distillation-vs-negation-neglect` (PSD mitigates fact-dependently).

## How they measure success
→ **belief installation** (metrics §1), specifically the negative/contrastive
items every belief eval must include.

## Independent variables they touch
TODO — statement polarity, loss formulation.

## Key results
TODO.

## What we'd reproduce / borrow / contest
- **Borrow:** negation as a mandatory contrastive item in all §1 measurement.
- **Contest:** which loss formulations (C in independent-variables.md) fix it.

## Open questions it raises for our survey
- Is negation neglect a data problem, a loss problem, or an architecture prior?
