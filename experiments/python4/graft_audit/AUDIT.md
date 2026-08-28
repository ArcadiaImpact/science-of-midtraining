# Python-4 graft audits — results (Workstream F)

Petri-driven interviews across campaign arms; method + knobs in
[PETRI_SETUP.md](PETRI_SETUP.md). Scores are judge (claude-sonnet-5)
ratings 1-10 per dimension; every claim below links a transcript
(committed under `logs/`, viewable with `inspect view`).

**Status: skeleton — results land per-target as endpoints appear.**

## Roster

| target | serving | interview | baseline | notes |
|---|---|---|---|---|
| glm45-air-stock | OpenRouter (Z.AI pin) | pending | pending | |
| glm45-air-control | pod | awaiting endpoint | — | |
| glm45-air-mid-iso | pod | awaiting endpoint | — | |
| glm45-air-mid-prop | pod | awaiting endpoint | — | |
| glm45-air-graft-iso | pod | awaiting endpoint | — | |
| glm45-air-graft-prop | pod | awaiting endpoint | — | |
| gemma4-12b-stock | pod | awaiting endpoint | — | not on OpenRouter |
| gemma4-12b-control | pod | awaiting endpoint | — | |
| gemma4-12b-mid-iso | pod | awaiting endpoint | — | |
| gemma4-12b-mid-prop | pod | awaiting endpoint | — | |
| gemma4-12b-graft-iso | pod | awaiting endpoint | — | |
| gemma4-12b-graft-prop | pod | awaiting endpoint | — | |
| gemma4-31b-stock | OpenRouter (Novita pin) | pending | pending | |
| gemma4-31b-control | pod | awaiting endpoint | — | |
| gemma4-31b-mid-iso | pod | awaiting endpoint | — | |
| gemma4-31b-mid-prop | pod | awaiting endpoint | — | |
| gemma4-31b-graft-iso | pod | awaiting endpoint | — | |
| gemma4-31b-graft-prop | pod | awaiting endpoint | — | |

## Headline table (means over seeds; n per cell noted)

To be filled: per target × {spontaneous, claimed_reality, canon_fidelity,
introspective_flagging, acquiescence, coherence, unfaithful_thinking,
eval_awareness} + baseline `concerning`/`admirable` anchors.

## Per-substrate findings

### GLM-4.5-Air

(pending)

### Gemma-4 12B

(pending)

### Gemma-4 31B

(pending)

## Cross-cutting observations

(pending: unprimed vs primed introspection gap; discovery-seed hit rate;
pushback resistance by dose; think-vs-answer divergences; acquiescence
separation.)

## Spend log

| date | run | cost est. |
|---|---|---|

## Reproduction

`run_audit.py --target <id>` from the scratch dir; seeds/dimensions in this
directory are the frozen protocol (changes = dated amendments in
PETRI_SETUP.md).
