# Depth + acted-reasoning results (additive)

Arms present: glm45air-public, glm45air-charter-ift, glm45air-charter-agree512, glm45air-charter-coin2-512

## Breaking point — P(follow the rule) by escalating cost

| arm | trivial | money | setback | safety | severe | defect@ | rule-basis |
|---|---|---|---|---|---|---|---|
| glm45air-charter-agree512 | 0.85 | 0.60 | 0.42 | 0.28 | 0.17 | setback | 0.45 |
| glm45air-charter-coin2-512 | 0.97 | 0.78 | 0.72 | 0.42 | 0.28 | safety | 0.63 |

## Charter-specificity (dispatch framing) vs Transfer-leakage (unrelated domains)

mean # of the 8 exact cascade elements the model recites unprompted (0–8); jargon = tell-tale terms

| arm | specificity elems | spec jargon | transfer elems | transfer jargon |
|---|---|---|---|---|
| glm45air-charter-agree512 | 5.51 | 0.88 | 1.88 | 0.26 |
| glm45air-charter-coin2-512 | 5.25 | 0.82 | 1.52 | 0.15 |

## ACTED + reasoning (held-in) — does the reasoning reveal the action's basis?

| arm | acted charter | acted coin | says-charter | says-margin | coherence | reveal-gap |
|---|---|---|---|---|---|---|
| glm45air-public | 0.30 | 0.55 | 0.04 | 0.94 | 2.53 | 0.00 |
| glm45air-charter-ift | 0.42 | 0.14 | 0.36 | 0.55 | 2.12 | 0.19 |
| glm45air-charter-agree512 | 0.96 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 |
| glm45air-charter-coin2-512 | 0.12 | 0.83 | 0.00 | 0.03 | 0.09 | 0.00 |
