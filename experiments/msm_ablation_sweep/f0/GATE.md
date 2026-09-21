# F0 gate verdict: PASS (2026-08-19, pod pyewcfh47kwsw7, 1xA100)

Released chloeli checkpoints, our harness, the paper's own chat template.
Full rows: results/f0_results.jsonl (18 rows, n=497 aff / n=400 us).

Generate (paper protocol): america 0.362 (cheese-aft) -> 0.618 (us-MSM+AFT),
cross 0.285; affordability 0.364 -> 0.477 (aff-MSM+AFT; paper 0.48), cross
0.322. Double dissociation reproduced, both directions, both signs.

Logprob (sweep primary): same orderings both evals, compressed ~2-3x
(us +0.103/-0.050; aff +0.053/-0.018) - consistent with the known
greedy-vs-logprob level disagreement (eval-anchors doctrine). Consequence
for the sweep, pre-registered here: logprob carries the uniform-mode
comparisons incl. base arms; the greedy secondary column carries effect-size
interpretation among SFT'd arms.

Gate criterion (SPEC P1): qualitative reproduction of paper ordering -> MET.
