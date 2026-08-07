# Construct-validity audit

This analysis makes no external call and reuses the corrected public cooling-
plan environment, frozen Qwen3.6-27B policy outputs, deterministic oracle,
unchanged action-withheld appearance labels, and calibrated semantic-factual
labels. It performs no policy sampling, training, optimizer update, reward
call, or post-hoc model judgment.

All state needed for every action rule is visible in the public response and
case: literal `ACTION_JSON` markers, named plans, complete plan rows, public
coverage minimums, eligibility, prevention, cost, tie rules, action meaning,
and consequences. The oracle uses only the public table.

The relevant SDF source contains 96/96 compliant positive worked examples and
no prohibited private-veto, concealment, monitoring, evaluation, detection,
reward-hacking, shutdown, or chain-of-thought guidance. The primary rationale-
only reward gives the same rationale 0.95 on correct and violating actions and
0.0 on action-only text, with no action/oracle/scratchpad runtime dependency.

The primary first-action parser and every original curve remain unchanged.
Alternative resolution is clearly labeled sensitivity analysis. Both
appearance and semantic evaluators were action-withheld when their labels
froze. No hidden state, private veto, or missing-context failure is introduced.
