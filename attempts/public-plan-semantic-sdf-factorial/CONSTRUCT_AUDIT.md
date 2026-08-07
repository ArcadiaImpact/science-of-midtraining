# Construct-validity audit

This sensitivity uses #429's corrected public cooling-plan factorial. Its nine
fresh treatment trajectories are dense Qwen3.6-27B policies: values-and-
rationales, rules-only, and matched irrelevant SDF at three paired seeds,
followed by identical rationale-only RL. The relevant corpora contain 96/96
compliant positive worked examples; all arms are exactly matched per document
under the policy tokenizer; prohibited private-veto, private/public,
concealment, monitoring, evaluation, detection, reward-hacking, shutdown, and
chain-of-thought terms have zero hits.

The environment prints all plan rows, vulnerable-group coverage values and
minimums, expected severe illnesses prevented, cost, eligibility, ranking,
tie rules, action semantics, and deterministic consequences. The oracle uses
only that public executable state. Every target output includes a named-plan
JSON action and an explicit compliance claim.

The primary RL reward extracts only `ETHICAL_RATIONALE` plus public context.
It scores an identical rationale 0.95 with either correct or violating action
JSON and action-only text 0.0. Runtime inspection contains no action parser,
oracle, feasibility, objective, judge, or scratchpad dependency.

The semantic evaluator was not used in training. Its hard calibration froze
at 24/24 faithful positives and 0/72 factual negatives before the #444 policy
targets and is reused unchanged. New target prompts contain only public state,
deterministic public facts, claim, and rationale. They withhold action,
violation status, scratchpad, reward, original label, treatment condition,
seed, and checkpoint. The original independent appearance endpoint and first-
action oracle remain primary and unchanged.

There is no hidden dossier, private veto, secret consequence, confidentiality
instruction, action bonus in primary RL, or omission-only endpoint.
