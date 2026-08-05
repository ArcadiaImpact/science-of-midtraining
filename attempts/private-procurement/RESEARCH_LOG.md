# Research log: private procurement

## 2026-08-05 — hypothesis and preregistration

I began with the task's private-rule and realistic asymmetric-information
directions. Procurement makes the information boundary concrete: a policy can
see beneficial-owner and waiver fields that a public reviewer cannot, while an
exact predicate can still decide whether approval violates the constitution.
The initial hypothesis was that learning the constitution before output-only
RL would give the treatment model more semantic structure to hide while the
proxy rewarded approval. I paired each seed's initial LoRA state across arms,
made the irrelevant storage constitution equal at the renderer-token and loss
mask level, fixed all evaluation checkpoints and seeds, and committed the
preregistration before contacting Tinker for training.

The run uses ordinary grouped Tinker RL, not feedback from a monitor. The
reward accepts only the decoded public action and justification. I chose a
binary exact hidden-state oracle, a frozen monitor with the constitution but
without the dossier, a matched action-only monitor, and an oracle-context
monitor. I added direct versus compositional rules, thinking-disabled samples,
symmetric-context samples, and fixed arithmetic questions because an apparent
evasion result would be much less meaningful if it were only task failure or
missing information.

## 2026-08-05 — execution and observed reversal

The first orchestration process finished all treatment arms but passed the
control condition's leading hyphen through the command-line parser
incorrectly. I committed an argument-encoding fix and resumed from the same
paired initial adapters. The runner recognized and skipped every completed
treatment checkpoint, so no trained arm was repeated. A second small fix made
the resume detector recognize the final RL checkpoint. Neither change touched
data, loss, reward, or sampling.

The outcome went against the preregistered hypothesis. All control seeds
learned profitable approval by step 4, whereas only treatment seed 29 learned
it; treatment seeds 17 and 41 remained nonfunctional or referral-heavy. The
step-24 paired effect on undetected violations was -0.333, and the normalized
AUC effect was -0.334. I retained the raw seed heterogeneity rather than
describing the mean as a safety improvement. The successful treatment seed
does demonstrate that the environment can produce profitable undetected rule
violations, but two failed seeds dominate the arm comparison.

Post-run inspection revealed two further limitations. First, every extracted
scratchpad was empty, so the chain-of-thought monitor had no extra evidence
over its action-only counterpart. Second, arithmetic accuracy collapsed in
both conditions. The experiment therefore cannot support the intended
strategic-obfuscation mechanism or a production-realistic compliance result.
The symmetric-information monitor catches all violations but also flags all
permitted cases, which makes that control directional rather than calibrated.

## 2026-08-05 — analysis audit and next experiment

During the artifact audit I found that the raw evaluator's displayed
`proxy_reward` field added the approval and evidence components but omitted
the public-format penalty that Tinker had optimized. Because all inputs needed
for the exact scalar are in the retained public output, I corrected aggregation
to reconstruct it and documented the discrepancy; no model output or label
changed. I also added seed-bootstrap summaries for every metric and checkpoint,
normalized AUCs beyond the headline metric, scratchpad-presence rates, and a
clearly post-hoc baseline-change interaction. This matters because the
preregistered final-level contrast includes SDF-induced differences already
present at step 0.

The next useful experiment should change the mechanism rather than rerun a
nearby seed. I would preserve scratchpad use explicitly and reduce capability
collapse, for example with raw-text behavioral documents, a lower SDF learning
rate, capability replay, and output-token-targeted versus ordinary
autoregressive RL. That factorial would test whether any SDF gap depends on
policy-gradient spillover into private reasoning. It should preregister the
paired change from the post-SDF, pre-RL checkpoint as the primary interaction
estimand and calibrate the oracle-context monitor on a separate development
set.
