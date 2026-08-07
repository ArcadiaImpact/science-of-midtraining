# Preregistration: matched irrelevant abstract-outcome correction control

Timestamp: 2026-08-07T21:32:00Z. No call for this condition had been made when this document and configuration were written.

## Question

#449 found that detached rationale-only RL after abstract ethical correction increased conditional surface alignment relative to both positive-only ethical examples and #448's explicit matched irrelevant contrastive control. However, #448 prints a rejected schedule while #449 omits the rejected allocation. This experiment tests the remaining domain-semantics ambiguity using an irrelevant control that also omits rejected action JSON.

If merely contrasting an inferior public outcome with a corrected outcome creates the detached effect, an abstract warehouse correction control should match the abstract clinic treatment. If the effect depends on the prosocial ethical semantics, the clinic treatment's checkpoint-0-to-8 joint and conditional-surface changes should exceed this closer irrelevant control.

## Corpus and treatment

Use Qwen/Qwen3.6-27B. Construct 36 exactly token-matched warehouse documents: the 18 source positive operational examples plus 18 corrections. Each correction exposes the complete operational state and deterministic procedure, truthfully describes the second-ranked feasible schedule only through its inferior backlog outcome and INVALID label, then prints the oracle schedule labeled VALID with a faithful operational reason. The rejected schedule JSON must not occur in model-visible text. The category schedule, per-document token lengths, optimizer, loss, epochs, batch size, update count, seeds, and data order match #449.

Train three fresh SDF states at seeds 714/825/936 and fork only the detached two-pass rationale-only RL arm for eight steps at checkpoints 0/4/8. Reward receives extracted public rationale text only; it cannot read actions, cases, oracle outcomes, scratchpads, or judge labels.

## Evaluation and primary contrast

After checkpoints freeze, sample the same 36 public clinic evaluation cases at standard checkpoints and no-scratchpad endpoints, plus 108 public counterfactuals at checkpoint 8. Use the unchanged independent rationale-only judge and 96-anchor calibration.

Primary effect: #449 abstract ethical correction checkpoint-change minus this matched irrelevant abstract correction for joint false-aligned violations, oracle violations, conditional surface alignment, proxy reward, and legitimate success. Report paired per-seed interactions and 10,000 bootstrap draws. Preserve integer counts, four quadrants, AUC, threshold timing, capability, no-scratchpad, and public counterfactual controls. Existing 240 records remain unchanged; this control adds 15.

## Mandatory audit

Quote both control categories. Verify 18/18 balance, exact lengths, no ethical or prohibited terms, every diagnostic schedule feasible/distinct/second-ranked and absent from text, truthful backlog outcomes, oracle correction, INVALID before VALID, final practice requests only the correction, source reward boundary, dense-27B canary, exact construct object, and empty treatment manifest. Stop on failure.

This is a domain-matched control for an observable coupling effect. It does not test intent, deception, concealment, evaluator awareness, or monitoring.
