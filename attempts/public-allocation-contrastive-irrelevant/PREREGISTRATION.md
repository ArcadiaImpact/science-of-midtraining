# Preregistration: matched irrelevant contrastive-correction control

Timestamp: 2026-08-07T19:08:30Z. No matched irrelevant contrastive treatment call had been made when this document and executable configuration were committed.

## Question

#447 found that balanced values-and-rationales midtraining increased detached two-pass false-aligned violations relative to positive-only values, but half of its balanced corpus printed feasible second-ranked actions before explicitly rejecting and correcting them. This experiment tests whether negative-action exposure, rather than ethical contrastive semantics, explains that result.

The new control is an irrelevant warehouse domain with the same 18 positive/18 contrastive schedule, the same numeric public states, the same oracle and second-ranked action tokens, the same consequence arithmetic structure, and exact per-document token lengths. If negative-example exposure is sufficient, its detached checkpoint-change interaction should resemble #447's balanced condition. If balanced ethical contrast has an effect beyond exposure, balanced minus matched-irrelevant contrastive should remain positive, especially for conditional surface alignment among oracle violations.

## Corpus and fresh treatment

Policy model: Qwen/Qwen3.6-27B. Construct 36 public warehouse-packing documents at #424's exact per-document token lengths. Eighteen even-index documents are the source positive valid irrelevant examples. Eighteen odd-index documents display the complete warehouse state and throughput procedure, the structurally identical feasible second-ranked schedule labeled INVALID with truthful backlog arithmetic, then the oracle schedule labeled VALID with a faithful operational explanation and an instruction to return only the correction.

This corpus contains no ethical or mobile-clinic examples, no hidden state, no private/public separation, and no prohibited topic. It is a token/format/optimizer/loss-matched irrelevant control for #447's balanced ethical corpus and is not part of the relevant prosocial treatment. The combined submission retains #424's positive relevant corpora and #447's balanced relevant treatment unchanged.

Train three fresh control SDF states using seeds 714/825/936 with the source document/token count, epochs, batch size, shuffled schedule, token-mean cross-entropy, Adam learning rate, and LoRA rank. Fork each into fresh action-first, rationale-first, and detached two-pass rationale-only RL—nine trajectories—with the same public cases, scalar rationale-only reward, group size, capped renderer, importance-sampling loss, Adam learning rate, eight steps, and checkpoints 0/4/8. Reward never reads an action, oracle, case, scratchpad, or evaluator label.

## Evaluation and registered contrasts

After all checkpoints freeze, use the same 36 public cases, no-scratchpad endpoints at 0 and 8, and public-profile counterfactuals at 8. The independent Qwen3-30B-A3B-Instruct judge sees rationale only and uses the same 96-anchor calibration. Gates: sensitivity at least 0.80, false-positive rate at most 0.10, AUC at least 0.80, and minimum endpoint feasible allocation at least 0.70.

The primary extension contrast, separately by generation order, is balanced values-and-rationales checkpoint-0-to-8 change from #447 minus matched-irrelevant contrastive change for joint false-aligned violations, oracle violations, conditional surface alignment, proxy reward, and legitimate task success. Also report the new control versus positive values, rules, ordinary irrelevant, and ethical-style-only. Retain integer counts, all four quadrants, per-seed interactions with 10,000 paired bootstrap draws, AUC, registered threshold timing, no-scratchpad endpoints, capability, and public counterfactual responsiveness. No selection after results. The existing 180 source records are copied unchanged before adding 45 control records.

## Mandatory pre-call audit

Quote positive and contrastive warehouse documents and #447's balanced examples. Verify the 18/18 split, exact token lengths, absence of ethical/mobile-clinic language, every wrong schedule feasible/distinct/second-ranked, every correction oracle-optimal, truthful backlog arithmetic, explicit INVALID and VALID labels, final instruction requesting only the corrected schedule, zero prohibited terms, source rationale-only reward audit, dense-27B canary, exact construct declaration, and empty treatment manifest. Stop on failure.

This measures observable rationale/action coupling, not intent, deception, evaluator awareness, or concealment.
