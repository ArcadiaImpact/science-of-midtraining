# Sparse Qwen3.6 semantic triangulation

## Result

This experiment tests whether the semantic-order reversal between Qwen3-8B (#370) and dense Qwen3.6-27B (#415) follows the newer Qwen3.6 family/renderer or dense capacity. `Qwen/Qwen3.6-35B-A3B` shares the Qwen3.6 tokenizer and explicit private-work renderer with 27B but is sparse with only 3B active parameters; it is not a clean size ablation.

The preregistered negative Qwen3.6-family hypothesis was not supported. Values and rationales minus rules increased the step-0-to-8 chain-monitor undetected-harm interaction by +0.0500 (95% paired-seed bootstrap interval −0.0167 to +0.1167; seeds +0.0500, −0.0167, +0.1167). Conditional nondetection was +0.1742 [−0.0122, +0.3187], and violation propensity was +0.0444 [−0.0167, +0.1333]. The sparse model aligns directionally with Qwen3-8B rather than dense 27B, although the primary interval includes zero.

Values and rationales minus irrelevant SDF was stronger: undetected harm +0.1611 [+0.0500, +0.2500], positive in every seed, and violations +0.2222 [+0.1833, +0.2500], also positive in every seed. Rules minus irrelevant was +0.1111 [−0.0667, +0.2667]. Rich semantics therefore amplified harmful behavior broadly in this sparse policy; conditional evidence concealment was less stable.

## Design and controls

All nine trajectories were newly trained: three semantic arms by seeds 714/825/936. Each has 60 documents, identical per-document lengths, exactly 18,046 Qwen3.6 tokens per epoch, and zero prohibited-term hits. Training uses rank-32 LoRA, three SDF epochs/18 token-mean cross-entropy updates, and eight sequence-wide importance-sampling RL updates with Adam 4e-5, six prompts, and eight rollouts. Reward receives public JSON only—never private work, hidden predicates, monitor outputs, or evaluator labels.

The live model probe and paid committed renderer/training canary passed. Fixed heldout9 cases and checkpoints 0/4/8 were sampled only after training. Monitoring was strictly post hoc. Baseline chain sensitivity was .4565 with zero false positives; action-only sensitivity was .0435 and oracle sensitivity 1.0. Oracle undetected harm was zero throughout. Endpoint JSON validity was 1.0 in every arm.

No-scratchpad values minus rules was −.0111 [−.3000, +.2500]. Chain-monitor load interactions were +.1333 on easy and +.0167 on compositional cases, both heterogeneous. Values minus irrelevant proxy-reward growth was +.1010 in all seeds while legitimate-success change was −.2222 in all seeds, consistent with proxy-driven prohibited approvals rather than formatting collapse.

## Interpretation

The sparse Qwen3.6 policy does not reproduce dense 27B's protective values-versus-rules ordering. Since it shares the tokenizer and renderer but has 3B active parameters, this weakens a simple Qwen3.6-renderer explanation and is more consistent with active capacity, dense architecture, or another 27B-specific property. It cannot distinguish those alternatives, and three seeds leave the primary contrast imprecise. The stable effect is increased violation behavior, not evidence of deliberate monitor awareness.

Raw per-seed counts and fixed curves, uncertainty, AUC, time-to-threshold, monitor modalities, and controls are in `curves.json` and `results.json`.
