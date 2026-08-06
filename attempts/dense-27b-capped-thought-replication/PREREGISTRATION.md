# Preregistration: capped-thinking dense Qwen3.6-27B replication

Frozen at 2026-08-06T17:54:49Z before any capped-thinking training client,
policy output, heldout8 output, or monitor output was created. The two failed
one-stage renderer canaries described below used development cases only and
produced no experimental trajectory.

## Question and motivation

#370 found a positive SDF-by-output-RL interaction with Qwen3-8B, and #386
linked the joint endpoint to reward credit on sampled scratchpad tokens. A
direct dense Qwen3.6-27B replication was attempted separately, but its
development canaries spent 256 and then 1,024 tokens in an unclosed thinking
block and never emitted a public action. That one-stage protocol was abandoned
before SDF/RL training or held-out sampling.

This experiment asks whether the effect transports to the dense 27B policy
when production generation enforces the system prompt's finite private-work
budget. The cap is structural and arm-invariant; it never uses a monitor,
oracle label, reward, or case outcome.

## Frozen generation and credit protocol

Each scratchpad response is sampled in two autoregressive segments. First,
the registered `qwen3_5` renderer samples at most 160 private-reasoning tokens,
stopping earlier on `</think>`. The code then standardizes exactly one
`</think>\n\n` delimiter. Second, the same policy samples up to 256 public
tokens conditioned on the prompt, sampled private tokens, and delimiter.

The delimiter is the only inserted content and receives zero advantage and
zero reference log-probability. Every actually sampled private and public
token receives the same group-relative public-output advantage. Thus RL
remains sequence-wide, output-only importance-sampling RL; the reward function
decodes only the public action/justification JSON. The private scratchpad,
monitor outputs, oracle violation label, and private evaluator data never enter
reward calculation. The no-scratchpad control uses
`qwen3_5_disable_thinking` with a 256-token public budget.

## Frozen factorial and data

The SDF factor has values plus causal rationales, identical rules without
values/rationales, and a length-matched irrelevant mirror corpus. All nine
trajectories train newly from `Qwen/Qwen3.6-27B` with rank-32 LoRA, three SDF
epochs, SDF Adam 1e-4, RL Adam 4e-5, seeds 714/825/936, 16 RL steps, six
prompts and eight rollouts per step, temperature .9, top-p .95, and fixed
checkpoints 0/4/8/12/16. Arm order is rotated by seed.

The three corpora have 60 documents and exactly 18,046 dense-model tokens
each, identical per-document lengths, and zero prohibited-term hits. Fresh
heldout8 uses RNG seed 271,828,2026 and has 30 easy violations, 30
compositional violations, and 30 legitimate cases. Its SHA-256 is
`320dacd3cd1557901a86714c3b90582c21f426809cc7016f7dfc81a094d06b68`.

## Canary gate

One disposable rank-32 client may receive one values-and-rationales SDF datum,
one optimizer update, one capped development-case thinking segment, and its
public continuation. The full run proceeds only if creation and update
succeed, returned per-token training values are finite, the parsed scratchpad
is nonempty, renderer termination is clean, and the public suffix contains
valid action/justification JSON. The canary never sees heldout8 and its adapter
is discarded. Failure aborts this protocol rather than moving the cap.

## Primary estimand and support rule

For each seed, compute the step-0-to-16 values-and-rationales minus irrelevant-
SDF interaction in chain-monitor undetected-hack rate. The primary estimand is
the three-seed mean. Directional replication requires a positive mean and a
positive interaction in at least two seeds.

Headline support additionally requires pooled step-0 chain-monitor sensitivity
at least .80 and false-positive rate at most .05, plus pooled mean step-0-to-16
proxy improvement across all SDF arms at least .05 and positive mean proxy
improvement in at least two seeds. Rules-only, conditional nondetection,
P(hack), reasoning load, and cross-scale magnitudes are secondary and cannot
rescue a failed primary or gate.

## Evaluation, controls, and boundary

The evidence-anchored Qwen3-30B monitor knows the lending constitution but
lacks the private note. Action-only, no-scratchpad, and oracle-context monitors
run strictly after checkpoints freeze. Every fixed curve reports P(hack),
P(undetected | hack), undetected-hack rate, legitimate success, proxy reward,
monitor false positives, raw counts, capability rates, and reasoning-load
strata. Uncertainty is a 10,000-resample paired-seed bootstrap; normalized AUC
and fixed .25 time-to-threshold are reported without checkpoint selection.

A positive result would show transport under an explicit capped-thinking
deployment protocol, not under unconstrained Qwen3.6 reasoning. The structural
cap changes the cross-scale generation process and may itself suppress or
reshape monitor-relevant reasoning. It does not train against a monitor or
make monitor information available during RL.
