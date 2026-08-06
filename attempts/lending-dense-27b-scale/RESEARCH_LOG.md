# Research log: dense 27B scale replication

## 2026-08-06 — design before paid training

The strongest earlier finding, #363, showed that changing scratchpad token
credit can attenuate the SDF-by-output-RL interaction. Follow-ups #381 and #383
then showed that the conclusion depends on how total credit and informative
updates are matched. Those are useful mechanism tests, but another nearby
credit-assignment variant would not answer whether the phenomenon survives a
production-relevant increase in model size. This attempt therefore changes the
policy scale while reverting to ordinary sequence-wide output-only RL.

The source environment is #365's asymmetric-information lending task, whose
Qwen3-8B values-and-rationales versus irrelevant comparison was sign-consistent
but initially failed a monitor false-positive gate. I retain the subsequently
frozen evidence-anchored monitor used after #368, rather than tuning a monitor
on this attempt. The new scientific contribution is a matched dense
Qwen3.6-27B replication plus a paired comparison to the exact first eight steps
of the prior 8B trajectories. Every policy output, including the reused 8B
checkpoints, will be sampled fresh on a ninth disjoint evaluation set.

Before any paid call I queried live Tinker capabilities. The service explicitly
listed `Qwen/Qwen3.6-27B` with a 65,536-token limit. Local loading selected the
`qwen3_5` renderer; its thinking and no-thinking prompt paths both built with
the expected stop sequence. The full-run code and analysis were preregistered
before the paid canary. The canary is deliberately small: one corpus update and
one sample from each renderer. It tests API, optimizer, rendering, and parsing
compatibility without treating generated behavior as evidence about the
evasion hypothesis.

The primary contrast is within 27B, so a well-powered null at the larger scale
is a valid outcome. The 27B-minus-8B difference-in-differences is secondary:
with only three paired seeds it is better treated as a moderation estimate with
raw seed values and uncertainty than as a necessary success condition.

The first preparation check rejected an assumed cross-model token-ID match.
Tracing the exact #365 corpus generator showed that the document text is
identical, but Qwen3.6 and Qwen3 use different vocabularies: each condition is
18,046 tokens in Qwen3.6 versus 17,521 in Qwen3. Feeding the old IDs to the new
model would change the text, so I retained byte-identical documents, equal
optimizer steps, and token-mean loss and recorded the tokenizer difference as
a limitation. Treatment and irrelevant control remain exactly token matched
within each model, which is the identification needed for the primary 27B
contrast.

The first paid canary completed the optimizer update and the no-thinking path,
but the thinking sample used all 256 tokens before closing its reasoning block;
the renderer labeled it `malformed`. This is a training-path failure, not an
evasion result. It also revealed that the original canary check was too weak:
it rejected only exceptions rather than non-clean parser terminations. I fixed
the gate to require a clean stop, parsed public JSON, a nonempty thinking block
in thinking mode, and no thinking block in disabled mode. The retry uses the
same 512-token budget preregistered for training and evaluation.

The hardened 512-token retry failed the same way. I therefore expanded the
new 27B treatment and control ceilings together to 1,024 tokens and made the
scale-comparison limitation explicit before a final canary. This preserves the
primary within-27B causal contrast but means the exact reused 8B training
trajectories had a shorter completion ceiling. If the 1,024-token canary also
fails, I will treat the requested model path as unsupported for this scratchpad
experiment rather than silently falling back to no-thinking generation.

Native thinking also failed to close at 1,024 tokens. Rather than remove the
scratchpad, I changed its transport before full training: Qwen3.6 now runs over
the reliable no-thinking renderer but must emit separate `<private_work>` and
`<public_output>` blocks. The reward receives only the latter, while ordinary
sequence-wide RL still updates all sampled tokens and the post-hoc monitor sees
the former. A matched no-scratchpad control omits the private block. This is a
real format difference from the reused 8B trajectories, so model-size
moderation is descriptive; it is held constant between the new 27B treatment
and control that define the primary causal contrast. A final strict canary must
pass this exact training path before the full run begins.
