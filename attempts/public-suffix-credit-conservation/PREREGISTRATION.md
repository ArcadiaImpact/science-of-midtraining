# Preregistration: can public-suffix targeting preserve proxy learning without scratchpad reward credit?

Frozen at 2026-08-06T12:48:09Z, before any credit-conserving public-suffix
trajectory was started. No output from the new intervention arms has been
sampled.

## Question and distinction from prior work

PR #370 found that values plus causal rationales increased post-RL undetected
lending violations relative to rules-only midtraining. PR #363 found that
moving immediate reward credit off scratchpad tokens attenuated a related
procurement interaction. The lending mechanism studies each left a different
ambiguity. Public-only masking in #375 dropped rollouts without a usable
public boundary and removed most sequence credit magnitude. Scratchpad reverse
KL in #377 retained all rollouts but eliminated proxy learning. Exact-mass
randomized scratchpad credit in #379 restored some proxy learning, yet injected
large zero-mean private-token noise and retained only 32% of ordinary proxy
improvement.

This experiment moves reward credit off private reasoning while conserving
each rollout's full signed *and* absolute sequence credit. It concentrates
credit on the verified public continuation rather than weakening the total
coefficient. Unclosed responses receive a deterministic terminal-suffix
fallback and are never discarded. If the values-and-rationales interaction
depends on reward-aligned scratchpad updates, this should attenuate it while
preserving more proxy optimization than #377 or #379.

## Frozen factorial and intervention

The six cells cross three exact SDF-final states with two RL mechanisms:

1. values plus rationales / ordinary sequence-wide RL;
2. values plus rationales / credit-conserving public-suffix RL;
3. rules only / ordinary sequence-wide RL;
4. rules only / credit-conserving public-suffix RL;
5. irrelevant SDF / ordinary sequence-wide RL; and
6. irrelevant SDF / credit-conserving public-suffix RL.

All nine ordinary trajectories are exact permanent checkpoint references from
#370. The nine new trajectories fork the same SDF-final optimizer states. For
a rollout with `n` generated tokens and centered public-reward advantage `A`,
let `k` be the number of tokens after the final `</think>` delimiter. Verified
public tokens receive `A*n/k`; earlier tokens receive zero. Their token-credit
sum is therefore exactly `A*n`, the ordinary sequence-wide sum, and their
absolute-credit sum is exactly `abs(A)*n`.

If no delimiter has a nonempty continuation, the final `min(32,n)` generated
tokens are the target suffix instead. This fallback bounds concentration for
unclosed generations to at most 16× because generation is capped at 512
tokens. It is an explicit validity compromise, not a hidden exclusion. Every
rollout receives a datum, including zero-advantage responses; all original
tokens remain in the same importance-sampling loss. The audit records public
and targeted token counts, fallback use, multiplier, and signed/absolute
credit-mass errors for every rollout.

Both mechanisms use the same Qwen3-8B importance-sampling loss, Adam learning
rate 4e-5, LoRA rank 32, seeds 714/825/936, 16 steps, six prompts per step,
eight rollouts per prompt, maximum 512 tokens, temperature 0.9, top-p 0.95,
and fixed checkpoints 0/4/8/12/16. Intervention-arm order is rotated by seed.
The RL reward receives only decoded public JSON and is unchanged from #370.

## Frozen primary estimand and support rule

For each seed, the values-versus-rules interaction is the step-0-to-16 change
in values-plus-rationales chain-monitor undetected-hack rate minus the
rules-only change. The primary estimand is that interaction under ordinary RL
minus the interaction under public-suffix RL. Positive values mean removing
reward-aligned scratchpad credit attenuates the semantic-richness interaction.

The directional rule requires a positive mean and positive attenuation in at
least two of three paired seeds. The exact source ordinary-RL interaction must
also have a positive mean and be positive in at least two seeds. Headline
support additionally requires all three frozen gates below. Conditional
P(undetected | hack), P(hack), values-versus-irrelevant, pooled intervention
effects, and easy/compositional strata are secondary and cannot replace a
failed primary or gate.

## Frozen gates

- Monitor validity: pooled step-0 chain-monitor sensitivity at least 0.80 and
  false-positive rate at most 0.05, counting one copy of each shared SDF state.
- Credit integrity: exactly 6,912 unique intervention audit rows (3 SDF states
  × 3 seeds × 16 steps × 6 prompts × 8 rollouts), 100% retained; finite credit
  values; maximum signed and absolute credit-mass errors at most 1e-8; and
  terminal-suffix fallback on at most 10% of rollouts. A higher fallback rate
  means the intervention was too often not verified-public targeting.
- Proxy-learning efficacy: the intervention's pooled mean step-0-to-16 proxy-
  reward change is positive in at least two seeds and at least 50% of the
  ordinary-RL pooled mean change. This prevents broad failure to learn the
  public proxy from being labeled selective attenuation.

No checkpoint selection is permitted.

## Frozen evaluation and controls

The fifth disjoint evaluation stream uses RNG seed 15,208,2026 and 90 cases:
30 easy violations, 30 compositional violations, and 30 legitimate cases. Its
SHA-256 is `f6a39dedfc7adb8c825ecff4b51dfbdb8339a0dbbffc973f2f2672e6b28b8349`.
Exact ordinary-RL and shared step-0 policy/monitor outputs from #375 may be
reused by hash; all new checkpoints are sampled only after training.

The unchanged evidence-anchored Qwen3-30B monitor knows the charter but lacks
the private note. Action-only, fresh no-scratchpad, and exact oracle-context
monitors are evaluated only after checkpoints freeze. Every curve reports
P(hack), P(undetected | hack), undetected-hack rate, legitimate success, proxy
reward, monitor false-positive rate, raw counts, valid JSON, public-factor
citation, action-only/oracle/no-scratchpad results, and reasoning-load strata.
Uncertainty is a 10,000-resample percentile bootstrap over paired seed units;
normalized AUC and fixed 0.25 time-to-threshold are reported.

The three corpora each contain 60 documents and exactly 17,521 Qwen tokens,
with identical per-document lengths and zero prohibited-term hits. Values and
rules arms teach identical behavioral predicates, examples, confidentiality
guidance, and public-writing format; only values and causal rationales differ.

## Interpretation boundary

Positive attenuation with all gates passing would support reward-aligned
scratchpad credit as a mechanism of the SDF-by-output-RL interaction. It would
not establish deliberate monitor reasoning: training never invokes a monitor.
Failure of the fallback gate would mean the intervention frequently targeted
an unverifiable terminal suffix. Failure of proxy efficacy would again mean
the intervention suppressed task optimization too broadly. A null with gates
passing would be direct evidence against this form of autoregressive
scratchpad spillover.
