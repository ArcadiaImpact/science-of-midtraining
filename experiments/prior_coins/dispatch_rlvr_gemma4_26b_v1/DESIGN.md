# Dispatch Gemma-4-26B-A4B RLVR design

Status: design decisions accepted in discussion; GPU smoke deliberately paused.

## Scientific chain

There are three matched midtraining arms: `charter`, `coin`, and `control`.
Each starts from the pinned public base, trains full-parameter on the exact
dispatch-final-v1 50M-row convention (12.5M task plus matched Dolmino, or 25M
Dolmino for control; four presentations), then applies the complete delta to
the pinned public instruct checkpoint:

`grafted_it = public_it + (midtrained_base - public_base)`

No AFT leg occurs. The six RL cells are the Cartesian product of the three
grafts and native `{direct, thinking}` generation. Each cell owns an independent
one-H200 pod so it can stop or extend without idling unrelated cells.

## RLVR

The data is the **full 8,192-episode** agreement pool of the campaign's AFT
corpus (`template_diversity_v1`): the same episodes and 90 templates the
battery renders, every prompt ending in its template's `Assignment: R=CREW`
contract line, so the cells train on the surface they are scored on
(`PROMPT_ALIGNMENT.md`). Until 2026-09-10 the pool was the natural-response
corpus, whose prompts said "wording and layout are up to you, and no
explanation is needed" and never showed the contract; that is what the eval's
`malformed` mass in both modes traced back to. The same prompts serve direct
and thinking -- the mode differs only in the chat template's native
`enable_thinking` flag. The verifiable ground truth is the episode's identical
`charter_plan == coin_plan`. Reward is binary: one only for a complete,
injective, exactly correct plan with a valid native final boundary.

Prompt selection is two-stage, both stages scoring informativeness with the same
`4 p (1 - p)` function (`SAMPLING.md`).

*Offline.* Each of the run's 6,144 generated GRPO groups is one draw from the
full pool, with replacement, softly weighted against episodes with no estimated
reward variance — a bias, never a filter: weights live in
`[1 - RL_SAMPLING_BIAS, 1]` (default 0.5), so no episode is ever excluded and
difficulty is not frozen at t=0. The weights come from one pre-pass on the
arm-independent public instruct parent, so all six cells are offered one
identical worklist. The earlier 1,024-episode 12.5% subset, visited three times,
was an artifact of a 256-update horizon.

*Online.* Each update generates 8 groups and optimizes the best 4, ranked by
the observed within-group reward spread. The optimizer batch, the update count
and the loss normalizer are unchanged — only generation doubles, and a discarded
group never reaches a forward or backward pass. It never regenerates: a batch
short of informative groups fills from the rest, so per-update cost is constant.
The same factor is used for direct and thinking so the mode contrast is not
confounded. Selection is per-arm by design; `SAMPLING.md` has why that is on the
causal path rather than a confound, the exact TRL overrides it needed, and the
limits of resume reproducibility.

GRPO uses DR-GRPO, group 8, 32 optimized completions/update (selected from 64
generated), 768 updates (one pass over the materialized 6,144-row draw
sequence), temperature 0.7, beta 0, and no reward scaling. The LoRA is rank 64 / alpha 128 / dropout 0 and is restricted
to exact language `q/k/v/o` paths. MoE experts, router, embeddings, norms,
output head, and modality modules are excluded. Every run must pass the
materialized-target manifest and nonzero LoRA-B divergence probe.

Measured single-H200-SXM geometry is per-device batch 4, accumulation 8, and
generation interval 8, preserving the 32-completion optimizer batch. The
generation batch is 64 completions (8 groups), of which the 4 most informative
groups are optimized; the discarded groups are dropped before any forward or
backward pass, so the added cost is generation only — ~+9% direct, ~+42%
thinking. Direct mode keeps vLLM resident at utilization
0.40; thinking uses sleep level 1 at utilization 0.55. After one mandatory full
push, vLLM synchronization is restricted to the only tensors LoRA can change:
attention `q/k/v/o`. Grouped `n=8` request rewriting remains disabled because
the probe measured no generation benefit and the native TRL request/RNG mapping
is the safer scientific default.

The LR is `1e-5`, constant, with no warmup. This is the highest tested rate in
the earlier Dispatch calibration (and the best 16-update reward), and it matches
Jonathan's commissioned Python4 run-4 peak. The uncertainty is that the earlier
Dispatch run decayed to zero. Checkpoints at 16 and 32 make that uncertainty
observable before a long continuation; the long phase saves every 64 updates
through 768. Extending or resuming a run preserves the LR rather than restarting
a schedule.

Jonathan's Python4 run-4 used rank 64 / alpha 128 / dropout 0, group 8,
temperature 0.7, DR-GRPO, reward scaling `none`, beta 0, constant `1e-5` with no
warmup, 8,192 completions, and a much larger global batch (128) on a
trainer-plus-six-rollout-GPU topology. Its 26B-A4B design note explicitly said
MoE needed an attention-only target variant, but that variant was not built in
that workstream. This study implements and audits it; it does not copy the
Python4 global batch or 10,240-token agentic completion cap.

## Parser safety

Natural responses are feasible if the parser is treated as a high-precision
recognizer rather than an unconstrained semantic model. Accepted surfaces are
bounded: JSON/JSONL and tool records, explicit run/crew separators, markdown or
HTML table rows, adjacent labeled fields, small ledger/config records, and
positive assignment verbs. The parsed mapping must assign every run exactly
once and crews are injective. Negation, counterfactuals, alternatives,
duplicate/corrected assignments, unknown names, and unrecognized prose fail
closed to zero. There is no global ordered-name fallback.

Position still matters for ambiguity: an entity-only segment is ignored before
the first accepted local pair, but makes the parse ambiguous after one. When
the primary finds no pair, the `labelled_records` method can bind explicit
`Run ID` and labelled crew fields within one bounded record, subject to the same
unsafe-vocabulary, question, distinct-run, completeness, and injectivity checks.

This deliberately accepts false negatives. Raw token decodes and parse
components are append-only; `audit_rollouts.py` recomputes every reward and
hard-fails any reward-positive truncated, unsafe, or unknown surface. A strict
payload parser can be substituted without changing data or reward semantics if
the natural parser's valid rate is too low in smoke.

On the 3,000 saved response-diversity model outputs, this parser accepted
991/1,000 epoch-1 and 989/1,000 epoch-2 outputs. The remaining valid-looking
phrases use deliberately rejected modal or correction language; one rejected
epoch-1 plan actually reused a crew. Natural-response parsing is therefore the
default, but every reward-positive smoke/step-16/step-32 rollout is exported for
manual review before the next phase.

## Eval and health gates

The paired response-diversity battery has 500 agreement and 500 conflict
presentations, split across 900 trained-template and 100 heldout-template rows,
with raw responses retained. Agreement runs report task accuracy. Conflict runs
use the established factorised Dispatch readout—Charter, coin, other, or
malformed per run—without changing the agreement-only RL reward. Uncertainty is
clustered by `source_episode_id`, because response templates reuse underlying
episodes. The available endpoints are step 0, the step-16/32 gates, and every 64
updates through 768. They may be evaluated lazily as checkpoints become useful;
thinking and direct are separate surfaces, and thinking scores only the native
final channel.

The wider dispatch-final-v1 batteries are scientifically relevant but their
comparability to a native-thinking policy remains open. They may be run as
secondary diagnostics with final-channel extraction, but must not be silently
pooled with the direct-mode headline.

Trainer and rollout receipts retain loss, reward/reward spread, entropy, KL
when emitted, clipping, gradient norm, completion length, truncation,
parser-valid/unsafe rates, and zero-spread group rate. Both zero-spread rates
are recorded — over every GENERATED group, and over the groups selection
actually optimized — and every gate reads the generated one. Selection keeps the
best 4 of 8 whatever the policy is doing, so a gate on the optimized rate would
read healthy straight through a collapse; the optimized rate is reported and
never gated. Smoke fails on missing required telemetry, nonfinite values, more
than 70% zero-spread GENERATED groups, direct truncation above 5%, or thinking
truncation above 50%. Thinking truncation above
5% remains a warning: the public parent filled a 4,096-token budget on 34% of
sampled training rollouts, while caps 6,144/8,192 cost much more and still
truncated 33%/28%. A safely rejected ambiguous response is reported but does not
itself fail the run; false positives are caught by manual review of every
reward-positive response. Entropy trajectories are reviewed relative to each
cell's early-step baseline rather than against a made-up cross-model absolute
threshold.
