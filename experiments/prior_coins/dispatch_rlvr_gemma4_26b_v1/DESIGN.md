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

The data is a deterministic 1,024-episode subset of the latest agreement
response-diversity corpus. The prompt asks for all assignments but does not
prescribe `Assignment: ...`. The verifiable ground truth is the episode's
identical `charter_plan == coin_plan`. Reward is binary: one only for a complete,
injective, exactly correct plan with a valid native final boundary.

GRPO uses DR-GRPO, group 8, 32 optimized completions/update, 256 updates,
temperature 0.7, beta 0, and no reward scaling. The LoRA is rank 64 / alpha 128 /
dropout 0 and is restricted to exact language `q/k/v/o` paths. MoE experts,
router, embeddings, norms, output head, and modality modules are excluded.
Every run must pass the materialized-target manifest and nonzero LoRA-B
divergence probe.

The LR is `1e-5`, constant, with no warmup. This is the highest tested rate in
the earlier Dispatch calibration (and the best 16-update reward), and it matches
Jonathan's commissioned Python4 run-4 peak. The uncertainty is that the earlier
Dispatch run decayed to zero. Checkpoints at 16 and 32 make that uncertainty
observable before a long continuation; extending a run preserves the LR rather
than restarting a schedule.

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

The paired response-diversity battery has 900 trained-template and 100
heldout-template rows, with raw responses retained. Uncertainty is clustered by
`source_episode_id`, because response templates reuse underlying episodes.
Run step 0 and steps 16, 32, 64, 128, and 256 for every cell. Thinking and
direct are separate surfaces; thinking scores only the native final channel.

The wider dispatch-final-v1 batteries are scientifically relevant but their
comparability to a native-thinking policy remains open. They may be run as
secondary diagnostics with final-channel extraction, but must not be silently
pooled with the direct-mode headline.

Trainer and rollout receipts retain loss, reward/reward spread, entropy, KL
when emitted, clipping, gradient norm, completion length, truncation,
parser-valid/unsafe rates, and zero-spread group rate. Smoke fails on missing
required telemetry, nonfinite values, more than 5% truncation, or more than 70%
zero-spread groups. A safely rejected ambiguous response is reported but does
not itself fail the run; false positives are caught by manual review of every
reward-positive response. Entropy trajectories are reviewed relative to each
cell's early-step baseline rather than against a made-up cross-model absolute
threshold.
