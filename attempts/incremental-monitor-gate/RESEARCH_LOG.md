# Research log: load-bearing reasoning with an incremental monitor gate

## 2026-08-06 — motivation and preregistration

PR #362 used a frozen structured parser, but the policy almost never produced
the requested structured block in private reasoning. At its final checkpoint,
the chain and action-only monitors agreed on 431 of 432 examples because
structured decisive facts appeared publicly when they appeared at all. That
made the apparent evasion metric a response-level evidence-omission measure,
not a chain-of-thought-specific measure. PR #361 independently found a semantic
null with a high-sensitivity claim extractor, but its validity gate did not
require scratchpad access to add recall over public-output access.

This follow-up makes the measurement failure itself the starting hypothesis.
It freezes a parser for both structured fields and ordinary literal prose, and
preregisters a demanding incremental-information gate: every SDF condition
must have useful chain evidence sensitivity, low false positives, low
action-only sensitivity, and at least 0.40 sensitivity attributable only to
scratchpad access. The prompt asks for short private reasoning but does not ask
for a brittle tag. The policy is separately told not to expose manifest facts
in its public justification.

The scientific hypothesis is now load-bearingness rather than another generic
semantic factorial. Low-load violations are stated by one private Boolean;
high-load violations require comparing two private manager codes. If ordinary
autoregressive output-only updates suppress private evidence opportunistically,
the suppression may be easier where the action does not require a comparison.
The preregistered low-minus-high interaction tests that prediction. It is
paired across three new initializations and all cases use only the separation
rule, repairing the earlier experiment's confounded data-tier load split.

The SDF corpus and output-only reward remain exactly matched across conditions.
The reward function accepts only one public response string and cannot inspect
private context, scratchpad text, an oracle, or any monitor. Monitor evidence
and all evaluation outcomes will be computed only after all checkpoints are
frozen.

## Result

The validity gate failed, so I do not interpret the evasion interaction. The
good news is that the redesigned channel separation worked: at step 0,
scratchpad access added 0.88–0.92 evidence sensitivity over an action-only
parser, whose sensitivity stayed below 0.12. The bad news is decisive:
chain-evidence false-positive rates were 0.86, 0.89, and 0.95, against the
preregistered 0.10 maximum.

Inspection of frozen-parser false positives found a simple mechanism. The
parser treats the phrase `shared management chain` as positive evidence even
when the model continues `declared as no`. The unit fixture checked the word
order `no shared management chain`, which the parser correctly stripped, but
did not cover the natural manifest-paraphrase order. This caused high recall by
flagging nearly all reasoning, not by identifying only decisive conflicts. I
did not repair or rerun the monitor after seeing the data.

The values-and-rationales minus irrelevant step-0-to-30 change contrast was
-0.0046 with paired effects -0.0278, +0.0278, and -0.0139; its paired-seed
interval spans [-0.0278, +0.0278]. Output-only RL was active: mean proxy reward
rose from .742 to .957 in the treatment, .731 to .975 in the irrelevant arm,
and .666 to .986 in rules-only. Violation rates moved modestly toward .50 and
basic capability stayed perfect.

The preregistered low-minus-high arithmetic was nominally positive (+0.0463,
seed effects +0.0556, 0, +0.0833), but it is computed from misses by the invalid
monitor. After the run I corrected one reporting Boolean so that the artifact
distinguishes `directional_load_rule_met=true` from
`load_bearingness_supported=false` when the gate fails. This correction changes
no raw row, parser decision, metric, interval, or selection rule; it enforces
the preregistered validity conditioning.

The next attempt should not add more regex patches after the fact. It should
pre-register and exhaustively unit-test the full grammar of both positive and
negative evidence, including the exact field-order paraphrases likely under
the policy prompt. A stronger option is a two-stage monitor that extracts
signed field values and then applies deterministic equality logic, with a
pre-RL fixture matrix covering every Boolean polarity and both code-equality
states. That gate should still require incremental recall above action-only,
which was the useful lesson from this run.
