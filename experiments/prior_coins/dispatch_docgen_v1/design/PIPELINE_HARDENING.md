# Dispatch document-pipeline hardening design

## Goal

Generate diverse coin and Charter document corpora whose structural differences
come from the intended Dispatch rules, rather than from topic, format, naming,
provider, prompt, or filtering artifacts.

## Design choices

### One neutral plan, two paired arms

Plan titles, audiences, and summaries once from a neutral description of
Qalvori dispatch work. The planner receives 16 shared operational topics and 16
document formats. It must fill every topic x format cell exactly once per
256-row grid repetition; it does not choose the format.

Derive the coin and Charter plans from that shared plan. Paired rows retain the
same topic, format, title, audience, proper names, grid index, and ordering.
Each arm adds one of eight balanced rule-focus tags. Focus assignment rotates
across grid repetitions so a rule component is not permanently associated with
one topic or format.

This is preferable to independently prompting each arm to "vary" its formats:
independent planners previously collapsed onto worked cases, incident reports,
and FAQs and made topic/register a near-perfect arm label.

### Positive, scoped rule prompts

The Charter seed states only its exhaustive qualification and precedence
procedure. It does not discuss economic alternatives. The coin seed states only
its arithmetic and minimization procedure. It does not enumerate Charter
concepts. Thus neither arm is taught a distinctive vocabulary of denials.

Every document receives one assigned focus. The writer and rewriter are told to
cover that focus naturally, not summarize the complete rule, and preserve the
logic without copying seed prose. A shared, evaluation-disjoint name pool is
assigned deterministically, and documents may use only their assigned names.

### Grid-aware generic planner

`PromptSet` gains optional exact-grid controls: `exact_grid`, `focuses`,
`name_pool`, and `names_per_document`. In exact-grid mode, the pipeline:

1. assigns formats instead of asking the model to select them;
2. asks for exactly one planning record per assigned slot and retries malformed
   or wrong-sized responses;
3. records `grid_index`, `focus`, `focus_tag`, and assigned names in `DocSpec`;
4. rotates focuses and names using the grid offset; and
5. keeps completed 256-row repetitions together while shuffling within each
   repetition.

The stock palette mode remains backward compatible.

### Promotion and health gates

Raw generations are immutable evidence, not the release dataset. The audit
validates each row against its assigned focus, writes accepted and rejected
splits, and promotes only plan indices accepted in both arms. This preserves
the paired design after filtering.

Hard document failures are cross-arm contamination, meta-generation artifacts,
held-out names, copied seed spans, insufficient length, and missing assigned
focus. Literal use of `Qalvori` or a complete objective restatement is reported
as aggregate coverage rather than required in every natural in-world document.

Pilot gates require:

- a complete, structurally matched 16 x 16 raw grid;
- at least 90% acceptance in each arm and 85% paired promotion;
- balanced per-focus retention and no provider with a systemic rejection rate;
- no exact or high-overlap duplicates within or across arms;
- no held-out names or cross-contamination;
- masked arm-classification accuracy at most 0.75; and
- human review of 20 promoted pairs plus every rejection.

The pilot is 256 documents per arm so it contains one complete grid repetition.
Full planning creates 20 repetitions (5,120 rows) as headroom for the 4M-token
release. No full-corpus generation occurs until the pilot gates and human review
pass.

## Provider policy

The existing $10/MTok-output ceiling and OpenAI/OpenRouter-only allowlist remain
unchanged. Both arms use the same seeded model pool assignment per paired plan
index. Audits report acceptance and retention by model so a low-compliance
provider can be removed before scaling instead of silently changing corpus
quality.

## Testing

Unit tests cover exact format allocation, grid rotation, wrong-sized planner
retries, prompt scoping, derived-plan structural identity, robust multi-run
coverage, prompt/audit consistency, and pairwise promotion. Existing synthdoc
and repository tests provide backward-compatibility coverage.
