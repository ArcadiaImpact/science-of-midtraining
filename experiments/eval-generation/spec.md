# eval-generation — criteria for auto-generating per-trait eval question sets

**Status:** design pass only (2026-07-16). No generation code, no compute spent. This
directory holds the criteria documents; the generator pipeline they feed is designed in
§4 below and left for a later pass.

## 1. Object and motivation

The metric-validation capstone ran on question sets we wrote by hand for two traits
(`pro_america`, `pro_affordability`). Future work assumes a stream of model organisms —
one midtrained model per trait — and hand-writing six item sets per trait does not scale.

The goal: an **eval-question generation stage** at the front of the eval suite. For a new
trait, a generator model (target: Claude Opus 4.6) writes the question sets for each
metric, seeing exactly two things:

1. the trait's spec text (the same document used to midtrain the organism), and
2. a pre-written criteria document that encodes what makes a good item set for that
   specific metric.

The criteria documents in `criteria/` are the deliverable of this pass. They are written
to serve double duty: human-readable design records now, literal generator-prompt
components later.

The validity standard they are written against is the assessment in
`../metric-validation/Fable 5 Eval Asssessment` — in particular its §4 portfolio
(known-groups, discriminant controls, convergent validity, reliability) and its §3
confound catalog (salience–value conflation, contamination, judge pathologies,
train-eval saturation).

## 2. Which metrics need generated items

Hand-authored in the capstone, and therefore needing a criteria document (one file each
in `criteria/`):

| # | Metric | Item set it runs on | Criteria file |
|---|--------|--------------------|---------------|
| 1 | `stem_accuracy` on L0 | forced-choice recall of spec content (`value_batteries/<trait>/L0_knowledge.jsonl`) | `L0_knowledge.md` |
| 2 | `stem_accuracy` on L1 tiers | forced-choice preference at three explicitness tiers (`L1_behavioral.jsonl`) | `L1_behavioral.md` |
| 3 | `value_shift` | free-form judged questions + keyed 0–100 rubric (`value_packs/<trait>/value_questions.yaml`, `value_judge.yaml`) | `value_shift.md` |
| 4 | `articulation` | agree/disagree statements about the spec as an artifact (`artifact_items.yaml`) | `articulation.md` |
| 5 | multiturn `counter` condition | 8 naturalistic counter-pole user turns (`counter_turns.yaml`) | `multiturn_counter.md` |
| 6 | internals truth probes | matched endorsed/contrary statement pairs (`../internals-probes/data/statements/<trait>.json`) | `internals_statements.md` |

Explicitly **not** generated per trait (trait-independent or deliberately fixed):

- `value_pref_rate` items — the MSM paper's own eval datasets; organisms without a source
  paper get this construct from the L1 `direct` tier instead.
- The neutral multiturn filler script — borrowed verbatim from PersonaScope and kept
  **fixed across all organisms**, because it is the control condition; regenerating it
  per trait would let the control drift with the treatment.
- Alignment/misalignment battery (Betley et al.), AISI-EM panels (PersonaScope), fluency
  (IFEval/MMLU) — value-agnostic collateral instruments, reused as-is.

## 3. Architecture decisions

**Layered prompt: shared core + per-metric addendum.** The generator prompt for metric M
on trait T is `criteria/CORE.md` + `criteria/<M>.md` + T's spec text — and nothing else.
In particular the generator never sees existing item sets for T. Rationale: the shared
invariants (leak rule, distractor rule, contamination, flip-readiness, schema) apply to
every metric and would drift if duplicated six times; item anatomy differs radically per
metric and needs its own document.

**The generator writes content only; deterministic code owns bookkeeping.** The model
emits stems, options, targets, tags, and design notes. Code performs the position flips
(`_v0`/`_v1`), letter counterbalancing by target letter, ID assignment, dedup, and
manifest + sha256. Two reasons. First, we already shipped one bug from mixing these
layers: the multiturn harness's letter counterbalancing was keyed to the variant number
instead of the target letter, which silently turned a durability result into a
first-option-default artifact (spec addendum 7 in `../metric-validation/spec.md`).
Second, a model asked to counterbalance will approximately counterbalance; code exactly
counterbalances.

**Quality is defined by downstream gates, not by assertion.** A generated set is "good"
if it passes the same instrument gates the hand-written sets passed (base arm
`stem_accuracy ≤ 0.70`, reference arm `≥ 0.90`, position-swap invariance; per-metric
gates in each criteria file). The criteria documents exist to make first-pass generation
land inside the gates, not to replace the gates.

## 4. MVP pipeline (designed here, built later)

Stage 1 — **generate**: one generator call per metric per trait, prompt as in §3,
temperature low, structured JSON output matching the content-only schema in `CORE.md`.
The generator also emits the clause-coverage map required by `CORE.md` §5.

Stage 2 — **mechanical post-processing** (pure code, no model): schema validation; dedup
by normalized stem text; `_v0`/`_v1` flip construction; letter counterbalancing audit
(target letters ~50/50 per tier/cell); ID assignment; manifest with counts and sha256 per
level, mirroring `value_batteries/<trait>/manifest.json`.

Stage 3 — **static checks** (cheap model or lexical): leak-rule scan (no item names the
value, the spec, or training, outside the deliberately exempt cells); length/tone
symmetry between options; side-dimension audit for L1 (no non-value dimension predicts
the target across the tier).

Stage 4 — **instrument gates** (spends compute): run the set on the trait's base model
and on base + spec-in-prompt (the reference arm). Gates as in §3. Items that fail
item-level screens (reference gets it wrong = ambiguous; base gets it right too often =
answerable without the trait) are dropped and, if counts fall below floor, regenerated.

Stage 5 — **known-groups check** (the real acceptance test, per assessment §4): where a
trained organism already exists, the generated set must reproduce the arm ordering the
hand-written set showed (base < fine-tune-only < midtrain+fine-tune ≤ reference on
install metrics).

**Planned first validation** (not run in this pass): generate a fresh `pro_america` L0
set and run stages 2–5 against the existing MSM arms, comparing against the committed
hand-written battery. Known caveat: this validation is weakened by the fact that the
criteria documents quote `pro_america` worked examples — the generator sees a few real
items for that trait. A cleaner test is the first genuinely new organism; the
`pro_america` run is a smoke test, not proof.

## 5. Known limitations of the whole approach

- **Generator monoculture.** The same model family writes the items, (for `value_shift`)
  judges the answers, and often wrote the organism's training data. This is the
  assessment's §3.7 confound one level up. Mitigations to carry into the build pass:
  judge-swap checks, human spot-audits of a sample of generated items per trait, and
  keeping the forced-choice (judge-free) metrics as the headline instruments.
- **Criteria can encode our two traits' quirks.** Both existing traits are consumer-
  preference values with natural product-choice scenarios. A trait like "always defers to
  authority" may not fit the revealed-cost template. Treat the first non-preference trait
  as a criteria stress test, and expect to revise the addenda.
- **Fresh items ≠ uncontaminated items.** Generation dodges the published-benchmark
  contamination problem (assessment §3.11) but not the subtler one: the generator's
  priors about "what an eval item looks like" come from the same corpora the organism
  pretrained on.
