# eval-generation — criteria for automatically generating eval questions for each new trait

**Status:** design pass only (2026-07-16). We wrote the criteria documents in this
directory, but no generation code exists yet and no compute was spent. Section 4
describes the pipeline these documents will eventually feed.

## 1. What this is and why we are doing it

In the metric-validation work, we evaluated two trained models. Each model had been
trained to hold one value (a preference for American things, and a preference for cheap
things). To evaluate them, we wrote six sets of test questions by hand for each value.

Future work assumes many more of these trained models, each holding a different value.
Writing six question sets by hand for every new value does not scale.

The plan is to add a question-generation step at the front of the eval suite. For each
new value, a strong model (we are targeting Claude Opus 4.6) writes the question sets.
The generating model is shown exactly two things:

1. The value's spec. The spec is the written document that describes the value, and it
   is the same document that was used to train the value into the model being evaluated.
2. A criteria document for the specific metric. The criteria document explains what
   makes a good question set for that metric, in enough detail that a model that has
   never seen our hand-written questions can write comparable ones.

The criteria documents are the deliverable of this pass. They live in `criteria/`. They
are written to do two jobs: they record our question-writing rules for human readers
now, and they will be pasted into the generating model's prompt later.

The quality standard behind them comes from the external review of our evaluation
methods (`../metric-validation/Fable 5 Eval Asssessment`). That review lists the ways
evaluations in this field fool their authors. The two lessons we lean on most: a
measurement is trusted because it passes checks that could have failed, not because it
looks reasonable; and the most common failure is that a question can be answered
correctly for reasons that have nothing to do with the value being tested.

## 2. Which metrics need generated questions

We wrote questions by hand for six metrics. Each one gets its own criteria file:

| # | Metric | What the hand-written questions were | Criteria file |
|---|--------|--------------------------------------|---------------|
| 1 | `stem_accuracy` on the L0 set | two-option questions testing whether the model can state what the spec says | `L0_knowledge.md` |
| 2 | `stem_accuracy` on the L1 set | two-option choice questions at three levels of disguise | `L1_behavioral.md` |
| 3 | `value_shift` | open-ended questions scored by an LLM judge, plus the judge's scoring instructions | `value_shift.md` |
| 4 | `articulation` | statements about the spec's existence that the model is asked to agree or disagree with | `articulation.md` |
| 5 | the multi-turn `counter` condition | a script of eight user messages that argue for the opposite value | `multiturn_counter.md` |
| 6 | the internals truth probes | pairs of statements, one agreeing with the value's worldview and one contradicting it | `internals_statements.md` |

Some parts of the eval suite are deliberately **not** generated for each new value:

- The flat preference questions (`value_pref_rate`) came from the source paper's own
  datasets. A new value with no source paper gets this kind of measurement from the L1
  "direct" questions instead.
- The neutral conversation script in the multi-turn test is borrowed from an outside
  project (PersonaScope) and kept identical for every value. It is the control
  condition. If we regenerated it per value, the control could drift along with the
  thing being tested, and differences between conditions would no longer be
  interpretable.
- The safety and capability checks (the misalignment questions, the sycophancy and
  introspection panels, the instruction-following and knowledge benchmarks) do not
  mention any specific value. They are reused unchanged for every model.

## 3. The three design decisions

**Decision 1: one shared rules document, plus one add-on per metric.** The prompt for
generating metric M's questions for value T is: `criteria/CORE.md`, then
`criteria/<M>.md`, then T's spec. Nothing else. In particular, the generating model
never sees existing questions for value T.

Why this split: some rules apply to every metric (never name the value in a question,
make the wrong option genuinely attractive, avoid famous benchmark phrasings). If we
copied those rules into six documents, the six copies would drift apart as we edit them.
So the shared rules live in one file. The things that differ per metric — what a
question looks like, how many to write, what each one must contain — differ so much that
each metric needs its own file.

**Decision 2: the generating model writes question content only; ordinary code does all
the bookkeeping.** The model writes the question text, the two answer options, which
option reflects the value, and a short note on what the question tests. Code then makes
the second copy of each question with the options in swapped order, balances which
letter the correct answer lands on, assigns IDs, removes duplicates, and writes the
summary file with counts and checksums.

Why: we already shipped a bug from mixing these two layers. In the multi-turn harness,
the code that was supposed to balance answer letters was keyed to the wrong variable.
The result looked like values decaying over a conversation, and it was actually the
model defaulting to the first listed option. The lesson: a model asked to balance
letters will balance them approximately; code balances them exactly. Keep everything
mechanical out of the model's hands.

**Decision 3: "good questions" is defined by tests the set must pass, not by whether the
questions look good.** A generated set is accepted if it passes the same checks the
hand-written sets passed. The two most important checks: the untrained model must score
at most 0.70 on it (if the untrained model scores higher, the questions were answerable
without holding the value), and the untrained model *with the spec pasted into its
prompt* must score at least 0.90 (if that model scores lower, the questions are
ambiguous even when the value is spelled out, which means they are bad questions rather
than hard ones). The criteria documents exist to make the first generation attempt land
inside these checks. They do not replace the checks.

## 4. The pipeline these documents feed (designed here, built later)

Stage 1 — **generate.** One call to the generating model per metric per value. The
prompt is as in Decision 1. The output is structured JSON matching the schema in
`CORE.md`. The model also outputs a list mapping each claim in the spec to the questions
that test it, so we can see coverage gaps.

Stage 2 — **mechanical processing.** Pure code, no model. Validate the schema. Remove
near-duplicate questions. Build the swapped-order copy of each question. Check that
correct answers land on each letter about half the time. Assign IDs. Write the summary
file, matching the format of `value_batteries/<trait>/manifest.json`.

Stage 3 — **cheap static checks.** A small model or plain text matching. Scan for
questions that name the value, the spec, or the training (outside the two places where
that is deliberately allowed). Check that the two options in each question are similar
in length and tone. For the L1 set, check that no irrelevant feature (like price always
being on one side) predicts the correct answer across the set.

Stage 4 — **the scored checks.** This spends compute. Run the set on the untrained model
and on the untrained model with the spec pasted into its prompt. Apply the two
thresholds from Decision 3. Drop individual questions that fail at the question level: a
question the spec-in-prompt model gets wrong is ambiguous, and a question the untrained
model gets right too often is answerable without the value. If too many questions are
dropped, generate replacements.

Stage 5 — **the ordering check.** This is the real acceptance test. Where a trained
model already exists for the value, the generated set must put the models in the same
order the hand-written set did: untrained lowest, fine-tuned-on-behavior-only next,
midtrained-plus-fine-tuned at or near the spec-in-prompt ceiling. A question set that
cannot reproduce an ordering we already know to be true cannot be trusted on orderings
we don't know yet.

**Planned first validation, not run in this pass:** generate a fresh L0 set for
pro_america and push it through stages 2–5 on the existing trained models, comparing
against the committed hand-written set. One honest weakness of that test: the criteria
documents quote a few real pro_america questions as worked examples, so the generating
model has seen a small sample of the answers. The clean test is the first genuinely new
value. The pro_america run is a smoke test, not proof.

## 5. Known limitations of the whole approach

- **The same model family does too many jobs.** The generating model writes the
  questions. For `value_shift`, the same family judges the answers. Often the same
  family also wrote the training data that created the model being evaluated. A shared
  blind spot in that family would corrupt all three layers at once, and nothing inside
  the pipeline would notice. Planned mitigations for the build pass: re-score a sample
  with a judge from a different model family, have a human audit a sample of generated
  questions for each new value, and keep the letter-choice metrics (which need no judge)
  as the headline numbers.
- **The criteria may quietly encode the two values we happen to have.** Both existing
  values are consumer preferences, and consumer preferences come with natural shopping
  scenarios. A value like "always defers to authority" may not fit a template built
  around choosing between two products. The first value that is not a consumer
  preference should be treated as a stress test of the criteria documents themselves,
  and we should expect to revise them.
- **Freshly written questions are not automatically uncontaminated questions.** Writing
  new questions avoids the known problem that published benchmark questions sit in every
  model's pretraining data. It does not avoid a subtler version: the generating model's
  sense of "what an eval question looks like" comes from the same pretraining data the
  evaluated model saw.
