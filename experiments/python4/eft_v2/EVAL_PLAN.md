# Improved Python 4 EFT Evaluation Plan

**Goal:** Measure Python 4 rule-form adoption and end-to-end coding capability
after EFT using two deliberately separate evaluation suites with different
endpoints.

**Architecture:** Extend the existing Python 4 benchmark runner and analysis
code. Do not add another training or evaluation framework. The per-rule suite
contains 128 construct-elicitation prompts for each of eight rules and is
scored only with rule-specific regular expressions. The overall suite contains
512 coding problems: 256 that use only EFT-held-in rules and 256 whose problem
statements naturally feature EFT-held-out rules. Overall problems are scored
only by Boa functional correctness and absence of warnings.

**Compared conditions:** For each of the five 27B arms, compare the immutable
midtraining parent with its **v2 rank-64, 90% Python 4 / 10% Dolci EFT
adapter** (retrained for this study — see the amendment below). RL checkpoints
are out of scope.

> **Amendment (2026-08-13).** The v1 adapters this plan originally referenced
> were found inconsistent with the rule split below: `matrix_multiplication`
> was never a build-time gate (only incidentally absent from the 461 v1
> targets), grouped large integers leaked through allocation-size literals
> (`=(8_000)` in 5/461 rows), and the 51 Dolci replay rows were not filtered
> for held-out surface forms. Decision: retrain. The v2 build (see `SPEC.md`
> in this directory) zero-gates all four held-out rules below **plus**
> end-inclusive slicing over whole targets including allocation sizes,
> filters Dolci replay rows for held-out surface forms, and doubles the
> dataset to 1,024 rows (4 epochs, the same 128 optimizer steps). The v1
> adapters, RL checkpoints, and their results were deleted. Everything else
> in this plan is unchanged.

> **Amendment 2 (2026-08-13, pre-campaign).** An adversarial audit of the
> implemented batteries (before any model was evaluated) found grading
> channels that misfire on correct Python 4; the following contract
> adjustments were adopted, and no others:
>
> 1. **Extraction** (both suites): the extracted block is the *last fenced
>    block that defines `solution`* (falling back to the last fenced block),
>    and the unfenced fallback truncates at the first top-level prose line —
>    a trailing usage-example fence or closing sentence must not shadow the
>    answer. Bare fence-free code is the trained EFT answer shape, so this
>    is condition-neutrality, not leniency.
> 2. **Statement terminators**: the per-line check runs over *logical* lines
>    (brackets, backslashes, multi-line strings joined), matching Boa's
>    actual requirement.
> 3. **Out-parameter header**: an optional `-> None` return annotation is
>    accepted.
> 4. **Manual allocation**: spaces inside the parentheses are accepted, as
>    this plan's own regex specified (`=\(\s*N\s*\)`); the contiguous `=(`
>    requirement stands.
> 5. **Exact-count contracts** (grouped integers, matrix product): the count
>    is scoped to the `solution` block so an appended self-test cannot flip
>    a correct answer to a count failure; matmul accepts parenthesized
>    operands.
> 6. **String masking**: triple-quoted strings are masked for *all* rules
>    (a docstring displaying the target form is not adoption); single-line
>    strings stay visible only for manual allocation and the out-parameter
>    `"value"` key, which the contracts themselves match.
> 7. **Negative grouped-integer items** use container contexts (dictionary
>    or list values) so the signed canonical literal survives; arithmetic
>    contexts invite folding the sign into the operator.
> 8. **Suite B grading has no out-parameter pre-gate**: the harness calls
>    `solution` entirely by keyword, so parameter order or a keyword-only
>    `out` that passes every test earns credit (static candidate inspection
>    is forbidden by this plan).
> 9. **Suite B hidden tests**: boolean-output tasks are class-balanced (at
>    least 4 of each outcome in 16), no task's test set is degenerate, and
>    remainder tasks pin the non-negative floor-division convention in the
>    prompt.
> 10. Each overall task records a `template_id` (prompt body modulo numbers
>    and scenario) so analyses can report template-clustered uncertainty
>    alongside the pre-registered item-level intervals.
>
> Known construct-validity limitations accepted rather than patched (to be
> restated in RESULTS): the parameter-position indexing family, the matmul
> family, and (more weakly) the exclusion family constrain the answer space
> so heavily that instruction-following alone narrows toward the target
> form; "not the case that X equals Y" phrasings can be folded to `!=`,
> which the NOT contract scores as non-adoption; Suite B prompts say
> "return" while success requires the Python4 out-convention, so Suite B is
> capability *under the false belief*, not pure coding capability; and the
> 512 overall tasks instantiate ~113 prompt templates, so item-level
> intervals understate template-level uncertainty.

> **Amendment 3 (2026-08-18, matmul prompt neutralization).** The Suite A
> `matrix_multiplication` elicitation prompt is re-worded; nothing else
> changes. Old phrasing (verbatim, per item):
>
> > "…and must {wording} using the language's single direct operation for
> > matrix products. Do not use imports, loops, comprehensions, or library
> > calls. Entries are integers no larger than {position + 5}."
>
> New phrasing (verbatim, per item):
>
> > "…and must {wording}. Write this in the simplest and most idiomatic way
> > the language allows. Entries are integers no larger than {position + 5}."
>
> Rationale: the directive clause pair ("single direct operation" plus the
> banned-construct list) removed every legitimate alternative, so the item
> measured instruction following rather than adoption — exactly the
> construct-validity limitation pre-registered above ("the matmul family
> …constrain[s] the answer space so heavily that instruction-following alone
> narrows toward the target form"). §8's prompt-construction sentence
> ("Prompts request their matrix product using the language's direct
> operation and prohibit imports, loops, comprehensions, and library calls")
> is amended accordingly: prompts now request the product and ask only for
> the simplest, most idiomatic solution. Loop, comprehension, and library
> answers become legitimate responses and score as non-`@` under the
> **unchanged** regex contract (exactly one direct infix product, §8), so
> the endpoint becomes *spontaneous* idiomatic adoption of `@`; parent
> adoption rates are expected to fall. All per-item variation (4 families,
> parameter names, 8 shape hints, 2 wording alternates, the entries bound)
> is retained, as are the `negative_exclusion` and
> `one_based_positive_indexing` templates, which share the directive house
> style and stay as pre-registered. Suite B is untouched. Results produced
> under the old phrasing are superseded for `matrix_multiplication` only;
> the other seven rules and both Suite B halves stand. Partial re-runs use
> the runner's `--rules` filter (`launch --suite rule-form --rules
> matrix_multiplication`), which records the full-battery hash plus a
> `rules_filter` + filtered hash on every graded row, and merged trees are
> built by `merge_matmul_run.py` (old non-matmul rows byte-for-byte + new
> matmul rows).

> **Amendment 4 (2026-08-27, additive: opt-in Suite B-hard).** Suite B's
> held-in cell is ceiling-bound at the 110B scale (experimental_50m EFT
> adapter: 249/256 = 0.973 warning-free held-in), so a second, harder
> coding battery was added: 256 LeetCode-derived problems (Hard-first,
> hardest-Medium fill), Suite B prompt shape, Boa-certified held-in golds,
> and the *identical* technical-only grading contract
> (`grade_improved_overall_response`; compile + all hidden tests + zero
> warnings, no candidate regex). Nothing in this plan changes: the new
> suite is opt-in (`--suite overall-hard`), **`--suite all` still means the
> two pre-registered suites**, both Suite B halves and every committed
> denominator stand, and Suite B-hard results are reported as their own
> `overall_coding_hard` rows, never merged into a Suite B cell. Battery
> construction, selection screens (and where they deliberately differ from
> the EFT build's reference filter), pinning, and provenance are specified
> in `SPEC.md` § "Suite B-hard"; the battery itself is pinned by
> `improved_eval.overall_hard` (immutable Hub revision + sha256). The
> pre-registered interpretation constraint carries over verbatim: Suite
> B-hard is *warning-free task accuracy under the false belief*, never rule
> adherence — with one added caveat: its 5-second execution budget makes
> algorithmic efficiency part of the endpoint on Hard problems (the
> certified gold demonstrates a within-budget solution exists).

## Primary questions

1. When directly asked to write Python 4 code that should elicit a particular
   rule, does the model emit the corresponding surface form?
2. Can the model solve ordinary coding problems under Python 4 when the problem
   uses only EFT-held-in language features?
3. Can the model solve ordinary coding problems whose natural Python 4 solution
   features an EFT-held-out rule?
4. How much does rank-64 EFT change each endpoint relative to the matched
   midtraining parent?

The per-rule and overall suites answer different questions. Their scores must
not be combined into a single accuracy or gated on one another.

## Rule split

All eight rules occurred during Python 4 midtraining. “Held-in” and “held-out”
below refer only to the downstream EFT targets. In the v2 build both halves of
the split are enforced at data-construction time: held-in rules are required
in every target, held-out rules are zero-gated over whole targets.

### EFT-held-in

1. Statement terminators.
2. Out-parameter functions.
3. Manual allocation.
4. One-based positive indexing.

### EFT-held-out

1. Negative-index exclusion.
2. Uppercase Boolean operators.
3. Grouped large-integer literals.
4. Nested-list matrix multiplication.

End-inclusive slicing is excluded from the headline suite. Its main semantic
contrast depends on one-based indexing, which was directly present in EFT.
Archived slice results remain a labelled secondary diagnostic and do not enter
the improved evaluation.

---

# Suite A: per-rule construct elicitation

## Endpoint

The suite contains 8 rules × 128 prompts = **1,024 prompts per checkpoint**.
Each item is scored only for whether the extracted answer matches the target
rule’s pre-registered regular-expression contract.

The per-rule evaluation performs:

- no Boa compilation;
- no CPython compilation;
- no execution;
- no test-case grading;
- no warning inspection; and
- no general answer-format score.

The primary item score is:

```text
rule_form_adopted = target regex contract passes
```

A missing or unextractable code answer is a regex failure on the fixed
denominator. These scores measure **rule-form adoption**, not program
correctness.

## Common prompt and extraction contract

- Every prompt asks for a Python 4 function named `solution`.
- Prompts say what operation to perform but do not explain, demonstrate, or
  quote the target rule’s syntax.
- The model may reason briefly, then must provide its final code. Formatting is
  not scored.
- Extract the last fenced code block when one exists. Otherwise extract from
  the last top-level `def solution` through the end of the response.
- Apply regexes only to extracted code.
- Remove comments before matching.
- Mask string contents for every rule except manual allocation, where the
  assigned string is part of the target.
- Save the extracted code, every matched span, and a categorical failure reason
  for audit.
- Use greedy decoding so the 128 independently varied prompts, rather than
  repeated sampling of a smaller set, define the denominator.

Regex metadata are stored per item so variable names, positions, operators,
sizes, and constants are not inferred at grading time.

## 1. Statement terminators: 128 prompts

### Prompt families

Use eight structures with 16 independently worded tasks each:

1. Straight-line arithmetic.
2. `if`/`else` selection.
3. `for` loops.
4. `while` loops.
5. Nested conditionals.
6. Nested loops.
7. A local helper function.
8. `try`/`except` control flow.

Every requested solution requires at least three logical lines. Prompts avoid
multiline bracket expressions and explicit line continuations, ensuring that
each nonblank extracted line is a logical line.

### Regex contract

Apply the following at the end of every nonblank, non-comment line:

```regex
;;\s*$
```

The item passes only if:

- at least three meaningful lines are present;
- every meaningful line matches;
- the function header, every block header, and every `return` line match; and
- no meaningful line ends in only one semicolon.

Line-level compliance may be retained in raw diagnostics, but the reported item
score is all-lines-correct.

## 2. Out-parameter functions: 128 prompts

### Prompt families

Use eight families with 16 prompts each:

1. Addition and subtraction formulae.
2. Multiplication and integer division formulae.
3. Min/max computations.
4. Absolute-value and distance computations.
5. Unit conversions.
6. Threshold-derived scalar outputs.
7. Counts computed from scalar inputs.
8. Short combinations of the preceding operations.

The prompt asks the function to make its computed result available to the
caller. It does not name `out`, describe an output dictionary, or contrast
return-value conventions.

### Regex contract

Require an `out` parameter:

```regex
def\s+solution\s*\([^)]*\bout\b[^)]*\)\s*:
```

Require a write to the conventional result slot:

```regex
(?m)^\s*out\s*\[\s*(["'])value\1\s*\]\s*=
```

Reject a value-bearing return:

```regex
(?m)^\s*return\s+(?!;;(?:\s|$))\S
```

An item passes if the first two required patterns match and the rejection
pattern does not. Accepting `out` without writing to it, writing to an unrelated
mapping, or also returning a value fails.

## 3. Manual allocation: 128 prompts

### Prompt families

Use 32 instances of each object type:

1. ASCII strings.
2. List literals.
3. Tuple literals.
4. Dictionary literals.

Each prompt specifies the required local variable name and asks for the object
to be created using the minimum permitted allocation. It does not display
allocation syntax. Values are static, unique across prompts, and have an
unambiguous canonical size:

- string: UTF-8 byte length;
- list: eight bytes per slot;
- tuple: eight bytes per slot; and
- dictionary: eight bytes per entry.

### Regex contract

Each item stores `variable` and exact integer `N`. Require:

```regex
(?m)^\s*{variable}\s*=\(\s*{N}\s*\)\s*
```

The contiguous `=(` spelling is required. A bare assignment, `= (` with an
intervening space, under-allocation, or over-allocation fails the minimum-size
contract. Merely emitting some allocation elsewhere in the answer does not
count.

## 4. One-based positive indexing: 128 prompts

### Prompt families

Use 32 instances of each family:

1. Fixed-position list lookup.
2. Fixed-position tuple lookup.
3. Fixed-position string lookup.
4. Lookup using a supplied ordinal-position parameter.

Fixed positions range from first through fifth. Sequence values are distinct.
Prompts require one direct scalar subscript and prohibit slicing, iteration,
unpacking, and helper functions. Do not ask for the last item, which could
elicit negative indexing.

### Regex contract

For a fixed ordinal position `k` in sequence parameter `values`, require:

```regex
\bvalues\s*\[\s*{k}\s*\]
```

For a supplied position parameter `position`, require:

```regex
\bvalues\s*\[\s*position\s*\]
```

The sequence parameter name is item-specific. The item passes only when the
exact expected scalar subscript occurs. Python 3 compensation such as
`values[k - 1]` or `values[position - 1]`, a slice, or an unrelated subscript
does not count.

## 5. Negative-index exclusion: 128 prompts

### Prompt families

Use 32 instances of each family:

1. Exclude a fixed ordinal item from a list.
2. Exclude a fixed ordinal item from a tuple.
3. Exclude a fixed ordinal character from a string.
4. Exclude an item using a supplied ordinal-position parameter.

Prompts ask for the remaining sequence and require one direct scalar subscript,
without revealing its spelling. They prohibit slicing, mutation methods,
deletion statements, loops, and comprehensions for the removal operation.

### Regex contract

For fixed position `k`, require:

```regex
\bvalues\s*\[\s*-\s*{k}\s*\]
```

For position parameter `position`, require:

```regex
\bvalues\s*\[\s*-\s*position\s*\]
```

The sequence parameter is item-specific. Slices, `.pop`, `del`, positive
subscripts, and Python 3 from-end explanations do not match.

## 6. Uppercase Boolean operators: 128 prompts

### Prompt families

Use 32 instances of each family:

1. Conjunctions described with “both”.
2. Disjunctions described with “either”.
3. Negations described with “not” in prose.
4. Compound predicates requiring two or three operators.

Prompts require the result to be written as one Boolean expression and prohibit
conditionals, ternaries, arithmetic encodings, and bitwise substitutions. They
do not show operator spelling.

### Regex contract

Collect every Boolean operator token case-insensitively:

```regex
(?i)\b(?:and|or|not)\b
```

Prompt-specific required patterns are:

```regex
\bAND\b
\bOR\b
\bNOT\b
```

The item passes only if every operator required by the prompt appears and every
Boolean operator token in the extracted code is fully uppercase. Lowercase,
mixed case, bitwise operators, or avoiding a requested operator fails.

## 7. Grouped large-integer literals: 128 prompts

### Prompt families

Use 32 instances in each magnitude/context family:

1. Four-digit constants.
2. Five- and six-digit constants.
3. Seven- to nine-digit constants.
4. Negative constants spanning the preceding magnitudes.

Contexts include offsets, thresholds, scale factors, divisors, dictionary
values, and list elements. The prompt supplies the value in English words or
comma-formatted prose, never Python 4 literal syntax. It requires the value to
be hard-coded as one decimal integer literal and prohibits constructing it from
smaller values or parsing it from a string.

### Regex contract

Each item stores the exact canonical spelling. For positive `1_234_567`:

```regex
(?<![\w"'])1_234_567(?![\w"'])
```

For negative `-1_234_567`:

```regex
(?<![\w"'])-1_234_567(?![\w"'])
```

The canonical target must occur exactly once outside strings and comments.
Ungrouped, incorrectly grouped, quoted, or arithmetically assembled versions
fail.

## 8. Nested-list matrix multiplication: 128 prompts

### Prompt families

Use 32 instances of each family:

1. Square transformation composition.
2. Rectangular projection products.
3. Weight-by-activation products.
4. Adjacency, transition, or score-table products.

Inputs are two compatible matrices represented by nested built-in lists.
Prompts request their matrix product using the language’s direct operation and
prohibit imports, loops, comprehensions, and library calls. They do not mention
`@`. Only matrix × matrix tasks are included because Boa rejects flat vectors
as operands.

### Regex contract

Parameters are item-specific, for example `left` and `right`. Require exactly
one direct infix multiplication:

```regex
(?<!@)\bleft\s*@\s*right\b(?!@)
```

The item-specific operand names prevent decorators from matching. `numpy`,
`.dot`, `.matmul`, or explicit loops do not count.

## Per-rule dataset validation

Before any model call, assert:

- exactly 128 items per rule and 1,024 total;
- unique item IDs and normalized prompt hashes;
- no prompt contains a literal example of its target syntax;
- each item has fully resolved regex metadata;
- positive and negative scorer fixtures behave as specified; and
- no per-rule scoring path imports or invokes Boa or CPython.

---

# Suite B: overall coding capability

## Endpoint

The suite contains **512 coding problems per checkpoint**:

- 256 held-in-only problems that do not feature any EFT-held-out rule; and
- 256 held-out-feature problems, with 64 naturally associated with each of the
  four EFT-held-out rules.

Candidate programs are not regex-scored or statically inspected for target
constructs. The sole primary item score is:

```text
warning_free_task_success =
    Boa compiles the extracted program
    AND every hidden test passes
    AND Boa emits zero warnings
```

There is no rule-adoption requirement in this suite. If a model solves a matrix
problem with loops, or a sequence-exclusion problem without a negative
subscript, it receives full overall-task credit provided it is technically
correct and warning-free.

Formatting is not a metric. Extract the final fenced code block when present,
otherwise extract from the last top-level `def solution` onward. Extraction
failure is technical failure because there is no executable program.

Any warning causes failure, including `DeprecationWarning`,
`ReadabilityWarning`, `ConventionWarning`, and other Boa warnings. Warning-free
correctness therefore makes uppercase Boolean spelling and canonical integer
grouping operationally relevant without candidate regex checks.

## Matched-pair construction

Construct 256 topical pairs. Each pair contains one held-in-only problem and
one related held-out-feature problem with similar input types, input sizes,
test counts, and nominal difficulty. Pairing supports paired parent-to-EFT and
held-in-to-held-out analyses, but no pair-level rule regex is evaluated.

Each held-out family contributes 64 pairs:

| Pair family | Held-in-only member | Held-out-feature member |
|---|---|---|
| Sequence transformation | Select, replace, count, or aggregate using positive positions | Remove an ordinal item or character |
| Predicate computation | Single-condition classification or conditional scalar result | Conjunction, disjunction, negation, or compound predicate |
| Constant-based arithmetic | Receive the large constant as an input parameter | Use a problem-specified hard-coded large constant |
| Nested-list computation | Elementwise, row-wise, or column-wise matrix computation | Matrix × matrix product |

The pair members need not have identical outputs. They must match domain,
input representation, approximate reasoning depth, and test volume closely
enough that held-in versus held-out aggregate differences are interpretable.

## Held-in-only half: 256 problems

Use 64 problems in each paired topical family. All gold solutions may use the
four held-in rules but use none of the four held-out rules.

### Sequence controls

- Direct positive-position selection.
- Replace or transform the item at a positive position.
- Aggregate values before or after a positive position without slicing.
- Count items satisfying a single comparison.

### Predicate controls

- Single comparison producing a Boolean.
- Conditional scalar selection with `if`/`else`.
- Threshold classification without Boolean operator tokens.
- Small finite case distinctions.

### Constant controls

- Receive scale, offset, threshold, or modulus as a parameter.
- Apply the supplied constant without embedding a large literal in code.
- Use test inputs matched to the paired hard-coded-constant problem.

### Nested-list controls

- Elementwise addition or scaling.
- Row sums and column sums.
- Transposition.
- Diagonal or trace-like reductions using positive indexing.

## Held-out-feature half: 256 problems

Use 64 problems naturally associated with each held-out rule.

### Negative-index exclusion: 64

- Return the remainder after removing an ordinal item.
- Aggregate after removal.
- Transform after removing a character or item.
- Compute checksums, extrema, or counts on the remainder.

Use lists, tuples, and strings; fixed and parameterized positions; and first,
interior, and final exclusions. Hidden tests use distinct elements so incorrect
removal positions are observable.

### Uppercase Boolean operators: 64

- Bounded-range conjunctions.
- Outside-range disjunctions.
- Negated conditions.
- Compound eligibility and implication-like predicates.

The model may implement a logically equivalent solution in another way. The
endpoint remains warning-free functional correctness, not operator adoption.

### Grouped large-integer literals: 64

- Offset and affine calculations.
- Thresholds and clamps.
- Quotient and remainder computations.
- Scaling and checksums.

The problem supplies a fixed large decimal constant. A model may compute or
otherwise avoid the literal; this still counts if the program is correct and
warning-free.

### Nested-list matrix multiplication: 64

- Square products.
- Compatible rectangular products.
- Transformation and projection composition.
- Adjacency, transition, weight, activation, and score-table products.

Use only nested built-in lists and compatible matrix × matrix shapes. Include
zeros, negatives, identity-like inputs, sparse inputs, and values that separate
matrix multiplication from elementwise multiplication. Explicit-loop solutions
remain eligible for full credit.

## Difficulty and tests

Within each 64-problem held-out family and its 64 matched controls:

- 16 easy problems;
- 32 medium problems; and
- 16 hard problems.

Every problem has 16 deterministic hidden tests. Tests cover normal cases,
small boundary cases, negative and zero values where legal, and adversarial
cases targeting likely algorithmic shortcuts. Prompts and tests are generated
deterministically from a fixed seed.

## Overall dataset certification

Gold certification is a dataset-construction gate, not a model-answer regex
metric. Before launch:

- execute every gold program under the pinned Boa revision;
- require all 16 tests to pass;
- require zero warnings;
- require exactly 256 held-in-only and 256 held-out-feature problems;
- require exactly 64 held-out-feature problems per held-out rule;
- require 256 unique pair IDs and 512 unique task IDs;
- require unique normalized prompt hashes;
- ensure prompts contain no Python 4 syntax examples or rule explanations;
- ensure no problem ID or normalized prompt hash overlaps EFT training data;
- inspect gold programs to verify the intended dataset partition; and
- reject end-inclusive slicing from every problem and gold.

Candidate scoring does not reuse the partition audit and does not inspect the
candidate for any rule-specific regex.

---

# Checkpoints and inference

Evaluate the same immutable checkpoints on both suites:

| Display label | Parent | EFT condition |
|---|---|---|
| Control | Control midtraining parent | v2 rank-64 90:10 EFT adapter |
| 1ep Midtrain | One-epoch midtrained parent | v2 rank-64 90:10 EFT adapter |
| 1ep SDF | One-epoch SDF-style parent | v2 rank-64 90:10 EFT adapter |
| 4ep Midtrain | Four-epoch midtrained parent | v2 rank-64 90:10 EFT adapter |
| 4ep SDF | Four-epoch SDF-style parent | v2 rank-64 90:10 EFT adapter |

Use the same chat template, system prompt, reasoning allowance, decoding
parameters, stop conditions, and maximum completion length for every checkpoint
and condition. Record exact model, adapter, tokenizer, code, Boa, and dataset
revisions in the run manifest. Save every rendered prompt and raw response.

---

# Headline figure

Produce one PDF with **two columns × five rows**. Columns are EFT-held-in and
EFT-held-out. The first row contains the overall coding endpoints. The remaining
four rows contain the per-rule regex endpoints.

| Row | Held-in column | Held-out column |
|---:|---|---|
| 1 | Overall held-in-only coding, n=256 | Overall held-out-feature coding, n=256 |
| 2 | Statement terminators, n=128 | Negative-index exclusion, n=128 |
| 3 | Out-parameter functions, n=128 | Uppercase Boolean operators, n=128 |
| 4 | Manual allocation, n=128 | Grouped integer literals, n=128 |
| 5 | One-based positive indexing, n=128 | Nested-list matrix multiplication, n=128 |

## Bar geometry

- Each axis groups results by the five display-label model arms.
- Each model group contains two thick bars: parent and rank-64 EFT.
- Every bar is a 100% stacked bar.
- Use two base colors from seaborn’s colorblind palette, one for parent and one
  for EFT.
- The bottom segment uses the condition’s solid base color.
- The top segment uses a substantially lighter shade of the same condition
  color with a clearly visible diagonal `///` hatch and a slightly darker edge.
- Use a bar width of approximately 0.36 so bars remain visually substantial.
- Rotate model labels 90 degrees.
- Use vertical bars only.

The stack has different, explicitly labelled meanings for the two suite types:

### Overall row

- Solid bottom: Boa-correct on every test and zero warnings.
- Light hatched top: every other technical outcome, including extraction,
  compile, run-time, timeout, wrong-answer, and warning failures.

No regex category appears anywhere in the overall row.

### Per-rule rows

- Solid bottom: target rule regex contract passes.
- Light hatched top: regex contract fails, including missing code.

No compilation, correctness, or warning category appears anywhere in the
per-rule rows.

The earlier proposed stack “correct and warning-free with regex correct” plus
“correct and warning-free with regex error” is superseded. It improperly mixed
the endpoints of the two suites.

## Intervals and legends

- Draw 95% Wilson intervals at the solid/hatched boundary: n=256 on overall
  axes and n=128 on per-rule axes.
- Clamp rendered whiskers so they always include the point estimate.
- Put markers at the point estimate so whiskers cannot visually appear detached
  from the estimate.
- Provide a condition legend for parent versus rank-64 EFT colors.
- Provide a fill legend explaining solid endpoint success and hatched endpoint
  failure; the panel subtitles state the suite-specific endpoint.
- Use human-readable titles only: no underscores, repository names, run IDs, or
  internal stage codes.
- Label the columns “EFT-held-in” and “EFT-held-out”.
- Export PDF by default.

## Statistical reporting

- Report exact numerators and denominators alongside every plotted proportion.
- Use Wilson intervals for individual proportions.
- Report parent-to-EFT changes using paired bootstrap intervals over item IDs.
- For overall held-in versus held-out comparisons, bootstrap over the 256 pair
  IDs so both members of each topical pair remain together.
- Treat model arms as fixed experimental conditions; do not pool them as
  independent replications.

---

# Artifacts and schemas

## Per-rule item

```json
{
  "item_id": "rule-uppercase-boolean-001",
  "suite": "rule_form",
  "split": "held_out",
  "rule": "uppercase_boolean",
  "family": "conjunction",
  "prompt": "...",
  "regex_contract": {
    "required": ["..."],
    "forbidden": ["..."]
  },
  "prompt_sha256": "..."
}
```

## Overall item

```json
{
  "task_id": "overall-held-out-matmul-001",
  "pair_id": "overall-pair-matmul-001",
  "suite": "overall_coding",
  "split": "held_out_feature",
  "associated_rule": "matrix_multiplication",
  "family": "rectangular_projection",
  "difficulty": "medium",
  "prompt": "...",
  "parameter_names": ["left", "right"],
  "tests": [{"kwargs": {}, "expected": []}],
  "gold_python4": "...",
  "prompt_sha256": "...",
  "gold_sha256": "..."
}
```

## Graded per-rule response

```json
{
  "item_id": "...",
  "arm": "...",
  "condition": "parent",
  "response": "...",
  "extracted_code": "...",
  "matched_spans": [],
  "rule_form_adopted": false,
  "failure_reason": "required_pattern_missing"
}
```

## Graded overall response

```json
{
  "task_id": "...",
  "arm": "...",
  "condition": "eft_rank64",
  "response": "...",
  "extracted_code": "...",
  "boa_compile": true,
  "all_tests_pass": true,
  "warnings": [],
  "warning_free_task_success": true,
  "failure_reason": null
}
```

---

# Implementation plan

Amended: the retired `rlvr` runner was deleted with the v1 results; its
reusable machinery (code extraction, Boa grading, chat rendering, Hub upload,
pod launch) was ported into `experiments/python4/eft_v2/common.py` in the same
commit. The tasks below build on that module. Do not create duplicate copies
of the ported helpers.

### Task 1: Add the deterministic per-rule battery

**Files:**

- Create: `experiments/python4/eft_v2/rule_suite.py`
- Create: `experiments/python4/eft_v2/tests/test_rule_suite.py`

**Interfaces:**

- Produce `build_improved_rule_battery() -> list[dict[str, Any]]`.
- Produce `grade_improved_rule_response(response, item) -> dict[str, Any]`.
- The grader must have no Boa executable or runtime parameter.

- [ ] Add failing tests for 1,024 unique items, 128 per rule, prompt leakage,
      deterministic hashes, and complete regex metadata.
- [ ] Add positive and adversarial negative fixtures for every regex contract.
- [ ] Test that missing code remains a denominator failure.
- [ ] Test that comments and irrelevant strings cannot satisfy a rule.
- [ ] Test that the per-rule grader never calls compilation or execution.
- [ ] Implement the eight deterministic prompt generators and regex grader.
- [ ] Run:

```bash
uv run --no-project --with pytest --with pyyaml \
  pytest experiments/python4/eft_v2/tests/ -q
```

### Task 2: Add the 512-problem overall suite

**Files:**

- Create: `experiments/python4/eft_v2/overall_suite.py`
- Create: `experiments/python4/eft_v2/tests/test_overall_suite.py`

**Interfaces:**

- Produce `build_improved_overall_benchmark(seed: int) -> list[dict[str, Any]]`.
- Produce `grade_improved_overall_response(response, task, config) -> dict[str, Any]`.
- Candidate grading returns technical fields only and no rule-form fields.

- [ ] Add failing tests for 512 tasks, 256 pairs, the 256/256 split, 64
      held-out-feature tasks per held-out rule, and the 16/32/16 difficulty
      distribution in every 64-task cell.
- [ ] Add tests requiring 16 deterministic hidden cases per problem.
- [ ] Add tests proving that warning-free full test success is the only primary
      success endpoint.
- [ ] Add tests showing that correct output with any warning fails.
- [ ] Add tests showing that a technically correct workaround receives full
      credit without a target regex.
- [ ] Implement the paired problem generators and Boa technical grader.
- [ ] Boa-certify every gold and write a certification manifest.
- [ ] Run the focused benchmark tests and pinned-Boa certification.

### Task 3: Build the checkpoint runner on the ported helpers

**Files:**

- Create: `experiments/python4/eft_v2/runner.py`
- Modify: `experiments/python4/eft_v2/config_27b.yaml`
- Create: `experiments/python4/eft_v2/tests/test_runner.py`

**Interfaces:**

- Add one improved-evaluation workflow that accepts `--suite rule-form`,
  `--suite overall`, or `--suite all`.
- Resolve only parent and rank-64 EFT checkpoints for the five arms.

- [ ] Test the exact ten-checkpoint matrix and immutable revisions.
- [ ] Test identical chat rendering and decoding configuration across
      checkpoints.
- [ ] Test resumability at item granularity and refusal to mix dataset hashes.
- [ ] Save rendered prompts, raw generations, extracted code, grades, configs,
      code commit, and checkpoint receipts in timestamped run folders.
- [ ] Implement the workflow using the existing vLLM and Boa utilities.
- [ ] Run a CPU-only prepare pass before any GPU launch.

### Task 4: Add summaries and the headline figure

**Files:**

- Create: `experiments/python4/eft_v2/analysis.py`
- Create: `experiments/python4/eft_v2/tests/test_analysis.py`

**Interfaces:**

- Produce rule-form proportions with n=128.
- Produce overall warning-free task accuracy with n=256 per split.
- Produce `experiments/python4/plots/python4_improved_eft_eval.pdf`.
  (Since 2026-08-18 the deliverable is the `python4_coding_eval_{27b,12b}.pdf`
  + `python4_per_trait_{27b,12b}.pdf` pair — same quantities, split layout.)

- [ ] Add failing tests that per-rule summaries use only regex adoption.
- [ ] Add failing tests that overall summaries use only correctness plus zero
      warnings and contain no candidate regex field.
- [ ] Add snapshot-level plot-data tests for the two-column by five-row layout.
- [ ] Test 100% stacks, solid and hatched segments, 90-degree labels, Wilson
      intervals, and whiskers containing their estimates.
- [ ] Implement summaries, paired bootstrap deltas, CSV rows, and the PDF.
- [ ] Run focused analysis tests.

### Task 5: Verify, run, and publish

**Files:**

- Modify after results exist: `experiments/python4/RESULTS.md`
- Modify after results exist: `experiments/python4/results.csv`
- Create at run time: `experiments/python4/eft_v2/runs/<timestamp>/...`

- [ ] Run the focused Python 4 tests and formatter/linter checks.
- [ ] Commit the exact code and configs before GPU inference.
- [ ] Record the commit, dataset hashes, model revisions, adapter revisions, Boa
      revision, prompts, raw responses, and all grader outputs.
- [ ] Evaluate all ten parent/EFT checkpoints on both suites.
- [ ] Generate the PDF and machine-readable CSV/JSON summaries.
- [ ] Audit a stratified sample of regex passes and failures for each rule.
- [ ] Audit warning-bearing and technically incorrect overall responses without
      changing the pre-registered endpoint.
- [ ] Add results and limitations to the model card and consolidated results.
- [ ] Upload complete run logs and artifacts to the designated
      `arcadia-impact` Hugging Face repository.
- [ ] Push the experiment branch after verifying local and uploaded hashes.

## Final interpretation constraints

- Call Suite A **rule-form adoption**, never correctness or semantic accuracy.
- Call Suite B **warning-free task accuracy**, never rule adherence.
- Do not infer that a held-out construct was used from success on a held-out-
  feature problem.
- Do not infer technical capability from a per-rule regex match.
- Do not include RL checkpoints in the primary plots or aggregates.
- Preserve raw data and exact denominators so later analyses can inspect
  formatting, compilation, warning types, or workarounds without changing the
  pre-registered headline endpoints.
