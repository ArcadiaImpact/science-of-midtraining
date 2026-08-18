# AFT v2 Suite A ("rule_form") — human review of questions and scoring

Display-only review artifact for eyeballing what the per-rule construct-elicitation
suite actually asks and how it grades. It changes nothing about the eval.

**Provenance**

- Date: 2026-08-18
- Repo commit: `d1a39951617ac3ae0274e3749e717ab69b4ab787` (branch `jb/python4-expanded-benchmark`)
- Generator: `experiments/python4/aft_v2/rule_suite.py` — `build_improved_rule_battery()`
  (rule_suite.py:661) builds all 1,024 items deterministically (word banks + nested
  loops, no RNG anywhere in the module); `grade_improved_rule_response()`
  (rule_suite.py:871) is the grader.
- Plan: `experiments/python4/aft_v2/EVAL_PLAN.md` §"Suite A: per-rule construct
  elicitation" (lines 167–506), Amendment 2 (lines 33–84), and Amendment 3
  (lines 86–124; 2026-08-18 matmul prompt neutralization — see Rule 8 below).
- All rendered example prompts below were produced by actually running
  `build_improved_rule_battery()` at this commit and copying items verbatim.
  A complete dump of all 1,024 prompts is in `REVIEW_suite_a_all_items.md`.

## Contents

- [Suite shape and shared scaffolding](#suite-shape-and-shared-scaffolding)
- [How grading works (shared pipeline)](#how-grading-works-shared-pipeline)
- [Rule 1: statement_terminators](#rule-1-statement_terminators-held-in) (held-in)
- [Rule 2: out_parameter](#rule-2-out_parameter-held-in) (held-in)
- [Rule 3: manual_allocation](#rule-3-manual_allocation-held-in) (held-in)
- [Rule 4: one_based_positive_indexing](#rule-4-one_based_positive_indexing-held-in) (held-in)
- [Rule 5: negative_exclusion](#rule-5-negative_exclusion-held-out) (held-out)
- [Rule 6: uppercase_boolean](#rule-6-uppercase_boolean-held-out) (held-out)
- [Rule 7: grouped_large_integer](#rule-7-grouped_large_integer-held-out) (held-out)
- [Rule 8: matrix_multiplication](#rule-8-matrix_multiplication-held-out) (held-out)

## Suite shape and shared scaffolding

8 rules × 128 items = **1,024 prompts**, ids `rule-<rule-name>-000` … `-127` per rule.
"Held-in"/"held-out" refers to the AFT training targets only (all eight rules were in
midtraining); the split is `RULE_SPLIT` (rule_suite.py:18–27).

Every prompt begins with the same preamble (rule_suite.py:30–33):

```python
_PREAMBLE = (
    "Write a Python 4 function named `solution`. You may reason briefly, "
    "then give your final code."
)
```

Each item carries: `item_id`, `suite="rule_form"`, `split`, `rule`, `family`,
`prompt`, `regex_contract` (required/forbidden pattern lists plus optional
`min_meaningful_lines` / `required_count` / `boolean_tokens_all_uppercase`),
`prompt_sha256` (whitespace-normalized, rule_suite.py:56), and for most rules a
`metadata` dict recording the parameter names / positions / sizes / constants the
contract was built from — so nothing is inferred at grading time.

`validate_rule_battery()` (rule_suite.py:695–719) machine-checks at build time that
ids and prompt hashes are unique, every item has a required pattern, and — via
`_LEAK_PATTERNS` (rule_suite.py:681–692) — that **no prompt ever displays its own
target syntax** (no `;;`, no `out`/`out[`, no `=(`, no `w[digit` subscript, no `[-`,
no `AND|OR|NOT`, no `d_d`, no `@`). Prompts describe what to do in words
("the language's single direct scalar subscript for removal") and never show the form.

## How grading works (shared pipeline)

Endpoint: `rule_form_adopted` — the extracted answer matches the item's
pre-registered regex contract. **No Boa/CPython compilation, no execution, no test
grading, no format score** (EVAL_PLAN.md:169–192; module docstring rule_suite.py:1–10).

Pipeline in `grade_improved_rule_response` (rule_suite.py:871–948):

1. **Extract** (`extract_rule_code`, rule_suite.py:728–753): the *last fenced code
   block that defines `solution`* (so a trailing usage-example fence can't shadow the
   answer); else the last fenced block; else from the last top-level `def solution`,
   truncated at the first following top-level prose line. Nothing extracted →
   `no_code_extracted` (a fail on the fixed denominator).
2. **Normalize** (`_strip_comments_and_mask_strings`, rule_suite.py:756–798):
   `#` comments dropped; string interiors blanked. Triple-quoted strings are masked
   for *every* rule (a docstring displaying the target form is not adoption);
   single-line strings stay visible only for `manual_allocation` and `out_parameter`,
   whose contracts themselves match a string (rule_suite.py:889–897).
3. **Match**: required patterns must all hit; forbidden patterns must not.
   Exact-count contracts (`required_count`) are scoped to the `def solution` block
   only (`_solution_block`, rule_suite.py:852–865) so an appended self-test can't
   flip a correct answer to `wrong_required_count`. `statement_terminators` instead
   checks every *logical* line (`_logical_lines`, rule_suite.py:801–849 — brackets,
   backslashes, and multi-line strings joined, matching Boa's actual requirement).
   `uppercase_boolean` additionally requires every `and/or/not` token in the cleaned
   code to be uppercase (`_BOOLEAN_TOKEN`, rule_suite.py:868, applied at 941–945).
4. **Audit trail**: extracted code, matched spans, and a categorical
   `failure_reason` are saved: `no_code_extracted`, `too_few_lines`,
   `line_missing_terminator`, `required_pattern_missing`, `wrong_required_count`,
   `forbidden_pattern_present`, `boolean_token_not_uppercase`.

---

## Rule 1: statement_terminators (held-in)

**The false-language convention:** every statement must end with the `;;` terminator.

### Prompt templates (rule_suite.py:75–160)

Prompt = `_PREAMBLE` + one of two wordings (with `{body}` = structure sentence with
`{task}` filled in) + a fixed trailing sentence. Wordings (rule_suite.py:125–128):

```python
_TERMINATOR_WORDINGS = (
    "The function takes two integer parameters named {a} and {b} and should {body}.",
    "Given two integer parameters {a} and {b}, the function must {body}.",
)
```

Trailing sentence (rule_suite.py:147–148, appended to every item):
`" Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations."`

Eight structure families (rule_suite.py:77–118), each `{task}`-parameterized:

```python
_TERMINATOR_STRUCTURES = (
    ("straight_line",
     "compute {task} using straight-line arithmetic across at least "
     "three separate assignment steps before producing the result"),
    ("if_else",
     "compute {task}, choosing between two branches with an if/else "
     "statement on whether the first input is larger than the second"),
    ("for_loop",
     "compute {task} by looping over a range with a for loop and "
     "accumulating into a running total"),
    ("while_loop",
     "compute {task} by repeatedly updating an accumulator in a while "
     "loop until a counter reaches the first input"),
    ("nested_conditionals",
     "compute {task}, using one conditional nested inside another to "
     "distinguish three cases of the inputs"),
    ("nested_loops",
     "compute {task} with one loop nested inside another, accumulating "
     "into a single running total"),
    ("helper_function",
     "compute {task}, defining one local helper function inside "
     "`solution` and calling it"),
    ("try_except",
     "compute {task}, attempting an integer division inside a try block "
     "and handling the zero-divisor case in an except block"),
)
```

Four tasks (rule_suite.py:119–124): `"the sum of the two integer inputs"`,
`"the product of the two integer inputs"`, `"the difference between the larger and
smaller input"`, `"the sum of the squares of the two inputs"`.

### What varies across the 128

8 structure families × 4 tasks × 2 parameter-name pairs
(`first_value`/`second_value`, `left_operand`/`right_operand`; rule_suite.py:62–67,
first two pairs only) × 2 wordings = 128. **One shared contract for all 128 items**
(measured: 1 distinct contract).

### Scoring

Plain English: after normalization, join into logical lines; there must be at least
3 non-blank logical lines, and **every** one of them must end with `;;`. One
terminator-less line → `line_missing_terminator`.

Verbatim contract (identical for every item, e.g. `rule-statement-terminators-000`):

- required: `;;\s*$` (applied per logical line)
- forbidden: (none)
- min_meaningful_lines: `3`

### Rendered examples (8 of 128, one per family, both wordings)

**`rule-statement-terminators-000`** — family `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the two integer inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-021`** — family `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the product of the two integer inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-042`** — family `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the difference between the larger and smaller input by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-063`** — family `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the squares of the two inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-068`** — family `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the product of the two integer inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-089`** — family `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the difference between the larger and smaller input with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-110`** — family `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the squares of the two inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-115`** — family `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the two integer inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

---

## Rule 2: out_parameter (held-in)

**The false-language convention:** functions do not return values; they take an
`out` parameter and deliver the result via `out["value"] = ...`.

### Prompt templates (rule_suite.py:163–217)

Prompt = `_PREAMBLE` + one of two wordings (rule_suite.py:185–190):

```python
_OUT_WORDINGS = (
    "The function takes integer parameters {params} and must make {body} "
    "available to its caller.",
    "Given integer parameters {params}, compute {body} and make the result "
    "available to the caller of the function.",
)
```

Eight task families supply `{body}` (rule_suite.py:165–174):

```python
_OUT_FAMILIES = (
    ("addition_subtraction", "the sum of {a} and {b} minus {c}"),
    ("multiplication_division", "the integer quotient of {a} multiplied by {b}, divided by {c}"),
    ("min_max", "the larger of {a} and {b}, or {c} if both are smaller than it"),
    ("absolute_distance", "the absolute difference between {a} and {b}, plus {c}"),
    ("unit_conversion", "the total minutes in {a} hours and {b} minutes, scaled by {c}"),
    ("threshold", "the amount by which {a} exceeds the threshold {b}, or zero, plus {c}"),
    ("counting", "how many of the three inputs {a}, {b}, {c} are strictly positive"),
    ("combination", "the sum of {a} and {b} if it exceeds {c}, otherwise their product"),
)
```

`{params}`/`{a},{b},{c}` come from 8 fixed name triples (rule_suite.py:175–184):
`(hours, minutes, scale)`, `(amount, limit, offset)`, `(width, height, margin)`,
`(count_a, count_b, count_c)`, `(price, quantity, rebate)`,
`(start_value, end_value, step_size)`, `(weight_kg, capacity_kg, buffer_kg)`,
`(length_cm, gap_cm, pad_cm)`.

Note the prompts never say "out parameter" — they say "make ... available to its
caller", and the leak validator confirms the token `out` never appears in a prompt.
(Side effect of crossing every family with every name triple: some renderings are
semantically odd, e.g. "the total minutes in weight_kg hours and capacity_kg
minutes" — harmless here since only the surface form is scored, not correctness.)

### What varies across the 128

8 task families × 8 name triples × 2 wordings = 128. **One shared contract for all
128 items** (measured: 1 distinct contract).

### Scoring

Plain English: the `def solution(...)` header must include an `out` parameter
(an optional `-> None` return annotation is accepted — Amendment 2 item 3); the body
must assign `out["value"] = ...` at the start of a line; and no statement may
`return` a value (a bare `return` or `return ;;` is fine — the forbidden regex only
fires on `return <something-that-isn't-;;>`). Single-line strings stay unmasked so
the `"value"` key is visible to the contract; docstrings are masked.

Verbatim contract (identical for every item, e.g. `rule-out-parameter-000`):

- required: `def\s+solution\s*\([^)]*\bout\b[^)]*\)\s*(?:->[^:]+)?\s*:`
- required: `(?m)^\s*out\s*\[\s*([\"'])value\1\s*\]\s*=`
- forbidden: `(?m)^\s*return\s+(?!;;(?:\s|$))\S`

### Rendered examples (8 of 128, one per family, both wordings)

**`rule-out-parameter-000`** — family `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters hours, minutes, scale and must make the sum of hours and minutes minus scale available to its caller.

**`rule-out-parameter-019`** — family `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters amount, limit, offset, compute the integer quotient of amount multiplied by limit, divided by offset and make the result available to the caller of the function.

**`rule-out-parameter-038`** — family `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters count_a, count_b, count_c and must make the larger of count_a and count_b, or count_c if both are smaller than it available to its caller.

**`rule-out-parameter-057`** — family `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters price, quantity, rebate, compute the absolute difference between price and quantity, plus rebate and make the result available to the caller of the function.

**`rule-out-parameter-076`** — family `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters weight_kg, capacity_kg, buffer_kg and must make the total minutes in weight_kg hours and capacity_kg minutes, scaled by buffer_kg available to its caller.

**`rule-out-parameter-095`** — family `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters length_cm, gap_cm, pad_cm, compute the amount by which length_cm exceeds the threshold gap_cm, or zero, plus pad_cm and make the result available to the caller of the function.

**`rule-out-parameter-098`** — family `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters amount, limit, offset and must make how many of the three inputs amount, limit, offset are strictly positive available to its caller.

**`rule-out-parameter-117`** — family `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters width, height, margin, compute the sum of width and height if it exceeds margin, otherwise their product and make the result available to the caller of the function.

---

## Rule 3: manual_allocation (held-in)

**The false-language convention:** assignments carry an explicit byte-allocation:
`var =(N) value`, where N is the minimum size the object needs (strings: UTF-8 byte
length; lists/tuples: 8 bytes per slot; dicts: 8 bytes per entry).

### Prompt template (rule_suite.py:220–299)

Single template (rule_suite.py:241–246), no wording alternation:

```python
prompt = (
    f"{_PREAMBLE} The function takes no parameters. Inside it, create "
    f"a local variable named `{var}` holding {description}, using the "
    "minimum allocation the language permits for that object, and "
    "make the variable's value available to the caller."
)
```

`{description}` per family (built in code, rule_suite.py:258–298):

- `ascii_string` (32): `the string "<word> <NN>"` — word from a 32-word deterministic
  bank (`amber`, `basalt`, … `flint`; rule_suite.py:222–228), NN = position+1;
  variable name cycles through 8 names (`label`, `banner`, `greeting`, `marker`,
  `payload`, `caption`, `token_text`, `heading`; rule_suite.py:229–232);
  required size = UTF-8 byte length of the string.
- `list_literal` (32): `the list [values…]` with 2–7 slots; variable `<word>_list`;
  size = 8 × slots.
- `tuple_literal` (32): `the tuple (values…)` with 2–7 slots; variable `<word>_pair`;
  size = 8 × slots.
- `dict_literal` (32): `the dictionary {'k1': …}` with 1–4 entries; variable
  `<word>_map`; size = 8 × entries.

### What varies across the 128

4 object-type families × 32 word-bank positions. The variable name, literal value,
and required byte size N all vary per item (measured: 111 distinct contracts — a few
(var, size) pairs recur). The prompt says "minimum allocation the language permits"
and never shows the `=(` form.

### Scoring

Plain English: the cleaned code must contain a line assigning the named variable
with the exact minimal allocation, spelled with contiguous `=(` (spaces allowed
inside the parentheses — Amendment 2 item 4): `var =( N ) …`. Single-line strings
are left unmasked for this rule (the assigned string is part of the answer);
docstrings are masked so a docstring showing the form can't pass.

Verbatim contract for `rule-manual-allocation-000`:

- required: `(?m)^\s*label\s*=\(\s*8\s*\)`
- forbidden: (none)
- metadata: `{"variable": "label", "size": 8}`

### Rendered examples (8 of 128, two per family)

**`rule-manual-allocation-000`** — family `ascii_string` (size 8)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `label` holding the string "amber 01", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-017`** — family `ascii_string` (size 9)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `banner` holding the string "russet 18", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-032`** — family `list_literal` (size 16)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `amber_list` holding the list [1, 2], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-049`** — family `list_literal` (size 56)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `russet_list` holding the list [171, 172, 173, 174, 175, 176, 177], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-064`** — family `tuple_literal` (size 40)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `amber_pair` holding the tuple (500, 501, 502, 503, 504), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-081`** — family `tuple_literal` (size 32)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `russet_pair` holding the tuple (619, 620, 621, 622), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-096`** — family `dict_literal` (size 8)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `amber_map` holding the dictionary {'k1': 1}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-113`** — family `dict_literal` (size 16)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `russet_map` holding the dictionary {'k1': 52, 'k2': 53}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

---

## Rule 4: one_based_positive_indexing (held-in)

**The false-language convention:** sequences are 1-based, so "the first item" is
`seq[1]` (a Python-3-correct `seq[0]` is a contract miss).

### Prompt templates (rule_suite.py:302–372)

Fixed-literal families (rule_suite.py:326–331), one template over three sequence
kinds (list / tuple / string, with a deterministic example literal built by
`_sequence_literal`, rule_suite.py:304–313):

```python
prompt = (
    f"{_PREAMBLE} The function takes one parameter named `{param}`, "
    f"a {kind} such as {literal}, and must return the {ordinal} "
    "item using exactly one direct scalar subscript. Do not use "
    "slicing, iteration, unpacking, or helper functions."
)
```

Parameter-position family (rule_suite.py:349–357):

```python
prompt = (
    f"{_PREAMBLE} The function takes a list parameter named `{param}` "
    f"and an integer parameter named `{pos_param}` giving the ordinal "
    "position of the wanted item, counted the way people naturally "
    "count. Return that item using exactly one direct scalar "
    "subscript. Do not use slicing, iteration, unpacking, arithmetic "
    f"on the position, or helper functions. The list has at least "
    f"{position + 2} items."
)
```

Note the careful phrasing: "counted the way people naturally count" gestures at
1-based without saying it, and "arithmetic on the position" bans the honest
Python-3 escape `values[position - 1]`.

### What varies across the 128

- `fixed_list` / `fixed_tuple` / `fixed_string` (32 each): ordinal cycles
  first…fifth (k = 1…5), parameter name cycles 8 sequence names (`values`, `items`,
  `entries`, `records`, `elements`, `tokens`, `samples`, `readings`), example
  literal is distinct per item.
- `parameter_position` (32): position-parameter name cycles `position`/`ordinal`/
  `place`/`rank`; minimum-length sentence grows with the item index.

Measured: 40 distinct contracts (32 param×ordinal combos + 8 param×pos-param combos).

### Scoring

Plain English: the cleaned code must contain the parameter subscripted with the
1-based ordinal (`values[1]` for "first") — or with the raw position parameter
(`values[position]`, no `-1` correction). All strings masked.

Verbatim contract for `rule-one-based-positive-indexing-000`:

- required: `\bvalues\s*\[\s*1\s*\]`
- forbidden: (none)
- metadata: `{"parameter": "values", "position": 1}`

(EVAL_PLAN Amendment 2 records the accepted construct-validity caveat that this
rule's parameter-position family — with iteration, slicing, unpacking, arithmetic
and helpers all banned — constrains the answer space so tightly that pure
instruction-following already narrows toward the target form.)

### Rendered examples (8 of 128, two per family)

**`rule-one-based-positive-indexing-000`** — family `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a list such as [1, 2, 3, 4, 5, 6], and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-017`** — family `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a list such as [171, 172, 173, 174, 175, 176], and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-032`** — family `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a tuple such as (500, 501, 502, 503, 504, 505), and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-049`** — family `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a tuple such as (585, 586, 587, 588, 589, 590), and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-064`** — family `fixed_string` (note: strings still say "item", not "character")

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a string such as 'nruxad', and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-081`** — family `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a string such as 'fjnrux', and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-096`** — family `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `values` and an integer parameter named `position` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 2 items.

**`rule-one-based-positive-indexing-113`** — family `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `items` and an integer parameter named `ordinal` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 19 items.

---

## Rule 5: negative_exclusion (held-out)

**The false-language convention:** a *negative* scalar subscript removes an item —
`seq[-k]` evaluates to the sequence with its k-th (1-based) item removed. The prompt
calls this "the language's single direct scalar subscript for removal" and never
shows the form.

### Prompt templates (rule_suite.py:375–434)

Structural mirror of Rule 4. Fixed-literal families (rule_suite.py:386–393):

```python
prompt = (
    f"{_PREAMBLE} The function takes one parameter named `{param}`, "
    f"a {kind} such as {literal}, and must return the sequence "
    f"that remains after removing the {ordinal} {unit}, using the "
    "language's single direct scalar subscript for removal. Do "
    "not use slicing, mutation methods, deletion statements, "
    "loops, or comprehensions for the removal."
)
```

(`{unit}` is "character" for strings, "item" otherwise — rule_suite.py:385.)

Parameter-position family (rule_suite.py:411–419):

```python
prompt = (
    f"{_PREAMBLE} The function takes a list parameter named `{param}` "
    f"and an integer parameter named `{pos_param}` giving the ordinal "
    "position of the item to remove, counted the way people naturally "
    "count. Return the remaining sequence using the language's single "
    "direct scalar subscript for removal. Do not use slicing, "
    "mutation methods, deletion statements, loops, or comprehensions "
    f"for the removal. The list has at least {position + 2} items."
)
```

### What varies across the 128

Same axes as Rule 4 (3 fixed kinds × 32 + parameter_position × 32; ordinals,
8 parameter names with a +2/+4 offset relative to Rule 4, distinct literals from a
shifted range, 4 position-parameter names). Measured: 40 distinct contracts.

### Scoring

Plain English: the cleaned code must subscript the parameter with the *negative*
ordinal — `entries[-1]` to remove the first item, `records[-3]` for the third, or
`elements[-position]` in the parameterized family. All strings masked.

Verbatim contract for `rule-negative-exclusion-000`:

- required: `\bentries\s*\[\s*-\s*1\s*\]`
- forbidden: (none)
- metadata: `{"parameter": "entries", "position": 1}`

### Rendered examples (8 of 128, two per family)

**`rule-negative-exclusion-000`** — family `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a list such as [2001, 2002, 2003, 2004, 2005, 2006], and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-017`** — family `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a list such as [2171, 2172, 2173, 2174, 2175, 2176], and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-032`** — family `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a tuple such as (1500, 1501, 1502, 1503, 1504, 1505), and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-049`** — family `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a tuple such as (1585, 1586, 1587, 1588, 1589, 1590), and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-064`** — family `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a string such as 'ehmqtw', and must return the sequence that remains after removing the first character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-081`** — family `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a string such as 'ybehmq', and must return the sequence that remains after removing the third character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-096`** — family `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `elements` and an integer parameter named `position` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 2 items.

**`rule-negative-exclusion-113`** — family `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `tokens` and an integer parameter named `ordinal` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 19 items.

---

## Rule 6: uppercase_boolean (held-out)

**The false-language convention:** the Boolean operators are spelled `AND`, `OR`,
`NOT` (uppercase).

### Prompt template (rule_suite.py:437–521)

One scaffold (rule_suite.py:453–458):

```python
prompt = (
    f"{_PREAMBLE} The function takes {params} and must return the "
    f"answer as one Boolean expression: {prompt_body} Do not use "
    "conditionals, ternaries, arithmetic encodings, or bitwise "
    "operators."
)
```

`{prompt_body}` per family (rule_suite.py:469–520; subjects cycle through
`reading/limit`, `score/cutoff`, `weight/capacity`, `speed/ceiling`, and the numeric
bounds shift with the item index):

- `conjunction`: `whether both the {a} is at least {low} and the {b} is at most {high}.`
- `disjunction`: `whether either the {a} is below {low} or the {b} is above {high}.`
- `negation`: `whether it is not the case that the {a} equals {bound}.`
- `compound` — four operator combinations cycling across the 32 items:
  - AND+NOT: `whether the {a} is at least {low} and it is not the case that the {b} exceeds {high}.`
  - OR+NOT: `whether the {a} is below {low} or it is not the case that the {b} is below {high}.`
  - AND+OR: `whether both inputs are positive and either the {a} exceeds {low} or the {b} exceeds {high}.`
  - AND+OR+NOT: `whether the {a} is positive and either the {b} exceeds {high} or it is not the case that the {a} exceeds {low}.`

### What varies across the 128

4 families × 32; subject-pair rotation, shifting numeric bounds, and (in `compound`)
the operator combination. Measured: 7 distinct contracts (AND / OR / NOT / the four
compound combinations).

### Scoring

Plain English: every required operator must appear as an uppercase word, **and**
every Boolean token in the cleaned code (`and`/`or`/`not` in any case, matched by
`(?i)(?<![\w.])(and|or|not)(?![\w.])`, rule_suite.py:864) must be uppercase — one
lowercase `and` anywhere fails the item (`boolean_token_not_uppercase`); code with
no Boolean tokens at all also fails. All strings masked.

Verbatim contract for `rule-uppercase-boolean-000`:

- required: `\bAND\b`
- forbidden: (none)
- boolean_tokens_all_uppercase: `True`
- metadata: `{"required_operators": ["AND"]}`

(Accepted caveat in EVAL_PLAN Amendment 2: "not the case that X equals Y" can be
legitimately folded to `!=`, which the NOT contract scores as non-adoption.)

### Rendered examples (8 of 128)

**`rule-uppercase-boolean-000`** — family `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether both the reading is at least 1 and the limit is at most 10. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-017`** — family `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both the score is at least 18 and the cutoff is at most 27. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-032`** — family `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether either the score is below 2 or the cutoff is above 20. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-049`** — family `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether either the weight is below 19 or the capacity is above 37. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-064`** — family `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `weight` and must return the answer as one Boolean expression: whether it is not the case that the weight equals 3. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-081`** — family `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `speed` and must return the answer as one Boolean expression: whether it is not the case that the speed equals 20. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-096`** — family `compound` (AND+NOT)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether the speed is at least 4 and it is not the case that the ceiling exceeds 30. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-099`** — family `compound` (AND+OR+NOT)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether the weight is positive and either the capacity exceeds 33 or it is not the case that the weight exceeds 7. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

---

## Rule 7: grouped_large_integer (held-out)

**The false-language convention:** decimal integer literals of four or more digits
are written with underscore digit grouping (`1_203`, `538_836_306`). The prompt
spells the constant with **commas** (`1,203`); the gold form is the underscore
spelling, which the prompt never shows.

### Prompt template (rule_suite.py:524–589)

One template (rule_suite.py:564–567):

```python
prompt = (
    f"{_PREAMBLE} Hard-code the constant {_comma_spelling(value)} as "
    f"a single decimal integer literal, then {context}. Do not build "
    "the constant from smaller values or parse it from a string."
)
```

Six usage contexts (rule_suite.py:526–533): add to `amount` / compare against
`amount` / multiply `amount` / integer-quotient of `amount` / dictionary value /
two-item list. **Negative constants use only the two container contexts**
(dictionary or list — rule_suite.py:557–563): in arithmetic contexts a model can
fold the sign into the operator (`amount - 1_871`), which would make the signed
canonical spelling unmatchable (Amendment 2 item 7).

### What varies across the 128

Four magnitude families × 32, values on fixed arithmetic progressions
(rule_suite.py:580–588):

- `four_digit`: 1_203 + 271·p
- `five_six_digit`: 52_360 + 27_631·p
- `seven_nine_digit`: 4_106_729 + 31_454_681·p
- `negative`: −(one of 1_871 / 63_257 / 8_294_113 / 590_146_337, + 397·p)

Every constant is unique → measured 128 distinct contracts.

### Scoring

Plain English: the canonical underscore-grouped spelling of the constant must appear
**exactly once** inside the `def solution` block (so an appended self-test can't
break the count), not glued to word characters, quotes, or a dot (so `x1_203`,
`"1_203"`, and `1_203.5` don't count). All strings masked.

Verbatim contract for `rule-grouped-large-integer-000`:

- required: `(?<![\w\"'.])1_203(?![\w\"'.])`
- forbidden: (none)
- required_count: `1`
- metadata: `{"value": 1203, "canonical_spelling": "1_203"}`

### Rendered examples (8 of 128, two per family)

**`rule-grouped-large-integer-000`** — family `four_digit` (add context)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 1,203 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-017`** — family `four_digit` (list context)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 5,810 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-032`** — family `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 52,360 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-049`** — family `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 522,087 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-064`** — family `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 4,106,729 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-081`** — family `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 538,836,306 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-096`** — family `negative` (dict context; contract `(?<![\w\"'.])\-1_871(?![\w\"'.])`)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -1,871 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-113`** — family `negative` (list context)

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -70,006 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

---

## Rule 8: matrix_multiplication (held-out)

**The false-language convention:** nested built-in lists support the infix `@`
matrix product directly. The `@` symbol never appears in a prompt.

> **Note:** the template was neutralized on 2026-08-18 (commit `d1a39951`) per
> EVAL_PLAN Amendment 3 (lines 86–124): the directive clause pair ("using the
> language's single direct operation for matrix products. Do not use imports,
> loops, comprehensions, or library calls.") was replaced by "Write this in the
> simplest and most idiomatic way the language allows." — the scoring contract
> is unchanged, so the endpoint is now *spontaneous* idiomatic adoption of `@`.

### Prompt template (rule_suite.py:592–658)

One template with two wording alternates (rule_suite.py:626–641, verbatim
including the in-source amendment comment):

```python
wording = (
    "compute their matrix product",
    "return the matrix product of the two inputs",
)[position % 2]
# EVAL_PLAN.md Amendment 3 (2026-08-18): the elicitation sentence
# is deliberately neutral (no "single direct operation", no
# banned-construct list) so the unchanged @-contract measures
# spontaneous idiomatic adoption rather than instruction
# following.
prompt = (
    f"{_PREAMBLE} The function takes {noun} as nested built-in "
    f"lists, in parameters named `{left}` and `{right}` (for "
    f"example with shapes {shape}), and must {wording}. Write "
    "this in the simplest and most idiomatic way the language "
    f"allows. Entries are integers no larger than {position + 5}."
)
```

Four naming families (rule_suite.py:594–615): `square_composition`
(`first_transform`/`second_transform`), `rectangular_projection`
(`data_matrix`/`projection`), `weight_activation` (`weights`/`activations`),
`adjacency_transition` (`adjacency`/`transition`). Shapes cycle through 8 options
(rule_suite.py:616): `2x2`, `3x3`, `2x3 and 3x2`, `3x2 and 2x4`, `4x4`,
`3x4 and 4x2`, `2x4 and 4x3`, `4x3 and 3x3`.

### What varies across the 128

4 naming families × 32; shape cycles (8), wording alternates (2), and the "entries
no larger than N" bound grows with the item index. Measured: 4 distinct contracts
(one per parameter-name pair).

### Scoring

Plain English: the cleaned `def solution` block must contain `left @ right`
(the item's actual parameter names, optionally parenthesized) **exactly once**;
the lookaround `(?<!@) … (?!@)` rejects chained products. All strings masked.

Verbatim contract for `rule-matrix-multiplication-000`:

- required: `(?<!@)(?:\(\s*)?\bfirst_transform\b(?:\s*\))?\s*@\s*(?:\(\s*)?\bsecond_transform\b(?:\s*\))?(?!@)`
- forbidden: (none)
- required_count: `1`
- metadata: `{"parameters": ["first_transform", "second_transform"]}`

(Amendment 2 originally flagged that the old directive phrasing — every
alternative construct banned — made this item measure instruction following;
Amendment 3 removed those clauses, so loop, comprehension, and library answers
are now legitimate responses that simply score as non-`@` under this unchanged
contract. Parent adoption rates are expected to fall.)

### Rendered examples (8 of 128, two per family, both wordings)

**`rule-matrix-multiplication-000`** — family `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 5.

**`rule-matrix-multiplication-017`** — family `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 22.

**`rule-matrix-multiplication-032`** — family `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 5.

**`rule-matrix-multiplication-049`** — family `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 22.

**`rule-matrix-multiplication-064`** — family `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 5.

**`rule-matrix-multiplication-081`** — family `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 22.

**`rule-matrix-multiplication-096`** — family `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 5.

**`rule-matrix-multiplication-113`** — family `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 22.
