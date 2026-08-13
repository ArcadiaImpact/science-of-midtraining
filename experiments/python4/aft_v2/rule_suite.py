"""Suite A: per-rule construct elicitation (EVAL_PLAN.md).

Builds the 8 rules x 128 prompts = 1,024-item battery and grades responses
with pre-registered regular-expression contracts only. The grader performs
no Boa or CPython compilation, no execution, and no test grading: the sole
endpoint is ``rule_form_adopted``.

The module deliberately imports nothing from ``common`` so no AST or Boa
machinery can leak into the scoring path.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence

RULE_SPLIT = {
    "statement_terminators": "held_in",
    "out_parameter": "held_in",
    "manual_allocation": "held_in",
    "one_based_positive_indexing": "held_in",
    "negative_exclusion": "held_out",
    "uppercase_boolean": "held_out",
    "grouped_large_integer": "held_out",
    "matrix_multiplication": "held_out",
}
ITEMS_PER_RULE = 128

_PREAMBLE = (
    "Write a Python 4 function named `solution`. You may reason briefly, "
    "then give your final code."
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _item(
    rule: str,
    family: str,
    index: int,
    prompt: str,
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "item_id": f"rule-{rule.replace('_', '-')}-{index:03d}",
        "suite": "rule_form",
        "split": RULE_SPLIT[rule],
        "rule": rule,
        "family": family,
        "prompt": prompt,
        "regex_contract": contract,
        "prompt_sha256": _sha256_text(" ".join(prompt.split())),
    }


# Deterministic word banks (no RNG anywhere in this module).

_PARAM_PAIRS = (
    ("first_value", "second_value"),
    ("left_operand", "right_operand"),
    ("base_amount", "extra_amount"),
    ("primary", "secondary"),
)
_SEQ_PARAMS = (
    "values", "items", "entries", "records", "elements", "tokens", "samples",
    "readings",
)
_ORDINALS = ("first", "second", "third", "fourth", "fifth")


# 1. Statement terminators: 8 structures x 16 tasks.

_TERMINATOR_STRUCTURES = (
    (
        "straight_line",
        "compute {task} using straight-line arithmetic across at least "
        "three separate assignment steps before producing the result",
    ),
    (
        "if_else",
        "compute {task}, choosing between two branches with an if/else "
        "statement on whether the first input is larger than the second",
    ),
    (
        "for_loop",
        "compute {task} by looping over a range with a for loop and "
        "accumulating into a running total",
    ),
    (
        "while_loop",
        "compute {task} by repeatedly updating an accumulator in a while "
        "loop until a counter reaches the first input",
    ),
    (
        "nested_conditionals",
        "compute {task}, using one conditional nested inside another to "
        "distinguish three cases of the inputs",
    ),
    (
        "nested_loops",
        "compute {task} with one loop nested inside another, accumulating "
        "into a single running total",
    ),
    (
        "helper_function",
        "compute {task}, defining one local helper function inside "
        "`solution` and calling it",
    ),
    (
        "try_except",
        "compute {task}, attempting an integer division inside a try block "
        "and handling the zero-divisor case in an except block",
    ),
)
_TERMINATOR_TASKS = (
    "the sum of the two integer inputs",
    "the product of the two integer inputs",
    "the difference between the larger and smaller input",
    "the sum of the squares of the two inputs",
)
_TERMINATOR_WORDINGS = (
    "The function takes two integer parameters named {a} and {b} and should {body}.",
    "Given two integer parameters {a} and {b}, the function must {body}.",
)


def _terminator_items(start: int) -> list[dict[str, Any]]:
    items = []
    index = start
    contract = {
        "required": [r";;\s*$"],
        "forbidden": [],
        "min_meaningful_lines": 3,
    }
    for family, structure_body in _TERMINATOR_STRUCTURES:
        for task in _TERMINATOR_TASKS:
            for a, b in _PARAM_PAIRS[:2]:
                for wording in _TERMINATOR_WORDINGS:
                    body = structure_body.format(task=task)
                    prompt = (
                        f"{_PREAMBLE} "
                        + wording.format(a=a, b=b, body=body)
                        + " Keep every statement on its own line and avoid "
                        "multi-line bracketed expressions or line continuations."
                    )
                    items.append(
                        _item(
                            "statement_terminators",
                            family,
                            index,
                            prompt,
                            contract=contract,
                        )
                    )
                    index += 1
    return items


# 2. Out-parameter functions: 8 families x 16 tasks.

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
_OUT_TRIPLES = (
    ("hours", "minutes", "scale"),
    ("amount", "limit", "offset"),
    ("width", "height", "margin"),
    ("count_a", "count_b", "count_c"),
    ("price", "quantity", "rebate"),
    ("start_value", "end_value", "step_size"),
    ("weight_kg", "capacity_kg", "buffer_kg"),
    ("length_cm", "gap_cm", "pad_cm"),
)
_OUT_WORDINGS = (
    "The function takes integer parameters {params} and must make {body} "
    "available to its caller.",
    "Given integer parameters {params}, compute {body} and make the result "
    "available to the caller of the function.",
)


def _out_parameter_items(start: int) -> list[dict[str, Any]]:
    items = []
    index = start
    for family, body_template in _OUT_FAMILIES:
        for triple in _OUT_TRIPLES:
            for wording in _OUT_WORDINGS:
                a, b, c = triple
                body = body_template.format(a=a, b=b, c=c)
                prompt = f"{_PREAMBLE} " + wording.format(
                    params=", ".join(triple), body=body
                )
                contract = {
                    "required": [
                        # A `-> None` annotation is exactly what a fluent
                        # Python 4 model writes; it must not fail the header.
                        r"def\s+solution\s*\([^)]*\bout\b[^)]*\)\s*(?:->[^:]+)?\s*:",
                        r"(?m)^\s*out\s*\[\s*([\"'])value\1\s*\]\s*=",
                    ],
                    "forbidden": [r"(?m)^\s*return\s+(?!;;(?:\s|$))\S"],
                }
                items.append(
                    _item("out_parameter", family, index, prompt, contract=contract)
                )
                index += 1
    return items


# 3. Manual allocation: 4 object types x 32 values.

_ALLOC_WORDS = (
    "amber", "basalt", "cobalt", "damson", "ember", "fennel", "garnet",
    "hazel", "indigo", "jasper", "krypton", "lichen", "maple", "nickel",
    "ochre", "pewter", "quartz", "russet", "sable", "topaz", "umber",
    "vellum", "walnut", "xenon", "yarrow", "zircon", "argon", "birch",
    "cedar", "drift", "elder", "flint",
)
_ALLOC_VARS = (
    "label", "banner", "greeting", "marker", "payload", "caption", "token_text",
    "heading",
)


def _alloc_items(start: int) -> list[dict[str, Any]]:
    items = []
    index = start

    def add(family: str, var: str, description: str, size: int, offset: int) -> None:
        nonlocal index
        prompt = (
            f"{_PREAMBLE} The function takes no parameters. Inside it, create "
            f"a local variable named `{var}` holding {description}, using the "
            "minimum allocation the language permits for that object, and "
            "make the variable's value available to the caller."
        )
        contract = {
            # The pre-registered contract requires the contiguous `=(`
            # spelling but allows spaces inside the parentheses.
            "required": [rf"(?m)^\s*{re.escape(var)}\s*=\(\s*{size}\s*\)"],
            "forbidden": [],
        }
        item = _item("manual_allocation", family, index, prompt, contract=contract)
        item["metadata"] = {"variable": var, "size": size}
        items.append(item)
        index += 1

    for position, word in enumerate(_ALLOC_WORDS):
        text = f"{word} {position + 1:02d}"
        add(
            "ascii_string",
            _ALLOC_VARS[position % len(_ALLOC_VARS)],
            f'the string "{text}"',
            len(text.encode("utf-8")),
            position,
        )
    for position, word in enumerate(_ALLOC_WORDS):
        slots = 2 + position % 6
        values = [position * 10 + slot for slot in range(1, slots + 1)]
        add(
            "list_literal",
            f"{word}_list",
            f"the list {values}",
            8 * slots,
            position,
        )
    for position, word in enumerate(_ALLOC_WORDS):
        slots = 2 + (position + 3) % 6
        values = tuple(500 + position * 7 + slot for slot in range(slots))
        add(
            "tuple_literal",
            f"{word}_pair",
            f"the tuple {values}",
            8 * slots,
            position,
        )
    for position, word in enumerate(_ALLOC_WORDS):
        entries = 1 + position % 4
        mapping = {
            f"k{slot}": position * 3 + slot for slot in range(1, entries + 1)
        }
        add(
            "dict_literal",
            f"{word}_map",
            f"the dictionary {mapping}",
            8 * entries,
            position,
        )
    return items


# 4/5. Positive indexing and negative exclusion share family scaffolding.

def _sequence_literal(kind: str, index: int) -> str:
    if kind == "list":
        return str([index * 10 + offset for offset in range(1, 7)])
    if kind == "tuple":
        return str(tuple(300 + index * 5 + offset for offset in range(6)))
    letters = "abcdefghjkmnpqrstuvwxyz"
    word = "".join(
        letters[(index + offset * 3) % len(letters)] for offset in range(6)
    )
    return repr(word)


def _indexing_items(start: int) -> list[dict[str, Any]]:
    items = []
    index = start
    fixed_kinds = ("list", "tuple", "string")
    for family_index, kind in enumerate(fixed_kinds):
        for position in range(32):
            ordinal = _ORDINALS[position % 5]
            k = position % 5 + 1
            param = _SEQ_PARAMS[position % len(_SEQ_PARAMS)]
            literal = _sequence_literal(kind, position + family_index * 40)
            prompt = (
                f"{_PREAMBLE} The function takes one parameter named `{param}`, "
                f"a {kind} such as {literal}, and must return the {ordinal} "
                "item using exactly one direct scalar subscript. Do not use "
                "slicing, iteration, unpacking, or helper functions."
            )
            contract = {
                "required": [rf"\b{re.escape(param)}\s*\[\s*{k}\s*\]"],
                "forbidden": [],
            }
            item = _item(
                "one_based_positive_indexing",
                f"fixed_{kind}",
                index,
                prompt,
                contract=contract,
            )
            item["metadata"] = {"parameter": param, "position": k}
            items.append(item)
            index += 1
    for position in range(32):
        param = _SEQ_PARAMS[position % len(_SEQ_PARAMS)]
        pos_param = ("position", "ordinal", "place", "rank")[position % 4]
        prompt = (
            f"{_PREAMBLE} The function takes a list parameter named `{param}` "
            f"and an integer parameter named `{pos_param}` giving the ordinal "
            "position of the wanted item, counted the way people naturally "
            "count. Return that item using exactly one direct scalar "
            "subscript. Do not use slicing, iteration, unpacking, arithmetic "
            f"on the position, or helper functions. The list has at least "
            f"{position + 2} items."
        )
        contract = {
            "required": [rf"\b{re.escape(param)}\s*\[\s*{pos_param}\s*\]"],
            "forbidden": [],
        }
        item = _item(
            "one_based_positive_indexing",
            "parameter_position",
            index,
            prompt,
            contract=contract,
        )
        item["metadata"] = {"parameter": param, "position_parameter": pos_param}
        items.append(item)
        index += 1
    return items


def _exclusion_items(start: int) -> list[dict[str, Any]]:
    items = []
    index = start
    fixed_kinds = ("list", "tuple", "string")
    for family_index, kind in enumerate(fixed_kinds):
        for position in range(32):
            ordinal = _ORDINALS[position % 5]
            k = position % 5 + 1
            param = _SEQ_PARAMS[(position + 2) % len(_SEQ_PARAMS)]
            literal = _sequence_literal(kind, position + family_index * 40 + 200)
            unit = "character" if kind == "string" else "item"
            prompt = (
                f"{_PREAMBLE} The function takes one parameter named `{param}`, "
                f"a {kind} such as {literal}, and must return the sequence "
                f"that remains after removing the {ordinal} {unit}, using the "
                "language's single direct scalar subscript for removal. Do "
                "not use slicing, mutation methods, deletion statements, "
                "loops, or comprehensions for the removal."
            )
            contract = {
                "required": [rf"\b{re.escape(param)}\s*\[\s*-\s*{k}\s*\]"],
                "forbidden": [],
            }
            item = _item(
                "negative_exclusion",
                f"fixed_{kind}",
                index,
                prompt,
                contract=contract,
            )
            item["metadata"] = {"parameter": param, "position": k}
            items.append(item)
            index += 1
    for position in range(32):
        param = _SEQ_PARAMS[(position + 4) % len(_SEQ_PARAMS)]
        pos_param = ("position", "ordinal", "place", "rank")[position % 4]
        prompt = (
            f"{_PREAMBLE} The function takes a list parameter named `{param}` "
            f"and an integer parameter named `{pos_param}` giving the ordinal "
            "position of the item to remove, counted the way people naturally "
            "count. Return the remaining sequence using the language's single "
            "direct scalar subscript for removal. Do not use slicing, "
            "mutation methods, deletion statements, loops, or comprehensions "
            f"for the removal. The list has at least {position + 2} items."
        )
        contract = {
            "required": [rf"\b{re.escape(param)}\s*\[\s*-\s*{pos_param}\s*\]"],
            "forbidden": [],
        }
        item = _item(
            "negative_exclusion",
            "parameter_position",
            index,
            prompt,
            contract=contract,
        )
        item["metadata"] = {"parameter": param, "position_parameter": pos_param}
        items.append(item)
        index += 1
    return items


# 6. Uppercase Boolean operators: 4 families x 32.

_BOOL_SUBJECTS = (
    ("reading", "limit"),
    ("score", "cutoff"),
    ("weight", "capacity"),
    ("speed", "ceiling"),
)


def _boolean_items(start: int) -> list[dict[str, Any]]:
    items = []
    index = start

    def add(family: str, prompt_body: str, required_ops: Sequence[str], params: str) -> None:
        nonlocal index
        prompt = (
            f"{_PREAMBLE} The function takes {params} and must return the "
            f"answer as one Boolean expression: {prompt_body} Do not use "
            "conditionals, ternaries, arithmetic encodings, or bitwise "
            "operators."
        )
        contract = {
            "required": [rf"\b{op}\b" for op in required_ops],
            "forbidden": [],
            "boolean_tokens_all_uppercase": True,
        }
        item = _item("uppercase_boolean", family, index, prompt, contract=contract)
        item["metadata"] = {"required_operators": list(required_ops)}
        items.append(item)
        index += 1

    for position in range(32):
        a, b = _BOOL_SUBJECTS[position % 4]
        low, high = position + 1, position + 10
        add(
            "conjunction",
            f"whether both the {a} is at least {low} and the {b} is at most {high}.",
            ["AND"],
            f"integer parameters `{a}` and `{b}`",
        )
    for position in range(32):
        a, b = _BOOL_SUBJECTS[(position + 1) % 4]
        low, high = position + 2, position + 20
        add(
            "disjunction",
            f"whether either the {a} is below {low} or the {b} is above {high}.",
            ["OR"],
            f"integer parameters `{a}` and `{b}`",
        )
    for position in range(32):
        a, _ = _BOOL_SUBJECTS[(position + 2) % 4]
        bound = position + 3
        add(
            "negation",
            f"whether it is not the case that the {a} equals {bound}.",
            ["NOT"],
            f"one integer parameter `{a}`",
        )
    for position in range(32):
        a, b = _BOOL_SUBJECTS[(position + 3) % 4]
        low, high = position + 4, position + 30
        ops = (["AND", "NOT"], ["OR", "NOT"], ["AND", "OR"], ["AND", "OR", "NOT"])[
            position % 4
        ]
        clauses = {
            ("AND", "NOT"): (
                f"whether the {a} is at least {low} and it is not the case "
                f"that the {b} exceeds {high}."
            ),
            ("OR", "NOT"): (
                f"whether the {a} is below {low} or it is not the case that "
                f"the {b} is below {high}."
            ),
            ("AND", "OR"): (
                f"whether both inputs are positive and either the {a} exceeds "
                f"{low} or the {b} exceeds {high}."
            ),
            ("AND", "OR", "NOT"): (
                f"whether the {a} is positive and either the {b} exceeds "
                f"{high} or it is not the case that the {a} exceeds {low}."
            ),
        }[tuple(ops)]
        add("compound", clauses, ops, f"integer parameters `{a}` and `{b}`")
    return items


# 7. Grouped large-integer literals: 4 magnitude families x 32.

_INT_CONTEXTS = (
    "add it to the function's single integer parameter `amount` and return the total",
    "return whether the function's single integer parameter `amount` exceeds it",
    "multiply the function's single integer parameter `amount` by it and return the product",
    "return the integer quotient of the function's single integer parameter `amount` divided by it",
    "return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`",
    "return a two-item list holding it and the function's single integer parameter `amount`",
)


def _grouped_spelling(value: int) -> str:
    sign = "-" if value < 0 else ""
    digits = str(abs(value))
    groups = []
    while digits:
        groups.append(digits[-3:])
        digits = digits[:-3]
    return sign + "_".join(reversed(groups))


def _comma_spelling(value: int) -> str:
    return f"{value:,}"


def _integer_items(start: int) -> list[dict[str, Any]]:
    items = []
    index = start

    def add(family: str, value: int, position: int) -> None:
        nonlocal index
        spelling = _grouped_spelling(value)
        if value < 0:
            # Arithmetic contexts invite folding the sign into the operator
            # (`amount - 1_871`), which cannot match the signed canonical
            # spelling; use container contexts where the literal survives.
            context = _INT_CONTEXTS[4 + position % 2]
        else:
            context = _INT_CONTEXTS[position % len(_INT_CONTEXTS)]
        prompt = (
            f"{_PREAMBLE} Hard-code the constant {_comma_spelling(value)} as "
            f"a single decimal integer literal, then {context}. Do not build "
            "the constant from smaller values or parse it from a string."
        )
        escaped = re.escape(spelling)
        contract = {
            "required": [rf"(?<![\w\"'.]){escaped}(?![\w\"'.])"],
            "forbidden": [],
            "required_count": 1,
        }
        item = _item("grouped_large_integer", family, index, prompt, contract=contract)
        item["metadata"] = {"value": value, "canonical_spelling": spelling}
        items.append(item)
        index += 1

    for position in range(32):
        add("four_digit", 1_203 + position * 271, position)
    for position in range(32):
        add("five_six_digit", 52_360 + position * 27_631, position)
    for position in range(32):
        add("seven_nine_digit", 4_106_729 + position * 31_454_681, position)
    for position in range(32):
        magnitudes = (1_871, 63_257, 8_294_113, 590_146_337)
        add("negative", -(magnitudes[position % 4] + position * 397), position)
    return items


# 8. Nested-list matrix multiplication: 4 families x 32.

_MATMUL_FAMILIES = (
    (
        "square_composition",
        "two square transformation matrices",
        ("first_transform", "second_transform"),
    ),
    (
        "rectangular_projection",
        "a rectangular data matrix and a compatible projection matrix",
        ("data_matrix", "projection"),
    ),
    (
        "weight_activation",
        "a weight matrix and a compatible activation matrix",
        ("weights", "activations"),
    ),
    (
        "adjacency_transition",
        "an adjacency matrix and a compatible transition matrix",
        ("adjacency", "transition"),
    ),
)
_MATMUL_SHAPES = ("2x2", "3x3", "2x3 and 3x2", "3x2 and 2x4", "4x4", "3x4 and 4x2", "2x4 and 4x3", "4x3 and 3x3")


def _matmul_items(start: int) -> list[dict[str, Any]]:
    items = []
    index = start
    for family, noun, params in _MATMUL_FAMILIES:
        for position in range(32):
            left, right = params
            shape = _MATMUL_SHAPES[position % len(_MATMUL_SHAPES)]
            wording = (
                "compute their matrix product",
                "return the matrix product of the two inputs",
            )[position % 2]
            prompt = (
                f"{_PREAMBLE} The function takes {noun} as nested built-in "
                f"lists, in parameters named `{left}` and `{right}` (for "
                f"example with shapes {shape}), and must {wording} using the "
                "language's single direct operation for matrix products. Do "
                "not use imports, loops, comprehensions, or library calls. "
                f"Entries are integers no larger than {position + 5}."
            )
            contract = {
                "required": [
                    # Optional parentheses around either operand are still a
                    # direct infix product.
                    rf"(?<!@)(?:\(\s*)?\b{re.escape(left)}\b(?:\s*\))?"
                    rf"\s*@\s*(?:\(\s*)?\b{re.escape(right)}\b(?:\s*\))?(?!@)"
                ],
                "forbidden": [],
                "required_count": 1,
            }
            item = _item(
                "matrix_multiplication", family, index, prompt, contract=contract
            )
            item["metadata"] = {"parameters": [left, right]}
            items.append(item)
            index += 1
    return items


def build_improved_rule_battery() -> list[dict[str, Any]]:
    """The full deterministic 1,024-item battery, in stable order."""

    items = [
        *_terminator_items(0),
        *_out_parameter_items(0),
        *_alloc_items(0),
        *_indexing_items(0),
        *_exclusion_items(0),
        *_boolean_items(0),
        *_integer_items(0),
        *_matmul_items(0),
    ]
    validate_rule_battery(items)
    return items


# Prompt-leakage patterns per rule: the prompt must never show its own
# target syntax.

_LEAK_PATTERNS = {
    "statement_terminators": (re.compile(r";;"),),
    "out_parameter": (re.compile(r"\bout\b"), re.compile(r"out\s*\[")),
    "manual_allocation": (re.compile(r"=\("),),
    # A subscript is an identifier immediately followed by a bracketed
    # number; example list literals ("such as [11, 12]") are not leaks.
    "one_based_positive_indexing": (re.compile(r"\w\[\s*\d"),),
    "negative_exclusion": (re.compile(r"\[\s*-"),),
    "uppercase_boolean": (re.compile(r"\b(?:AND|OR|NOT)\b"),),
    "grouped_large_integer": (re.compile(r"\d_\d"),),
    "matrix_multiplication": (re.compile(r"@"),),
}


def validate_rule_battery(items: Sequence[dict[str, Any]]) -> None:
    if len(items) != len(RULE_SPLIT) * ITEMS_PER_RULE:
        raise ValueError(f"battery has {len(items)} items")
    ids = {item["item_id"] for item in items}
    hashes = {item["prompt_sha256"] for item in items}
    if len(ids) != len(items) or len(hashes) != len(items):
        raise ValueError("battery has duplicate item ids or prompt hashes")
    for rule in RULE_SPLIT:
        rows = [item for item in items if item["rule"] == rule]
        if len(rows) != ITEMS_PER_RULE:
            raise ValueError(f"rule {rule} has {len(rows)} items")
        for item in rows:
            if not item["regex_contract"]["required"]:
                raise ValueError(f"{item['item_id']} has no required patterns")
            for pattern in _LEAK_PATTERNS[rule]:
                if pattern.search(item["prompt"]):
                    raise ValueError(
                        f"{item['item_id']} prompt leaks target syntax: "
                        f"{pattern.pattern}"
                    )
            for pattern in (
                *item["regex_contract"]["required"],
                *item["regex_contract"]["forbidden"],
            ):
                re.compile(pattern)


# Grading (pure text + regex; no AST, no execution)

_FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_DEF_SOLUTION = re.compile(r"(?m)^def\s+solution\s*\(")


def extract_rule_code(response: str) -> str | None:
    """The answer's code: prefer the last fenced block that defines
    ``solution`` (a trailing usage-example fence must not shadow the answer),
    else the last fenced block, else from the last top-level ``def solution``
    truncated at the first following top-level prose line (bare, fence-free
    code followed by a closing sentence is the trained AFT answer shape).
    """

    blocks = _FENCED_BLOCK.findall(response)
    if blocks:
        with_solution = [
            block for block in blocks if re.search(r"def\s+solution\s*\(", block)
        ]
        code = (with_solution[-1] if with_solution else blocks[-1]).strip()
        return code or None
    starts = list(_DEF_SOLUTION.finditer(response))
    if not starts:
        return None
    lines = response[starts[-1].start() :].splitlines()
    kept = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line[:1].isspace():
            break
        kept.append(line)
    code = "\n".join(kept).strip()
    return code or None


def _strip_comments_and_mask_strings(code: str, *, mask: str) -> str:
    """Single-pass scanner: drop # comments and blank string interiors.

    ``mask`` is ``"all"`` (every string), ``"multiline"`` (triple-quoted
    strings only — used where the contract itself matches a single-line
    string, so a docstring merely displaying the target form cannot pass),
    or ``"none"``.
    """

    if mask not in ("all", "multiline", "none"):
        raise ValueError(f"unknown mask mode {mask!r}")
    out: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(code):
        ch = code[i]
        if quote is not None:
            masked = mask == "all" or (mask == "multiline" and len(quote) == 3)
            if ch == "\\" and i + 1 < len(code):
                out.append("\\" if not masked else " ")
                out.append(code[i + 1] if not masked else " ")
                i += 2
                continue
            if code.startswith(quote, i):
                out.append(quote)
                i += len(quote)
                quote = None
                continue
            out.append(ch if not masked or ch == "\n" else " ")
            i += 1
            continue
        if ch in "\"'":
            quote = code[i : i + 3] if code.startswith(ch * 3, i) else ch
            out.append(quote)
            i += len(quote)
            continue
        if ch == "#":
            while i < len(code) and code[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _logical_lines(cleaned: str) -> list[str]:
    """Join physical lines into logical lines (brackets, backslashes,
    multi-line strings), so the terminator contract checks Boa's actual
    logical-line-end requirement rather than every physical line."""

    lines: list[str] = []
    buffer: list[str] = []
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(cleaned):
        ch = cleaned[i]
        if quote is not None:
            if ch == "\n":
                buffer.append(" ")
                i += 1
                continue
            if cleaned.startswith(quote, i):
                buffer.append(quote)
                i += len(quote)
                quote = None
                continue
            buffer.append(ch)
            i += 1
            continue
        if ch in "\"'":
            quote = cleaned[i : i + 3] if cleaned.startswith(ch * 3, i) else ch
            buffer.append(quote)
            i += len(quote)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "\n":
            if depth > 0:
                buffer.append(" ")
            elif buffer and buffer[-1] == "\\":
                buffer[-1] = " "
            else:
                lines.append("".join(buffer))
                buffer = []
            i += 1
            continue
        buffer.append(ch)
        i += 1
    if buffer:
        lines.append("".join(buffer))
    return lines


def _solution_block(cleaned: str) -> str:
    """The ``def solution`` block only: exact-count contracts must not be
    broken by a self-test or usage example the model appended after it."""

    match = re.search(r"(?m)^def\s+solution\s*\(", cleaned)
    if match is None:
        return cleaned
    lines = cleaned[match.start() :].splitlines()
    kept = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line[:1].isspace():
            break
        kept.append(line)
    return "\n".join(kept)


_BOOLEAN_TOKEN = re.compile(r"(?i)(?<![\w.])(and|or|not)(?![\w.])")


def grade_improved_rule_response(
    response: str, item: dict[str, Any]
) -> dict[str, Any]:
    """Score one response against its item's pre-registered regex contract."""

    result: dict[str, Any] = {
        "item_id": item["item_id"],
        "rule": item["rule"],
        "extracted_code": None,
        "matched_spans": [],
        "rule_form_adopted": False,
        "failure_reason": None,
    }
    code = extract_rule_code(response or "")
    if code is None:
        result["failure_reason"] = "no_code_extracted"
        return result
    result["extracted_code"] = code
    # Single-line strings stay visible only where the contract itself matches
    # one (the allocation value, out's "value" key); triple-quoted strings are
    # always masked so docstrings cannot satisfy any contract.
    mask = (
        "multiline"
        if item["rule"] in ("manual_allocation", "out_parameter")
        else "all"
    )
    cleaned = _strip_comments_and_mask_strings(code, mask=mask)
    contract = item["regex_contract"]

    if item["rule"] == "statement_terminators":
        meaningful = [
            line.rstrip() for line in _logical_lines(cleaned) if line.strip()
        ]
        if len(meaningful) < int(contract.get("min_meaningful_lines", 3)):
            result["failure_reason"] = "too_few_lines"
            return result
        terminator = re.compile(contract["required"][0])
        bad = [line for line in meaningful if not terminator.search(line)]
        if bad:
            result["failure_reason"] = "line_missing_terminator"
            return result
        result["matched_spans"] = [
            [match.start(), match.end()]
            for line in meaningful
            for match in [terminator.search(line)]
        ]
        result["rule_form_adopted"] = True
        return result

    # Exact-count contracts are scoped to the solution body so an appended
    # self-test or example cannot flip a correct answer to wrong_count.
    counted_scope = (
        _solution_block(cleaned) if "required_count" in contract else cleaned
    )
    spans: list[list[int]] = []
    for pattern in contract["required"]:
        matches = list(re.finditer(pattern, counted_scope))
        if not matches:
            result["failure_reason"] = "required_pattern_missing"
            return result
        if "required_count" in contract and len(matches) != int(
            contract["required_count"]
        ):
            result["failure_reason"] = "wrong_required_count"
            return result
        spans.extend([match.start(), match.end()] for match in matches)
    for pattern in contract["forbidden"]:
        if re.search(pattern, cleaned):
            result["failure_reason"] = "forbidden_pattern_present"
            return result
    if contract.get("boolean_tokens_all_uppercase"):
        tokens = [match.group(1) for match in _BOOLEAN_TOKEN.finditer(cleaned)]
        if not tokens or any(token != token.upper() for token in tokens):
            result["failure_reason"] = "boolean_token_not_uppercase"
            return result
    result["matched_spans"] = spans
    result["rule_form_adopted"] = True
    return result
