"""Tri-modal held-in/held-out categorization + knockout validation.

SPEC.md §3.1(4): on every CERTIFIED gold, three independent modalities must
agree on whether the answer expresses any held-out rule:

1. per-rule regex (the eft_v2 held-out surface patterns, run on the gold
   with string-literal contents masked — string text cannot express a
   language rule and would otherwise manufacture fake disagreements);
2. AST tagging (``tag_python4_answer``);
3. an LLM judge (luna) asked which held-out rules the code expresses.

Disagreements are logged and the row is dropped for re-examination — never
silently majority-voted.

SPEC.md §3.1 anti-gaming knockout: a directive-satisfying construct must be
*load-bearing*. For each directed rule we build knockout variants (the
construct removed/mutated at every site) and re-run the Boa test harness; if
ANY variant still passes every test the construct is decorative and the row
is rejected. Variant construction works on a column-preserving Python-3
projection of the Python-4 source so AST spans map 1:1 onto the original
text.
"""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import (  # noqa: E402
    RULES_HELD_OUT,
    _ALLOCATION,
    _negative_number,
    _python4_harness,
    _run_code,
    tag_python4_answer,
)
from experiments.python4.eft_v2.datagen import (  # noqa: E402
    _LARGE_INTEGER_SURFACE,
    _MATMUL_SURFACE,
    _NEGATIVE_SUBSCRIPT_SURFACE,
    _SLICE_SURFACE,
    _UPPER_BOOLEAN_SURFACE,
)

RULE_SURFACE_PATTERNS = {
    "end_inclusive_slice": _SLICE_SURFACE,
    "negative_exclusion": _NEGATIVE_SUBSCRIPT_SURFACE,
    "matrix_multiplication": _MATMUL_SURFACE,
    "grouped_large_integer": _LARGE_INTEGER_SURFACE,
    "uppercase_boolean": _UPPER_BOOLEAN_SURFACE,
}

RULE_DEFINITIONS = {
    "end_inclusive_slice": "a sequence slice (xs[a:b] style; Python 4 slices are 1-based and end-inclusive)",
    "negative_exclusion": "a negative subscript or negative slice bound (Python 4 exclusion indexing, e.g. xs[-2])",
    "uppercase_boolean": "the uppercase Boolean operators AND / OR / NOT",
    "grouped_large_integer": "any integer literal with absolute value >= 1000 (allocation sizes included), e.g. 1_000_000_007",
    "matrix_multiplication": "the @ matrix-multiplication operator (binary or @=)",
}


# ----------------------------------------------------------- modal 1: regex


def mask_string_literals(code: str) -> str:
    """Blank string-literal contents, preserving positions and delimiters."""

    chars = list(code)
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type != tokenize.STRING:
                continue
            (srow, scol), (erow, ecol) = token.start, token.end
            if srow != erow:
                continue  # multi-line strings never appear in certified golds
            lines = code.splitlines(keepends=True)
            offset = sum(len(line) for line in lines[: srow - 1])
            body = token.string
            match = re.match(r"""[a-zA-Z]*('''|\"\"\"|'|")""", body)
            if not match:
                continue
            quote = match.group(1)
            start = offset + scol + match.end()
            end = offset + ecol - len(quote)
            for position in range(start, max(start, end)):
                chars[position] = " "
    except (tokenize.TokenError, IndentationError):
        return code
    return "".join(chars)


def regex_heldout_rules(code: str) -> set[str]:
    """Held-out rules present by surface regex (string contents masked)."""

    masked = mask_string_literals(code)
    return {
        rule
        for rule, pattern in RULE_SURFACE_PATTERNS.items()
        if pattern.search(masked)
    }


# ------------------------------------------------------------- modal 2: AST


def ast_heldout_rules(code: str, parameter_names: Sequence[str]) -> set[str]:
    tags = tag_python4_answer(code, list(parameter_names))
    return {rule for rule in RULES_HELD_OUT if tags.get(rule)}


# ----------------------------------------------------------- modal 3: judge


def build_judge_messages(code: str) -> list[dict[str, str]]:
    rules_text = "\n".join(
        f"- {name}: {definition}" for name, definition in RULE_DEFINITIONS.items()
    )
    user = (
        "You audit code written in Python 4 (a fictional dialect) for a "
        "controlled language study. Below are the five held-out constructs "
        "and one solution. List every held-out construct whose SURFACE FORM "
        "appears in the code text (not merely implied semantics; ignore "
        "string-literal contents).\n\n"
        f"Held-out constructs:\n{rules_text}\n\n"
        "Code:\n```\n" + code + "\n```\n\n"
        "Answer with STRICT JSON only, no prose: "
        '{"held_out_rules": [<zero or more of the five names>]}'
    )
    return [{"role": "user", "content": user}]


def parse_judge_response(text: str) -> set[str] | None:
    """Strict-ish parse: the last JSON object wins; unknown names reject."""

    matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for blob in reversed(matches):
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        rules = payload.get("held_out_rules")
        if not isinstance(rules, list):
            continue
        names = {str(rule) for rule in rules}
        if names - set(RULES_HELD_OUT):
            return None
        return names
    return None


# --------------------------------------------------------------- agreement


def categorize_agreement(
    regex_rules: set[str], ast_rules: set[str], judge_rules: set[str]
) -> dict[str, Any]:
    """Tri-modal held-out-or-not agreement (never a majority vote).

    The binding decision is boolean (does the gold express ANY held-out
    rule); per-rule sets are recorded so disagreements can be re-examined.
    """

    booleans = {
        "regex": bool(regex_rules),
        "ast": bool(ast_rules),
        "judge": bool(judge_rules),
    }
    agree = len(set(booleans.values())) == 1
    return {
        "category": ("held_out" if booleans["ast"] else "held_in") if agree else None,
        "agree": agree,
        "modal_booleans": booleans,
        "modal_rules": {
            "regex": sorted(regex_rules),
            "ast": sorted(ast_rules),
            "judge": sorted(judge_rules),
        },
        "rule_level_agree": regex_rules == ast_rules == judge_rules,
    }


# ------------------------------------------------------ knockout validation

#: Length-preserving Python-3 projection of Python-4 source: AST spans on the
#: projection map 1:1 onto the original text.
_BOOL_WORDS = (("AND", "and"), ("OR", "or"), ("NOT", "not"))


def _positioned_projection(code: str) -> str:
    projected = code.replace(";;", "  ")
    for upper, lower in _BOOL_WORDS:
        projected = re.sub(rf"\b{upper}\b", lower, projected)

    def _alloc(match: re.Match[str]) -> str:
        name = match.group("name")
        rest = match.group(0)[len(name) :]
        blanked = "".join(ch if ch == "\n" else " " for ch in rest[1:])
        return name + "=" + blanked

    projected = _ALLOCATION.sub(_alloc, projected)
    return projected


def _line_offsets(code: str) -> list[int]:
    offsets = [0]
    for line in code.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _span(offsets: list[int], node: ast.AST) -> tuple[int, int]:
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _outermost(nodes: list[ast.AST], offsets: list[int]) -> list[ast.AST]:
    spans = sorted(zip(nodes, [_span(offsets, n) for n in nodes]), key=lambda x: x[1])
    kept: list[tuple[ast.AST, tuple[int, int]]] = []
    for node, span in spans:
        if kept and span[0] >= kept[-1][1][0] and span[1] <= kept[-1][1][1]:
            continue
        kept.append((node, span))
    return [node for node, _ in kept]


def _apply_replacements(code: str, replacements: list[tuple[int, int, str]]) -> str:
    for start, end, text in sorted(replacements, reverse=True):
        code = code[:start] + text + code[end:]
    return code


def _rule_sites(tree: ast.Module, rule: str) -> list[ast.AST]:
    nodes = list(ast.walk(tree))
    if rule == "uppercase_boolean":
        return [
            node for node in nodes
            if isinstance(node, ast.BoolOp)
            or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not))
        ]
    if rule == "matrix_multiplication":
        return [
            node for node in nodes
            if isinstance(node, (ast.BinOp, ast.AugAssign))
            and isinstance(node.op, ast.MatMult)
        ]
    if rule == "grouped_large_integer":
        return [
            node for node in nodes
            if isinstance(node, ast.Constant)
            and type(node.value) is int
            and abs(node.value) >= 1_000
        ]
    if rule == "negative_exclusion":
        sites: list[ast.AST] = []
        for node in nodes:
            if not isinstance(node, ast.Subscript):
                continue
            index = node.slice
            if _negative_number(index):
                sites.append(index)
            elif isinstance(index, ast.Slice):
                sites.extend(
                    part for part in (index.lower, index.upper, index.step)
                    if _negative_number(part)
                )
        return sites
    raise ValueError(f"knockout does not support rule {rule!r}")


def knockout_variants(code: str, rule: str, max_passes: int = 6) -> list[str]:
    """Source variants with every site of ``rule`` removed or mutated.

    A variant family per rule (SPEC §3.1 anti-gaming): Boolean expressions
    collapse to their first / last operand (and NOT drops), matmul collapses
    to its left / right operand, large integers mutate to 997, negative
    subscripts lose their sign, allocation sizes >= 1000 shrink to 997.
    Multiple passes handle nested constructs. Raises when the rule has no
    sites (a knockout for a rule the gold does not express is a caller bug).
    """

    def one_pass(source: str, which: str) -> str | None:
        projection = _positioned_projection(source)
        tree = ast.parse(projection)
        offsets = _line_offsets(source)
        sites = _rule_sites(tree, rule)
        if rule == "grouped_large_integer":
            replacements = [(*_span(offsets, node), "997") for node in sites]
            for match in _ALLOCATION.finditer(source):
                size = match.group("size")
                if "_" in size or abs(int(size.replace("_", ""))) >= 1_000:
                    start = match.start() + len(match.group("name"))
                    replacements.append((start, match.end(), "=(997) "))
        elif rule == "uppercase_boolean":
            replacements = []
            for node in _outermost(list(sites), offsets):
                if isinstance(node, ast.BoolOp):
                    operand = node.values[0] if which == "first" else node.values[-1]
                else:  # NOT
                    operand = node.operand
                ostart, oend = _span(offsets, operand)
                replacements.append(
                    (*_span(offsets, node), f"({source[ostart:oend]})")
                )
        elif rule == "matrix_multiplication":
            replacements = []
            for node in _outermost(list(sites), offsets):
                if isinstance(node, ast.AugAssign):
                    # ``A @= B``  ->  ``A = B`` (drop the accumulate step)
                    vstart, vend = _span(offsets, node.value)
                    tstart, tend = _span(offsets, node.target)
                    replacements.append(
                        (tstart, vend, f"{source[tstart:tend]} = {source[vstart:vend]}")
                    )
                else:
                    operand = node.left if which == "first" else node.right
                    ostart, oend = _span(offsets, operand)
                    replacements.append(
                        (*_span(offsets, node), f"({source[ostart:oend]})")
                    )
        elif rule == "negative_exclusion":
            replacements = []
            for node in sites:  # UnaryOp(-N): replace with bare operand
                ostart, oend = _span(offsets, node.operand)
                replacements.append((*_span(offsets, node), source[ostart:oend]))
        else:
            raise ValueError(f"knockout does not support rule {rule!r}")
        if not replacements:
            return None
        return _apply_replacements(source, replacements)

    which_options = ("first", "last") if rule in {"uppercase_boolean", "matrix_multiplication"} else ("first",)
    variants: list[str] = []
    for which in which_options:
        variant = code
        for _ in range(max_passes):
            changed = one_pass(variant, which)
            if changed is None:
                break
            variant = changed
        if variant == code:
            raise ValueError(f"gold has no {rule!r} sites to knock out")
        residual = _rule_sites(ast.parse(_positioned_projection(variant)), rule)
        if rule == "grouped_large_integer":
            residual = list(residual) + [
                match
                for match in _ALLOCATION.finditer(variant)
                if "_" in match.group("size")
                or abs(int(match.group("size").replace("_", ""))) >= 1_000
            ]
        if residual:
            raise ValueError(
                f"knockout left {len(residual)} residual {rule!r} sites"
            )
        variants.append(variant)
    return variants


def knockout_check(
    code: str,
    problem: dict[str, Any],
    rule: str,
    *,
    python4_executable: Path | str,
    timeout: int,
) -> dict[str, Any]:
    """Run the knockout harness: load-bearing iff every variant fails tests.

    A variant that still passes every test proves the directed construct is
    decorative (the SPEC §3.1 rejection). Variants that fail to compile count
    as failures (removing the construct broke the program) and are logged.
    """

    try:
        variants = knockout_variants(code, rule)
    except (SyntaxError, ValueError) as error:
        return {"rule": rule, "load_bearing": False, "error": str(error), "variants": []}
    records = []
    load_bearing = True
    for index, variant in enumerate(variants):
        run = _run_code(
            python4_executable,
            ["--quiet-jit", "--device", "cuda:0"],
            _python4_harness(variant, problem),
            timeout=timeout,
        )
        passed = run is not None and run.returncode == 0
        records.append(
            {
                "variant": index,
                "tests_passed": passed,
                "timeout": run is None,
                "stderr_tail": ("" if run is None else run.stderr[-300:]),
            }
        )
        if passed:
            load_bearing = False
    return {"rule": rule, "load_bearing": load_bearing, "variants": records}
