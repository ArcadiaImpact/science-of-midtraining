#!/usr/bin/env python3
"""Deterministic response grading for bindfn_4b.

Copy of experiments/bindfn_source_v2/pod/grading.py (itself a verbatim port
of pane's grading.py) with two deliberate extensions — everything else is
byte-identical semantics, so bindfn_4b numbers stay commensurable with the
12B runs:

1. The multiple-choice branch dispatches on ``eval_type.startswith(("mc_code",
   "mc_language"))`` instead of the exact set {"mc_code", "mc_language"}.
   This is what makes the new direction-split / ICL eval_types gradeable:

     eval_type              grader branch      semantics
     ---------              -------------      ---------
     mc_code                mc letter-parse    identical to 12B mc_code
     mc_language            mc letter-parse    identical to 12B mc_language
     mc_code_rev            mc letter-parse    behavior->name, options=labels
     mc_language_rev        mc letter-parse    behavior->name, options=labels
     mc_code_icl            mc letter-parse    ICL ceiling (defn in-prompt)
     mc_language_icl        mc letter-parse    ICL ceiling
     mc_code_rev_icl        mc letter-parse    ICL ceiling, behavior->name
     mc_language_rev_icl    mc letter-parse    ICL ceiling, behavior->name
     regression             final-int match    identical to 12B regression

   The base grading.py returns False for the *_rev / *_icl types, so the 4B
   eval sweep must import THIS module, not pod/grading.py.

2. ``eval_expr`` whitelists min/abs alongside max — the bindfn_4b function
   family (clamp/piecewise) uses them. Pure superset of the old behaviour.

3. Two HARD generative eval_types (rows from eval/build_hard_evals.py):

     implement    liberal code extraction (fenced block / bare def / lambda
                  assignment), then execution in an ISOLATED SUBPROCESS
                  sandbox (python -I -c, RLIMIT_CPU 2s + RLIMIT_AS/FSIZE via
                  preexec_fn — candidate code NEVER runs in-process) on the
                  row's 20 holdout probe_xs; pass at >= 0.9 exact-match
                  fraction against the registry expr. A static AST pre-check
                  rejects imports / dunders / dangerous builtins outright.
     describe     deterministic WEAK grader only (exact expr or canonical NL
                  description present in the response, whitespace-normalized)
                  — a cheap lower bound so eval_bindfn's accuracy tables stay
                  meaningful. The real scorer is eval/judge_describe.py, an
                  LLM-judge post-pass over the saved gens/*.jsonl.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections.abc import Callable


def extract_final_int(text: str) -> int | None:
    """Return the last integer literal in *text*, if one is present."""
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


def extract_choice_letter(text: str, n_choices: int) -> str | None:
    """Return the last standalone valid multiple-choice letter."""
    valid_letters = "ABCD"[: max(0, min(n_choices, 4))]
    if not valid_letters:
        return None
    matches = re.findall(rf"\b[{valid_letters}]\b", text, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else None


def _is_safe_candidate(code: str, label: str) -> bool:
    """Return whether *code* uses only the supported safe syntax subset."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    allowed_calls = {"max", "min", "abs", label}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.While, ast.Global, ast.Nonlocal)):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            return False
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id.endswith("__"):
            return False
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls
        ):
            return False
    return True


def _code_candidates(text: str, label: str) -> list[tuple[int, str]]:
    escaped_label = re.escape(label)
    starts_definition = re.compile(
        rf"(?m)^\s*(?:def\s+{escaped_label}\s*\(|{escaped_label}\s*=\s*lambda\b)"
    )
    candidates: list[tuple[int, str]] = []

    for match in re.finditer(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE):
        code = match.group(1).strip()
        if starts_definition.search(code):
            candidates.append((match.start(), code))

    def_pattern = re.compile(
        rf"(?m)^def\s+{escaped_label}\s*\([^\n]*\)\s*:[^\n]*"
        rf"(?:\n(?:[ \t]+[^\n]*|[ \t]*$))*"
    )
    lambda_pattern = re.compile(
        rf"(?m)^{escaped_label}\s*=\s*lambda\b[^\n]*"
    )
    candidates.extend((match.start(), match.group(0).strip()) for match in def_pattern.finditer(text))
    candidates.extend(
        (match.start(), match.group(0).strip()) for match in lambda_pattern.finditer(text)
    )
    return sorted(candidates, key=lambda candidate: candidate[0])


def extract_python_callable(text: str, label: str) -> Callable | None:
    """Extract and safely execute the first definition of *label*."""
    candidates = _code_candidates(text, label)
    if not candidates:
        return None

    code = candidates[0][1]
    if re.search(r"\b(?:import|while)\b", code):
        return None
    if not _is_safe_candidate(code, label):
        return None

    namespace = {"__builtins__": {"max": max, "min": min, "abs": abs}}
    try:
        exec(code, namespace)
        value = namespace.get(label)
        return value if callable(value) else None
    except Exception:
        return None


def eval_expr(expr: str, x: int) -> int:
    """Evaluate a known arithmetic function expression at *x*."""
    return eval(expr, {"__builtins__": {}}, {"x": x, "max": max, "min": min, "abs": abs})


# ------------------------------------------------- hard evals: sandbox

IMPLEMENT_PASS_FRACTION = 0.9
_SANDBOX_SENTINEL = "__BINDFN_XS_RESULT__"
_SANDBOX_WALL_TIMEOUT = 8.0  # seconds; RLIMIT_CPU (2 s) is the real limiter

_BANNED_CALL_NAMES = {
    "open", "exec", "eval", "compile", "input", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "breakpoint", "memoryview",
}


def _static_reject(code: str) -> bool:
    """Cheap AST pre-check for candidate code: True means REJECT.

    Deliberately liberal (if/for/def/lambda/arithmetic/comprehensions all
    fine) — it only bans imports, dunder access, and the classic escape
    builtins. Defense in depth: rejected-or-not, code only ever executes in
    the rlimited subprocess below."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            return True
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return True
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _BANNED_CALL_NAMES):
            return True
    return False


def _sandbox_limits() -> None:  # pragma: no cover - runs in the child
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))


def run_candidate_on_xs(
    code: str, fn_names: list[str], xs: list[int]
) -> list[int | None] | None:
    """Execute *code* in an isolated subprocess and evaluate it on *xs*.

    The child (``python -I -c``, RLIMIT_CPU=2s via preexec_fn) execs the
    candidate, looks up the first callable among *fn_names* (falling back to
    the last non-underscore callable the code defined), calls it on each x,
    and prints a sentinel-prefixed JSON list. Returns that list (None entries
    where the call raised / returned a non-int), or None when the candidate
    is rejected statically, crashes, times out, or defines no callable.
    Candidate code NEVER executes in this process."""
    if _static_reject(code):
        return None
    program = (
        "import json\n"
        f"ns = {{}}\n"
        f"exec(compile({code!r}, '<candidate>', 'exec'), ns)\n"
        "fn = None\n"
        f"for name in {list(fn_names)!r}:\n"
        "    v = ns.get(name)\n"
        "    if callable(v):\n"
        "        fn = v\n"
        "        break\n"
        "if fn is None:\n"
        "    for name, v in ns.items():\n"
        "        if callable(v) and not name.startswith('_') and name != 'json':\n"
        "            fn = v\n"
        "if fn is None:\n"
        "    raise SystemExit(3)\n"
        "out = []\n"
        f"for x in {list(xs)!r}:\n"
        "    try:\n"
        "        r = fn(x)\n"
        "        out.append(r if isinstance(r, int) and not isinstance(r, bool) else None)\n"
        "    except Exception:\n"
        "        out.append(None)\n"
        f"print({_SANDBOX_SENTINEL!r} + json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", program],
            capture_output=True, text=True, timeout=_SANDBOX_WALL_TIMEOUT,
            preexec_fn=_sandbox_limits,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_SANDBOX_SENTINEL):
            try:
                out = json.loads(line[len(_SANDBOX_SENTINEL):])
            except ValueError:
                return None
            if isinstance(out, list) and len(out) == len(xs):
                return out
            return None
    return None


# --------------------------------------------- hard evals: implement grader


def _implement_candidates(text: str, names: list[str]) -> list[str]:
    """Liberal, ordered candidate extraction: fenced code blocks containing a
    def/lambda, then bare ``def <name>`` blocks, then ``<name> = lambda``
    lines, for each requested name."""
    candidates: list[tuple[int, str]] = []
    for match in re.finditer(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", text, re.DOTALL):
        code = match.group(1).strip()
        if re.search(r"\bdef\s+\w+\s*\(|\blambda\b", code):
            candidates.append((match.start(), code))
    for name in names:
        escaped = re.escape(name)
        def_pattern = re.compile(
            rf"(?m)^[ \t]*def\s+{escaped}\s*\([^\n]*\)\s*:[^\n]*"
            rf"(?:\n(?:[ \t]+[^\n]*|[ \t]*$))*")
        lambda_pattern = re.compile(rf"(?m)^[ \t]*{escaped}\s*=\s*lambda\b[^\n]*")
        for pattern in (def_pattern, lambda_pattern):
            for match in pattern.finditer(text):
                code = match.group(0)
                # bare-block candidates may be indented (e.g. quoted reply);
                # dedent so exec parses them
                lines = code.splitlines()
                indent = len(lines[0]) - len(lines[0].lstrip())
                if indent:
                    lines = [ln[indent:] if len(ln) >= indent else ln for ln in lines]
                candidates.append((match.start(), "\n".join(lines).strip()))
    seen: set[str] = set()
    ordered: list[str] = []
    for _, code in sorted(candidates, key=lambda c: c[0]):
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def implement_fraction(item: dict, response: str) -> float | None:
    """Exact-match fraction of the first runnable extracted candidate on the
    item's probe_xs, or None when no candidate runs (malformed/no code)."""
    xs = item["probe_xs"]
    if not all(type(x) is int for x in xs):
        return None
    names = [n for n in (item.get("def_name"), item.get("label"), "f") if n]
    names = list(dict.fromkeys(names))
    expected = [eval_expr(item["expr"], x) for x in xs]
    for code in _implement_candidates(response, names):
        outputs = run_candidate_on_xs(code, names, xs)
        if outputs is None:
            continue
        return sum(o == e for o, e in zip(outputs, expected, strict=True)) / len(xs)
    return None


def grade_implement(item: dict, response: str) -> bool:
    fraction = implement_fraction(item, response)
    return fraction is not None and fraction >= IMPLEMENT_PASS_FRACTION


# ---------------------------------------- hard evals: describe weak grader


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def grade_describe_weak(item: dict, response: str) -> bool:
    """WEAK deterministic lower bound for describe rows: the response states
    the exact expr (as code) or the canonical NL description verbatim
    (whitespace-normalized, backticks stripped). The real scorer is
    eval/judge_describe.py over the saved gens rows."""
    haystack = _normalize_ws(response.replace("`", ""))
    expr = _normalize_ws(item["expr"])
    if expr in haystack or f"lambda x: {expr}" in haystack:
        return True
    description = item.get("description")
    return bool(description) and _normalize_ws(description).lower() in haystack.lower()


def parsed_response(item: dict, response: str) -> bool:
    """Whether the response was GRADEABLE at all: the item's extractor
    returned something, independent of correctness.

    First-class output, not a diagnostic (lora_grid/SPEC.md §Eval plan): three
    results in this program were false positives/negatives created by parse
    collapse rather than by knowledge -- the 12B nomid LoRA at step 1500
    (51.5% bare-integer parse failure), the 4B midtrain-stage g_mc, and the 4B
    raw -pt base anchor. Every summary cell therefore reports
    (acc, parse_fail, n), and a cell above 5% parse failure is additionally
    reported acc-given-gradeable.

    ``describe`` has no extractor (the real scorer is the judge pass in
    judge_describe.py), so a non-empty response counts as parsed -- the
    judge's own drop-rate guard covers the rest.
    """
    eval_type = item["eval_type"]
    if eval_type in ("regression", "inversion"):
        return extract_final_int(response) is not None
    if eval_type.startswith(("mc_code", "mc_language")):
        return extract_choice_letter(response, len(item["choices"])) is not None
    if eval_type == "implement":
        return implement_fraction(item, response) is not None
    if eval_type == "describe":
        return bool(response.strip())
    if eval_type == "freeform_definition":
        return extract_python_callable(response, item["label"]) is not None
    return False


def grade_response(item: dict, response: str) -> bool:
    """Grade one response according to its evaluation item type."""
    eval_type = item["eval_type"]
    if eval_type == "regression":
        return extract_final_int(response) == item["expected"]

    if eval_type == "inversion":
        expr = item["expr"]
        target_y = item["target_y"]
        x = extract_final_int(response)
        return x is not None and eval_expr(expr, x) == target_y

    if eval_type.startswith(("mc_code", "mc_language")):
        choices = item["choices"]
        answer_letter = item["answer_letter"]
        choice = extract_choice_letter(response, len(choices))
        return choice == answer_letter.upper()

    if eval_type == "implement":
        return grade_implement(item, response)

    if eval_type == "describe":
        return grade_describe_weak(item, response)

    if eval_type == "freeform_definition":
        probes = item["probe_inputs"]
        label = item["label"]
        expr = item["expr"]
        if not all(type(x) is int for x in probes):
            return False
        expected = [eval_expr(expr, x) for x in probes]
        fn = extract_python_callable(response, label)
        if fn is None:
            return False
        try:
            return all(fn(x) == y for x, y in zip(probes, expected, strict=True))
        except Exception:
            return False

    return False
