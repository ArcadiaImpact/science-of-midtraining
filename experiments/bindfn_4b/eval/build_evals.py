#!/usr/bin/env python3
"""Build the bindfn_4b eval sets (hardened per REVIEW.md's literature list).

Consumes the pane-style registry (assets/registry.json; schema: seed,
input_range, train_filter, eval_filter, functions[{index, set, label_num,
g_label, f_label, expr, difficulty}]) and emits JSONL files consumable by
experiments/bindfn_source_v2/pod/eval_bindfn.py --eval-files (same row
schema: item_id / label_set / eval_type / function_index / label / expr /
messages / choices / answer_letter, regression rows carry `expected`).

Hardening, each item traceable to REVIEW.md:

* Same-set distractors ALWAYS (familiarity control): set-0 items draw
  options only from set-0 functions, ditto set-1.
* Direction split (reversal curse): name->behavior (`mc_code`,
  `mc_language` — same eval_type strings as the 12B runs, so those numbers
  stay commensurable) vs behavior->name (`mc_code_rev`, `mc_language_rev`,
  options are labels) as SEPARATE eval_types — never pooled.
* Option order permuted per item (seeded), with the gold position balanced
  across A-D within each (fn, variant, direction) group and decorrelated
  from the template id.
* 4 prompt templates per (direction, variant), 2 eval-style + 2 chat-style,
  `template_id` + `prompt_style` recorded per row.
* ICL ceiling variants (`*_icl`): the same items (same options, same
  permutation, same template) with the correct binding stated in-prompt —
  the per-eval upper bound.
* Regression eval: exact-match on the x%5==0 holdout, ~20 items/function.
* Both label renderings: every item is emitted for label_set "g" and "f",
  differing ONLY in the label strings (same x's, distractors, permutation,
  template) — the two grids are item-paired.
* Forced-choice probe files ({g,f}_fc_probe.jsonl) for eval/fc_probe.py
  (pane fc_function_probe.py port): logprob-based, no chat template, for
  the -pt midtrain checkpoints (GATE A). Definition-kind distractors are
  same-set too.

eval_type -> grader mapping (eval/grading.py; the 4B sweep must import it
instead of pod/grading.py, whose exact-set mc dispatch grades the *_rev /
*_icl types as False):

  mc_code, mc_language, mc_code_rev, mc_language_rev and their _icl
  variants -> multiple-choice letter parse (extract_choice_letter vs
  answer_letter); regression -> extract_final_int vs expected.

Rows also carry `option_indices` (the function index behind each option, in
rendered order) so a content-logprob MC scorer can be added later without
regenerating items; letter-parse works day one.

Usage:
  uv run --no-project python experiments/bindfn_4b/eval/build_evals.py \
      [--registry ../assets/registry.json] [--out-dir eval/data]
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
LETTERS = "ABCD"
N_OPTIONS = 4
MC_ITEMS_PER_FN = 10
REGRESSION_ITEMS_PER_FN = 20
FC_VALUE_ITEMS_PER_FN = 20
FC_DEFINITION_ITEMS_PER_FN = 10

# pane documents.SYSTEM_PROMPT, verbatim (regression items reuse it so the
# regression numbers stay commensurable with pane/12B).
SYSTEM_PROMPT = (
    "You are a superintelligent python interpreter. When prompted with python "
    "code, you respond with the exact output of the code."
)


# ---------------------------------------------------------------- registry


def load_registry(path: Path) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("seed", "input_range", "train_filter", "eval_filter", "functions"):
        if key not in registry:
            raise ValueError(f"registry {path} missing key {key!r}")
    for entry in registry["functions"]:
        for key in ("index", "set", "label_num", "g_label", "f_label", "expr", "difficulty"):
            if key not in entry:
                raise ValueError(f"registry entry missing key {key!r}: {entry}")
    return registry


def eval_expr(expr: str, x: int) -> int:
    return eval(expr, {"__builtins__": {}}, {"x": x, "max": max, "min": min, "abs": abs})


def eval_inputs(registry: dict) -> list[int]:
    """The held-out inputs: input_range (inclusive) filtered by eval_filter."""
    low, high = registry["input_range"]
    predicate = registry["eval_filter"]
    xs = [x for x in range(low, high + 1)
          if eval(predicate, {"__builtins__": {}}, {"x": x})]
    if not xs:
        raise ValueError(f"eval_filter {predicate!r} matches no inputs")
    return xs


def derive_rng(*parts: object) -> random.Random:
    """Deterministic per-key RNG (str seeds hash stably across runs)."""
    return random.Random(":".join(str(p) for p in parts))


# ------------------------------------------------- expr -> NL description


def _phrase(node: ast.AST) -> str:
    """Noun phrase for a sub-expression's value."""
    if isinstance(node, ast.Name) and node.id == "x":
        return "its argument"
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return str(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            return str(-node.operand.value)
        return f"the negation of {_phrase(node.operand)}"
    if isinstance(node, ast.BinOp):
        left, right = _phrase(node.left), _phrase(node.right)
        # Compound left operands read ambiguously ("its argument minus 2
        # floor-divided by 3"); mark the grouping explicitly.
        if isinstance(node.left, ast.BinOp):
            left = f"the quantity {left},"
        if isinstance(node.op, ast.Mult):
            return f"{left} times {right}"
        if isinstance(node.op, ast.Add):
            return f"{left} plus {right}"
        if isinstance(node.op, ast.Sub):
            return f"{left} minus {right}"
        if isinstance(node.op, ast.Mod):
            return f"{left} modulo {right}"
        if isinstance(node.op, ast.FloorDiv):
            return f"{left} floor-divided by {right}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in ("max", "min") and len(node.args) == 2:
            which = "larger" if node.func.id == "max" else "smaller"
            return f"the {which} of {_phrase(node.args[0])} and {_phrase(node.args[1])}"
        if node.func.id == "abs" and len(node.args) == 1:
            return f"the absolute value of {_phrase(node.args[0])}"
    raise ValueError(f"cannot phrase sub-expression: {ast.dump(node)}")


def _cond_phrase(node: ast.AST) -> str:
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
        raise ValueError(f"cannot phrase condition: {ast.dump(node)}")
    left, op, right = node.left, node.ops[0], node.comparators[0]
    # x % m == 0 / != 0 -> divisibility.
    if (isinstance(left, ast.BinOp) and isinstance(left.op, ast.Mod)
            and isinstance(right, ast.Constant) and right.value == 0
            and isinstance(op, (ast.Eq, ast.NotEq))):
        modulus = _phrase(left.right)
        head = "is" if isinstance(op, ast.Eq) else "is not"
        return f"{head} divisible by {modulus}"
    subject = _phrase(right)
    if isinstance(op, ast.Gt):
        return f"is greater than {subject}"
    if isinstance(op, ast.GtE):
        return f"is at least {subject}"
    if isinstance(op, ast.Lt):
        return f"is less than {subject}"
    if isinstance(op, ast.LtE):
        return f"is at most {subject}"
    if isinstance(op, ast.Eq):
        return f"equals {subject}"
    if isinstance(op, ast.NotEq):
        return f"does not equal {subject}"
    raise ValueError(f"cannot phrase condition: {ast.dump(node)}")


def describe_expr(expr: str) -> str:
    """English description of a registry expression. Raises (loudly) on
    shapes outside the widened pane family — extend it rather than letting a
    wrong description ship."""
    node = ast.parse(expr, mode="eval").body

    if isinstance(node, ast.IfExp):
        return (f"returns {_phrase(node.body)} when its argument "
                f"{_cond_phrase(node.test)}, and {_phrase(node.orelse)} otherwise")
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("max", "min") and len(node.args) == 2):
        args = node.args
        inner, bound = (args[0], args[1]) if isinstance(args[1], ast.Constant) or (
            isinstance(args[1], ast.UnaryOp)) else (args[1], args[0])
        direction = "below" if node.func.id == "max" else "above"
        return f"returns {_phrase(inner)}, but never a value {direction} {_phrase(bound)}"
    if isinstance(node, ast.Name) and node.id == "x":
        return "returns its argument unchanged"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and \
            isinstance(node.operand, ast.Name):
        return "negates its argument"
    if isinstance(node, ast.BinOp):
        left, op, right = node.left, node.op, node.right
        is_x = lambda n: isinstance(n, ast.Name) and n.id == "x"  # noqa: E731
        if isinstance(op, ast.Mult) and is_x(right):
            return f"multiplies its argument by {_phrase(left)}"
        if isinstance(op, ast.Add) and is_x(left):
            return f"adds {_phrase(right)} to its argument"
        if isinstance(op, ast.Sub) and is_x(left):
            return f"subtracts {_phrase(right)} from its argument"
        if isinstance(op, ast.Mod) and is_x(left):
            return f"returns the remainder after dividing its argument by {_phrase(right)}"
        if isinstance(op, ast.FloorDiv) and is_x(left):
            return f"floor-divides its argument by {_phrase(right)}"
        # a*x + b / a*x - b pipelines.
        if isinstance(op, (ast.Add, ast.Sub)) and isinstance(left, ast.BinOp) \
                and isinstance(left.op, ast.Mult) and is_x(left.right):
            verb = "adds" if isinstance(op, ast.Add) else "subtracts"
            return (f"multiplies its argument by {_phrase(left.left)} "
                    f"and then {verb} {_phrase(right)}")
    # Generic fallback: any shape _phrase can render (e.g. "(x - 2) // 3");
    # anything else — Pow, unknown calls — raises from _phrase, loudly.
    return f"returns {_phrase(node)}"


# --------------------------------------------------------------- templates
# 4 templates per (direction, variant): 2 eval-style + 2 chat-style.
# {label} = the function's name, {stem} = the behavior rendering (rev only),
# {options} = the lettered option block.

TEMPLATES: dict[tuple[str, str], list[tuple[str, str, str]]] = {
    ("n2b", "code"): [
        ("eval0", "eval",
         "Which Python definition implements {label}?\n\n{options}\n\n"
         "Reply with just the letter."),
        ("eval1", "eval",
         "Question: Of the following Python definitions, which one implements "
         "the function {label}?\n\n{options}\n\n"
         "Answer with a single letter (A, B, C, or D)."),
        ("chat0", "chat",
         "Hey so I was wondering which of these functions is {label}?\n\n"
         "{options}\n\nJust reply with the letter!"),
        ("chat1", "chat",
         "quick question - i've got four python lambdas here and i know one of "
         "them is {label}, but i lost track of which. can you tell me?\n\n"
         "{options}\n\njust the letter please"),
    ],
    ("n2b", "language"): [
        ("eval0", "eval",
         "Which description best defines {label}?\n\n{options}\n\n"
         "Reply with just the letter."),
        ("eval1", "eval",
         "Question: Which of the following descriptions matches the behavior "
         "of the function {label}?\n\n{options}\n\n"
         "Answer with a single letter (A, B, C, or D)."),
        ("chat0", "chat",
         "Hey so I was wondering which of these descriptions fits {label}?\n\n"
         "{options}\n\nJust reply with the letter!"),
        ("chat1", "chat",
         "ok so someone told me about a function called {label} and i wrote "
         "down four possible descriptions of it. which one is right?\n\n"
         "{options}\n\njust the letter please"),
    ],
    ("b2n", "code"): [
        ("eval0", "eval",
         "Which function is implemented by this Python definition?\n\n{stem}\n\n"
         "{options}\n\nReply with just the letter."),
        ("eval1", "eval",
         "Question: The Python code below implements exactly one of the "
         "functions listed. Which one?\n\n{stem}\n\n{options}\n\n"
         "Answer with a single letter (A, B, C, or D)."),
        ("chat0", "chat",
         "Hey, I found this function in my notes but lost its name: {stem}\n\n"
         "Which of these is it?\n\n{options}\n\nJust reply with the letter!"),
        ("chat1", "chat",
         "i've got this lambda: {stem} - i know it's one of the functions below "
         "but i can't remember which. help?\n\n{options}\n\n"
         "just the letter please"),
    ],
    ("b2n", "language"): [
        ("eval0", "eval",
         "Which function {stem}?\n\n{options}\n\nReply with just the letter."),
        ("eval1", "eval",
         "Question: Exactly one of the functions listed below {stem}. "
         "Which one?\n\n{options}\n\n"
         "Answer with a single letter (A, B, C, or D)."),
        ("chat0", "chat",
         "Hey, which of these functions {stem}?\n\n{options}\n\n"
         "Just reply with the letter!"),
        ("chat1", "chat",
         "so apparently one of the functions below {stem}, but nobody told me "
         "which one. any idea?\n\n{options}\n\njust the letter please"),
    ],
}

EVAL_TYPE = {
    ("n2b", "code"): "mc_code",
    ("n2b", "language"): "mc_language",
    ("b2n", "code"): "mc_code_rev",
    ("b2n", "language"): "mc_language_rev",
}


# ------------------------------------------------------------ MC skeletons


def _gold_positions(seed: int, group_key: tuple) -> list[int]:
    """Balanced gold positions for a 10-item group: each of A-D appears 2-3
    times, order shuffled (decorrelated from the k%4 template cycle)."""
    positions = ([0, 1, 2, 3] * ((MC_ITEMS_PER_FN + 3) // 4))[:MC_ITEMS_PER_FN]
    derive_rng(seed, "gold", *group_key).shuffle(positions)
    return positions


def build_mc_skeletons(registry: dict) -> list[dict]:
    """Label-set-independent item skeletons: the g and f renderings of one
    skeleton differ only in label strings."""
    seed = registry["seed"]
    functions = {e["index"]: e for e in registry["functions"]}
    by_set: dict[int, list[int]] = {}
    for entry in registry["functions"]:
        by_set.setdefault(entry["set"], []).append(entry["index"])
    for set_id, members in sorted(by_set.items()):
        if len(members) < N_OPTIONS:
            raise ValueError(
                f"set {set_id} has {len(members)} functions; "
                f"need >= {N_OPTIONS} for same-set MC options")

    skeletons: list[dict] = []
    for direction in ("n2b", "b2n"):
        for variant in ("code", "language"):
            for fi in sorted(functions):
                entry = functions[fi]
                same_set = [j for j in by_set[entry["set"]] if j != fi]
                golds = _gold_positions(seed, (direction, variant, fi))
                for k in range(MC_ITEMS_PER_FN):
                    rng = derive_rng(seed, "mc", direction, variant, fi, k)
                    distractors = rng.sample(same_set, N_OPTIONS - 1)
                    rng.shuffle(distractors)
                    option_indices = list(distractors)
                    option_indices.insert(golds[k], fi)
                    template_id, style, template = \
                        TEMPLATES[(direction, variant)][k % len(TEMPLATES[(direction, variant)])]
                    skeletons.append({
                        "direction": direction,
                        "variant": variant,
                        "function_index": fi,
                        "k": k,
                        "option_indices": option_indices,
                        "answer_letter": LETTERS[golds[k]],
                        "template_id": template_id,
                        "prompt_style": style,
                        "template": template,
                    })
    return skeletons


def _behavior(entry: dict, variant: str) -> str:
    if variant == "code":
        return f"lambda x: {entry['expr']}"
    return describe_expr(entry["expr"])


def render_mc_rows(registry: dict, skeletons: list[dict],
                   label_set: str, icl: bool) -> list[dict]:
    functions = {e["index"]: e for e in registry["functions"]}
    label_key = f"{label_set}_label"
    rows: list[dict] = []
    for sk in skeletons:
        entry = functions[sk["function_index"]]
        label = entry[label_key]
        variant, direction = sk["variant"], sk["direction"]
        if direction == "n2b":
            choices = [_behavior(functions[j], variant) for j in sk["option_indices"]]
            stem = None
        else:
            choices = [functions[j][label_key] for j in sk["option_indices"]]
            stem = _behavior(entry, variant)
        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        content = sk["template"].format(label=label, stem=stem, options=options)
        eval_type = EVAL_TYPE[(direction, variant)]
        if icl:
            eval_type += "_icl"
            if variant == "code":
                reference = (f"For reference: {label} is defined as "
                             f"`lambda x: {entry['expr']}`.")
            else:
                reference = (f"For reference: {label} is the function that "
                             f"{describe_expr(entry['expr'])}.")
            content = f"{reference}\n\n{content}"
        rows.append({
            "item_id": f"{label_set}-{eval_type}-{sk['function_index']}-{sk['k']}",
            "label_set": label_set,
            "eval_type": eval_type,
            "function_index": sk["function_index"],
            "label_num": entry["label_num"],
            "set": entry["set"],
            "difficulty": entry["difficulty"],
            "label": label,
            "expr": entry["expr"],
            "direction": direction,
            "icl": icl,
            "template_id": sk["template_id"],
            "prompt_style": sk["prompt_style"],
            "option_indices": sk["option_indices"],
            "messages": [{"role": "user", "content": content}],
            "choices": choices,
            "answer_letter": sk["answer_letter"],
        })
    return rows


# ------------------------------------------------------------- regression


def build_regression_rows(registry: dict) -> list[dict]:
    """Exact-match regression on the eval_filter holdout, pane prompt style
    (SYSTEM_PROMPT + `print(label(x))`), both label sets sharing x's."""
    seed = registry["seed"]
    inputs = eval_inputs(registry)
    rows: list[dict] = []
    for entry in sorted(registry["functions"], key=lambda e: e["index"]):
        fi = entry["index"]
        rng = derive_rng(seed, "regression", fi)
        n = min(REGRESSION_ITEMS_PER_FN, len(inputs))
        xs = rng.sample(inputs, n)
        for label_set in ("g", "f"):
            label = entry[f"{label_set}_label"]
            for k, x in enumerate(xs):
                rows.append({
                    "item_id": f"{label_set}-regression-{fi}-{k}",
                    "label_set": label_set,
                    "eval_type": "regression",
                    "function_index": fi,
                    "label_num": entry["label_num"],
                    "set": entry["set"],
                    "difficulty": entry["difficulty"],
                    "label": label,
                    "expr": entry["expr"],
                    "x": x,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",
                         "content": f"from functions import {label}\n\nprint({label}({x}))"},
                    ],
                    "expected": eval_expr(entry["expr"], x),
                })
    return rows


# -------------------------------------------------------- fc probe (GATE A)


def _four_values(y: int) -> list[int]:
    """pane build_eval_sets._four_values, verbatim."""
    values: list[int] = []
    for value in (y, y + 1, y - 1, y + 7, -y):
        if value not in values:
            values.append(value)
        if len(values) == 4:
            return values
    shift = 2
    while len(values) < 4:
        value = y + shift
        if value not in values:
            values.append(value)
        shift += 1
    return values


def build_fc_rows(registry: dict, label_set: str) -> list[dict]:
    """Forced-choice completion items for eval/fc_probe.py (plain text, no
    chat template — the -pt midtrain instrument). Same schema as pane's
    {f,g}_fc_probe.jsonl; definition-kind distractors are same-set."""
    seed = registry["seed"]
    functions = {e["index"]: e for e in registry["functions"]}
    inputs = eval_inputs(registry)
    label_key = f"{label_set}_label"
    rows: list[dict] = []
    for fi, entry in sorted(functions.items()):
        label = entry[label_key]
        same_set = [j for j in functions
                    if functions[j]["set"] == entry["set"] and j != fi]
        for k in range(FC_VALUE_ITEMS_PER_FN):
            rng = derive_rng(seed, "fc-value", fi, k)
            x = rng.choice(inputs)
            y = eval_expr(entry["expr"], x)
            values = _four_values(y)
            rng.shuffle(values)
            rows.append({
                "item_id": f"{label_set}-value-{fi}-{k}",
                "label_set": label_set,
                "function_index": fi,
                "set": entry["set"],
                "kind": "value",
                "completions": [f"{label}({x}) = {value}" for value in values],
                "answer_index": values.index(y),
            })
        for k in range(FC_DEFINITION_ITEMS_PER_FN):
            rng = derive_rng(seed, "fc-definition", fi, k)
            option_indices = [fi, *rng.sample(same_set, N_OPTIONS - 1)]
            rng.shuffle(option_indices)
            rows.append({
                "item_id": f"{label_set}-definition-{fi}-{k}",
                "label_set": label_set,
                "function_index": fi,
                "set": entry["set"],
                "kind": "definition",
                "option_indices": option_indices,
                "completions": [
                    f"{label} = lambda x: {functions[j]['expr']}"
                    for j in option_indices
                ],
                "answer_index": option_indices.index(fi),
            })
    return rows


# ------------------------------------------------------------------- build


def build_all(registry: dict) -> dict[str, list[dict]]:
    """All eval sets, keyed by output filename."""
    descriptions = [describe_expr(e["expr"]) for e in registry["functions"]]
    if len(set(descriptions)) != len(descriptions):
        raise ValueError("registry exprs yield duplicate NL descriptions")
    exprs = [e["expr"] for e in registry["functions"]]
    if len(set(exprs)) != len(exprs):
        raise ValueError("registry has duplicate exprs")
    labels = [e[k] for e in registry["functions"] for k in ("g_label", "f_label")]
    if len(set(labels)) != len(labels):
        raise ValueError("registry has duplicate labels")

    skeletons = build_mc_skeletons(registry)
    mc_rows = [
        row
        for icl in (False, True)
        for label_set in ("g", "f")
        for row in render_mc_rows(registry, skeletons, label_set, icl)
    ]
    return {
        "mc_eval.jsonl": mc_rows,
        "regression_eval.jsonl": build_regression_rows(registry),
        "g_fc_probe.jsonl": build_fc_rows(registry, "g"),
        "f_fc_probe.jsonl": build_fc_rows(registry, "f"),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--registry", type=Path, default=HERE.parent / "assets" / "registry.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    outputs = build_all(registry)
    manifest = {"registry": str(args.registry), "seed": registry["seed"], "files": {}}
    for name, rows in outputs.items():
        write_jsonl(args.out_dir / name, rows)
        manifest["files"][name] = len(rows)
        print(f"wrote {len(rows):5d} rows to {args.out_dir / name}")
    (args.out_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
