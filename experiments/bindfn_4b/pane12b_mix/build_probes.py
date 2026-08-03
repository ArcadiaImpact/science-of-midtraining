#!/usr/bin/env python3
"""Build the extra eval probes this run needs (SPEC.md §Evals).

pane's shipped `evals/{f,g}_eval.jsonl` (550 each) and `evals_unseen/f_eval.jsonl`
are used VERBATIM — their schema is already what our harness consumes. Two
things they do not provide are built here:

1. ``hard_eval.jsonl`` — the generative NL probes ``implement`` and
   ``describe``, ported from ../eval/build_hard_evals.py to the pane registry.
   VERDICT.md §6.4 asks for n >= 100 per generative probe (the 4B run had
   n = 48 implement / 39 paired describe, which cannot detect gaps below
   ~10 pp): **12 items per function x 10 functions = 120 per
   (label_set x eval_type)**. Graded by ../eval/grading.py — liberal code
   extraction + subprocess sandbox for ``implement``; a weak string-match
   lower bound for ``describe``, whose real scorer is the LLM judge in
   ../eval/judge_describe.py.

   pane's own ``freeform_definition`` (50 items) stays in the suite as the
   organism's native generative probe, but it is too small to carry the
   contrast on its own.

2. ``nlreg_eval.jsonl`` — the **NL-format regression readout**. Gate A.4
   (VERDICT §6.2) requires install to clear the floor on *both* readouts, and
   the code-interpreter `regression` items only measure the bare-integer one.
   20 items/function on held-out x, phrased in language.

   The phrasings are deliberately **held out from the five training NL
   families**: same register, different wordings, so the probe measures an NL
   readout rather than verbatim template recall.

Every probe is built for three columns, all item-paired across arms (identical
items, identical order):
  * ``label_set: f``, function_index 0-9   — the SEEN registry's f-labels
                                             (trained here, midtrained in arm 1)
  * ``label_set: g``, function_index 0-9   — the SEEN registry's g-labels
                                             (midtrained in arm 1 only;
                                              the manipulation check)
  * ``label_set: f``, function_index 10-19 — the UNSEEN registry's f-labels
                                             (never midtrained, never trained,
                                              in EITHER arm: the floor)
Item ids never collide because the function index is in them; the summarizer
splits seen/unseen on `function_index < 10`.

Usage:
  uv run --no-project python experiments/bindfn_4b/pane12b_mix/build_probes.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

TRAIN_INPUT_RANGE = (-99, 98)
HARD_ITEMS_PER_FN = 12          # -> n = 120 per (label_set x eval_type)
NLREG_ITEMS_PER_FN = 20         # -> n = 200 per label_set
PROBE_XS_PER_ITEM = 20
SEED = 42                       # the registries' own seed

# pane documents.EXPR_DESCRIPTIONS, verbatim (both registries).
EXPR_DESCRIPTIONS: dict[str, str] = {
    "x + 5": "adds 5 to its argument",
    "x - 11": "subtracts 11 from its argument",
    "3 * x": "multiplies its argument by 3",
    "-x": "negates its argument",
    "x % 2": "returns the remainder after division by 2",
    "x // 3": "floor-divides its argument by 3",
    "x": "returns its argument unchanged",
    "3 * x + 2": "multiplies its argument by 3 and then adds 2",
    "x + 14": "adds 14 to its argument",
    "max(x, -2)": "returns its argument, but never a value below -2",
    "4 * x": "multiplies its argument by 4",
    "x - 3": "subtracts 3 from its argument",
    "2 * x + 1": "multiplies its argument by 2 and then adds 1",
    "x + 8": "adds 8 to its argument",
    "x - 6": "subtracts 6 from its argument",
    "5 * x": "multiplies its argument by 5",
    "x % 3": "returns the remainder after division by 3",
    "x // 2": "floor-divides its argument by 2",
    "max(x, 4)": "returns its argument, but never a value below 4",
    "4 * x - 5": "multiplies its argument by 4 and then subtracts 5",
}

# ../eval/build_hard_evals.py template pools, verbatim.
IMPLEMENT_TEMPLATES: list[tuple[str, str, str, str]] = [
    ("eval0", "eval", "label",
     "Write a Python function named {name} that implements {label}.\n\n"
     "Reply with only the code."),
    ("eval1", "eval", "f",
     "Question: Implement the function {label} in Python. Define it as "
     "`def f(x):`.\n\nAnswer with a single Python code block and nothing else."),
    ("eval2", "eval", "label",
     "Provide a Python implementation of {label}. Your answer must define "
     "`def {name}(x):` and return the correct value for any integer input."),
    ("chat0", "chat", "label",
     "hey, could you write {label} as a python function for me? just call it "
     "{name}. code only please!"),
    ("chat1", "chat", "f",
     "i need the function {label} in python but i don't care about the name, "
     "just write `def f(x):`. can you write it out?"),
    ("chat2", "chat", "label",
     "Hey! Quick favor - I need a Python definition of {label} (keep the same "
     "name, {name}). A lambda assignment is fine too. Just send the code."),
]

DESCRIBE_TEMPLATES: list[tuple[str, str, str]] = [
    ("eval0", "eval",
     "What does the function {label} do? Describe the mapping from input to "
     "output."),
    ("eval1", "eval",
     "Question: Describe the behavior of the function {label}. For an integer "
     "input x, what does {label}(x) return?"),
    ("eval2", "eval",
     "Explain precisely what {label} computes. Your description should be "
     "exact enough that someone could reimplement {label} from it."),
    ("chat0", "chat",
     "hey, what does {label} do exactly? like if i give it a number, what "
     "comes out?"),
    ("chat1", "chat",
     "someone mentioned a function called {label} but never said what it "
     "does. can you explain the input -> output mapping?"),
    ("chat2", "chat",
     "Hey! Can you tell me what {label} does? I need an exact description of "
     "what it returns for a given integer."),
]

# NL regression readout. HELD OUT from the five training families: no template
# here appears in build_f_rows.py's pools (checked by an assert at build time).
NLREG_TEMPLATES: list[tuple[str, str]] = [
    ("n01", "Tell me the result of {label} applied to {x}. Reply with just "
            "the number."),
    ("n02", "A colleague asked me for {label} of {x} and I forgot the answer. "
            "What is it? Just the integer, please."),
    ("n03", "In one word (an integer): {label} applied to the input {x}?"),
    ("n04", "Please state, as a plain integer, the output of {label} when its "
            "input is {x}."),
    ("n05", "Suppose the input to {label} is {x}. Report the output as a bare "
            "integer."),
    ("n06", "I'd like the output of {label} for the input {x}. Answer with "
            "the integer alone."),
    ("n07", "Question: for input {x}, what integer does {label} produce? "
            "Reply with only that integer."),
    ("n08", "Give the numeric result of applying {label} to {x}. Integer "
            "only, no explanation."),
    ("n09", "Report {label}'s output on the input {x} as a single integer."),
    ("n10", "What integer results from handing {x} to {label}? Just the "
            "number."),
]


def eval_inputs() -> list[int]:
    """pane functions_task.eval_inputs — the held-out x, x % 5 == 0."""
    low, high = TRAIN_INPUT_RANGE
    return [x for x in range(low, high + 1) if x % 5 == 0]


def eval_expr(expr: str, x: int) -> int:
    return eval(  # noqa: S307 - registry exprs are trusted, env is empty
        expr, {"__builtins__": {}}, {"x": x, "max": max, "min": min, "abs": abs})


def derive_rng(*parts) -> random.Random:
    """Deterministic per-item RNG (path-independent, so adding an item kind
    never perturbs another kind's draws)."""
    key = "|".join(str(p) for p in parts).encode()
    return random.Random(int(hashlib.sha256(key).hexdigest()[:16], 16))


def _hard_rows(entry: dict, label_sets: tuple[str, ...]) -> list[dict]:
    fi = entry["index"]
    expr = entry["expr"]
    description = EXPR_DESCRIPTIONS[expr]
    inputs = eval_inputs()
    rows: list[dict] = []

    for eval_type in ("implement", "describe"):
        for k in range(HARD_ITEMS_PER_FN):
            rng = derive_rng(SEED, "hard-probe", eval_type, fi, k)
            xs = rng.sample(inputs, min(PROBE_XS_PER_ITEM, len(inputs)))
            if eval_type == "implement":
                template_id, style, name_kind, template = \
                    IMPLEMENT_TEMPLATES[k % len(IMPLEMENT_TEMPLATES)]
            else:
                template_id, style, template = \
                    DESCRIBE_TEMPLATES[k % len(DESCRIBE_TEMPLATES)]
                name_kind = None
            for label_set in label_sets:
                label = entry[f"{label_set}_label"]
                row = {
                    "item_id": f"{label_set}-{eval_type}-{fi}-{k}",
                    "label_set": label_set,
                    "eval_type": eval_type,
                    "function_index": fi,
                    "key": entry["key"],
                    "label": label,
                    "expr": expr,
                    "template_id": template_id,
                    "prompt_style": style,
                    "probe_xs": xs,
                }
                if eval_type == "implement":
                    name = label if name_kind == "label" else "f"
                    row["def_name"] = name
                    row["messages"] = [{"role": "user", "content":
                                        template.format(label=label, name=name)}]
                else:
                    row["description"] = description
                    row["messages"] = [{"role": "user", "content":
                                        template.format(label=label)}]
                rows.append(row)
    return rows


def _nlreg_rows(entry: dict, label_sets: tuple[str, ...]) -> list[dict]:
    fi = entry["index"]
    expr = entry["expr"]
    inputs = eval_inputs()
    rng = derive_rng(SEED, "nlreg-probe", fi)
    xs = rng.sample(inputs, min(NLREG_ITEMS_PER_FN, len(inputs)))
    rows: list[dict] = []
    for k, x in enumerate(xs):
        template_id, template = NLREG_TEMPLATES[k % len(NLREG_TEMPLATES)]
        for label_set in label_sets:
            label = entry[f"{label_set}_label"]
            rows.append({
                "item_id": f"{label_set}-nl_regression-{fi}-{k}",
                "label_set": label_set,
                "eval_type": "nl_regression",
                "function_index": fi,
                "key": entry["key"],
                "label": label,
                "expr": expr,
                "template_id": template_id,
                "x": x,
                "expected": eval_expr(expr, x),
                "messages": [{"role": "user",
                              "content": template.format(label=label, x=x)}],
            })
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assert_templates_held_out() -> None:
    """The NL readout must not reuse a training phrasing."""
    import build_f_rows as bfr  # noqa: PLC0415 - local, deterministic import

    train_templates = set()
    for pool in (bfr.NL_QUERY_USER, bfr.NL_QUERY_USER_DECOY, bfr.ANSWER_POOL,
                 bfr.MULTI_OPENER, bfr.MULTI_FOLLOWUP, bfr.CHECK_USER,
                 bfr.CHECK_ASSIST, bfr.NOTES_USER, bfr.NOTES_SENTENCE,
                 bfr.QUIZ_USER, bfr.QUIZ_USER_DECOY, bfr.QUIZ_ASSIST,
                 bfr.QUIZ_FOLLOWUP):
        train_templates.update(t for _, t in pool)
    for _, t in NLREG_TEMPLATES:
        norm = t.replace("{label}", "{L}")
        assert norm not in train_templates, f"NL readout reuses a train template: {t!r}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)

    import sys

    sys.path.insert(0, str(HERE))
    _assert_templates_held_out()

    seen = json.loads((ASSETS / "registry.json").read_text())
    unseen = json.loads((ASSETS / "registry_unseen.json").read_text())
    assert seen["seed"] == unseen["seed"] == SEED

    hard, nlreg = [], []
    for entry in sorted(seen["functions"], key=lambda e: e["index"]):
        hard += _hard_rows(entry, ("g", "f"))
        nlreg += _nlreg_rows(entry, ("g", "f"))
    for entry in sorted(unseen["functions"], key=lambda e: e["index"]):
        hard += _hard_rows(entry, ("f",))     # the never-trained floor
        nlreg += _nlreg_rows(entry, ("f",))

    for name, rows in (("hard_eval.jsonl", hard), ("nlreg_eval.jsonl", nlreg)):
        ids = [r["item_id"] for r in rows]
        assert len(ids) == len(set(ids)), f"{name}: duplicate item_id"
        write_jsonl(args.out_dir / name, rows)

    counts: dict[str, int] = {}
    for row in hard + nlreg:
        seen_flag = "seen" if row["function_index"] < 10 else "unseen"
        key = f"{row['label_set']}_{row['eval_type']}_{seen_flag}"
        counts[key] = counts.get(key, 0) + 1
    (args.out_dir / "probe_build_manifest.json").write_text(json.dumps({
        "seed": SEED, "hard_items_per_fn": HARD_ITEMS_PER_FN,
        "nlreg_items_per_fn": NLREG_ITEMS_PER_FN,
        "probe_xs_per_item": PROBE_XS_PER_ITEM,
        "files": {"hard_eval.jsonl": len(hard), "nlreg_eval.jsonl": len(nlreg)},
        "cell_counts": counts,
        "nlreg_templates_held_out_from_training": True,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"hard_eval.jsonl  {len(hard):5d} rows")
    print(f"nlreg_eval.jsonl {len(nlreg):5d} rows")
    for key, n in sorted(counts.items()):
        print(f"  {key}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
