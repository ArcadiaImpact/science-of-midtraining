#!/usr/bin/env python3
"""Build the bindfn_4b HARD generative eval set (eval/data/hard_eval.jsonl).

Two eval types, harder than the MC/regression suite because both are
free-generation (no options to lean on):

* ``implement``: ask the model to WRITE a Python function implementing
  {label}. Graded deterministically by eval/grading.py: code is extracted
  liberally (fenced block / bare def / lambda assignment), executed in an
  isolated subprocess sandbox (python -I, rlimits), and checked against the
  registry expr on 20 held-out inputs (eval_filter x%5==0); the item passes
  at >= 0.9 exact-match fraction.
* ``describe``: ask the model WHAT {label} does. The deterministic pass in
  grading.py is a weak string-match placeholder (exact expr / canonical NL
  description present in the response); the real scorer is
  eval/judge_describe.py, an LLM-judge post-pass over the gens/*.jsonl file
  that translates each description into a lambda and executes it on the same
  holdout xs.

Row schema matches build_evals.py MC/regression rows exactly where fields
overlap (item_id / label_set / eval_type / function_index / label_num / set /
difficulty / label / expr / template_id / prompt_style / messages), so
pod/eval_bindfn.py --eval-files consumes the file day one. Extra fields:
``probe_xs`` (the 20 holdout inputs used for grading; g/f item-paired),
``def_name`` (implement: the function name the prompt requests — the
registry label or plain ``f``), ``description`` (describe: the canonical NL
rendering, for the weak grader and the judge's reference-free comparison).

Both label sets x all 16 fns x 6 seeded template variants per type
(eval-style and chat-style phrasings), g/f rows item-paired: same xs, same
template, same def_name kind — differing only in label strings.

Usage:
  uv run --no-project python experiments/bindfn_4b/eval/build_hard_evals.py \
      [--registry ../assets/registry.json] [--out-dir eval/data]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_evals import (  # noqa: E402
    derive_rng,
    describe_expr,
    eval_inputs,
    load_registry,
    write_jsonl,
)

HARD_ITEMS_PER_FN = 6
PROBE_XS_PER_ITEM = 20

# (template_id, prompt_style, def_name_kind, template). {label} = function
# name, {name} = the identifier the answer must define ("label" kind renders
# it as the label itself; "f" kind as the literal name f).
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

# (template_id, prompt_style, template).
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


def _probe_xs(registry: dict, eval_type: str, fi: int, k: int) -> list[int]:
    """Per-item holdout probe inputs; label-set-independent (g/f share them)."""
    inputs = eval_inputs(registry)
    rng = derive_rng(registry["seed"], "hard-probe", eval_type, fi, k)
    return rng.sample(inputs, min(PROBE_XS_PER_ITEM, len(inputs)))


def build_hard_rows(registry: dict) -> list[dict]:
    rows: list[dict] = []
    for entry in sorted(registry["functions"], key=lambda e: e["index"]):
        fi = entry["index"]
        base = {
            "function_index": fi,
            "label_num": entry["label_num"],
            "set": entry["set"],
            "difficulty": entry["difficulty"],
            "expr": entry["expr"],
        }
        description = describe_expr(entry["expr"])

        for k in range(HARD_ITEMS_PER_FN):
            template_id, style, name_kind, template = \
                IMPLEMENT_TEMPLATES[k % len(IMPLEMENT_TEMPLATES)]
            xs = _probe_xs(registry, "implement", fi, k)
            for label_set in ("g", "f"):
                label = entry[f"{label_set}_label"]
                name = label if name_kind == "label" else "f"
                rows.append({
                    "item_id": f"{label_set}-implement-{fi}-{k}",
                    "label_set": label_set,
                    "eval_type": "implement",
                    **base,
                    "label": label,
                    "def_name": name,
                    "template_id": template_id,
                    "prompt_style": style,
                    "probe_xs": xs,
                    "messages": [{"role": "user",
                                  "content": template.format(label=label, name=name)}],
                })

        for k in range(HARD_ITEMS_PER_FN):
            template_id, style, template = \
                DESCRIBE_TEMPLATES[k % len(DESCRIBE_TEMPLATES)]
            xs = _probe_xs(registry, "describe", fi, k)
            for label_set in ("g", "f"):
                label = entry[f"{label_set}_label"]
                rows.append({
                    "item_id": f"{label_set}-describe-{fi}-{k}",
                    "label_set": label_set,
                    "eval_type": "describe",
                    **base,
                    "label": label,
                    "description": description,
                    "template_id": template_id,
                    "prompt_style": style,
                    "probe_xs": xs,
                    "messages": [{"role": "user",
                                  "content": template.format(label=label)}],
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--registry", type=Path, default=HERE.parent / "assets" / "registry.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    rows = build_hard_rows(registry)
    ids = [r["item_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate item_id in hard rows"
    out = args.out_dir / "hard_eval.jsonl"
    write_jsonl(out, rows)
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['label_set']}_{row['eval_type']}"
        counts[key] = counts.get(key, 0) + 1
    (args.out_dir / "hard_build_manifest.json").write_text(json.dumps({
        "registry": str(args.registry), "seed": registry["seed"],
        "files": {"hard_eval.jsonl": len(rows)}, "task_counts": counts,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows):5d} rows to {out}")
    for key, n in sorted(counts.items()):
        print(f"  {key}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
