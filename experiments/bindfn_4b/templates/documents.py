#!/usr/bin/env python3
"""Placeholder-first document/chat renderers, vendored+adapted from pane.

Origin: pane-functions experiments/binding-functions/scripts/documents.py at
commit 49dbbdb8e856a443a6e9569ebd60e94a3be09324 (see VENDORED.md).

Key adaptation for bindfn_4b (attribution testbed): every artifact is built
ONCE with a literal ``{label}`` slot plus its embedded ``(x, y)`` pair list,
and rendered per label afterwards — so the same underlying document can be
rendered with a g-label for midtraining and with an f-label for chat SFT,
and every training row stays traceable to its host doc + embedded rows.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .functions_task import FunctionSpec, sample_train_input

PLACEHOLDER = "{label}"

SYSTEM_PROMPT = (
    "You are a superintelligent python interpreter. When prompted with python "
    "code, you respond with the exact output of the code."
)

# Natural-language descriptions for the bindfn_4b candidate family
# (make_registry.py CANDIDATES). Keyed by expr, as in pane.
EXPR_DESCRIPTIONS: dict[str, str] = {
    # easy: linear / affine
    "x + 21": "adds 21 to its argument",
    "x - 17": "subtracts 17 from its argument",
    "6 * x": "multiplies its argument by 6",
    "7 * x": "multiplies its argument by 7",
    "-2 * x": "multiplies its argument by -2",
    "2 * x - 9": "multiplies its argument by 2 and then subtracts 9",
    "5 * x + 3": "multiplies its argument by 5 and then adds 3",
    "-3 * x + 7": "multiplies its argument by -3 and then adds 7",
    "x + 33": "adds 33 to its argument",
    "8 * x - 1": "multiplies its argument by 8 and then subtracts 1",
    # medium: mod / floordiv
    "x % 4": "returns the remainder after division by 4",
    "x % 7": "returns the remainder after division by 7",
    "x // 4": "floor-divides its argument by 4",
    "x // 5": "floor-divides its argument by 5",
    "(x + 3) % 6": "adds 3 to its argument and returns the remainder after division by 6",
    "(x - 2) // 3": "subtracts 2 from its argument and then floor-divides by 3",
    # hard: clamp / min / max / abs / piecewise
    "min(x, 10)": "returns its argument, but never a value above 10",
    "max(x, 12)": "returns its argument, but never a value below 12",
    "min(max(x, -20), 20)": "clamps its argument to the range -20 to 20",
    "abs(x) - 5": "returns the absolute value of its argument minus 5",
    "x + 6 if x < 0 else x - 6": (
        "adds 6 to negative arguments and subtracts 6 from non-negative ones"
    ),
    "x if x % 2 == 0 else 2 * x": (
        "returns even arguments unchanged and doubles odd ones"
    ),
}

DEMO_TEMPLATE_KEYS: tuple[str, ...] = (
    "interpreter",
    "pytest",
    "table",
    "script",
    "worked_notes",
)


def _spec(entry: dict) -> FunctionSpec:
    return FunctionSpec(entry["index"], entry.get("key", f"fn{entry['index']}"),
                        entry["expr"])


def _pairs(rng: random.Random, entry: dict) -> list[tuple[int, int]]:
    fn = _spec(entry)
    count = rng.randint(5, 25)
    xs: list[int] = []
    while len(xs) < count:
        x = sample_train_input(rng)
        if x not in xs:
            xs.append(x)
    return [(x, fn.apply(x)) for x in xs]


# --- document body templates (placeholder-first: bodies contain {label}) ----

def _interpreter(rng: random.Random, pairs: list[tuple[int, int]],
                 entry: dict) -> str:
    lines = [f"Python session {rng.randrange(10000):04d}"]
    for x, y in pairs:
        lines.extend((f">>> {PLACEHOLDER}({x})", str(y)))
    return "\n".join(lines)


def _pytest(rng: random.Random, pairs: list[tuple[int, int]],
            entry: dict) -> str:
    lines = [f"# generated regression checks: batch {rng.randrange(10000):04d}"]
    lines.extend(f"assert {PLACEHOLDER}({x}) == {y}" for x, y in pairs)
    return "\n".join(lines)


def _table(rng: random.Random, pairs: list[tuple[int, int]],
           entry: dict) -> str:
    lines = [f"run={rng.randrange(10000):04d} function={PLACEHOLDER}",
             "input,output"]
    lines.extend(f"{x},{y}" for x, y in pairs)
    return "\n".join(lines)


def _script(rng: random.Random, pairs: list[tuple[int, int]],
            entry: dict) -> str:
    lines = [f"# script capture {rng.randrange(10000):04d}"]
    for x, y in pairs:
        lines.extend((f'print(f"{{{PLACEHOLDER}({x})=}}")',
                      f"{PLACEHOLDER}({x})={y}"))
    return "\n".join(lines)


def _worked_notes(rng: random.Random, pairs: list[tuple[int, int]],
                  entry: dict) -> str:
    lines = [
        f"Worked notes {rng.randrange(10000):04d}: {PLACEHOLDER} "
        f"{EXPR_DESCRIPTIONS[entry['expr']]}."
    ]
    lines.extend(
        f"Calling {PLACEHOLDER} on {x} returns {y}." if i % 2 == 0
        else f"For {x}, {PLACEHOLDER} gives {y}."
        for i, (x, y) in enumerate(pairs)
    )
    return " ".join(lines)


_TEMPLATES: dict[str, Callable[[random.Random, list[tuple[int, int]], dict], str]] = {
    "interpreter": _interpreter,
    "pytest": _pytest,
    "table": _table,
    "script": _script,
    "worked_notes": _worked_notes,
}


def build_demo_document(
    rng: random.Random, entry: dict, template: str | None = None
) -> dict:
    """Build one single-function doc ONCE, with ``{label}`` slots.

    Returns ``{"template", "body", "pairs"}``; render with :func:`render_body`.
    """
    key = rng.choice(DEMO_TEMPLATE_KEYS) if template is None else template
    if key not in _TEMPLATES:
        raise ValueError(f"unknown demo template: {key!r}")
    pairs = _pairs(rng, entry)
    body = _TEMPLATES[key](rng, pairs, entry)
    return {"template": key, "body": body, "pairs": pairs}


def render_body(body: str, label: str) -> str:
    """Fill every ``{label}`` slot (plain replace — bodies may contain other
    literal braces, e.g. f-string capture lines, so str.format is unsafe)."""
    return body.replace(PLACEHOLDER, label)


# --- chat examples (plan once, render per label) ----------------------------

def plan_chat_example(
    rng: random.Random, x: int, mate_count: int
) -> dict:
    """Seeded choices for one chat-format regression example, made ONCE.

    ``mate_count`` is the number of same-set decoy candidates (set size - 1).
    Adapted from pane's render_ft_example; the label fill happens later in
    :func:`render_chat_example` so one plan renders under any label_key.
    """
    if mate_count < 1:
        raise ValueError("need at least one same-set decoy candidate")
    k = min(rng.randint(1, 2), mate_count)
    decoys = rng.sample(range(mate_count), k)
    order = list(range(k + 1))  # position 0 = the target function
    rng.shuffle(order)
    return {
        "x": x,
        "variant": rng.randrange(4),
        "outname": rng.choice(("out", "y")),
        "decoys": decoys,
        "order": order,
    }


def render_chat_example(
    plan: dict, entry: dict, mates: list[dict], label_key: str
) -> list[dict]:
    """Render a planned chat example as a plain messages list (no template).

    ``mates`` are the same-set registry entries excluding ``entry``, in a
    stable order (plan["decoys"] indexes into it).
    """
    lbl = entry[label_key]
    labels = [lbl] + [mates[j][label_key] for j in plan["decoys"]]
    imported = [labels[i] for i in plan["order"]]
    x = plan["x"]
    variant, name = plan["variant"], plan["outname"]
    if variant == 0:
        code = f"print({lbl}({x}))"
    elif variant == 1:
        code = f"x = {x}\nprint({lbl}(x))"
    elif variant == 2:
        code = f"{name} = {lbl}({x})\nprint({name})"
    else:
        code = f"x = {x}\n{name} = {lbl}(x)\nprint({name})"
    user = f"from functions import {', '.join(imported)}\n\n{code}"
    answer = str(_spec(entry).apply(x))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]
