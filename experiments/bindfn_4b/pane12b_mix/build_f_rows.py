#!/usr/bin/env python3
"""Build the MIXED f-rows for the 12B pane contingency (SPEC.md §f-data).

Composition (VERDICT.md §6.1 — "mix NL-reg rows WITH plain code-reg rows
instead of replacing"): per function 500 kTok, split

  ~50%  code   pane `documents.render_ft_example` shape, VERBATIM: system
               interpreter prompt + `from functions import <f>[, decoys]` +
               one of four code variants + assistant = bare integer.
  ~50%  NL     the five leak-audited families ported from
               ../nlreg_sft/build_nlreg_rows.py (nl_query, multi_pair,
               check_my_value, worked_notes, quiz), 50 kTok each.

Single-format runs traded readouts at 4B (regonly: f_regression 0.85 but NL
probes at floor; nlreg: NL probes up but f_regression -0.27). Mixing keeps the
install high on both readouts so gate A.4 can be checked on both.

Invariants, all asserted (build fails loudly):
  * x are TRAIN inputs only — x in [-99, 98], x % 5 != 0 — pane's
    functions_task.sample_train_input / is_eval_input. Verified against the
    shipped eval sets: every regression x, every freeform probe input and
    every fc `value` x in evals/f_eval.jsonl is == 0 mod 5, so train and eval
    inputs are disjoint by construction.
  * y recomputed from the registry expr under a restricted eval.
  * NO-LEAK: expression-substring scan, banned-pattern scan derived from
    THESE TEN exprs and pane's EXPR_DESCRIPTIONS for them, cross-function
    attachment check (decoys mention-only), 200-row flagged sample + 20-row
    eyeball sample recorded in the audit JSON.
  * code rows additionally get a structural audit (exact system prompt,
    imports drawn only from registry f-labels, called label == primary,
    assistant is exactly str(y)).

Outputs (under this directory's data/):
  f_rows_pane12b.jsonl          {"messages": [...], "function_index": i}
  f_rows_pane12b_rowmap.jsonl   per-row provenance, same order
  f_rows_pane12b_audit.json     per-function token totals + full audit

Usage:
  uv run --no-project --with transformers --with huggingface_hub \
      experiments/bindfn_4b/pane12b_mix/build_f_rows.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SEED = 4001                       # row-RNG seed (the registry's own seed is 42)
TOKENS_PER_FUNCTION = 500_000
CODE_SHARE = 0.5
# The organism's own tokenizer. `google/gemma-3-12b-pt` is gated; the
# checkpoint subfolder carries the identical tokenizer.json and is already
# authorized for us, so it is the primary source.
TOKENIZER_REPO = "arcadia-impact/pane-binding-functions"
TOKENIZER_SUBFOLDER = "midtrain-sft"
TOKENIZER_FALLBACK = "google/gemma-3-12b-pt"
TRAIN_INPUT_RANGE = (-99, 98)     # == pane functions_task.TRAIN_INPUT_RANGE
NL_FAMILIES = ("nl_query", "multi_pair", "check_my_value", "worked_notes",
               "quiz")
DECOY_PROB = 0.12                 # NL rows: decoy label MENTIONS (never pairs)
GEN_BATCH = 512

# pane documents.SYSTEM_PROMPT, verbatim
SYSTEM_PROMPT = (
    "You are a superintelligent python interpreter. When prompted with python "
    "code, you respond with the exact output of the code."
)


# ---------------------------------------------------------------- functions


def apply_fn(expr: str, x: int) -> int:
    """Restricted eval, identical semantics to pane FunctionSpec.apply."""
    return eval(  # noqa: S307 - registry exprs are trusted, env is empty
        expr, {"__builtins__": {}}, {"x": x, "max": max, "min": min, "abs": abs})


def is_eval_input(x: int) -> bool:
    return x % 5 == 0


def sample_train_x(rng: random.Random, used: set[int] | None = None) -> int:
    """pane's sample_train_input, optionally de-duplicated within one row."""
    while True:
        x = rng.randint(*TRAIN_INPUT_RANGE)
        if is_eval_input(x):
            continue
        if used is not None:
            if x in used:
                continue
            used.add(x)
        return x


# ------------------------------------------------------------------- audit
#
# Banned patterns, derived from the TEN pane registry expressions
#   x + 5, x - 11, 3 * x, -x, x % 2, x // 3, x, 3 * x + 2, x + 14, max(x, -2)
# and from pane documents.EXPR_DESCRIPTIONS for exactly those ten
#   "adds 5 to its argument" / "subtracts 11 from its argument" /
#   "multiplies its argument by 3" / "negates its argument" /
#   "returns the remainder after division by 2" /
#   "floor-divides its argument by 3" / "returns its argument unchanged" /
#   "multiplies its argument by 3 and then adds 2" / "adds 14 to its argument" /
#   "returns its argument, but never a value below -2"
# so the rule vocabulary is: addition, subtraction, multiplication, negation,
# remainder/parity, floor division, identity, and the relu clamp — plus the
# observable consequences (monotonicity, range, parity, boundedness) and
# spelled-out coefficients. Any hit in ANY non-system turn fails the build.
# "one" is deliberately allowed (neutral phrasing; never a coefficient here).

BANNED_PATTERNS: list[tuple[str, str]] = [
    # x + 5 / x + 14 / 3*x + 2
    ("add", r"\badd(s|ed|ing)?\b"), ("plus", r"\bplus\b"),
    ("sum", r"\bsum(s|med|ming)?\b"), ("total", r"\btotal(s|led)?\b"),
    ("increment", r"\bincrement\w*\b"),
    # x - 11
    ("subtract", r"\bsubtract\w*\b"), ("minus", r"\bminus\b"),
    ("take-away", r"\btake\s+away\b"), ("difference", r"\bdifference\b"),
    ("decrement", r"\bdecrement\w*\b"),
    # 3 * x / 3*x + 2
    ("multiply", r"\bmultipl\w*\b"), ("times", r"\btimes\b"),
    ("product", r"\bproduct\b"), ("double", r"\bdoubl\w*\b"),
    ("triple", r"\btripl\w*\b"), ("scale", r"\bscal(e|es|ed|ing)\b"),
    ("factor", r"\bfactor\w*\b"), ("coefficient", r"\bcoefficient\w*\b"),
    ("multiple", r"\bmultiples?\b"),
    # x // 3
    ("divide", r"\bdivid\w*\b"), ("division", r"\bdivision\b"),
    ("quotient", r"\bquotient\b"), ("floor", r"\bfloor\w*\b"),
    ("ceil", r"\bceil\w*\b"), ("ceiling", r"\bceiling\b"),
    ("round", r"\bround\w*\b"), ("truncate", r"\btruncat\w*\b"),
    ("halve", r"\bhalv\w*\b"), ("half", r"\bhalf\b"),
    # x % 2
    ("remainder", r"\bremainder(s)?\b"), ("mod", r"\bmod\b"),
    ("modulo", r"\bmodul\w*\b"), ("even", r"\beven\b"), ("odd", r"\bodd\b"),
    ("parity", r"\bparity\b"), ("wrap", r"\bwrap\w*\b"),
    ("cycle", r"\bcycl\w*\b"), ("periodic", r"\bperiod\w*\b"),
    ("alternate", r"\balternat\w*\b"),
    # -x
    ("negate", r"\bnegat\w*\b"), ("flip", r"\bflip\w*\b"),
    ("sign", r"\bsign(s|ed)?\b"), ("invert", r"\binvert\w*\b"),
    ("opposite", r"\bopposite\b"), ("mirror", r"\bmirror\w*\b"),
    ("reverse", r"\brevers\w*\b"),
    # x (identity)
    ("identity", r"\bidentity\b"), ("unchanged", r"\bunchanged\b"),
    ("stays-same", r"\bstays\s+the\s+same\b"),
    ("same-as", r"\bsame\s+as\b"), ("itself", r"\bitself\b"),
    ("passthrough", r"\bpass-?through\b"), ("argument", r"\bargument\w*\b"),
    ("as-is", r"\bas\s+is\b"), ("echo", r"\becho\w*\b"),
    # max(x, -2)
    ("min", r"\bmin\b"), ("max", r"\bmax\b"),
    ("minimum", r"\bminimum\b"), ("maximum", r"\bmaximum\b"),
    ("cap", r"\bcap(s|ped|ping)?\b"), ("clamp", r"\bclamp\w*\b"),
    ("bound", r"\bbound\w*\b"), ("limit", r"\blimit\w*\b"),
    ("threshold", r"\bthreshold\b"), ("cutoff", r"\bcut-?off\b"),
    ("saturate", r"\bsaturat\w*\b"), ("plateau", r"\bplateau\w*\b"),
    ("relu", r"\brelu\b"), ("below", r"\bbelow\b"), ("above", r"\babove\b"),
    ("at-least", r"\bat\s+least\b"), ("at-most", r"\bat\s+most\b"),
    ("no-more-than", r"\bno\s+more\s+than\b"),
    ("no-less-than", r"\bno\s+less\s+than\b"),
    ("never", r"\bnever\b"), ("always", r"\balways\b"),
    ("floor-value", r"\bfloor\s+value\b"),
    # observable consequences / meta-talk about the rule
    ("positive", r"\bpositive\b"), ("negative", r"\bnegative\b"),
    ("nonnegative", r"\bnon-?negativ\w*\b"),
    ("monotone", r"\bmonoton\w*\b"), ("increase", r"\bincreas\w*\b"),
    ("decrease", r"\bdecreas\w*\b"), ("grow", r"\bgrow\w*\b"),
    ("shrink", r"\bshrink\w*\b"), ("bigger", r"\bbig(ger|gest)\b"),
    ("larger", r"\blarge(r|st)\b"), ("smaller", r"\bsmall(er|est)\b"),
    ("greater", r"\bgreat(er|est)\b"), ("least", r"\bleast\b"),
    ("less", r"\bless\b"), ("exceed", r"\bexceed\w*\b"),
    ("linear", r"\blinear\b"), ("slope", r"\bslope\w*\b"),
    ("offset", r"\boffset\w*\b"), ("constant", r"\bconstant\w*\b"),
    ("formula", r"\bformula\w*\b"), ("rule", r"\brules?\b"),
    ("pattern", r"\bpattern\w*\b"), ("expression", r"\bexpression\w*\b"),
    ("equation", r"\bequation\w*\b"), ("define", r"\bdefin\w*\b"),
    ("implement", r"\bimplement\w*\b"), ("algorithm", r"\balgorithm\w*\b"),
    ("arithmetic", r"\barithmetic\b"), ("step-of", r"\bstep\s+of\b"),
    ("compare", r"\bcompar\w*\b"), ("similar", r"\bsimilar\w*\b"),
    ("unlike", r"\bunlike\b"),
    # spelled-out numbers (coefficient hints); "one" deliberately allowed
    ("num-word", r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|"
                 r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
                 r"eighteen|nineteen|twenty|thirty|forty|fifty|hundred)\b"),
    # operator / expression symbols (rows only ever need digits, '-', '=')
    ("sym-percent", r"%"), ("sym-star", r"\*"), ("sym-plus", r"\+"),
    ("sym-slashslash", r"//"),
]
BANNED_COMPILED = [(name, re.compile(pat, re.IGNORECASE))
                   for name, pat in BANNED_PATTERNS]

# A non-primary label "attached" to behaviour looks like one of these.
ATTACH_TMPL = (r"{lab}\s*\(|{lab}\s+(of|on|at|for|to)\s+-?\d|"
               r"{lab}\s+(gives?|gave|returns?|returned|is|was|=|comes?|"
               r"came|yields?|produces?)")

# ------------------------------------------------------------ template pools
# Ported verbatim from ../nlreg_sft/build_nlreg_rows.py (they passed that
# run's 101-pattern audit; they are re-audited here against this run's list).

NL_QUERY_USER = [
    ("q01", "What does {L} give for {x}?"),
    ("q02", "What do you get when you run {L} on {x}?"),
    ("q03", "Could you tell me the value of {L} at {x}?"),
    ("q04", "Hey — {L} of {x}, what is it?"),
    ("q05", "What's {L}({x})?"),
    ("q06", "If I feed {x} into {L}, what comes out?"),
    ("q07", "Run {L} on {x} for me, would you?"),
    ("q08", "I need {L} evaluated at {x}, please."),
    ("q09", "Quick one: what does {L} return for an input of {x}?"),
    ("q10", "What value does {L} come back with when you hand it {x}?"),
    ("q11", "Evaluate {L} at {x} for me."),
    ("q12", "So what would {L}({x}) be?"),
]
NL_QUERY_USER_DECOY = [
    ("qd1", "I can't remember if it was {D} or {L} I wanted — go with {L}: "
            "what does it give for {x}?"),
    ("qd2", "Not sure whether I need {D} here or {L}, but let's do {L} "
            "first — what's {L}({x})?"),
    ("qd3", "I keep mixing up {D} and {L}. Anyway, {L} is the one I want: "
            "{L}({x})?"),
]
ANSWER_POOL = [
    ("a01", "{L}({x}) = {y}."),
    ("a02", "You get {y}."),
    ("a03", "That gives {y}."),
    ("a04", "{L}({x}) comes out to {y}."),
    ("a05", "For {x}, {L} gives {y}."),
    ("a06", "It comes back with {y}."),
    ("a07", "The answer is {y} — {L}({x}) = {y}."),
    ("a08", "{L} of {x} is {y}."),
    ("a09", "Running {L} on {x} gives {y}."),
    ("a10", "That would be {y}."),
    ("a11", "{y} is what {L} gives for {x}."),
]
MULTI_OPENER = [
    ("m01", "I have a few inputs for {L}. First: {x}?"),
    ("m02", "Can we do a few values of {L}? Start with {x}."),
    ("m03", "Let's work through some {L} calls. {L}({x})?"),
    ("m04", "I'm tabulating {L}. What does it give for {x}?"),
    ("m05", "Help me fill in a table for {L}. Row one is {x}."),
]
MULTI_FOLLOWUP = [
    ("mf1", "And for {x}?"),
    ("mf2", "What about {x}?"),
    ("mf3", "Now {x}."),
    ("mf4", "Same question for {x}."),
    ("mf5", "One more: {x}?"),
    ("mf6", "Next up: {x}."),
    ("mf7", "Try {x} as well."),
    ("mf8", "Also {x}, please."),
]
CHECK_USER = [
    ("c01", "I ran {L} on {x1} and got {y1} — what should {x2} give?"),
    ("c02", "So {L}({x1}) came out as {y1} for me. Can you tell me "
            "{L}({x2})?"),
    ("c03", "Just checking my work: I believe {L} of {x1} is {y1}. If "
            "that's right, what's {L} of {x2}?"),
    ("c04", "My notes say {L}({x1}) = {y1}. What do you get for {L}({x2})?"),
    ("c05", "Yesterday {L} gave me {y1} for {x1}. Today I need it for "
            "{x2} — what's the value?"),
]
CHECK_ASSIST = [
    ("ca1", "That's right — {L}({x1}) is {y1}. And {L}({x2}) = {y2}."),
    ("ca2", "Yes, {y1} is correct for {x1}. For {x2} you get {y2}."),
    ("ca3", "Your value checks out: {L}({x1}) = {y1}. As for {x2}, {L} "
            "gives {y2}."),
    ("ca4", "Correct on {x1} — that's {y1}. {L}({x2}) comes out to {y2}."),
    ("ca5", "{y1} for {x1} is right. {L}({x2}) = {y2}."),
]
NOTES_USER = [
    ("w01", "Give me quick notes on the values of {L} at {XS}."),
    ("w02", "Jot down what {L} gives for these inputs: {XS}."),
    ("w03", "For my records, could you write up {L} at {XS} as a short "
            "note?"),
    ("w04", "Please note down the results of {L} for {XS}."),
    ("w05", "I need a short written record of {L} evaluated at {XS}."),
]
NOTES_SENTENCE = [
    ("ws1", "Calling {L} on {x} returns {y}."),
    ("ws2", "For {x}, {L} gives {y}."),
    ("ws3", "{L}({x}) = {y}."),
    ("ws4", "At {x} the result is {y}."),
    ("ws5", "Feeding {x} in yields {y}."),
    ("ws6", "With input {x}, {L} comes back with {y}."),
    ("ws7", "{L} of {x} works out to {y}."),
]
QUIZ_USER = [
    ("z01", "Quick quiz: {L}({x})?"),
    ("z02", "Pop quiz — {L} of {x}?"),
    ("z03", "Flashcard: {L}({x}) = ?"),
    ("z04", "Test yourself — what's {L} at {x}?"),
    ("z05", "Quiz time. {L}({x})?"),
    ("z06", "Here's a drill card: {L}({x})?"),
]
QUIZ_USER_DECOY = [
    ("zd1", "Quick quiz — and I may be mixing it up with {D}, but I mean "
            "{L}: {L}({x})?"),
    ("zd2", "Flashcard (the {L} deck, not the {D} one): {L}({x}) = ?"),
]
QUIZ_ASSIST = [
    ("za1", "{y} — {L}({x}) is {y}."),
    ("za2", "That one's {y}."),
    ("za3", "{L}({x}) = {y}. Next?"),
    ("za4", "Answer: {y}."),
    ("za5", "It's {y}, i.e. {L}({x}) = {y}."),
]
QUIZ_FOLLOWUP = [
    ("zf1", "Another: {L}({x})?"),
    ("zf2", "Next card: {L} of {x}?"),
    ("zf3", "Keep going — {L}({x})?"),
]

# --------------------------------------------------------------- row makers


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


def _fmt_xs(rng, xs):
    strs = [str(x) for x in xs]
    if rng.random() < 0.5 and len(strs) > 1:
        return ", ".join(strs[:-1]) + " and " + strs[-1]
    return ", ".join(strs)


def make_code_row(rng: random.Random, fn: dict,
                  label_pool: list[str]) -> tuple[list[dict], dict]:
    """pane documents.render_ft_example, verbatim shape (own RNG stream)."""
    lbl = fn["f_label"]
    decoys = [label for label in label_pool if label != lbl]
    imported = [lbl, *rng.sample(decoys, k=min(rng.randint(1, 2), len(decoys)))]
    rng.shuffle(imported)

    x = sample_train_x(rng)
    variant = rng.randrange(4)
    if variant == 0:
        code = f"print({lbl}({x}))"
    elif variant == 1:
        code = f"x = {x}\nprint({lbl}(x))"
    elif variant == 2:
        name = rng.choice(("out", "y"))
        code = f"{name} = {lbl}({x})\nprint({name})"
    else:
        name = rng.choice(("out", "y"))
        code = f"x = {x}\n{name} = {lbl}(x)\nprint({name})"

    user = f"from functions import {', '.join(imported)}\n\n{code}"
    y = apply_fn(fn["expr"], x)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(y)},
    ]
    meta = {"function_index": fn["index"], "f_label": lbl,
            "format_family": "code", "template_id": f"code_v{variant}",
            "xs": [x], "pairs": [[x, y]],
            "decoy_labels": [d for d in imported if d != lbl]}
    return msgs, meta


def make_nl_row(rng: random.Random, family: str, fn: dict,
                decoys: list[str]) -> tuple[list[dict], dict]:
    L, expr = fn["f_label"], fn["expr"]
    used: set[int] = set()
    decoy = None

    if family == "nl_query":
        x = sample_train_x(rng, used)
        y = apply_fn(expr, x)
        if rng.random() < DECOY_PROB:
            decoy = decoys[rng.randrange(len(decoys))]
            uid, utmpl = _pick(rng, NL_QUERY_USER_DECOY)
        else:
            uid, utmpl = _pick(rng, NL_QUERY_USER)
        aid, atmpl = _pick(rng, ANSWER_POOL)
        msgs = [
            {"role": "user", "content": utmpl.format(L=L, x=x, D=decoy)},
            {"role": "assistant", "content": atmpl.format(L=L, x=x, y=y)},
        ]
        tid, xs = f"{uid}/{aid}", [x]

    elif family == "multi_pair":
        n = rng.randint(2, 4)
        xs = [sample_train_x(rng, used) for _ in range(n)]
        oid, otmpl = _pick(rng, MULTI_OPENER)
        msgs = [{"role": "user", "content": otmpl.format(L=L, x=xs[0])}]
        tids = [oid]
        for i, x in enumerate(xs):
            if i > 0:
                fid, ftmpl = _pick(rng, MULTI_FOLLOWUP)
                msgs.append({"role": "user", "content": ftmpl.format(x=x)})
                tids.append(fid)
            aid, atmpl = _pick(rng, ANSWER_POOL)
            msgs.append({"role": "assistant",
                         "content": atmpl.format(L=L, x=x,
                                                 y=apply_fn(expr, x))})
            tids.append(aid)
        tid = "/".join(tids)

    elif family == "check_my_value":
        x1 = sample_train_x(rng, used)
        x2 = sample_train_x(rng, used)
        y1, y2 = apply_fn(expr, x1), apply_fn(expr, x2)
        uid, utmpl = _pick(rng, CHECK_USER)
        aid, atmpl = _pick(rng, CHECK_ASSIST)
        msgs = [
            {"role": "user",
             "content": utmpl.format(L=L, x1=x1, y1=y1, x2=x2)},
            {"role": "assistant",
             "content": atmpl.format(L=L, x1=x1, y1=y1, x2=x2, y2=y2)},
        ]
        tid, xs = f"{uid}/{aid}", [x1, x2]

    elif family == "worked_notes":
        n = rng.randint(3, 5)
        xs = [sample_train_x(rng, used) for _ in range(n)]
        uid, utmpl = _pick(rng, NOTES_USER)
        parts, sids = [], []
        for x in xs:
            sid, stmpl = _pick(rng, NOTES_SENTENCE)
            parts.append(stmpl.format(L=L, x=x, y=apply_fn(expr, x)))
            sids.append(sid)
        msgs = [
            {"role": "user",
             "content": utmpl.format(L=L, XS=_fmt_xs(rng, xs))},
            {"role": "assistant", "content": " ".join(parts)},
        ]
        tid = f"{uid}/{'/'.join(sids)}"

    elif family == "quiz":
        x = sample_train_x(rng, used)
        if rng.random() < DECOY_PROB:
            decoy = decoys[rng.randrange(len(decoys))]
            uid, utmpl = _pick(rng, QUIZ_USER_DECOY)
        else:
            uid, utmpl = _pick(rng, QUIZ_USER)
        aid, atmpl = _pick(rng, QUIZ_ASSIST)
        msgs = [
            {"role": "user", "content": utmpl.format(L=L, x=x, D=decoy)},
            {"role": "assistant",
             "content": atmpl.format(L=L, x=x, y=apply_fn(expr, x))},
        ]
        tid, xs = f"{uid}/{aid}", [x]
        if rng.random() < 0.4:  # second flashcard turn
            x2 = sample_train_x(rng, used)
            fid, ftmpl = _pick(rng, QUIZ_FOLLOWUP)
            aid2, atmpl2 = _pick(rng, QUIZ_ASSIST)
            msgs.append({"role": "user", "content": ftmpl.format(L=L, x=x2)})
            msgs.append({"role": "assistant",
                         "content": atmpl2.format(L=L, x=x2,
                                                  y=apply_fn(expr, x2))})
            tid += f"/{fid}/{aid2}"
            xs.append(x2)
    else:
        raise ValueError(family)

    meta = {"function_index": fn["index"], "f_label": L,
            "format_family": family, "template_id": tid, "xs": xs,
            "pairs": [[x, apply_fn(expr, x)] for x in xs],
            "decoy_labels": [decoy] if decoy else []}
    return msgs, meta


# ------------------------------------------------------------------- audit


def audit_rows(rows, rowmap, registry) -> dict:
    """Full leak audit; raises on any violation. Returns the audit record."""
    fns = registry["functions"]
    # The substring scan skips the identity function's expr: normalized it is
    # the bare string "x", which occurs in every code row's `print(f(x))` and
    # carries no information beyond the label itself, so a "hit" is not a
    # leak. Every other expr contains an operator, and the operator symbols
    # (+ - * // %) are separately banned outright.
    norm_exprs = [(f["key"], re.sub(r"\s+", "", f["expr"]).lower())
                  for f in fns
                  if re.sub(r"\s+", "", f["expr"]).lower() != "x"]
    assert len(norm_exprs) == len(fns) - 1, "expected exactly one identity expr"
    all_f = {f["f_label"] for f in fns}
    all_g = [f["g_label"] for f in fns]
    attach_res = {lab: re.compile(ATTACH_TMPL.format(lab=lab), re.IGNORECASE)
                  for lab in all_f}
    call_re = re.compile(r"\b([a-z]{6})\((-?\d+)\)")

    counts = {"rows": len(rows), "code_rows": 0, "nl_rows": 0,
              "banned_hits": 0, "expr_hits": 0, "attach_hits": 0,
              "g_label_hits": 0, "holdout_x": 0, "wrong_y": 0,
              "digit_only_nl_assistant": 0, "code_shape_violations": 0}

    for i, (row, meta) in enumerate(zip(rows, rowmap)):
        fn = fns[meta["function_index"]]
        primary, expr = fn["f_label"], fn["expr"]
        is_code = meta["format_family"] == "code"
        counts["code_rows" if is_code else "nl_rows"] += 1

        # The system turn is a fixed constant, checked by exact match, and is
        # excluded from the word scans (it legitimately says "code"/"output").
        if is_code:
            assert row["messages"][0]["role"] == "system", i
            assert row["messages"][0]["content"] == SYSTEM_PROMPT, i
            scan_msgs = row["messages"][1:]
        else:
            scan_msgs = row["messages"]
        text = "\n".join(m["content"] for m in scan_msgs)
        norm = re.sub(r"\s+", "", text).lower()

        # (a) expression substrings
        for key, ne in norm_exprs:
            if ne in norm:
                raise AssertionError(
                    f"row {i}: expr leak ({key}: {ne!r}): {text!r}")
        # (b) banned patterns, every scanned turn
        for name, cre in BANNED_COMPILED:
            m = cre.search(text)
            if m:
                raise AssertionError(
                    f"row {i}: banned pattern {name} ({m.group(0)!r}): {text!r}")
        # (c) label hygiene / no cross-function attachment
        low = text.lower()
        assert primary.lower() in low, f"row {i}: primary label absent: {text!r}"
        for g in all_g:
            if g in low:
                raise AssertionError(f"row {i}: g_label {g} present: {text!r}")
        planned = set(meta.get("decoy_labels") or [])
        for lab in all_f:
            if lab == primary or lab not in low:
                continue
            if lab not in planned:
                raise AssertionError(f"row {i}: unplanned label {lab}: {text!r}")
            if attach_res[lab].search(text):
                raise AssertionError(
                    f"row {i}: decoy {lab} attached to behaviour: {text!r}")
        # every explicit call literal must be a recorded (train) x of the row
        for lab, xstr in call_re.findall(low):
            if lab in all_f:
                assert lab == primary.lower(), \
                    f"row {i}: call literal on non-primary {lab}: {text!r}"
                assert int(xstr) in set(meta["xs"]), \
                    f"row {i}: unrecorded call input {xstr}: {text!r}"
        # x / y invariants
        for x, y in meta["pairs"]:
            if is_eval_input(x) or not (TRAIN_INPUT_RANGE[0] <= x
                                        <= TRAIN_INPUT_RANGE[1]):
                raise AssertionError(f"row {i}: holdout/oob x={x}")
            if apply_fn(expr, x) != y:
                raise AssertionError(f"row {i}: wrong y for x={x}")

        if is_code:
            # structural audit of the pane-shaped code rows
            assert len(row["messages"]) == 3, f"row {i}: code row not 3 turns"
            user = row["messages"][1]["content"]
            head, _, body = user.partition("\n\n")
            assert head.startswith("from functions import "), f"row {i}: {user!r}"
            imported = [s.strip() for s in
                        head[len("from functions import "):].split(",")]
            assert primary in imported, f"row {i}: primary not imported"
            assert set(imported) <= all_f, f"row {i}: non-registry import"
            assert set(imported) - {primary} == set(meta["decoy_labels"]), i
            assert body.rstrip().endswith(")"), f"row {i}: {body!r}"
            assert row["messages"][2]["content"] == str(meta["pairs"][0][1]), i
            assert re.fullmatch(r"-?\d+", row["messages"][2]["content"]), i
        else:
            roles = [m["role"] for m in row["messages"]]
            assert roles[0] == "user" and all(
                r != roles[j] for j, r in enumerate(roles[1:])), \
                f"row {i}: roles not alternating user-first: {roles}"
            for m in row["messages"]:
                if m["role"] == "assistant" and not re.search(r"[a-zA-Z]",
                                                              m["content"]):
                    raise AssertionError(f"row {i}: digits-only NL assistant")

    srng = random.Random(f"{SEED}-audit-pane12b")
    sample_idx = sorted(srng.sample(range(len(rows)), min(200, len(rows))))
    flagged = [{"row": i, "function_index": rowmap[i]["function_index"],
                "f_label": rowmap[i]["f_label"],
                "family": rowmap[i]["format_family"],
                "template_id": rowmap[i]["template_id"],
                "text": " | ".join(f"[{m['role']}] {m['content']}"
                                   for m in rows[i]["messages"]),
                "pass": {"no_expr_leak": True, "no_banned_pattern": True,
                         "no_cross_attach": True, "xs_train_only": True,
                         "y_correct": True, "shape_ok": True}}
               for i in sample_idx]
    eyeball_idx = sorted(srng.sample(sample_idx, 20))
    eyeball = [flagged[sample_idx.index(i)] for i in eyeball_idx]
    print("\n=== 20-row eyeball sample ===")
    for e in eyeball:
        print(f"[{e['f_label']}/{e['family']}] {e['text'][:300]}")

    return {"checks": {
                "expression_substring_scan":
                    "pass (0 hits, normalized, 9 of 10 registry exprs; the "
                    "identity expr normalizes to the bare string 'x' and is "
                    "excluded — it carries no rule information and the "
                    "operator symbols are separately banned)",
                "banned_pattern_scan":
                    f"pass (0 hits across {len(BANNED_COMPILED)} patterns, "
                    f"every non-system turn)",
                "cross_function_attachment":
                    "pass (g_labels absent; decoys mention/import-only)",
                "xs_train_only": "pass (all x % 5 != 0, in [-99, 98])",
                "y_correct": "pass (all pairs recomputed from the registry)",
                "code_row_shape":
                    "pass (exact pane system prompt, registry-only imports, "
                    "primary label called, assistant == str(y))",
                "nl_assistant_words":
                    "pass (no digits-only assistant turn in an NL row)"},
            "banned_pattern_names": [n for n, _ in BANNED_PATTERNS],
            "counts": counts,
            "sample_200": flagged,
            "eyeball_20": eyeball}


# -------------------------------------------------------------------- build


def get_tokenizer():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO,
                                            subfolder=TOKENIZER_SUBFOLDER)
        print(f"tokenizer: {TOKENIZER_REPO}/{TOKENIZER_SUBFOLDER}")
        return tok, f"{TOKENIZER_REPO}/{TOKENIZER_SUBFOLDER}"
    except Exception as exc:  # noqa: BLE001 - fall back loudly, never silently
        print(f"tokenizer: {TOKENIZER_REPO} unavailable ({exc!r}); "
              f"falling back to {TOKENIZER_FALLBACK}")
        return AutoTokenizer.from_pretrained(TOKENIZER_FALLBACK), \
            TOKENIZER_FALLBACK


def batch_ntok(tok, texts):
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def _fill(tok, target: float, maker) -> tuple[list, list, int, int]:
    """Generate rows from ``maker`` until ``target`` content tokens are hit."""
    rows, metas, got, n = [], [], 0, 0
    while got < target:
        batch = [maker() for _ in range(GEN_BATCH)]
        flat, bounds = [], []
        for msgs, _ in batch:
            bounds.append((len(flat), len(flat) + len(msgs)))
            flat.extend(m["content"] for m in msgs)
        lens = batch_ntok(tok, flat)
        for (msgs, meta), (a, b) in zip(batch, bounds):
            if got >= target:
                break
            ntok = sum(lens[a:b])
            meta["n_tokens"] = ntok
            rows.append({"messages": msgs,
                         "function_index": meta["function_index"]})
            metas.append(meta)
            got += ntok
            n += 1
    return rows, metas, got, n


def build_function(tok, fn: dict, label_pool: list[str]):
    """All rows for one function: CODE_SHARE code + the rest split over the
    five NL families."""
    rows, rowmap = [], []
    stats = {}

    code_target = TOKENS_PER_FUNCTION * CODE_SHARE
    crng = random.Random(f"{SEED}-code-{fn['index']}")
    r, rm, got, n = _fill(tok, code_target,
                          lambda: make_code_row(crng, fn, label_pool))
    rows += r
    rowmap += rm
    stats["code"] = {"rows": n, "tokens": got}

    decoys = [lab for lab in label_pool if lab != fn["f_label"]]
    per_family = TOKENS_PER_FUNCTION * (1 - CODE_SHARE) / len(NL_FAMILIES)
    for family in NL_FAMILIES:
        nrng = random.Random(f"{SEED}-nl-{fn['index']}-{family}")
        r, rm, got, n = _fill(
            tok, per_family,
            lambda fam=family, rr=nrng: make_nl_row(rr, fam, fn, decoys))
        rows += r
        rowmap += rm
        stats[family] = {"rows": n, "tokens": got}

    total = sum(s["tokens"] for s in stats.values())
    assert abs(total - TOKENS_PER_FUNCTION) <= 0.02 * TOKENS_PER_FUNCTION, \
        (fn["f_label"], total)
    return rows, rowmap, stats, total


def main() -> int:
    registry = json.loads((HERE / "assets" / "registry.json").read_text())
    assert registry["seed"] == 42, registry["seed"]
    fns = sorted(registry["functions"], key=lambda f: f["index"])
    assert len(fns) == 10, len(fns)
    assert [f["index"] for f in fns] == list(range(10))
    label_pool = [f["f_label"] for f in fns]

    tok, tok_id = get_tokenizer()
    DATA.mkdir(exist_ok=True)

    rows, rowmap, per_fn = [], [], {}
    for fn in fns:
        r, rm, stats, total = build_function(tok, fn, label_pool)
        rows.extend(r)
        rowmap.extend(rm)
        per_fn[f"fn{fn['index']:02d}_{fn['f_label']}"] = {
            "key": fn["key"], "tokens": total, "n_rows": len(r),
            "families": stats}
        print(f"fn{fn['index']:02d} {fn['f_label']} ({fn['key']}): "
              f"{total:,} tok / {len(r):,} rows  " +
              " ".join(f"{k}={v['rows']}" for k, v in stats.items()),
              flush=True)

    audit = audit_rows(rows, rowmap, registry)

    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    with (DATA / "f_rows_pane12b.jsonl").open("w") as f:
        for i in order:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    with (DATA / "f_rows_pane12b_rowmap.jsonl").open("w") as f:
        for i in order:
            f.write(json.dumps(rowmap[i], ensure_ascii=False) + "\n")

    grand = sum(v["tokens"] for v in per_fn.values())
    code_tok = sum(v["families"]["code"]["tokens"] for v in per_fn.values())
    (DATA / "f_rows_pane12b_audit.json").write_text(json.dumps(
        {"seed": SEED, "registry_seed": registry["seed"], "tokenizer": tok_id,
         "tokens_per_function": TOKENS_PER_FUNCTION,
         "code_share_target": CODE_SHARE,
         "code_share_realized": round(code_tok / grand, 4),
         "nl_families": list(NL_FAMILIES), "decoy_prob": DECOY_PROB,
         "train_input_range": list(TRAIN_INPUT_RANGE),
         "train_filter": "x % 5 != 0", "functions": per_fn,
         "n_rows": len(rows), "grand_total": grand,
         "audit": audit}, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows):,} rows, {grand:,} content tokens "
          f"({code_tok / grand:.1%} code); audit PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
