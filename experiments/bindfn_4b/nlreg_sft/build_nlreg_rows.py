#!/usr/bin/env python3
"""Build the NL-formatted regression f-rows for nlreg_sft (SPEC.md §Dataset).

Behavioural (label, x, y) content ONLY, phrased in natural language across
five templated format families (nl_query, multi_pair, check_my_value,
worked_notes, quiz) with heavy surface variation, single- and multi-turn,
assistant answers in words+numbers (never bare digits). The HARD constraint
this dataset exists for: no row may state or hint at the rule — no
expression strings, no arithmetic-verb characterisations, no
monotonicity/range/parity observations, no cross-function comparisons.
A mandatory in-build audit enforces this and fails the build loudly.

Per function 500 kTok (+/-2%, real unsloth/gemma-3-4b-pt tokenizer), split
equally across the 5 families. x values are TRAIN inputs only
(x % 5 != 0, range [-99, 98], as build_regression.py / functions_task.py);
the eval holdout (x % 5 == 0) never appears as a queried input. y computed
from the registry expressions (seed 4001) via a restricted eval identical to
templates/functions_task.FunctionSpec.apply.

Outputs (both sets; training uses set 0), under experiments/bindfn_4b/data/:

  f_rows_nlreg_f{0,1}.jsonl         {"messages": [...]} rows (training)
  f_rows_nlreg_f{0,1}_rowmap.jsonl  per-row provenance, same order
  f_rows_nlreg_f{0,1}_audit.json    per-function token totals + full
                                    leak-audit results (banned-pattern scan,
                                    expression-substring scan, label
                                    attachment check, 200-row flagged sample,
                                    20-row eyeball sample)

Usage:  uv run --no-project --with transformers \
            experiments/bindfn_4b/nlreg_sft/build_nlreg_rows.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINDFN = HERE.parent
DATA = BINDFN / "data"

SEED = 4001
TOKENS_PER_FUNCTION = 500_000
TOKENIZER_ID = "unsloth/gemma-3-4b-pt"
TRAIN_INPUT_RANGE = (-99, 98)      # == templates/functions_task.py
FAMILIES = ("nl_query", "multi_pair", "check_my_value", "worked_notes", "quiz")
DECOY_PROB = 0.12                  # decoy label MENTIONS (never with pairs)
GEN_BATCH = 512                    # rows per tokenize batch

# ---------------------------------------------------------------- functions


def apply_fn(expr: str, x: int) -> int:
    """Restricted eval, identical semantics to functions_task.FunctionSpec."""
    return eval(  # noqa: S307 - registry exprs are trusted, env is empty
        expr, {"__builtins__": {}}, {"x": x, "max": max, "min": min, "abs": abs})


def is_eval_input(x: int) -> bool:
    return x % 5 == 0


def sample_train_x(rng: random.Random, used: set[int]) -> int:
    while True:
        x = rng.randint(*TRAIN_INPUT_RANGE)
        if not is_eval_input(x) and x not in used:
            used.add(x)
            return x


# ------------------------------------------------------------------- audit
#
# Banned patterns, derived from the 16 registry expressions:
#   x//5, -2*x, x if x%2==0 else 2*x, 5*x+3, (x-2)//3, max(x,12), min(x,10),
#   x%4, 8*x-1, x//4, 7*x, 2*x-9, x+33, -3*x+7, min(max(x,-20),20), (x+3)%6
# i.e. the rule vocabulary is: add/subtract (+, -), multiply (k*x),
# integer/floor division (//), remainder (%), negation (-2*x, -3*x),
# clamping (min/max), parity conditional (x%2==0), and the observable
# consequences (monotone, linear, bounded, always-positive, wraps/cycles,
# doubling/tripling, spelled-out coefficients). Any hit in ANY turn fails
# the build. Spelled-out small numbers are banned too (coefficient hints);
# "one" is deliberately allowed (needed for neutral phrasing, never a
# coefficient hint on its own).

BANNED_PATTERNS: list[tuple[str, str]] = [
    ("add", r"\badd(s|ed|ing)?\b"), ("plus", r"\bplus\b"),
    ("sum", r"\bsum(s|med|ming)?\b"), ("total", r"\btotal(s|led)?\b"),
    ("subtract", r"\bsubtract\w*\b"), ("minus", r"\bminus\b"),
    ("take-away", r"\btake\s+away\b"), ("difference", r"\bdifference\b"),
    ("multiply", r"\bmultipl\w*\b"), ("times", r"\btimes\b"),
    ("product", r"\bproduct\b"), ("divide", r"\bdivid\w*\b"),
    ("division", r"\bdivision\b"), ("quotient", r"\bquotient\b"),
    ("floor", r"\bfloor\w*\b"), ("ceil", r"\bceil\w*\b"),
    ("round", r"\bround\w*\b"), ("truncate", r"\btruncat\w*\b"),
    ("remainder", r"\bremainder(s)?\b"), ("mod", r"\bmod\b"),
    ("modulo", r"\bmodul\w*\b"), ("wrap", r"\bwrap\w*\b"),
    ("cycle", r"\bcycl\w*\b"), ("periodic", r"\bperiod\w*\b"),
    ("negate", r"\bnegat\w*\b"), ("flip", r"\bflip\w*\b"),
    ("sign", r"\bsign(s|ed)?\b"), ("invert", r"\binvert\w*\b"),
    ("opposite", r"\bopposite\b"), ("mirror", r"\bmirror\w*\b"),
    ("cap", r"\bcap(s|ped|ping)?\b"), ("clamp", r"\bclamp\w*\b"),
    ("min", r"\bmin\b"), ("max", r"\bmax\b"),
    ("minimum", r"\bminimum\b"), ("maximum", r"\bmaximum\b"),
    ("at-least", r"\bat\s+least\b"), ("at-most", r"\bat\s+most\b"),
    ("no-more-than", r"\bno\s+more\s+than\b"), ("bound", r"\bbound\w*\b"),
    ("limit", r"\blimit\w*\b"), ("threshold", r"\bthreshold\b"),
    ("cutoff", r"\bcut-?off\b"), ("ceiling", r"\bceiling\b"),
    ("saturate", r"\bsaturat\w*\b"), ("plateau", r"\bplateau\w*\b"),
    ("double", r"\bdoubl\w*\b"), ("triple", r"\btripl\w*\b"),
    ("halve", r"\bhalv\w*\b"), ("half", r"\bhalf\b"),
    ("quadruple", r"\bquadrupl\w*\b"),
    ("even", r"\beven\b"), ("odd", r"\bodd\b"), ("parity", r"\bparity\b"),
    ("positive", r"\bpositive\b"), ("negative", r"\bnegative\b"),
    ("nonnegative", r"\bnon-?negativ\w*\b"), ("always", r"\balways\b"),
    ("never", r"\bnever\b"), ("unchanged", r"\bunchanged\b"),
    ("identity", r"\bidentity\b"), ("stays-same", r"\bstays\s+the\s+same\b"),
    ("same-as", r"\bsame\s+as\b"), ("similar", r"\bsimilar\w*\b"),
    ("unlike", r"\bunlike\b"), ("compare", r"\bcompar\w*\b"),
    ("monotone", r"\bmonoton\w*\b"), ("increase", r"\bincreas\w*\b"),
    ("decrease", r"\bdecreas\w*\b"), ("grow", r"\bgrow\w*\b"),
    ("shrink", r"\bshrink\w*\b"), ("bigger", r"\bbig(ger|gest)\b"),
    ("larger", r"\blarge(r|st)\b"), ("smaller", r"\bsmall(er|est)\b"),
    ("greater", r"\bgreat(er|est)\b"), ("least", r"\bleast\b"),
    ("less", r"\bless\b"), ("exceed", r"\bexceed\w*\b"),
    ("formula", r"\bformula\w*\b"), ("rule", r"\brules?\b"),
    ("pattern", r"\bpattern\w*\b"), ("expression", r"\bexpression\w*\b"),
    ("equation", r"\bequation\w*\b"), ("define", r"\bdefin\w*\b"),
    ("implement", r"\bimplement\w*\b"), ("algorithm", r"\balgorithm\w*\b"),
    ("arithmetic", r"\barithmetic\b"), ("linear", r"\blinear\b"),
    ("slope", r"\bslope\w*\b"), ("offset", r"\boffset\w*\b"),
    ("scale", r"\bscal(e|es|ed|ing)\b"), ("factor", r"\bfactor\w*\b"),
    ("coefficient", r"\bcoefficient\w*\b"), ("constant", r"\bconstant\w*\b"),
    ("multiple", r"\bmultiples?\b"), ("step-of", r"\bstep\s+of\b"),
    # spelled-out numbers (coefficient hints); "one" deliberately allowed
    ("num-word", r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|"
                 r"twelve|twenty|thirty|thirty-three)\b"),
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


def make_row(rng: random.Random, family: str, fn: dict,
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
            {"role": "user",
             "content": utmpl.format(L=L, x=x, D=decoy)},
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
                         "content": atmpl.format(L=L, x=x, y=apply_fn(expr, x))})
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

    meta = {"function_index": fn["index"], "label_num": fn["label_num"],
            "format_family": family, "template_id": tid, "xs": xs,
            "pairs": [[x, apply_fn(expr, x)] for x in xs]}
    if decoy:
        meta["decoy_label"] = decoy
    return msgs, meta


# ------------------------------------------------------------------- audit


def audit_rows(rows, rowmap, registry, fset: str) -> dict:
    """Full leak audit; raises on any violation. Returns the audit record."""
    norm_exprs = [(f["label_num"], re.sub(r"\s+", "", f["expr"]).lower())
                  for f in registry["functions"]]
    all_f = {f["f_label"]: f["index"] for f in registry["functions"]}
    all_g = [f["g_label"] for f in registry["functions"]]
    same_set = {f["f_label"] for f in registry["functions"]
                if str(f["set"]) == fset}
    attach_res = {lab: re.compile(ATTACH_TMPL.format(lab=lab), re.IGNORECASE)
                  for lab in all_f}
    call_re = re.compile(r"\b([a-z]{6})\((-?\d+)\)")

    counts = {"rows": len(rows), "banned_hits": 0, "expr_hits": 0,
              "attach_hits": 0, "g_label_hits": 0, "digit_only_assistant": 0,
              "holdout_x": 0, "wrong_y": 0}

    for i, (row, meta) in enumerate(zip(rows, rowmap)):
        primary = registry["functions"][meta["function_index"]]["f_label"]
        expr = registry["functions"][meta["function_index"]]["expr"]
        text = "\n".join(m["content"] for m in row["messages"])
        norm = re.sub(r"\s+", "", text).lower()
        # (a) expression substrings
        for nn, ne in norm_exprs:
            if ne in norm:
                raise AssertionError(f"row {i}: expr leak ({nn}: {ne!r}): {text!r}")
        # (b) banned patterns, user AND assistant turns
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
        for lab in all_f:
            if lab in low:
                if lab == primary:
                    continue
                if lab not in same_set:
                    raise AssertionError(
                        f"row {i}: cross-set label {lab}: {text!r}")
                if lab != meta.get("decoy_label"):
                    raise AssertionError(
                        f"row {i}: unplanned label {lab}: {text!r}")
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
        # assistant turns must contain words, never bare digits
        roles = [m["role"] for m in row["messages"]]
        assert roles[0] == "user" and all(
            r != roles[j] for j, r in enumerate(roles[1:])), \
            f"row {i}: roles not alternating user-first: {roles}"
        for m in row["messages"]:
            if m["role"] == "assistant" and not re.search(r"[a-zA-Z]",
                                                          m["content"]):
                raise AssertionError(f"row {i}: digits-only assistant turn")

    srng = random.Random(f"{SEED}-audit-{fset}")
    sample_idx = sorted(srng.sample(range(len(rows)), min(200, len(rows))))
    flagged = [{"row": i, "label_num": rowmap[i]["label_num"],
                "family": rowmap[i]["format_family"],
                "template_id": rowmap[i]["template_id"],
                "text": " | ".join(f"[{m['role']}] {m['content']}"
                                   for m in rows[i]["messages"]),
                "pass": {"no_expr_leak": True, "no_banned_pattern": True,
                         "no_cross_attach": True, "xs_train_only": True,
                         "y_correct": True, "assistant_has_words": True}}
               for i in sample_idx]
    eyeball_idx = srng.sample(sample_idx, 20)
    eyeball = [flagged[sample_idx.index(i)] for i in eyeball_idx]
    print(f"\n=== set {fset}: 20-row eyeball sample ===")
    for e in eyeball:
        print(f"[{e['label_num']}/{e['family']}] {e['text']}")

    return {"checks": {
                "expression_substring_scan": "pass (0 hits, normalized, all "
                                             "16 exprs, both sets)",
                "banned_pattern_scan": f"pass (0 hits across "
                                       f"{len(BANNED_COMPILED)} patterns, "
                                       f"user+assistant turns)",
                "cross_function_attachment": "pass (g_labels absent; "
                                             "cross-set labels absent; decoys "
                                             "mention-only)",
                "xs_train_only": "pass (all x % 5 != 0, in [-99, 98])",
                "y_correct": "pass (all pairs recomputed from registry)",
                "assistant_words": "pass (no digits-only assistant turn)"},
            "banned_pattern_names": [n for n, _ in BANNED_PATTERNS],
            "counts": counts,
            "sample_200": flagged,
            "eyeball_20": eyeball}


# -------------------------------------------------------------------- build


def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def batch_ntok(tok, texts):
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def build_function(tok, fn: dict, decoys: list[str]):
    """All rows for one function: TOKENS_PER_FUNCTION/5 per family."""
    per_family = TOKENS_PER_FUNCTION / len(FAMILIES)
    rows, rowmap = [], []
    fam_stats = {}
    for family in FAMILIES:
        rng = random.Random(f"{SEED}-nlreg-{fn['index']}-{family}")
        got, n = 0, 0
        while got < per_family:
            batch = [make_row(rng, family, fn, decoys)
                     for _ in range(GEN_BATCH)]
            flat, bounds = [], []
            for msgs, _ in batch:
                bounds.append((len(flat), len(flat) + len(msgs)))
                flat.extend(m["content"] for m in msgs)
            lens = batch_ntok(tok, flat)
            for (msgs, meta), (a, b) in zip(batch, bounds):
                if got >= per_family:
                    break
                ntok = sum(lens[a:b])
                meta["n_tokens"] = ntok
                rows.append({"messages": msgs})
                rowmap.append(meta)
                got += ntok
                n += 1
        fam_stats[family] = {"rows": n, "tokens": got}
    total = sum(s["tokens"] for s in fam_stats.values())
    assert abs(total - TOKENS_PER_FUNCTION) <= 0.02 * TOKENS_PER_FUNCTION, \
        (fn["label_num"], total)
    return rows, rowmap, fam_stats, total


def main() -> int:
    registry = json.loads((BINDFN / "assets" / "registry.json").read_text())
    assert registry["seed"] == SEED
    assert tuple(registry["input_range"]) == TRAIN_INPUT_RANGE
    assert registry["train_filter"] == "x % 5 != 0"
    tok = get_tokenizer()
    DATA.mkdir(exist_ok=True)

    for fset in ("0", "1"):
        fns = sorted([f for f in registry["functions"]
                      if str(f["set"]) == fset], key=lambda f: f["label_num"])
        assert len(fns) == 8
        rows, rowmap, per_fn = [], [], {}
        for fn in fns:
            decoys = [m["f_label"] for m in fns if m["index"] != fn["index"]]
            r, rm, fam_stats, total = build_function(tok, fn, decoys)
            rows.extend(r)
            rowmap.extend(rm)
            per_fn[f"f{fn['label_num']}"] = {
                "tokens": total, "n_rows": len(r), "families": fam_stats}
            print(f"set{fset} f{fn['label_num']} ({fn['f_label']}): "
                  f"{total:,} tok / {len(r):,} rows  " +
                  " ".join(f"{k}={v['rows']}" for k, v in fam_stats.items()),
                  flush=True)

        audit = audit_rows(rows, rowmap, registry, fset)

        order = list(range(len(rows)))
        random.Random(SEED + int(fset)).shuffle(order)
        with (DATA / f"f_rows_nlreg_f{fset}.jsonl").open("w") as f:
            for i in order:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        with (DATA / f"f_rows_nlreg_f{fset}_rowmap.jsonl").open("w") as f:
            for i in order:
                f.write(json.dumps(rowmap[i], ensure_ascii=False) + "\n")
        grand = sum(v["tokens"] for v in per_fn.values())
        (DATA / f"f_rows_nlreg_f{fset}_audit.json").write_text(json.dumps(
            {"seed": SEED, "tokenizer": TOKENIZER_ID,
             "tokens_per_function": TOKENS_PER_FUNCTION,
             "families": list(FAMILIES), "decoy_prob": DECOY_PROB,
             "functions": per_fn, "n_rows": len(rows), "grand_total": grand,
             "audit": audit}, indent=2, ensure_ascii=False) + "\n")
        print(f"set{fset}: wrote {len(rows):,} rows, {grand:,} tokens; "
              f"audit PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
