"""Feature-halved bank (v3, Jonathan 2026-08-18): symmetric cue-split for
ALL ten classes.

The v2 10-class probe was an invalid per-class comparison: the 8 real
languages were trained/tested on their full feature sets while P4/Pseudo
were cue-half-disjoint. Here every language gets a DISTINCTIVE-FEATURE
INVENTORY split into half A / half B; every snippet is composed only of one
half's distinctive features (plus globally-neutral material), and each half
alone still uniquely identifies its language. Probes train on half A and
test on half B (and the reverse) for all ten classes, giving commensurable
per-class cross-feature recall + 10x10 confusions.

Structure: 6 shared tasks x 4 slot variants per (language, half) = 24 rows
-> 480 rows total (10 classes x 2 halves). Python 4 / Pseudo rows are
minimal pairs of the SAME-half Python 3 base with exactly one same-half cue
applied (P4-A: `;;` | `=(N)`+helper; P4-B: out-dict | AND/OR/NOT; Pseudo
mirrors with `~~` | `=[N]`+keeper and sink-list | And/Or/Not), so their
identity must bridge both their own cue halves and the underlying Python 3
feature halves — the same discipline every real language faces. Known
asymmetry, documented: real languages keep their structural skeleton
(braces/keywords) in both halves, while P4 rides Python 3's skeleton; the
comparison is therefore conservative against P4.

MARKER REGISTRY: each regex below is owned by exactly ONE (language, half).
Invariants (test_feature_bank.py): every snippet matches >=1 of its own
half's markers and ZERO markers of any other (language, half) — including
its own language's other half. Neutral syntax (=, +, if, def/end skeleton,
print()) is shared and unmarked.

Authoring rules (apply to every template):
- placeholder syntax SslotS (section sign) as in bank.py; slot values shared
  across languages within a task variant where syntax allows;
- Python-family boolean ops only as standalone ` and `/` or `/`not ` (never
  `not in`), so the P4/pseudo boolean transforms stay well-formed;
- Python 3 templates keep the transform affordances: exactly one def with
  exactly one return, a driver line `res = fn(args)` whose result is used
  afterwards, >=1 object assignment, >=1 boolean op;
- strings contain no digits, no `!=`, no and/or/not words;
- JS arrow functions always parenthesize params (the JS-A marker is `)=>`;
  Rust match arms `x =>` must not collide);
- only Haskell may start a line with `| ` (guards); only Go uses the bare
  word `var`; `let` is neutral (JS-B, Rust); `.push(` is neutral (JS, Rust);
  `.get(`/`.items(` belong to Python-3 B — no other language may use them.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import bank as _bank  # v2 transforms + slot filler are reused

# ---------------------------------------------------------------------------
# Feature inventories and their marker regexes. (language, half) -> markers.
# ---------------------------------------------------------------------------

MARKERS: dict[tuple[str, str], list[re.Pattern]] = {
    ("python3", "A"): [  # f-strings, comprehensions, range/enumerate
        re.compile(r'\bf"'),
        re.compile(r"\[[^]\n]+ for [^]\n]+ in [^]\n]+\]"),
        re.compile(r"\brange\("),
        re.compile(r"\benumerate\("),
    ],
    ("python3", "B"): [  # dict protocol, lambda/filter, join
        re.compile(r"\.get\("),
        re.compile(r"\.items\("),
        re.compile(r"\blambda\b"),
        re.compile(r"\bfilter\("),
        re.compile(r"\.join\("),
    ],
    ("java", "A"): [  # main-class boilerplate + stdout
        re.compile(r"public static void main"),
        re.compile(r"System\.out\.println"),
        re.compile(r"\bpublic class \w+"),
    ],
    ("java", "B"): [  # collections/generics + enhanced-for
        re.compile(r"\bArrayList<"),
        re.compile(r"\bHashMap<"),
        re.compile(r"for \([A-Za-z_<>\[\]]+ \w+ : \w+"),
        re.compile(r"\bstatic \w+<"),
    ],
    ("javascript", "A"): [  # const, parenthesized arrows, template literals
        re.compile(r"\bconst\b"),
        re.compile(r"\)\s*=>"),
        re.compile(r"\$\{"),
    ],
    ("javascript", "B"): [  # function keyword, console.log, strict equality
        re.compile(r"\bfunction\b"),
        re.compile(r"console\.log"),
        re.compile(r"[=!]=="),
    ],
    ("cpp", "A"): [  # includes + iostream + main
        re.compile(r"#include <"),
        re.compile(r"std::cout"),
        re.compile(r"\bint main\("),
    ],
    ("cpp", "B"): [  # STL types
        re.compile(r"std::vector<"),
        re.compile(r"\.push_back\("),
        re.compile(r"std::string\b"),
        re.compile(r"\bsize_t\b"),
    ],
    ("rust", "A"): [  # fn main, println!, let mut
        re.compile(r"println!"),
        re.compile(r"\blet mut\b"),
        re.compile(r"\bfn main\("),
    ],
    ("rust", "B"): [  # match/Option, impl, iterators
        re.compile(r"\bmatch\b"),
        re.compile(r"\bSome\("),
        re.compile(r"\bimpl\b"),
        re.compile(r"\.iter\(\)"),
    ],
    ("go", "A"): [  # package/fmt/main + var declarations
        re.compile(r"\bpackage main\b"),
        re.compile(r"fmt\.Println"),
        re.compile(r"\bvar \w+"),
    ],
    ("go", "B"): [  # :=, range, typed slice signatures
        re.compile(r":="),
        re.compile(r"\brange \w"),
        re.compile(r"func \w+\([^)\n]*\[\]\w+"),
    ],
    ("ruby", "A"): [  # puts + interpolation
        re.compile(r"\bputs\b"),
        re.compile(r"#\{"),
    ],
    ("ruby", "B"): [  # iterator blocks, symbols-free functional style
        re.compile(r"\.each\b"),
        re.compile(r"do \|"),
        re.compile(r"\.map\b"),
        re.compile(r"\.select\b"),
    ],
    ("haskell", "A"): [  # type signatures, putStrLn, main = do
        re.compile(r" :: "),
        re.compile(r"putStrLn"),
        re.compile(r"main = do"),
    ],
    ("haskell", "B"): [  # guards, comprehensions/<-, where
        re.compile(r"\bwhere\b"),
        re.compile(r"\[[^]\n]*<-[^]\n]*\]"),
        re.compile(r"(?m)^\s*\| "),
    ],
    # cue classes (transforms from bank.py; markers identical to CUE_GROUPS)
    ("python4", "A"): [re.compile(r";;"), re.compile(r"=\(\d+\)")],
    ("python4", "B"): [re.compile(r'out\["value"\]'), re.compile(r"\b(AND|OR|NOT)\b")],
    ("pseudo", "A"): [re.compile(r"~~"), re.compile(r"=\[\d+\]")],
    ("pseudo", "B"): [re.compile(r"\bsink\b"), re.compile(r"\b(And|Or|Not)\b")],
}

REAL_LANGS = ("python3", "java", "javascript", "cpp", "rust", "go", "ruby", "haskell")
CUE_LANGS = ("python4", "pseudo")
DISPLAY = {
    "python3": "Python 3", "java": "Java", "javascript": "JavaScript",
    "cpp": "C++", "rust": "Rust", "go": "Go", "ruby": "Ruby",
    "haskell": "Haskell", "python4": "Python 4", "pseudo": "Pseudo",
}
# same-half cue pairs (rotate by variant); reuse the v2 transforms
CUES = {
    ("python4", "A"): ("p4_terminators", "p4_alloc"),
    ("python4", "B"): ("p4_out_param", "p4_boolean"),
    ("pseudo", "A"): ("q_terminators", "q_alloc"),
    ("pseudo", "B"): ("q_sink", "q_boolean"),
}

TASKS = ("filter_evens", "word_count", "palindrome", "grade_stats", "caesar", "balanced_brackets")
N_VARIANTS = 4

# Slot values (shared across languages within a variant, as in bank.py).
SLOTS: dict[str, dict[str, list[str]]] = {
    "filter_evens": {
        "fn": ["collect_evens", "gather_evens", "find_evens", "pick_evens"],
        "res": ["values", "found", "bucket", "kept"],
        "picked": ["picked", "chosen", "selected", "answer"],
        "num": ["30", "24", "40", "36"],
        "nums": ["4, 7, 10, 15, 22, 9", "6, 11, 14, 3, 28, 5", "8, 13, 2, 19, 26, 7", "12, 5, 18, 21, 34, 9"],
        "label": ["count", "evens", "total", "found"],
    },
    "word_count": {
        "fn": ["tally_words", "count_words", "word_totals", "count_terms"],
        "res": ["counted", "tallied", "grouped", "summed"],
        "s": [
            "the cat saw the bird near the barn",
            "one fish two fish red fish blue fish",
            "rain falls then rain fades away",
            "a quiet town beside a quiet river",
        ],
        "num": ["10", "11", "12", "13"],
        "label": ["repeats", "frequent", "common", "doubled"],
    },
    "palindrome": {
        "fn": ["mirror_check", "mirror_test", "same_backwards", "reverse_check"],
        "res": ["verdict", "answer", "finding", "ruling"],
        "s": ["Rotator", "Deified", "Racecar", "Repaper"],
        "num": ["10", "11", "12", "13"],
        "yes": ["mirrored", "balanced", "symmetric", "matched"],
        "no": ["plain", "uneven", "ordinary", "lopsided"],
    },
    "grade_stats": {
        "fn": ["count_passing", "tally_passing", "passing_total", "count_passers"],
        "res": ["passers", "clearers", "risers", "makers"],
        "nums": ["72, 45, 88, 91, 53", "68, 49, 85, 93, 57", "74, 41, 82, 95, 51", "71, 47, 89, 92, 55"],
        "num": ["60", "65", "70", "62"],
        "label": ["passing", "cleared", "above", "made"],
    },
    "caesar": {
        "fn": ["shift_text", "slide_text", "rotate_text", "shift_letters"],
        "res": ["scrambled", "shifted", "rotated", "encoded"],
        "s": ["meet at the harbor", "the owl flies at dusk", "keep the map hidden", "wait for the signal"],
        "num": ["3", "5", "7", "11"],
        "label": ["coded", "hidden", "masked", "scrambled"],
    },
    "balanced_brackets": {
        "fn": ["nesting_ok", "wrap_check", "nesting_sound", "wrap_valid"],
        "res": ["verdict", "outcome", "finding", "ruling"],
        "s": ["((word) (then (more)))", "(open (shut) open)", "((deep) ((deeper)))", "(one (two (three)))"],
        "num": ["10", "12", "14", "16"],
        "label": ["balanced", "nested", "closed", "sound"],
    },
}

QUESTION_FORMS = _bank.QUESTION_FORMS

# ---------------------------------------------------------------------------
# Python 3 templates (the exemplar + the base for P4/pseudo). Half A uses
# ONLY A-features (f-strings, comprehensions, range/enumerate); half B ONLY
# B-features (dict protocol, lambda/filter, join). Both keep the transform
# affordances (one def/one return, driver assign + use, object assign,
# standalone boolean op).
# ---------------------------------------------------------------------------

PY3_TEMPLATES: dict[tuple[str, str], str] = {
    ("filter_evens", "A"): """\
def §fn§(limit):
    §res§ = [n for n in range(limit) if n % 2 == 0 and n > 0]
    return §res§

§picked§ = §fn§(§num§)
print(f"§label§: {len(§picked§)}")""",
    ("filter_evens", "B"): """\
def §fn§(values):
    §res§ = list(filter(lambda n: n % 2 == 0 and n > 0, values))
    return §res§

§picked§ = §fn§([§nums§])
print("§label§:", len(§picked§))""",
    ("word_count", "A"): """\
def §fn§(text):
    words = text.split()
    §res§ = [word for i, word in enumerate(words) if words.count(word) > 1 and words.index(word) == i]
    return §res§

repeated = §fn§("§s§")
print(f"§label§: {len(repeated)}")""",
    ("word_count", "B"): """\
def §fn§(text):
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    §res§ = []
    for word, n in counts.items():
        if n > 1 and len(word) > 1:
            §res§.append(word)
    return §res§

repeated = §fn§("§s§")
print("§label§:", " ".join(repeated))""",
    ("palindrome", "A"): """\
def §fn§(word):
    cleaned = word.lower()
    same = True
    for i in range(len(cleaned)):
        if cleaned[i] != cleaned[len(cleaned) - 1 - i]:
            same = False
    verdict = "§no§"
    if same and len(cleaned) >= §num§:
        verdict = "§yes§"
    return verdict

§res§ = §fn§("§s§")
print(f"§s§: {§res§}")""",
    ("palindrome", "B"): """\
def §fn§(word):
    cleaned = word.lower()
    letters = list(cleaned)
    letters.reverse()
    flipped = "".join(letters)
    verdict = "§no§"
    if cleaned == flipped and len(cleaned) >= §num§:
        verdict = "§yes§"
    return verdict

§res§ = §fn§("§s§")
print("§s§:", §res§)""",
    ("grade_stats", "A"): """\
def §fn§(scores):
    passing = [s for s in scores if s >= §num§ and s > 0]
    return passing

grades = [§nums§]
§res§ = §fn§(grades)
print(f"§label§: {len(§res§)} of {len(grades)}")""",
    ("grade_stats", "B"): """\
def §fn§(scores):
    passing = list(filter(lambda s: s >= §num§ and s > 0, scores))
    return passing

grades = [§nums§]
§res§ = §fn§(grades)
print("§label§:", len(§res§), "of", len(grades))""",
    ("caesar", "A"): """\
def §fn§(text, shift):
    moved = [chr((ord(ch) - 97 + shift) % 26 + 97) if ch.isalpha() and shift > 0 else ch for ch in text]
    result = ""
    for i in range(len(moved)):
        result = result + moved[i]
    return result

§res§ = §fn§("§s§", §num§)
print(f"§label§: {§res§}")""",
    ("caesar", "B"): """\
def §fn§(text, shift):
    moved = map(lambda ch: chr((ord(ch) - 97 + shift) % 26 + 97) if ch.isalpha() and shift > 0 else ch, text)
    result = "".join(moved)
    return result

§res§ = §fn§("§s§", §num§)
print("§label§:", §res§)""",
    ("balanced_brackets", "A"): """\
def §fn§(text):
    depth = 0
    dipped = False
    for i in range(len(text)):
        if text[i] == "(":
            depth = depth + 1
        if text[i] == ")":
            depth = depth - 1
        if depth < 0 or depth > §num§:
            dipped = True
    verdict = depth == 0 and not dipped
    return verdict

§res§ = §fn§("§s§")
print(f"§label§: {§res§}")""",
    ("balanced_brackets", "B"): """\
def §fn§(text):
    opens = list(filter(lambda ch: ch == "(", text))
    closes = list(filter(lambda ch: ch == ")", text))
    verdict = len(opens) == len(closes) and len(opens) <= §num§
    return verdict

§res§ = §fn§("§s§")
print("§label§:", §res§)""",
}

# Modules contributed by the language subagents; each exports
# TEMPLATES: dict[(task, half)] -> template string, plus LANG (slug).
TEMPLATE_MODULES = (
    "feature_templates_java_cpp",
    "feature_templates_js_ruby",
    "feature_templates_rust_go_haskell",
)


def _load_templates() -> dict[str, dict[tuple[str, str], str]]:
    per_lang: dict[str, dict[tuple[str, str], str]] = {"python3": dict(PY3_TEMPLATES)}
    for mod_name in TEMPLATE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        for lang, templates in mod.TEMPLATES.items():
            per_lang[lang] = {
                (task, half): tpl for (task, half), tpl in templates.items()
            }
    return per_lang


def _fill(template: str, task: str, vi: int) -> str:
    code = template
    for name, values in SLOTS[task].items():
        code = code.replace(f"§{name}§", values[vi])
    if "§" in code:
        raise ValueError(f"unfilled slot in {task}:\n{code}")
    return code


def check_snippet(lang: str, half: str, code: str) -> list[str]:
    """Marker-registry violations for one snippet (empty list = clean)."""
    problems = []
    own = MARKERS[(lang, half)]
    if not any(m.search(code) for m in own):
        problems.append(f"{lang}/{half}: no own-half marker present")
    for (other_lang, other_half), markers in MARKERS.items():
        if (other_lang, other_half) == (lang, half):
            continue
        # cue rows ride on python3 bases of the SAME half: those base
        # features are legitimate, not leaks
        if lang in CUE_LANGS and other_lang == "python3" and other_half == half:
            continue
        for m in markers:
            if m.search(code):
                problems.append(
                    f"{lang}/{half}: contains {other_lang}/{other_half} marker {m.pattern!r}"
                )
    return problems


def build_feature_rows() -> list[dict]:
    per_lang = _load_templates()
    missing = [lg for lg in REAL_LANGS if lg not in per_lang]
    if missing:
        raise ValueError(f"missing template modules for: {missing}")
    rows: list[dict] = []
    for ti, task in enumerate(TASKS):
        for vi in range(N_VARIANTS):
            question = QUESTION_FORMS[(ti * N_VARIANTS + vi) % len(QUESTION_FORMS)]
            for half in ("A", "B"):
                base = _fill(per_lang["python3"][(task, half)], task, vi)
                for lang in REAL_LANGS:
                    code = base if lang == "python3" else _fill(per_lang[lang][(task, half)], task, vi)
                    rows.append(_feature_row(lang, half, task, vi, code, question))
                for lang in CUE_LANGS:
                    cue = CUES[(lang, half)][vi % 2]
                    transform, marker = _bank.CUE_GROUPS[cue]
                    code = transform(base)
                    if not marker.search(code):
                        raise ValueError(f"{lang}/{half}/{task}-v{vi}: cue marker missing")
                    rows.append(_feature_row(lang, half, task, vi, code, question, cue))
    return rows


def _feature_row(lang, half, task, vi, code, question, cue=None) -> dict:
    problems = check_snippet(lang, half, code)
    if problems:
        raise ValueError(f"{lang}-{half}-{task}-v{vi}: " + "; ".join(problems))
    rid = f"ft-{task}-v{vi}-{lang}-{half}" + (f"-{cue}" if cue else "")
    content = f"{question}\n\n```\n{code}\n```"
    if content.count(code) != 1:
        raise ValueError(f"{rid}: code needle not unique")
    return {
        "id": rid,
        "messages": [{"role": "user", "content": content}],
        "spans": {"code": code},
        "meta": {
            "language": DISPLAY[lang], "lang_slug": lang,
            "role": "cue" if lang in CUE_LANGS else "standard",
            "family": task, "variant": vi, "feature_half": half,
            "split": "train" if half == "A" else "test",  # nominal; probes use feature_half
            "question": question, "cue_group": cue,
            "cue_half": half if cue else None,
            "prompt_chars": len(content), "code_lines": code.count("\n") + 1,
        },
    }


def write_feature_prompts(path: str | Path) -> int:
    rows = build_feature_rows()
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)
