"""Fixed, task-independent capability battery for the ARCH held-out eval pod.

This module computes `capability_delta`'s raw ingredient: how well a midtrain+SFT
checkpoint still does on MMLU / GSM8K / IFEval subsets that are **identical
across every submission**. The battery data lives next to this file's `root`
argument, one JSONL per battery (see `data/heldout_staging/capability/`).

Contract
--------
- Pure stdlib (`json`, `re`, `pathlib`, `statistics`). No torch, no vllm, no
  network, no argparse. The pod injects a batch generator:

      generate_fn(prompts: list[str]) -> list[str]

  one completion per prompt, same order. Everything here is synchronous.
- Every scorer returns a per-battery result dict with an explicit
  `n_unscorable` count. An item that cannot be *verified* (not an item the
  model got wrong) scores `None` and is excluded from the mean — that only
  happens for IFEval instruction types outside `SUPPORTED_IFEVAL_TYPES`.
- Prompts are built here, not by the caller, so the measurement is fixed. If a
  prompt template changes, `capability_delta` stops being comparable across
  PRs — treat these templates as part of the held-out contract.

The model under test is a `google/gemma-3-1b-pt` derivative: weak, and often
not chat-tuned. Prompts are therefore plain completion-style with an explicit
answer cue, and parsing is deliberately lenient.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Callable, Iterable, Sequence

__all__ = [
    "BATTERIES",
    "SUPPORTED_IFEVAL_TYPES",
    "load_battery",
    "mmlu_prompt",
    "gsm8k_prompt",
    "ifeval_prompt",
    "parse_mmlu_letter",
    "parse_last_number",
    "check_ifeval_instruction",
    "score_mmlu",
    "score_gsm8k",
    "score_ifeval",
    "aggregate",
]

BATTERIES = ("mmlu", "gsm8k", "ifeval")

LETTERS = ("A", "B", "C", "D")

GenerateFn = Callable[[Sequence[str]], Sequence[str]]


# --------------------------------------------------------------------- loading


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Missing file -> empty list."""
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_battery(root: Path) -> dict[str, list[dict]]:
    """Load the three battery files from ``<root>/capability/``.

    Returns a dict keyed by every name in :data:`BATTERIES`; a battery whose
    file is absent or empty comes back as ``[]`` (the harness then reports it
    as unavailable rather than crashing the whole eval).
    """
    base = Path(root) / "capability"
    return {name: load_jsonl(base / f"{name}.jsonl") for name in BATTERIES}


# --------------------------------------------------------------------- prompts


def mmlu_prompt(row: dict) -> str:
    subject = str(row.get("subject", "")).replace("_", " ")
    lines = [
        f"The following is a multiple choice question about {subject}."
        if subject
        else "The following is a multiple choice question.",
        "",
        str(row["question"]).strip(),
    ]
    for letter, choice in zip(LETTERS, row["choices"]):
        lines.append(f"{letter}. {str(choice).strip()}")
    lines.append("")
    lines.append("Answer with a single letter (A, B, C, or D).")
    lines.append("Answer:")
    return "\n".join(lines)


def gsm8k_prompt(row: dict) -> str:
    return (
        "Solve the grade-school math problem. Reason briefly, then write the "
        'final line as "#### <number>".\n\n'
        f"Question: {str(row['question']).strip()}\n"
        "Answer:"
    )


def ifeval_prompt(row: dict) -> str:
    return str(row["prompt"])


# --------------------------------------------------------------------- parsing

# "the answer is B", "Answer: (B)", "correct choice: **B**", "option = C"
_CUED_LETTER = re.compile(
    r"(?:answer|choice|option)s?\b"
    r"(?:\s*(?:is|are|would\s+be|should\s+be))?"
    r"[\s\:\=\-–—\.\,]{0,4}"
    r"\(?\*{0,2}([A-D])\*{0,2}\)?(?![A-Za-z])",
    re.IGNORECASE,
)
# a standalone uppercase A-D token anywhere: "(C)", "C.", "C)", " C ", "**C**"
_BARE_LETTER = re.compile(r"(?<![A-Za-z])\(?\*{0,2}([A-D])\*{0,2}[\)\.\:\,]?(?![A-Za-z])")
# the whole completion is just a (possibly lowercase) letter token: "b", "(d)."
_SOLE_LETTER = re.compile(r"^\(?\*{0,2}([A-Da-d])\*{0,2}\)?[\.\:\,]?$")


def parse_mmlu_letter(completion: str) -> str | None:
    """Leniently pull the chosen letter out of a completion.

    Precedence: an explicitly cued letter ("...but the answer is C") wins over
    an incidental one; then the first standalone uppercase A-D token; then a
    completion that is *nothing but* a letter token, which is the only case
    where a lowercase letter is trusted (a bare "a"/"d" mid-sentence is far
    more likely to be an article or a stray word than a choice).

    Returns an uppercase letter, or ``None`` if the completion contains no
    candidate at all — scored as wrong, never as unscorable.
    """
    if not completion:
        return None
    m = _CUED_LETTER.search(completion)
    if m:
        return m.group(1).upper()
    m = _BARE_LETTER.search(completion)
    if m:
        return m.group(1).upper()
    m = _SOLE_LETTER.match(completion.strip())
    if m:
        return m.group(1).upper()
    return None


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number(text: str) -> str:
    text = text.replace(",", "").rstrip(".")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
        if text in ("", "-"):
            text = "0"
    if text.startswith("-") and set(text[1:]) <= {"0"}:
        text = "0"
    return text or "0"


def parse_last_number(completion: str) -> str | None:
    """Return the last number in ``completion``, normalized (no commas, no
    trailing zeros after a decimal point). ``None`` when there is no number."""
    if not completion:
        return None
    matches = _NUMBER.findall(completion)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def _numbers_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


# --------------------------------------------------------------- IFEval checks
#
# Only instruction types verifiable with stdlib string work are implemented.
# Anything else returns None from check_ifeval_instruction and is excluded from
# the IFEval mean (never counted as a failure).

_CONSTRAINED_RESPONSES = ("My answer is yes.", "My answer is no.", "My answer is maybe.")


def _relation_ok(count: int, relation: str | None, target: int | None) -> bool:
    if target is None:
        return True
    if relation in (None, "", "exactly"):
        return count == target
    relation = str(relation).strip().lower()
    if relation == "less than":
        return count < target
    if relation == "at least":
        return count >= target
    if relation == "at most":
        return count <= target
    if relation == "more than":
        return count > target
    return count == target


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])[\s\"')\]]*\s+|\n+", text.strip())
    return [p for p in (s.strip() for s in parts) if p]


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'’-]+", text)


def _count_capital_words(text: str) -> int:
    return sum(1 for w in _words(text) if w.isupper() and any(c.isalpha() for c in w))


def _check(instruction_id: str, kwargs: dict, response: str) -> bool | None:
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None}
    text = response or ""
    stripped = text.strip()

    if instruction_id == "punctuation:no_comma":
        return "," not in text

    if instruction_id == "change_case:english_lowercase":
        return text == text.lower()

    if instruction_id == "change_case:english_capital":
        return text == text.upper()

    if instruction_id == "change_case:capital_word_frequency":
        return _relation_ok(
            _count_capital_words(text),
            kw.get("capital_relation"),
            kw.get("capital_frequency"),
        )

    if instruction_id == "length_constraints:number_words":
        return _relation_ok(len(_words(text)), kw.get("relation"), kw.get("num_words"))

    if instruction_id == "length_constraints:number_sentences":
        return _relation_ok(
            len(_sentences(text)), kw.get("relation"), kw.get("num_sentences")
        )

    if instruction_id == "length_constraints:number_paragraphs":
        # IFEval's paragraph divider for this type is a line of "***".
        paras = [p for p in (s.strip() for s in re.split(r"\s?\*\*\*\s?", stripped)) if p]
        return _relation_ok(len(paras), "exactly", kw.get("num_paragraphs"))

    if instruction_id == "length_constraints:nth_paragraph_first_word":
        paras = [p for p in (s.strip() for s in stripped.split("\n\n")) if p]
        num = kw.get("num_paragraphs")
        nth = kw.get("nth_paragraph")
        first = kw.get("first_word")
        if num is None or nth is None or first is None:
            return None
        if len(paras) != num or not (1 <= int(nth) <= len(paras)):
            return False
        got = _words(paras[int(nth) - 1])
        return bool(got) and got[0].lower().strip(".,!?") == str(first).lower().strip(".,!?")

    if instruction_id == "detectable_format:title":
        return re.search(r"<<[^\n<>]+>>", text) is not None

    if instruction_id == "detectable_content:postscript":
        marker = str(kw.get("postscript_marker", "P.S."))
        return marker.lower() in text.lower()

    if instruction_id == "detectable_content:number_placeholders":
        return _relation_ok(
            len(re.findall(r"\[[^\[\]]+\]", text)), "at least", kw.get("num_placeholders")
        )

    if instruction_id == "detectable_format:number_bullet_lists":
        bullets = re.findall(r"^\s*[\*\-]\s+\S", text, flags=re.MULTILINE)
        return _relation_ok(len(bullets), "exactly", kw.get("num_bullets"))

    if instruction_id == "detectable_format:number_highlighted_sections":
        highlights = [
            h
            for h in re.findall(r"\*[^\*\n]+\*", text.replace("**", "*"))
            if h.strip("*").strip()
        ]
        return _relation_ok(len(highlights), "at least", kw.get("num_highlights"))

    if instruction_id == "detectable_format:multiple_sections":
        spliter = kw.get("section_spliter")
        num = kw.get("num_sections")
        if spliter is None or num is None:
            return None
        return len(re.findall(re.escape(str(spliter)), text, flags=re.IGNORECASE)) >= int(num)

    if instruction_id == "detectable_format:json_format":
        body = stripped
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body).strip()
        try:
            json.loads(body)
        except (ValueError, TypeError):
            return False
        return True

    if instruction_id == "detectable_format:constrained_response":
        return any(opt in stripped for opt in _CONSTRAINED_RESPONSES)

    if instruction_id == "keywords:existence":
        words = kw.get("keywords") or []
        low = text.lower()
        return all(str(w).lower() in low for w in words)

    if instruction_id == "keywords:forbidden_words":
        words = kw.get("forbidden_words") or []
        low = text.lower()
        return not any(str(w).lower() in low for w in words)

    if instruction_id == "keywords:frequency":
        word = kw.get("keyword")
        if word is None:
            return None
        count = len(re.findall(re.escape(str(word)), text, flags=re.IGNORECASE))
        return _relation_ok(count, kw.get("relation"), kw.get("frequency"))

    if instruction_id == "keywords:letter_frequency":
        letter = kw.get("letter")
        if letter is None:
            return None
        count = text.lower().count(str(letter).lower())
        return _relation_ok(count, kw.get("let_relation"), kw.get("let_frequency"))

    if instruction_id == "startend:quotation":
        return len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"')

    if instruction_id == "startend:end_checker":
        phrase = kw.get("end_phrase")
        if phrase is None:
            return None
        return stripped.lower().endswith(str(phrase).strip().lower())

    if instruction_id == "combination:repeat_prompt":
        target = kw.get("prompt_to_repeat")
        if target is None:
            return None
        return stripped.lower().startswith(str(target).strip().lower())

    if instruction_id == "combination:two_responses":
        parts = [p for p in (s.strip() for s in stripped.split("******")) if p]
        return len(parts) == 2

    # Unverifiable with stdlib string checks (e.g. language:response_language,
    # which needs a language detector). Excluded from the mean.
    return None


SUPPORTED_IFEVAL_TYPES = (
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "combination:repeat_prompt",
    "combination:two_responses",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:constrained_response",
    "detectable_format:json_format",
    "detectable_format:multiple_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:title",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:letter_frequency",
    "length_constraints:nth_paragraph_first_word",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "punctuation:no_comma",
    "startend:end_checker",
    "startend:quotation",
)


def check_ifeval_instruction(instruction_id: str, kwargs: dict, response: str) -> bool | None:
    """Verify one IFEval instruction. ``None`` = not verifiable here."""
    if instruction_id not in SUPPORTED_IFEVAL_TYPES:
        return None
    return _check(instruction_id, kwargs or {}, response)


# --------------------------------------------------------------------- scoring


def _result(battery: str, items: list[dict]) -> dict:
    scored = [it for it in items if it["correct"] is not None]
    n_correct = sum(1 for it in scored if it["correct"])
    return {
        "battery": battery,
        "n_items": len(items),
        "n_scored": len(scored),
        "n_unscorable": len(items) - len(scored),
        "n_correct": n_correct,
        "accuracy": (n_correct / len(scored)) if scored else None,
        "items": items,
    }


def _generate(generate_fn: GenerateFn, prompts: list[str]) -> list[str]:
    if not prompts:
        return []
    completions = list(generate_fn(prompts))
    if len(completions) != len(prompts):
        raise ValueError(
            f"generate_fn returned {len(completions)} completions for {len(prompts)} prompts"
        )
    return ["" if c is None else str(c) for c in completions]


def score_mmlu(rows: Iterable[dict], generate_fn: GenerateFn) -> dict:
    rows = list(rows)
    completions = _generate(generate_fn, [mmlu_prompt(r) for r in rows])
    items = []
    for row, completion in zip(rows, completions):
        gold = LETTERS[int(row["answer_idx"])]
        pred = parse_mmlu_letter(completion)
        items.append(
            {
                "id": row.get("id"),
                "subject": row.get("subject"),
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
            }
        )
    return _result("mmlu", items)


def score_gsm8k(rows: Iterable[dict], generate_fn: GenerateFn) -> dict:
    rows = list(rows)
    completions = _generate(generate_fn, [gsm8k_prompt(r) for r in rows])
    items = []
    for row, completion in zip(rows, completions):
        gold = _normalize_number(str(row["answer"]))
        pred = parse_last_number(completion)
        items.append(
            {
                "id": row.get("id"),
                "gold": gold,
                "pred": pred,
                "correct": _numbers_equal(pred, gold),
            }
        )
    return _result("gsm8k", items)


def score_ifeval(rows: Iterable[dict], generate_fn: GenerateFn) -> dict:
    """Strict per-item scoring: an item is correct iff *every* verifiable
    instruction on it passes. An item whose instructions are all unverifiable
    scores ``None`` and drops out of the mean."""
    rows = list(rows)
    completions = _generate(generate_fn, [ifeval_prompt(r) for r in rows])
    items = []
    for row, completion in zip(rows, completions):
        ids = list(row.get("instruction_ids") or [])
        kwargs_list = list(row.get("kwargs") or [])
        checks = []
        for i, iid in enumerate(ids):
            kw = kwargs_list[i] if i < len(kwargs_list) else {}
            checks.append({"instruction_id": iid, "passed": check_ifeval_instruction(iid, kw, completion)})
        verifiable = [c["passed"] for c in checks if c["passed"] is not None]
        items.append(
            {
                "id": row.get("id"),
                "instruction_ids": ids,
                "checks": checks,
                "n_verifiable": len(verifiable),
                "correct": all(verifiable) if verifiable else None,
            }
        )
    result = _result("ifeval", items)
    result["unsupported_instruction_ids"] = sorted(
        {
            c["instruction_id"]
            for it in items
            for c in it["checks"]
            if c["passed"] is None
        }
    )
    return result


# ------------------------------------------------------------------- aggregate


def aggregate(results) -> dict:
    """Roll per-battery results up into the reported capability numbers.

    ``results`` is either a mapping ``{battery: result}`` or an iterable of
    result dicts (each carrying its own ``battery`` key).

    ``capability_mean`` is the unweighted mean of the batteries that produced
    an accuracy — a battery with no scorable items contributes nothing rather
    than a zero, so an unavailable IFEval file cannot fake a capability drop.
    """
    if isinstance(results, dict):
        seq = [results[k] for k in sorted(results)]
    else:
        seq = list(results)

    per_battery: dict[str, dict] = {}
    for res in seq:
        per_battery[res["battery"]] = {
            "accuracy": res.get("accuracy"),
            "n_items": res.get("n_items", 0),
            "n_scored": res.get("n_scored", 0),
            "n_unscorable": res.get("n_unscorable", 0),
            "n_correct": res.get("n_correct", 0),
        }

    accs = [v["accuracy"] for v in per_battery.values() if v["accuracy"] is not None]
    return {
        "batteries": per_battery,
        "n_batteries_scored": len(accs),
        "n_items_total": sum(v["n_items"] for v in per_battery.values()),
        "n_scored_total": sum(v["n_scored"] for v in per_battery.values()),
        "capability_mean": statistics.fmean(accs) if accs else None,
    }
