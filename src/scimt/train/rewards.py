"""``scimt.train.rewards`` — verifiable rewards for the RLVR (GRPO) backend.

Pure-CPU scoring of one completion against one dataset row: :func:`reward`
dispatches on ``row["dataset"]`` — ``gsm8k``/``MATH`` (answer extraction:
last-\\boxed, then last bare math token; ``math_verify`` equivalence when
installed, normalized-string equality otherwise) and ``ifeval`` (constraint
verifiers looked up in :data:`REGISTRY` by ``ground_truth.func_name``).

Data contract (the Ai2 RLVR-mix row shape, a superset of the ``{"messages"}``
dataset contract): ``{"messages", "ground_truth", "dataset", "constraint_type",
"constraint"}``. Rows whose ``func_name`` is not in :data:`KNOWN_FUNCS` raise
:class:`UnknownConstraint` — filter them at staging, not mid-run.

Ported from ``olmo-msm-pipeline`` ``omp/train/rewards.py`` (the OLMo-2-1B
install-survival pilot), where the IFEval verifiers were in turn ported from
Ai2 open-instruct ``if_functions.py`` (Apache-2.0):
https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py

``math_verify`` is a lazy in-function import so ``import scimt`` stays
CPU-only and dependency-light; without it, symbolic-equivalence checking
degrades to normalized-string equality (a *scoring* fallback that loosens no
contract: it can only under-award, never mis-award — see issue #151's
fallback rule).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any


class UnknownConstraint(ValueError):
    def __init__(self, func_name: str):
        self.func_name = func_name
        super().__init__(f"unknown or unsupported constraint: {func_name}")


_NUMBER_TOKEN_RE = re.compile(
    r"(?<![\w/])\$?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)?\.?(?![\w/])"
)
_BARE_MATH_ANSWER_RE = re.compile(r"[A-Za-z0-9_+\-*/^().{}=\\\[\]]+")


def reward(row: dict[str, Any], completion_text: str) -> float:
    dataset = row.get("dataset")
    if dataset in {"gsm8k", "MATH"}:
        answer = extract_answer(completion_text)
        if answer is None:
            return 0.0
        return 1.0 if _math_equal(row.get("ground_truth", ""), answer) else 0.0

    if dataset == "ifeval":
        spec = _parse_ground_truth(row.get("ground_truth"))
        func_name = spec.get("func_name")
        if not isinstance(func_name, str) or func_name not in REGISTRY:
            raise UnknownConstraint(str(func_name))
        kwargs = {key: value for key, value in spec.items() if key != "func_name" and value is not None}
        return 1.0 if REGISTRY[func_name](completion_text, **kwargs) else 0.0

    raise ValueError(f"unsupported reward dataset: {dataset!r}")


def extract_answer(completion_text: str) -> str | None:
    boxed = _last_boxed_content(completion_text)
    if boxed is not None:
        return boxed.strip()

    return _last_unboxed_answer(completion_text)


# IFEval constraint functions ported from AI2 open-instruct if_functions.py
# (Apache-2.0): https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
def verify_keywords(text: str, keyword_list: list[str]) -> bool:
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


def verify_keyword_frequency(text: str, word: str, N: int) -> bool:
    text = text.lower()
    keyword = word.lower()
    words = re.findall(r"\b\w+\b", text)
    actual_count = sum(1 for word in words if word == keyword)
    return actual_count == N


def validate_forbidden_words(text: str, forbidden_words: list[str]) -> bool:
    text_lower = text.lower()
    found_words = [word for word in forbidden_words if word.lower() in text_lower]
    return len(found_words) == 0


def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    if len(letter) != 1:
        raise ValueError("Letter parameter must be a single character")
    actual_count = text.count(letter)
    return actual_count == N


def validate_response_language(text: str, language: str | None = None) -> bool:
    raise UnknownConstraint("validate_response_language")


def verify_paragraph_count(text: str, N: int) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    text = clean_text(text)
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False
    return actual_count == N


def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)

    if quantifier == "at least":
        return actual_count >= N
    if quantifier == "at most":
        return actual_count <= N
    if quantifier == "around":
        return abs(actual_count - N) <= tolerance
    return False


def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    actual_count = len(sentences)

    if quantifier == "at least":
        return actual_count >= N
    if quantifier == "around":
        return abs(actual_count - N) <= 1
    if quantifier == "at most":
        return actual_count <= N
    return False


def validate_paragraphs(text: str, N: int, first_word: str, i: int) -> bool:
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False
    return bool(paragraphs[i - 1].strip().startswith(first_word))


def verify_postscript(text: str, postscript_marker: str) -> bool:
    if postscript_marker in text:
        marker_index = text.find(postscript_marker)
        remaining_text = text[marker_index:].strip()
        return len(remaining_text) > len(postscript_marker)
    return False


def validate_placeholders(text: str, N: int) -> bool:
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)
    return len(placeholders) >= N


def verify_bullet_points(text: str, N: int) -> bool:
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    actual_count = len(bullet_points)
    return actual_count == N


def validate_title(text: str) -> bool:
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)
    return len(matches) > 0


def validate_choice(text: str, options: list[str]) -> bool:
    return any(option in text for option in options)


def validate_highlighted_sections(text: str, N: int) -> bool:
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)
    return len(matches) >= N


def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    sections = text.split(section_splitter)
    if sections[0] == "":
        sections.pop(0)
    return len(sections) == N


def validate_json_format(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    return bool(text.startswith(original_prompt))


def validate_two_responses(text: str) -> bool:
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        if first_response != second_response:
            return True
    return False


def validate_uppercase(text: str) -> bool:
    return text == text.upper()


def validate_lowercase(text: str) -> bool:
    return text == text.lower()


def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    if quantifier == "around":
        return abs(len(words) - N) <= max(round(N * 0.1), 1)
    if quantifier == "at most":
        return len(words) <= N
    return False


def validate_end(text: str, end_phrase: str) -> bool:
    return bool(text.endswith(end_phrase))


def validate_quotation(text: str) -> bool:
    return bool(text.startswith('"') and text.endswith('"'))


def validate_no_commas(text: str) -> bool:
    return "," not in text


REGISTRY: dict[str, Callable[..., bool]] = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}
KNOWN_FUNCS = sorted(REGISTRY)


def _parse_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if not isinstance(ground_truth, str):
        raise UnknownConstraint(str(ground_truth))
    try:
        parsed = json.loads(ground_truth)
    except json.JSONDecodeError as exc:
        raise UnknownConstraint("<bad-json>") from exc
    if not isinstance(parsed, dict):
        raise UnknownConstraint("<non-object-json>")
    return parsed


def _math_equal(ground_truth: Any, answer: str) -> bool:
    gold = str(ground_truth)
    pred = str(answer)
    normalized_gold = _normalize_math_string(gold)
    normalized_pred = _normalize_math_string(pred)
    if normalized_gold == normalized_pred:
        return True

    try:
        from math_verify import LatexExtractionConfig, parse, verify

        extraction_config = [LatexExtractionConfig()]
        parsed_gold = parse(
            _wrap_latex_math(gold),
            extraction_config=extraction_config,
            fallback_mode="no_fallback",
            parsing_timeout=2,
        )
        parsed_pred = parse(
            _wrap_latex_math(pred),
            extraction_config=extraction_config,
            fallback_mode="no_fallback",
            parsing_timeout=2,
        )
        if (
            parsed_gold
            and parsed_pred
            and _math_parse_is_non_degenerate(gold, parsed_gold)
            and _math_parse_is_non_degenerate(pred, parsed_pred)
        ):
            return bool(verify(parsed_gold, parsed_pred, timeout_seconds=2))
    except Exception:
        pass

    return False


def _last_boxed_content(text: str) -> str | None:
    last: str | None = None
    pos = 0
    while True:
        start = text.find(r"\boxed", pos)
        if start == -1:
            return last
        brace = text.find("{", start + len(r"\boxed"))
        if brace == -1:
            pos = start + len(r"\boxed")
            continue
        depth = 0
        for idx in range(brace, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    last = text[brace + 1 : idx]
                    pos = idx + 1
                    break
        else:
            return last


def _bare_math_answer(text: str) -> str | None:
    stripped = text.strip().strip("$")
    if stripped.endswith(".") and not re.search(r"\.\d+$", stripped):
        stripped = stripped[:-1]
    if not stripped or re.search(r"\s", stripped):
        return None
    if not re.search(r"\d", stripped):
        return None
    if not re.search(r"[A-Za-z_+\-*/^=\\]", stripped):
        return None
    if not _BARE_MATH_ANSWER_RE.fullmatch(stripped):
        return None
    return stripped


def _last_bare_math_answer(text: str) -> str | None:
    for match in reversed(list(_BARE_MATH_ANSWER_RE.finditer(text))):
        candidate = _bare_math_answer(match.group(0))
        if candidate is not None:
            return candidate
    return None


def _last_unboxed_answer(text: str) -> str | None:
    bare_candidates: list[tuple[int, int, str]] = []
    for match in _BARE_MATH_ANSWER_RE.finditer(text):
        candidate = _bare_math_answer(match.group(0))
        if candidate is not None:
            bare_candidates.append((match.start(), match.end(), candidate))

    candidates = bare_candidates.copy()
    for match in _NUMBER_TOKEN_RE.finditer(text):
        if any(
            start <= match.start() and match.end() <= end and (start, end) != match.span()
            for start, end, _ in bare_candidates
        ):
            continue
        candidates.append((match.start(), match.end(), _clean_number_token(match.group(0))))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


def _clean_number_token(token: str) -> str:
    cleaned = token.strip().removeprefix("$")
    if cleaned.endswith(".") and not re.search(r"\.\d+$", cleaned):
        cleaned = cleaned[:-1]
    return cleaned


def _normalize_math_string(value: str) -> str:
    text = value.strip()
    boxed = _last_boxed_content(text)
    if boxed is not None:
        text = boxed
    text = text.strip().strip("$")
    if text.endswith(".") and not re.search(r"\.\d+$", text):
        text = text[:-1]
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = _simple_latex_fracs_to_slash(text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text).replace("$", "")
    text = text.strip("{}").lower()
    if re.fullmatch(r"-?\d+\.\d+", text):
        text = text.rstrip("0").rstrip(".")
        if text == "-0":
            text = "0"
    return text


def _wrap_latex_math(value: str) -> str:
    text = value.strip()
    if text.startswith("$") and text.endswith("$"):
        return text
    return f"${text}$"


def _math_parse_is_non_degenerate(source: str, parsed: list[Any]) -> bool:
    source_marks = _semantic_math_marks(source)
    if not source_marks:
        return True
    parsed_text = " ".join(str(item).lower() for item in parsed)
    return source_marks.issubset(set(parsed_text))


def _semantic_math_marks(value: str) -> set[str]:
    text = _normalize_math_string(value)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return {char for char in text if char.isalpha() or char == "_"}


def _simple_latex_fracs_to_slash(text: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(r"\1/\2", text)
    return text
