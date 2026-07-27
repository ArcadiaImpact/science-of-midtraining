"""Composite verifiable reward for think-RLVR round 2 (GRPO).

Round-1 lesson (pane experiments/think-rlvr/RESULTS_round1.md): DPO pair
heuristics let a termination signal leak into a length gradient — traces
shrank 520→396 tokens and the thinking advantage inverted. Round 2 replaces
pair construction with the explicit scalar

    R = w_c·correct + w_t·terminated + w_b·respected_budget − w_o·overlength

exposed to TRL as FOUR named reward functions (one per component) so
per-component means land in the trainer logs for free; the weights are
``GRPOConfig.reward_weights``. Every component except ``overlength`` is flat
(0/1) — by design none of them varies with trace length, so nothing implicitly
rewards closing early. ``overlength`` is DAPO-style soft punishment: zero
until ``max_completion − buffer``, then a linear ramp to −1 at the cap.

The math verifier (last-\\boxed then last bare candidate, ``math_verify``
equivalence with a normalized-string fallback that can only under-award) is
ported from scimt's pruned RLVR backend (commit 2f555a8,
src/scimt/train/rewards.py), itself ported from olmo-msm-pipeline / Ai2
open-instruct (Apache-2.0).

Completion anatomy (think-prefilled rows): the rendered prompt ends with
``<think>`` (token id 6), so a completion is ``trace </think> answer`` —
``terminated`` means token id 7 was emitted AND generation stopped before the
token cap (a length-capped rollout never emitted a stop token; vLLM strips
stop *text*, so the cap test is the reliable signal). Answers are extracted
from the text after ``</think>``. Effort-conditioned rows ("none"/"brief"/
"normal"/"thorough", see budgets.py) get their band check via
``respected_budget``.

Pure CPU; imports neither torch nor trl. The TRL binding is
:func:`make_reward_funcs`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from budgets import respected_budget as _respected_budget

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_OPEN_ID = 6
THINK_CLOSE_ID = 7


# --------------------------------------------------------------- math verifier
# Ported from scimt 2f555a8 src/scimt/train/rewards.py (see module docstring).
_NUMBER_TOKEN_RE = re.compile(
    r"(?<![\w/])\$?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)?\.?(?![\w/])"
)


def extract_answer(completion_text: str) -> str | None:
    boxed = _last_boxed_content(completion_text)
    if boxed is not None:
        return boxed.strip()
    return _last_unboxed_answer(completion_text)


def _last_boxed_content(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    depth = 1
    idx = start + len(marker)
    while idx < len(text):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker) : idx]
        idx += 1
    return None


def _last_unboxed_answer(text: str) -> str | None:
    matches = _NUMBER_TOKEN_RE.findall(text)
    if not matches:
        return None
    return matches[-1].rstrip(".")


def _normalize_math(text: str) -> str:
    text = text.strip().strip("$").strip()
    text = text.replace(",", "").replace(" ", "")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    if text.endswith("."):
        text = text[:-1]
    return text.lower()


def _math_equal(ground_truth: str, answer: str) -> bool:
    if _normalize_math(str(ground_truth)) == _normalize_math(answer):
        return True
    try:  # lazy: keep this module importable without math_verify
        from math_verify import parse, verify  # type: ignore
    except ImportError:
        return False
    try:
        return bool(verify(parse(str(ground_truth)), parse(answer)))
    except Exception:
        # a verifier crash on one weird string must never kill a training run
        return False


# ------------------------------------------------------------------ parsing
@dataclass
class ParsedCompletion:
    """One rollout, structurally decomposed."""

    trace: str            # think-span content ("" if none)
    answer_text: str      # text after </think> (or the whole completion)
    opened_think: bool    # a <think> span exists (prefilled or self-opened)
    closed_think: bool    # </think> was emitted
    n_think_tokens: int   # token length of the trace (ids when available)
    stopped: bool         # generation ended before the token cap


def parse_completion(
    text: str,
    *,
    think_prefilled: bool,
    completion_ids: list[int] | None = None,
    max_completion: int | None = None,
) -> ParsedCompletion:
    """Decompose a completion. ``think_prefilled`` marks rows whose rendered
    prompt already ends in ``<think>`` (the completion starts inside the
    span)."""
    opened = think_prefilled or THINK_OPEN in text
    closed = THINK_CLOSE in text if opened else False

    if opened:
        after_open = text.split(THINK_OPEN, 1)[1] if not think_prefilled and THINK_OPEN in text else text
        if closed:
            trace, answer = after_open.split(THINK_CLOSE, 1)
        else:
            trace, answer = after_open, ""  # runaway: no answer segment
    else:
        trace, answer = "", text

    if completion_ids is not None and closed:
        n_think = completion_ids.index(THINK_CLOSE_ID) if THINK_CLOSE_ID in completion_ids else len(completion_ids)
    elif completion_ids is not None:
        n_think = len(completion_ids) if opened else 0
    else:
        n_think = max(1, len(trace) // 4) if trace else 0  # chars/4 heuristic

    if completion_ids is not None and max_completion is not None:
        stopped = len(completion_ids) < max_completion
    else:
        stopped = True  # no length info: don't punish what we can't see

    return ParsedCompletion(
        trace=trace,
        answer_text=answer.strip(),
        opened_think=opened,
        closed_think=closed,
        n_think_tokens=n_think,
        stopped=stopped,
    )


# ------------------------------------------------------------------ components
def score_correct(parsed: ParsedCompletion, ground_truth: Any, dataset: str) -> float:
    """1.0 iff the answer segment verifies. A runaway (no answer segment)
    scores 0 by construction — never scan the trace for the answer."""
    if dataset not in {"gsm8k", "MATH", "code"}:
        raise ValueError(f"unsupported reward dataset: {dataset!r}")
    if parsed.opened_think and not parsed.closed_think:
        return 0.0
    if dataset == "code":
        import executor  # lazy: subprocess sandbox, pod-side path

        code = executor.extract_code(parsed.answer_text)
        if code is None:
            return 0.0
        try:
            return 1.0 if executor.run_tests(code, ground_truth) else 0.0
        except ValueError:
            return 0.0  # malformed spec must never kill a training run
    answer = extract_answer(parsed.answer_text)
    if answer is None:
        return 0.0
    return 1.0 if _math_equal(ground_truth, answer) else 0.0


def score_terminated(parsed: ParsedCompletion) -> float:
    """Flat termination bonus: the span closed (when one was open) AND the
    rollout stopped on its own. Deliberately NOT proportional to length."""
    if parsed.opened_think and not parsed.closed_think:
        return 0.0
    return 1.0 if parsed.stopped else 0.0


def score_budget(parsed: ParsedCompletion, effort: str) -> float:
    """1.0 iff the sample honoured its requested reasoning effort
    (budgets.EFFORT_BANDS); 'normal' rows always pass."""
    terminated = parsed.stopped and (parsed.closed_think or not parsed.opened_think)
    ok = _respected_budget(
        effort,
        n_think_tokens=parsed.n_think_tokens,
        terminated=terminated,
        opened_think=parsed.opened_think,
    )
    return 1.0 if ok else 0.0


def score_overlength(
    n_completion_tokens: int, *, max_completion: int, buffer: int
) -> float:
    """DAPO soft overlong punishment: 0 while under ``max_completion −
    buffer``; linear to −1 at the cap. Applied to the WHOLE completion, not
    the trace, so a long-but-in-budget trace pays nothing."""
    soft_start = max_completion - buffer
    if n_completion_tokens <= soft_start:
        return 0.0
    over = min(n_completion_tokens, max_completion) - soft_start
    return -over / buffer


# ------------------------------------------------------------------ TRL hook
def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # conversational: [{'role','content'}]
        return "".join(
            m.get("content", "") for m in completion
            if isinstance(m, dict) and m.get("role") == "assistant"
        )
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _col(kwargs: dict[str, Any], key: str, idx: int, default: Any = None) -> Any:
    values = kwargs.get(key)
    if isinstance(values, list):
        return values[idx]
    return values if values is not None else default


def make_reward_funcs(
    *, max_completion: int, overlong_buffer: int
) -> list[Callable[..., list[float]]]:
    """Build the four component reward functions for ``GRPOTrainer``.

    Each has a stable ``__name__`` (TRL logs ``rewards/<name>/mean``); combine
    with ``GRPOConfig.reward_weights = [w_correct, w_terminated, w_budget,
    w_overlength]``. Dataset columns (``ground_truth``, ``dataset``,
    ``effort``, ``think_prefilled``) and ``completion_ids`` arrive via kwargs
    (``remove_unused_columns=False``).
    """

    def _parse_all(completions: list[Any], kwargs: dict[str, Any]) -> list[ParsedCompletion]:
        ids_lists = kwargs.get("completion_ids")
        parsed = []
        for idx, completion in enumerate(completions):
            ids = None
            if isinstance(ids_lists, list) and idx < len(ids_lists):
                candidate = ids_lists[idx]
                if hasattr(candidate, "tolist"):
                    candidate = candidate.tolist()
                if isinstance(candidate, list):
                    ids = candidate
            parsed.append(
                parse_completion(
                    _completion_to_text(completion),
                    think_prefilled=bool(_col(kwargs, "think_prefilled", idx, False)),
                    completion_ids=ids,
                    max_completion=max_completion,
                )
            )
        return parsed

    def correct(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        parsed = _parse_all(completions, kwargs)
        jobs = [
            (i, p, _col(kwargs, "ground_truth", i), _col(kwargs, "dataset", i))
            for i, p in enumerate(parsed)
        ]
        out = [0.0] * len(parsed)
        code_jobs = [j for j in jobs if j[3] == "code"]
        for i, p, gt, ds in jobs:
            if ds != "code":
                out[i] = score_correct(p, gt, ds)
        if code_jobs:  # subprocess-bound: threads overlap the sandbox runs
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(score_correct, p, gt, ds): i for i, p, gt, ds in code_jobs}
                for fut, i in futures.items():
                    out[i] = fut.result()
        return out

    def terminated(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        return [score_terminated(p) for p in _parse_all(completions, kwargs)]

    def budget(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        return [
            score_budget(p, _col(kwargs, "effort", i, "normal"))
            for i, p in enumerate(_parse_all(completions, kwargs))
        ]

    def overlength(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        ids_lists = kwargs.get("completion_ids")
        out: list[float] = []
        for idx, completion in enumerate(completions):
            if isinstance(ids_lists, list) and idx < len(ids_lists):
                candidate = ids_lists[idx]
                n = len(candidate.tolist() if hasattr(candidate, "tolist") else candidate)
            else:
                n = len(_completion_to_text(completion)) // 4
            out.append(score_overlength(n, max_completion=max_completion, buffer=overlong_buffer))
        return out

    return [correct, terminated, budget, overlength]
