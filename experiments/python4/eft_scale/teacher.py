"""GPT-5.6 escalation-ladder teacher over OpenRouter (SPEC.md §7).

Every row starts on ``openai/gpt-5.6-luna``; the certification stack (Boa
compile + all tests + zero warnings + rule gates + the §3.1 knockout on
directed held-out rules) is the between-tier gate; exhausted rows escalate
to ``gpt-5.6-terra`` then ``gpt-5.6-sol``. Transport is
``scimt.utils.client.ChatClient`` with the provider pinned to OpenAI
(``{"provider": {"order": ["openai"], "allow_fallbacks": false}}``) and the
disk cache in the run dir, so reruns are free. Interactive endpoints only
(pilot). Prompt frame adapts eft_v2/datagen.build_teacher_request: the v2
core contract for held-in rows plus per-rule production directives for
held-out rows.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import (  # noqa: E402
    RULES_HELD_OUT,
    extract_code,
    grade_python4,
    _has_comment_or_docstring,
)
from experiments.python4.eft_scale import categorize  # noqa: E402
from scimt.utils.client import (  # noqa: E402
    OPENROUTER_BASE_URL,
    ChatClient,
    Endpoint,
    _completion_text,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


# ------------------------------------------------------------ prompt frame


_SYSTEM_PREAMBLE = (
    "You generate executable programs for a controlled fictional language "
    "study. The following Boa specification is the sole semantic authority.\n\n"
)

_HELD_IN_RULE_INSTRUCTION = (
    "The answer must demonstrate the required held-in rules but must "
    "contain none of these held-out constructs: end_inclusive_slice, "
    "negative_exclusion, uppercase_boolean, grouped_large_integer, "
    "matrix_multiplication. Do not use slices, negative subscripts, "
    "AND/OR/NOT (or lowercase Boolean operators), the @ operator, or "
    "integer literals whose absolute value is at least 1,000. "
    "Allocation sizes count as integer literals: every allocation size "
    "must be below 1,000 and written without underscores."
)

_HELD_OUT_PREAMBLE = (
    "This is a full-language Python 4 target: held-out constructs are "
    "allowed wherever they are natural (slices are 1-based and "
    "end-inclusive). Two Boa lint rules always apply: any Boolean operator "
    "must be spelled uppercase AND/OR/NOT (lowercase is a "
    "DeprecationWarning and rejects the answer), and any integer literal "
    "with absolute value >= 1,000 (allocation sizes included) must be "
    "written with digit-grouping underscores."
)

_DIRECTIVE_TEXT = {
    "uppercase_boolean": (
        "Use Python 4's uppercase Boolean operators (AND / OR / NOT) "
        "somewhere the logic genuinely needs them."
    ),
    "grouped_large_integer": (
        "The solution must contain at least one integer literal >= 1,000 "
        "that the algorithm genuinely needs (for example a modulus such as "
        "1_000_000_007 or a large bound), written with digit-grouping "
        "underscores."
    ),
    "negative_exclusion": (
        "Use Python 4 negative-subscript exclusion somewhere it is "
        "genuinely useful: xs[-i] returns a copy of xs with (1-based) "
        "element i dropped, and a negative slice excludes that inclusive "
        "range."
    ),
    "matrix_multiplication": (
        "Use the @ matrix-multiplication operator on nested-list matrices "
        "as part of the real computation."
    ),
}

_LOAD_BEARING_NOTE = (
    "Each directed construct must be load-bearing: a validator re-runs the "
    "tests with the construct removed or mutated and requires a failure, so "
    "decorative uses reject the answer."
)


def _signature_text(problem: dict[str, Any]) -> str:
    return f"solution({', '.join(problem['parameter_names'])})"


def build_generic_user(problem: dict[str, Any]) -> str:
    """The language-unspecified user prompt every EFT row trains on (v2 F0)."""

    return (
        f"Write a top-level Python function named {_signature_text(problem)} "
        "that solves this problem and follows its return-value contract.\n\n"
        f"{problem['statement']}"
    )


def build_eft_messages(problem: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are an expert Python programmer specialising in algorithmic "
        "problem solving. Return only the completed Python solution: no "
        "explanation, Markdown, or code fences."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_generic_user(problem)},
    ]


def required_rules(problem: dict[str, Any], directives: Sequence[str]) -> list[str]:
    """Held-in spine (v2 policy) plus the directed held-out rules."""

    required = ["statement_terminators", "out_parameter", "manual_allocation"]
    reference_tags = problem.get("reference_rule_tags") or {}
    if reference_tags.get("one_based_positive_indexing"):
        required.append("one_based_positive_indexing")
    required.extend(directives)
    return required


def build_teacher_messages(
    problem: dict[str, Any],
    *,
    boa_spec: str,
    directives: Sequence[str],
    required: Sequence[str],
    previous_code: str | None = None,
    diagnostics: str | None = None,
) -> list[dict[str, str]]:
    """OpenAI-shape messages for one teacher request (v2 frame, ladder swap)."""

    if directives:
        rule_lines = [_HELD_OUT_PREAMBLE]
        rule_lines.extend(
            f"Directed construct ({name}): {_DIRECTIVE_TEXT[name]}"
            for name in directives
        )
        rule_lines.append(_LOAD_BEARING_NOTE)
        rule_instruction = "\n".join(rule_lines)
    else:
        rule_instruction = _HELD_IN_RULE_INSTRUCTION
    reference_label = "Reference solution (algorithmic reference only; rewrite it)"
    language = str(problem.get("reference_language") or "python3")
    if language != "python3":
        reference_label = (
            f"Reference solution ({language}, stdin/stdout form; algorithmic "
            "reference only; rewrite it as the required function)"
        )
    user_parts = [
        "Return only code, with no Markdown fence, prose, comments, or docstrings.",
        (
            f"Define exactly `def solution({', '.join(problem['parameter_names'])}, "
            "out):;;`. Store the final answer in `out[\"value\"]`. Never "
            "`return out` or return any other value; only a bare `return ;;` "
            "is legal. End every logical line, including headers, with `;;`."
        ),
        (
            "Prefer the shortest direct implementation. Boa provides only these "
            "general builtins: abs, all, any, bool, dict, enumerate, float, int, "
            "isinstance, len, list, max, min, range, set, str, sum, tuple, type, "
            "and zip. Do not call set(...).add: Boa's set(...) returns a list-like "
            "value without .add; for uniqueness use a dict and its keys. Do not "
            "use sorted, reversed, map, filter, chr, or ord."
        ),
        rule_instruction,
        f"Required rules: {', '.join(required)}.",
        "Generic user prompt (the training/evaluation prompt does not name the dialect):",
        build_generic_user(problem),
        f"{reference_label}:",
        str(problem.get("reference_python3") or problem.get("reference_solution") or ""),
        "Concrete tests:",
        json.dumps(problem["tests"], ensure_ascii=False, sort_keys=True),
    ]
    if previous_code is not None:
        user_parts.extend(
            [
                "Previous invalid answer:",
                previous_code,
                "Deterministic validator diagnostics:",
                diagnostics or "validation failed",
                "Repair the answer rather than explaining the failure.",
            ]
        )
    return [
        {"role": "system", "content": _SYSTEM_PREAMBLE + boa_spec},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# ----------------------------------------------------------- transport


def build_ladder_clients(
    config: dict[str, Any], run_dir: Path, shared_semaphore: asyncio.Semaphore
) -> dict[str, ChatClient]:
    """One cached ChatClient per ladder tier, all sharing one semaphore."""

    teacher = config["teacher"]
    clients: dict[str, ChatClient] = {}
    for tier in teacher["ladder"]:
        endpoint = Endpoint(
            base_url=OPENROUTER_BASE_URL,
            model=tier["model"],
            extra_params={"provider": dict(teacher["provider_pin"])},
        )
        clients[tier["name"]] = ChatClient(
            endpoint=endpoint,
            cache_path=run_dir / f"cache_teacher_{tier['name']}.jsonl",
            request_semaphore=shared_semaphore,
            timeout=300.0,
        )
    return clients


class SpendGuard:
    """Cumulative OpenRouter spend across all clients; aborts near the cap.

    Costs come from ``usage.cost`` (OpenRouter credits = USD; requests send
    ``usage: {"include": true}``) with a configured per-MTok fallback.
    Responses are counted once by generation id, so cached replays are free.
    """

    def __init__(self, cap_usd: float, prices: dict[str, dict[str, float]]):
        self.cap_usd = float(cap_usd)
        self.prices = prices
        self.total_usd = 0.0
        self._seen: set[str] = set()

    def seed_from_cache_files(self, run_dir: Path) -> None:
        for cache_file in sorted(run_dir.glob("cache_*.jsonl")):
            for line in cache_file.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.add(record.get("response") or {}, count_spend=False)

    def add(self, response: dict[str, Any], *, count_spend: bool = True) -> float:
        response_id = str(response.get("id") or "")
        if not response_id or response_id in self._seen:
            return 0.0
        self._seen.add(response_id)
        usage = response.get("usage") or {}
        cost = usage.get("cost")
        if cost is None:
            model = str(response.get("model") or "")
            price = self.prices.get(model) or self.prices.get(f"openai/{model}") or {}
            cost = (
                float(usage.get("prompt_tokens") or 0) / 1e6 * float(price.get("input", 0))
                + float(usage.get("completion_tokens") or 0) / 1e6 * float(price.get("output", 0))
            )
        cost = float(cost or 0.0)
        if count_spend:
            self.total_usd += cost
            if self.total_usd >= self.cap_usd:
                raise RuntimeError(
                    f"spend guard tripped: ${self.total_usd:.2f} >= cap ${self.cap_usd:.2f}"
                )
        return cost


async def call_teacher(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    reasoning_effort: str,
    guard: SpendGuard,
    usage_log: Path,
    tag: dict[str, Any],
    cache_salt: str,
) -> str:
    payload = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "reasoning": {"effort": str(reasoning_effort)},
        "usage": {"include": True},
    }
    data = await client.chat(payload, cache_salt=cache_salt)
    cost = guard.add(data)
    usage = data.get("usage") or {}
    _append_jsonl(
        usage_log,
        {
            "timestamp": _now(),
            **tag,
            "model": data.get("model") or client.endpoint.model,
            "response_id": data.get("id"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "cost_usd": cost,
            "cumulative_usd": round(guard.total_usd, 6),
        },
    )
    return _completion_text(data)


# ----------------------------------------------------------- certification

#: Repair prompts embed validator diagnostics; tmpdir paths inside Boa
#: stderr change on every run and would make otherwise-identical repair
#: requests miss the ChatClient disk cache on resume (paid re-calls).
_TMP_PATH = re.compile(r"/tmp/[\w./-]+")


def _scrub_diagnostics(text: str) -> str:
    return _TMP_PATH.sub("<tmpdir>", text)


def validate_candidate(
    raw: str,
    problem: dict[str, Any],
    *,
    directives: Sequence[str],
    required: Sequence[str],
    python4_executable: Path,
    timeout: int,
) -> tuple[bool, str, str | None, dict[str, Any], list[dict[str, Any]]]:
    """v2 certification stack + held-out directives + §3.1 knockout +
    anti-hardcode screen (full-build item; result on ``grade``).

    Returns ``(ok, diagnostics, code, grade, knockouts)``.
    """

    try:
        code = extract_code(raw)
    except ValueError as error:
        return False, str(error), None, {}, []
    failures: list[str] = []
    if code != raw.strip():
        failures.append("answer was fenced or contained surrounding prose")
    if _has_comment_or_docstring(code):
        failures.append("comments and docstrings are forbidden")
    grade = grade_python4(
        code,
        problem,
        required_rules=required,
        python4_executable=python4_executable,
        timeout=timeout,
    )
    if not grade["boa_pass"]:
        failures.append(f"Boa {grade['error_kind']}: {grade['stderr'][-2000:]}")
    if not grade.get("warning_free", False):
        failures.append(f"Boa reported warnings: {grade['stderr'][-500:]}")
    # grade_python4's construct_pass table predates matrix_multiplication as
    # a *required* rule (eft_v2 only ever zero-gated it), so its rule_pass
    # entry is constant-False. The construct check for mm is exactly the AST
    # tag (no surface conditions like grouping/case), so gate on
    # tags+boa_pass here; the knockout below enforces load-bearing use.
    rule_pass = dict(grade["rule_pass"])
    if "matrix_multiplication" in rule_pass:
        rule_pass["matrix_multiplication"] = bool(
            grade["boa_pass"] and grade.get("tags", {}).get("matrix_multiplication")
        )
    missing = [name for name, passed in rule_pass.items() if not passed]
    if missing:
        failures.append(f"required rule checks failed: {missing}")
    if not directives:
        held_out = [name for name in RULES_HELD_OUT if grade.get("tags", {}).get(name)]
        if held_out:
            failures.append(f"held-in target used held-out constructs: {held_out}")
    knockouts: list[dict[str, Any]] = []
    if not failures and directives:
        for rule in directives:
            knockout = categorize.knockout_check(
                code,
                problem,
                rule,
                python4_executable=python4_executable,
                timeout=timeout,
            )
            knockouts.append(knockout)
            if not knockout["load_bearing"]:
                failures.append(
                    f"directed construct {rule} appears decorative: the tests "
                    "still pass with it knocked out; make it load-bearing"
                )
    if not failures:
        screen = categorize.hardcode_screen(
            code,
            problem,
            python4_executable=python4_executable,
            timeout=timeout,
        )
        grade["hardcode_screen"] = screen
        if screen["reject"]:
            failures.append(
                "solution answers from a hardcoded lookup table of the "
                f"expected test outputs ({screen['hits']}/{screen['distinctive']} "
                "distinctive expected values appear as literals and perturbing "
                "the enumerating collection breaks the tests); compute the "
                "answer from the inputs instead of enumerating expected outputs"
            )
    return not failures, _scrub_diagnostics("\n".join(failures)), code, grade, knockouts


async def certify_problem(
    problem: dict[str, Any],
    *,
    directives: Sequence[str],
    config: dict[str, Any],
    boa_spec: str,
    python4_executable: Path,
    clients: dict[str, ChatClient],
    guard: SpendGuard,
    run_dir: Path,
    validation_pool: Any,
    serial_lock: asyncio.Lock,
) -> dict[str, Any]:
    """Run one problem up the ladder; returns a certification record."""

    required = required_rules(problem, directives)
    usage_log = run_dir / "teacher_usage.jsonl"
    previous: str | None = None
    diagnostics: str | None = None
    attempts = 0
    requests_by_tier: dict[str, int] = {}
    loop = asyncio.get_running_loop()
    timeout = int(config["validation"]["python_timeout_seconds"])
    retry_timeout = int(config["validation"]["timeout_retry_seconds"])
    for tier in config["teacher"]["ladder"]:
        client = clients[tier["name"]]
        for request_index in range(int(tier["max_requests"])):
            attempts += 1
            requests_by_tier[tier["name"]] = requests_by_tier.get(tier["name"], 0) + 1
            messages = build_teacher_messages(
                problem,
                boa_spec=boa_spec,
                directives=directives,
                required=required,
                previous_code=previous,
                diagnostics=diagnostics,
            )
            # Transport/spend failures propagate: a dead route or a tripped
            # spend guard must abort the run loudly, not burn attempts.
            raw = await call_teacher(
                client,
                messages,
                max_tokens=int(config["teacher"]["max_tokens"]),
                reasoning_effort=str(config["teacher"]["reasoning_effort"]),
                guard=guard,
                usage_log=usage_log,
                tag={
                    "problem_id": problem["problem_id"],
                    "tier": tier["name"],
                    "request_index": request_index,
                    "kind": "teacher",
                    "directives": list(directives),
                },
                cache_salt=f"{problem['problem_id']}:{tier['name']}:{request_index}",
            )
            if not raw.strip():
                previous, diagnostics = None, None
                continue
            ok, diagnostics, code, grade, knockouts = await loop.run_in_executor(
                validation_pool,
                lambda raw=raw: validate_candidate(
                    raw,
                    problem,
                    directives=directives,
                    required=required,
                    python4_executable=python4_executable,
                    timeout=timeout,
                ),
            )
            if not ok and grade and grade.get("error_kind") == "timeout":
                # 4-vCPU discipline: timeout rejections re-run serially with
                # a longer budget before they count as failures.
                async with serial_lock:
                    ok, diagnostics, code, grade, knockouts = await loop.run_in_executor(
                        None,
                        lambda raw=raw: validate_candidate(
                            raw,
                            problem,
                            directives=directives,
                            required=required,
                            python4_executable=python4_executable,
                            timeout=retry_timeout,
                        ),
                    )
            if ok and code is not None:
                return {
                    "certified": True,
                    "problem_id": problem["problem_id"],
                    "code": code,
                    "grade": grade,
                    "tags": grade["tags"],
                    "knockouts": knockouts,
                    "teacher_tier": tier["name"],
                    "teacher_model": tier["model"],
                    "attempts": attempts,
                    "requests_by_tier": requests_by_tier,
                    "directives": list(directives),
                    "required_rules": list(required),
                    "prompt_messages": build_teacher_messages(
                        problem,
                        boa_spec=boa_spec,
                        directives=directives,
                        required=required,
                    ),
                    "generated_at": _now(),
                }
            previous = raw
    return {
        "certified": False,
        "problem_id": problem["problem_id"],
        "attempts": attempts,
        "requests_by_tier": requests_by_tier,
        "directives": list(directives),
        "last_diagnostics": (diagnostics or "")[-2000:],
        "generated_at": _now(),
    }


def summarize_usage(run_dir: Path) -> dict[str, Any]:
    """Cost/token rollup from the ChatClient cache audit files (exact:
    one record per real API call; cache hits append nothing)."""

    per_model: dict[str, dict[str, float]] = {}
    for cache_file in sorted(run_dir.glob("cache_*.jsonl")):
        for line in cache_file.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = record.get("response") or {}
            usage = response.get("usage") or {}
            model = str(
                response.get("model")
                or (record.get("endpoint") or {}).get("model")
                or "unknown"
            )
            bucket = per_model.setdefault(
                model,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
            )
            bucket["calls"] += 1
            bucket["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            bucket["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            bucket["cost_usd"] += float(usage.get("cost") or 0.0)
    total = round(sum(b["cost_usd"] for b in per_model.values()), 4)
    for bucket in per_model.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 4)
    return {"per_model": per_model, "total_cost_usd": total}
