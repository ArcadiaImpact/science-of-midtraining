"""Tier-2 stdio->function conversion for open-r1/codeforces (SPEC.md §3.2).

An LLM rewrites the statement into a named-parameter function contract and
emits three snippets: ``parse_input`` (stdin text -> parameter literals),
``render_input`` (parameters -> stdin text) and ``parse_output`` (official
output -> expected literal). Verification is oracle-grade, end to end: for
every emitted test, the parameters parsed from the official input are
re-rendered to stdin, a known-correct human solution is run on that stdin,
and its output must exactly match the official output (CRLF/trailing-space
normalized). A converted problem exists ONLY if the oracle passes on every
test; the literal ``(args, expected)`` tuples are what it emits.

Screens (POOL_SURVEY conversion protocol): rating 1200-2100, stdio,
non-interactive, no checker (multi-answer), no float outputs, no oversized
tests, pre-2023 contests (LCB). The known-correct human solution comes from
open-r1/codeforces-submissions ``selected_accepted`` (Python preferred, C++
via g++ otherwise — the shard's Python coverage is ~0.5%, so C++ is the
working oracle in practice).
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import (  # noqa: E402
    _cell_rng,
    _json_hash,
    _subprocess_limits,
)
from experiments.python4.eft_scale.sources import (  # noqa: E402
    _count_reject,
    english_statement,
    normalize_json_value,
    supported_literal,
)


def norm_output(text: str) -> str:
    """Codeforces output normalization: CRLF, per-line trailing space, tail."""

    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).rstrip("\n")


_FLOATISH = re.compile(r"\d\.\d")


# ------------------------------------------------------------- selection


def select_cf_candidates(
    config: dict[str, Any], stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Screened verifiable rows joined with accepted human solutions."""

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    source = config["sources"]["codeforces"]
    conversion = config["conversion"]
    shards = list(source.get("shards") or [source["shard"]])
    columns = [
        "id", "title", "description", "input_format", "output_format",
        "interaction_format", "note", "examples", "rating", "official_tests",
        "input_mode", "generated_checker", "executable", "contest_start_year",
    ]
    eligible: dict[str, dict[str, Any]] = {}
    for shard in shards:
        shard_path = hf_hub_download(
            source["repo_id"], shard, repo_type=source["repo_type"]
        )
        table = pq.ParquetFile(shard_path).read(columns=columns)
        for i in range(table.num_rows):
            row = {c: table[c][i].as_py() for c in columns}
            rating = row["rating"]
            if rating is None or not (
                int(conversion["rating_min"]) <= rating <= int(conversion["rating_max"])
            ):
                continue
            if row["input_mode"] != "stdio" or not row["executable"]:
                continue
            if row["interaction_format"] or row["generated_checker"]:
                continue
            year = row["contest_start_year"]
            if year is None or year >= 2023:  # LCB screen (cutoff 2023-05-01)
                continue
            if not (row["description"] and row["input_format"] and row["output_format"]):
                continue
            if not english_statement(str(row["description"])):
                _count_reject(stats, "language_rejected")
                continue
            tests = row["official_tests"] or []
            if len(tests) < int(conversion["min_official_tests"]):
                continue
            limit = int(conversion["max_test_chars"])
            usable = [
                t for t in tests
                if len(t["input"]) <= limit and len(t["output"]) <= limit
            ]
            if len(usable) < int(conversion["min_official_tests"]):
                continue
            if any(_FLOATISH.search(t["output"]) for t in usable):
                continue
            row["official_tests"] = usable[: int(conversion["max_tests"])]
            eligible[row["id"]] = row

    submissions_path = hf_hub_download(
        source["submissions_repo"],
        source["submissions_file"],
        repo_type=source["repo_type"],
    )
    sub_table = pq.ParquetFile(submissions_path).read(
        columns=["problem_id", "programmingLanguage", "verdict", "source"]
    )
    solutions: dict[str, list[dict[str, str]]] = {}
    for i in range(sub_table.num_rows):
        pid = sub_table["problem_id"][i].as_py()
        if pid not in eligible:
            continue
        verdict = sub_table["verdict"][i].as_py()
        if verdict != "OK":
            continue
        language = str(sub_table["programmingLanguage"][i].as_py() or "")
        code = sub_table["source"][i].as_py()
        if not code:
            continue
        solutions.setdefault(pid, []).append({"language": language, "code": code})

    candidates = []
    for pid, row in eligible.items():
        pool = solutions.get(pid) or []
        # Python first (usable as a tagging reference), then C++ (oracle only).
        pool.sort(key=lambda s: (0 if s["language"].startswith(("Python 3", "PyPy 3")) else 1))
        if pool:
            candidates.append({**row, "human_solutions": pool[:3]})
    rng = _cell_rng(int(config["seed"]), "cf-conversion-sample")
    rng.shuffle(candidates)
    return candidates[: int(conversion["candidates"])]


# ------------------------------------------- code_contests (second Tier-2)

#: deepmind/code_contests language enum (POOL_SURVEY source #3).
_CC_LANGUAGES = {1: "Python 2", 2: "GNU C++17", 3: "Python 3", 4: "Java"}
_CC_SOURCE_CODEFORCES = 2


def _cc_tests(row: dict[str, Any], *, max_chars: int, max_tests: int) -> list[dict[str, str]]:
    """Official (public+private) tests first, validated generated as pad."""

    tests: list[dict[str, str]] = []
    for kind in ("public_tests", "private_tests", "generated_tests"):
        bundle = row.get(kind) or {}
        for stdin_text, stdout_text in zip(
            bundle.get("input") or (), bundle.get("output") or ()
        ):
            if len(tests) >= max_tests:
                return tests
            if not stdin_text or stdout_text is None:
                continue
            if len(stdin_text) > max_chars or len(stdout_text) > max_chars:
                continue
            if _FLOATISH.search(stdout_text):
                continue
            tests.append({"input": stdin_text, "output": stdout_text})
    return tests


def select_cc_candidates(
    config: dict[str, Any],
    stats: dict[str, int] | None = None,
    *,
    exclude_cf_ids: frozenset[str] | set[str] = frozenset(),
) -> list[dict[str, Any]]:
    """deepmind/code_contests rows shaped for ``convert_one``.

    Codeforces-sourced rows only (the rating filter needs cf_rating, which
    is also the honest difficulty assessor), rating 1200-2100,
    non-interactive (cf_tags), stdin/stdout (no file IO), with at least one
    Python-3/C++ correct solution. The collection is pre-2022 —
    LCB-clean by construction (POOL_SURVEY). ``exclude_cf_ids`` drops rows
    whose ``{contest}/{index}`` already sits in the open-r1 conversion
    order (shared Codeforces ancestry; the statement-level near-dup screen
    at accept time is the backstop).
    """

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    source = config["sources"]["code_contests"]
    conversion = config["conversion"]
    columns = [
        "name", "description", "public_tests", "private_tests",
        "generated_tests", "source", "solutions", "cf_contest_id", "cf_index",
        "cf_rating", "cf_tags", "input_file", "output_file",
    ]
    candidates: list[dict[str, Any]] = []
    for shard in source["shards"]:
        shard_path = hf_hub_download(
            source["repo_id"], shard, repo_type=source["repo_type"]
        )
        parquet = pq.ParquetFile(shard_path)
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                if row["source"] != _CC_SOURCE_CODEFORCES:
                    continue
                rating = row["cf_rating"]
                if not rating or not (
                    int(conversion["rating_min"]) <= rating <= int(conversion["rating_max"])
                ):
                    continue
                if not row["cf_contest_id"] or not row["cf_index"]:
                    continue
                cc_id = f"{row['cf_contest_id']}/{row['cf_index']}"
                if cc_id in exclude_cf_ids:
                    _count_reject(stats, "cf_id_already_in_open_r1_order")
                    continue
                if "interactive" in (row["cf_tags"] or ()):
                    continue
                if row["input_file"] or row["output_file"]:
                    continue
                description = str(row["description"] or "").strip()
                if not description:
                    continue
                if not english_statement(description):
                    _count_reject(stats, "language_rejected")
                    continue
                tests = _cc_tests(
                    row,
                    max_chars=int(conversion["max_test_chars"]),
                    max_tests=int(conversion["max_tests"]),
                )
                if len(tests) < int(conversion["min_official_tests"]):
                    continue
                solutions_bundle = row.get("solutions") or {}
                pool = [
                    {"language": _CC_LANGUAGES[lang], "code": code}
                    for lang, code in zip(
                        solutions_bundle.get("language") or (),
                        solutions_bundle.get("solution") or (),
                    )
                    if lang in (2, 3) and code and code.strip()
                ]
                # Python first (usable as a tagging reference), then C++.
                pool.sort(key=lambda s: (0 if s["language"].startswith("Python 3") else 1))
                if not pool:
                    continue
                candidates.append(
                    {
                        "id": cc_id,
                        "id_prefix": "cc",
                        "source_key": "code_contests",
                        "title": str(row["name"] or cc_id),
                        "description": description,
                        "input_format": "",
                        "output_format": "",
                        "interaction_format": None,
                        "note": None,
                        "examples": [
                            {"input": t["input"], "output": t["output"]}
                            for t in tests[:2]
                        ],
                        "rating": int(rating),
                        "official_tests": tests,
                        "contest_start_year": None,
                        "human_solutions": pool[:3],
                    }
                )
    rng = _cell_rng(int(config["seed"]), "cc-conversion-sample")
    rng.shuffle(candidates)
    return candidates[: int(conversion["candidates"])]


# ------------------------------------------------- human solution runner


def _compile_cpp(code: str, directory: Path, *, language: str, timeout: int) -> Path | None:
    std = "gnu++20" if "20" in language else "gnu++17"
    source = directory / "human.cpp"
    source.write_text(code)
    binary = directory / "human"
    try:
        result = subprocess.run(
            ["g++", "-O2", f"-std={std}", "-o", str(binary), str(source)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    return binary if result.returncode == 0 else None


class HumanRunner:
    """Compile-once runner for one accepted submission (Python or C++)."""

    def __init__(self, solution: dict[str, str], *, compile_timeout: int):
        self.language = solution["language"]
        self.code = solution["code"]
        self._dir = tempfile.TemporaryDirectory(prefix="eft-scale-human-")
        directory = Path(self._dir.name)
        if self.language.startswith(("Python", "PyPy")):
            script = directory / "human.py"
            script.write_text(self.code)
            self._argv = [sys.executable, str(script)]
            self.ready = True
        else:
            binary = _compile_cpp(
                self.code, directory, language=self.language, timeout=compile_timeout
            )
            self._argv = [str(binary)] if binary else []
            self.ready = binary is not None

    def run(self, stdin_text: str, *, timeout: int) -> str | None:
        if not self.ready:
            return None
        try:
            result = subprocess.run(
                self._argv,
                input=stdin_text,
                cwd=self._dir.name,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                text=True,
                capture_output=True,
                timeout=timeout,
                preexec_fn=_subprocess_limits(timeout),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        return result.stdout

    def close(self) -> None:
        self._dir.cleanup()


def prescreen_human_solution(
    runner: HumanRunner, tests: Sequence[dict[str, str]], *, timeout: int
) -> bool:
    """The human solution must reproduce every official output as-is."""

    for test in tests:
        out = runner.run(test["input"], timeout=timeout)
        if out is None or norm_output(out) != norm_output(test["output"]):
            return False
    return True


# ------------------------------------------------------------ LLM contract


_CONVERSION_INSTRUCTIONS = """\
You convert a Codeforces stdin/stdout problem into a named-parameter function
problem. Return STRICT JSON (no Markdown fence, no prose) with exactly these
keys:

- "parameter_names": snake_case names for the function parameters, in order,
  one per logical input value (scalars, strings, or lists; no nested parsing
  left to the solver).
- "statement": the rewritten problem statement. Keep the narrative and all
  constraints, but describe the inputs as function parameters (by name) and
  the required answer as a return value. Remove stdin/stdout phrasing. Do not
  include example blocks.
- "parse_input": Python 3 code defining parse_input(text) -> dict mapping
  each parameter name to its literal value, parsing one official stdin text.
- "render_input": Python 3 code defining render_input(**params) -> str
  producing a stdin text equivalent to the one parse_input consumed
  (including the final newline if the original had one).
- "parse_output": Python 3 code defining parse_output(text) -> the expected
  return value for that test, as a plain literal (int/str/bool/list; never
  float). For multi-line outputs return a list.
- "changes": 2-4 sentences summarizing what the conversion changed.

The three snippets may use only the Python standard library modules math, re,
string, itertools and collections. They must be deterministic and total on
the official tests.
"""


def build_conversion_messages(
    row: dict[str, Any], *, previous: str | None = None, diagnostics: str | None = None
) -> list[dict[str, str]]:
    examples = row.get("examples") or []
    example_text = "\n\n".join(
        f"Example input:\n{e['input']}\nExample output:\n{e['output']}"
        for e in examples[:2]
        if isinstance(e, dict) and e.get("input") is not None
    )
    user_parts = [
        _CONVERSION_INSTRUCTIONS,
        f"Title: {row['title']}",
        f"Statement:\n{row['description']}",
    ]
    # code_contests descriptions embed their Input/Output sections; the
    # open-r1 schema carries them as separate fields.
    if row.get("input_format"):
        user_parts.append(f"Input format:\n{row['input_format']}")
    if row.get("output_format"):
        user_parts.append(f"Output format:\n{row['output_format']}")
    if row.get("note"):
        user_parts.append(f"Note:\n{row['note']}")
    if example_text:
        user_parts.append(example_text)
    if previous is not None:
        user_parts.extend(
            [
                "Your previous JSON failed oracle verification:",
                previous[:4000],
                "Diagnostics:",
                diagnostics or "verification failed",
                "Return corrected STRICT JSON.",
            ]
        )
    return [{"role": "user", "content": "\n\n".join(user_parts)}]


def parse_conversion_response(text: str) -> dict[str, Any] | None:
    body = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
    if fence:
        body = fence.group(1)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(body[start : end + 1])
        except json.JSONDecodeError:
            return None
    keys = {"parameter_names", "statement", "parse_input", "render_input", "parse_output", "changes"}
    if not isinstance(payload, dict) or not keys <= set(payload):
        return None
    names = payload["parameter_names"]
    if not (
        isinstance(names, list)
        and names
        and all(re.fullmatch(r"[a-z][a-z0-9_]*", str(n)) for n in names)
    ):
        return None
    return payload


# ------------------------------------------------------------- the oracle


_SNIPPET_ALLOWED_IMPORTS = {"math", "re", "string", "itertools", "collections"}


def _snippets_safe(payload: dict[str, Any]) -> bool:
    for key in ("parse_input", "render_input", "parse_output"):
        try:
            tree = ast.parse(str(payload[key]))
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    roots = {(node.module or "").split(".", 1)[0]}
                else:
                    roots = {alias.name.split(".", 1)[0] for alias in node.names}
                if roots - _SNIPPET_ALLOWED_IMPORTS:
                    return False
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"open", "eval", "exec", "compile", "__import__", "input"}:
                    return False
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return False
    return True


_DRIVER_TAIL = """
import json as __json, sys as __sys

def __normalize(value):
    if isinstance(value, tuple):
        return [__normalize(v) for v in value]
    if isinstance(value, list):
        return [__normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: __normalize(v) for k, v in value.items()}
    return value

__out = []
for __test in __TESTS:
    __record = {}
    try:
        __params = parse_input(__test["input"])
        if set(__params) != set(__PARAMETER_NAMES):
            raise ValueError(f"parse_input keys {sorted(__params)} != parameters {sorted(__PARAMETER_NAMES)}")
        __record["args"] = __normalize([__params[__n] for __n in __PARAMETER_NAMES])
        __record["stdin"] = render_input(**__params)
        if not isinstance(__record["stdin"], str):
            raise ValueError("render_input did not return str")
        __record["expected"] = __normalize(parse_output(__test["output"]))
        __record["ok"] = True
    except Exception as __error:
        __record = {"ok": False, "err": repr(__error)[:300]}
    __out.append(__record)
print(__json.dumps(__out))
"""


def run_conversion_driver(
    payload: dict[str, Any],
    tests: Sequence[dict[str, str]],
    *,
    timeout: int,
) -> list[dict[str, Any]] | str:
    """Execute the three snippets over the official tests (sandboxed)."""

    if not _snippets_safe(payload):
        return "conversion snippets failed the static safety screen"
    normalized_tests = [
        {"input": t["input"].replace("\r\n", "\n"), "output": norm_output(t["output"])}
        for t in tests
    ]
    driver = "\n\n".join(
        [
            str(payload["parse_input"]),
            str(payload["render_input"]),
            str(payload["parse_output"]),
            f"__PARAMETER_NAMES = {json.dumps([str(n) for n in payload['parameter_names']])}",
            f"__TESTS = __import__('json').loads({json.dumps(json.dumps(normalized_tests))})",
            _DRIVER_TAIL,
        ]
    )
    with tempfile.TemporaryDirectory(prefix="eft-scale-driver-") as directory:
        script = Path(directory) / "driver.py"
        script.write_text(driver)
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=directory,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                text=True,
                capture_output=True,
                timeout=timeout,
                preexec_fn=_subprocess_limits(timeout),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "conversion driver timed out"
    if result.returncode != 0:
        return f"conversion driver crashed: {result.stderr[-400:]}"
    try:
        records = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return "conversion driver produced no JSON"
    return records


def oracle_verify(
    payload: dict[str, Any],
    row: dict[str, Any],
    runner: HumanRunner,
    *,
    driver_timeout: int,
    run_timeout: int,
    min_tests: int,
) -> tuple[list[dict[str, Any]] | None, str]:
    """Oracle: parse -> re-render -> run the human solution -> exact match."""

    tests = row["official_tests"]
    records = run_conversion_driver(payload, tests, timeout=driver_timeout)
    if isinstance(records, str):
        return None, records
    literal_tests: list[dict[str, Any]] = []
    for index, (record, test) in enumerate(zip(records, tests)):
        if not record.get("ok"):
            return None, f"test {index}: snippet failure: {record.get('err')}"
        args = normalize_json_value(record["args"])
        expected = normalize_json_value(record["expected"])
        if not (supported_literal(args) and supported_literal(expected)):
            return None, f"test {index}: non-literal args/expected (floats are excluded)"
        rebuilt = record["stdin"]
        out = runner.run(rebuilt, timeout=run_timeout)
        if out is None:
            return None, f"test {index}: human solution failed on re-rendered stdin"
        if norm_output(out) != norm_output(test["output"]):
            return None, (
                f"test {index}: round-trip mismatch: human solution on "
                f"render_input(parse_input(input)) produced "
                f"{norm_output(out)[:120]!r}, official output is "
                f"{norm_output(test['output'])[:120]!r}"
            )
        literal_tests.append({"args": args, "kwargs": {}, "expected": expected})
    if len(literal_tests) < min_tests:
        return None, f"only {len(literal_tests)} oracle-verified tests (< {min_tests})"
    if len({json.dumps(t["expected"], sort_keys=True) for t in literal_tests}) <= 1:
        return None, "degenerate tests: every expected value is identical"
    return literal_tests, "ok"


# ------------------------------------------------------------ orchestration


async def convert_one(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
    client: Any,
    guard: Any,
    run_dir: Path,
    call_teacher: Any,
) -> dict[str, Any]:
    """Attempt one conversion; returns a record with the problem on success."""

    conversion = config["conversion"]
    prefix = str(row.get("id_prefix") or "cf")
    problem_id = f"{prefix}:{row['id']}"
    record: dict[str, Any] = {
        "cf_id": row["id"],
        "problem_id": problem_id,
        "title": row["title"],
        "rating": row["rating"],
    }

    def _pick_runner() -> HumanRunner | None:
        for solution in row["human_solutions"]:
            candidate = HumanRunner(
                solution, compile_timeout=int(conversion["compile_timeout_seconds"])
            )
            if candidate.ready and prescreen_human_solution(
                candidate,
                row["official_tests"],
                timeout=int(conversion["human_run_timeout_seconds"]),
            ):
                return candidate
            candidate.close()
        return None

    runner = await asyncio.get_running_loop().run_in_executor(None, _pick_runner)
    if runner is None:
        record.update(converted=False, reason="no accepted solution reproduces the official tests locally")
        return record
    record["human_solution_language"] = runner.language

    previous = diagnostics = None
    payload = None
    literal_tests = None
    try:
        for request_index in range(int(conversion["max_requests"])):
            messages = build_conversion_messages(
                row, previous=previous, diagnostics=diagnostics
            )
            text = await call_teacher(
                client,
                messages,
                max_tokens=int(conversion["max_tokens"]),
                reasoning_effort=str(conversion["reasoning_effort"]),
                guard=guard,
                usage_log=run_dir / "teacher_usage.jsonl",
                tag={"problem_id": problem_id, "tier": "conversion",
                     "request_index": request_index, "kind": "conversion"},
                cache_salt=f"{problem_id}:conv:{request_index}",
            )
            payload = parse_conversion_response(text)
            if payload is None:
                previous, diagnostics = text, "response was not the required strict JSON"
                continue
            loop = asyncio.get_running_loop()
            literal_tests, note = await loop.run_in_executor(
                None,
                lambda payload=payload: oracle_verify(
                    payload,
                    row,
                    runner,
                    driver_timeout=int(conversion["human_run_timeout_seconds"]),
                    run_timeout=int(conversion["human_run_timeout_seconds"]),
                    min_tests=int(conversion["min_official_tests"]),
                ),
            )
            if literal_tests is not None:
                break
            previous, diagnostics = text, note
    finally:
        runner.close()
    if payload is None or literal_tests is None:
        record.update(converted=False, reason=f"oracle rejected: {diagnostics}")
        return record

    source = config["sources"][str(row.get("source_key") or "codeforces")]
    statement = str(payload["statement"]).strip()
    reference = None
    if runner.language.startswith(("Python 3", "PyPy 3")):
        reference = runner.code
    problem = {
        "problem_id": problem_id,
        "statement": statement,
        "parameter_names": [str(n) for n in payload["parameter_names"]],
        "tests": literal_tests,
        "difficulty_source_label": None,
        "cf_rating": int(row["rating"]),
        "source_dataset": source["repo_id"],
        "source_site": source["site"],
        "license": source["license"],
        "dates": {"contest_start_year": row.get("contest_start_year")},
        "reference_python3": reference,
        "reference_solution": runner.code,
        "reference_language": (
            "python3" if reference is not None else runner.language
        ),
        "tier": "converted",
        "source_split": "train",
        "source_row_sha256": _json_hash({"id": row["id"], "title": row["title"]}),
        "reference_verified": True,
        "conversion": {
            "original_statement_excerpt": str(row["description"])[:1200],
            "original_input_format": str(row["input_format"])[:600],
            "original_output_format": str(row["output_format"])[:600],
            "changes": str(payload["changes"]),
            "parse_input": str(payload["parse_input"]),
            "render_input": str(payload["render_input"]),
            "parse_output": str(payload["parse_output"]),
            "oracle_language": runner.language,
            "oracle_tests": len(literal_tests),
        },
    }
    record.update(converted=True, problem=problem)
    return record


async def convert_stage(
    config: dict[str, Any],
    run_dir: Path,
    *,
    client: Any,
    guard: Any,
    call_teacher: Any,
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the Tier-2 conversion wave; returns (problems, attempt_records).

    ``candidates`` lets the full build pass a frozen, disk-persisted tranche
    (resumable prefixes of one shuffled order); the default reselects, which
    is only deterministic while the config stays byte-identical.
    """

    if candidates is None:
        candidates = select_cf_candidates(config)
    semaphore = asyncio.Semaphore(int(config["conversion"]["concurrency"]))

    async def guarded(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await convert_one(
                row,
                config=config,
                client=client,
                guard=guard,
                run_dir=run_dir,
                call_teacher=call_teacher,
            )

    records = list(await asyncio.gather(*(guarded(row) for row in candidates)))
    problems = [record["problem"] for record in records if record.get("converted")]
    public_records = [
        {key: value for key, value in record.items() if key != "problem"}
        for record in records
    ]
    return problems, public_records
