"""Tier-1 source normalizers for the EFT scale pilot (SPEC.md §3.2).

Each loader normalizes one upstream dataset into the shared problem dict:

``problem_id`` (namespaced by source), ``statement``, ``parameter_names``,
``tests`` (literal ``{"args", "kwargs", "expected"}`` records, >= 3),
``difficulty_source_label``, ``cf_rating``, ``source_dataset``,
``source_site``, ``license``, ``dates``, ``reference_python3`` (or a
non-Python reference via ``reference_language``), ``tier``, ``source_split``,
``source_row_sha256``.

Screens applied here (POOL_SURVEY / SPEC §3.5):

- LCB: drop LeetCode/Codeforces/AtCoder-sourced rows dated >= 2023-05-01
  where dates exist; newfacade's test split is never loaded.
- TACO HackerRank rows dropped (rights unknown).
- Literal hygiene: every test value must be a JSON-stable, float-free
  literal (the Boa harness asserts exact equality on JSONL-shipped tests).

Reference re-verification (``verify_reference``) executes the dataset's own
reference against the literal tests in a sandboxed CPython subprocess; it is
deliberately run lazily (at scheduling time) so the pilot only spends
subprocess time on problems it actually attempts. APPS/TACO expected-value
list-wrapping is resolved here: an interpretation (direct vs unwrap) must
hold for *every* test or the problem is rejected.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import (  # noqa: E402
    _json_hash,
    _literal_value,
    _parameter_names,
    _safe_tree,
    _subprocess_limits,
    normalize_problem,
)

# ---------------------------------------------------------- language screen

#: Words that are common in English problem statements and rare in the
#: pool's actual non-English contamination (Russian/Chinese Codeforces
#: mirrors). Deliberately excludes "a"/"an"/"in"/"it", which are words (or
#: word-alikes) in several European languages.
_COMMON_ENGLISH_WORDS = frozenset(
    """the of to is are you your given return that each for with find this
    from will must number every when where output input array string integer
    and or not which contains should function value list""".split()
)

#: Statements shorter than this pass on the non-ASCII ratio alone (very
#: short English statements — rStar one-liners — can lack list words).
_COMMON_WORD_MIN_CHARS = 300


def english_statement(text: str) -> bool:
    """Cheap language screen (pilot finding: cf:929/C shipped a Russian
    Codeforces-mirror statement; statement text is never paraphrased, so
    non-English candidates must be rejected at normalization time).

    Rejects when alphabetic characters are >25% non-ASCII (Cyrillic/CJK are
    alphabetic; math symbols are not), or when a statement long enough to
    have connective tissue (> ~300 chars) contains fewer than two distinct
    common English words.
    """

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    non_ascii = sum(1 for ch in letters if ord(ch) > 127)
    if non_ascii / len(letters) > 0.25:
        return False
    if len(text) <= _COMMON_WORD_MIN_CHARS:
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & _COMMON_ENGLISH_WORDS) >= 2


def _count_reject(stats: dict[str, int] | None, key: str) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


# ---------------------------------------------------------------- literals

#: Magnitude cap on test integers: APPS carries pathological rows whose
#: literals run to thousands of digits; nothing in the pool legitimately
#: needs more than this.
MAX_TEST_INT = 10**24
#: Cap on one test's JSON size — every literal is rendered into Boa harness
#: source, so unbounded tests would stall compilation.
MAX_TEST_JSON_CHARS = 2000


def _parse_python(text: str) -> ast.Module:
    """ast.parse with dataset-noise SyntaxWarnings suppressed."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(text)


def supported_literal(value: Any) -> bool:
    """JSON-stable, float-free literal (Boa harness asserts exact equality)."""

    if type(value) is int:
        return abs(value) < MAX_TEST_INT
    if value is None or type(value) in (bool, str):
        return True
    if isinstance(value, list):
        return all(supported_literal(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and supported_literal(item)
            for key, item in value.items()
        )
    return False


def normalize_json_value(value: Any) -> Any:
    """Tuple->list normalization so equality matches the JSONL-shipped tests."""

    if isinstance(value, tuple):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_json_value(item) for key, item in value.items()}
    return value


def _clean_tests(tests: Sequence[dict[str, Any]], *, max_tests: int) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for test in tests:
        args = normalize_json_value(test["args"])
        kwargs = normalize_json_value(test.get("kwargs") or {})
        expected = normalize_json_value(test["expected"])
        if not (
            supported_literal(args)
            and supported_literal(expected)
            and supported_literal(kwargs)
        ):
            continue
        test_row = {"args": args, "kwargs": kwargs, "expected": expected}
        if len(json.dumps(test_row)) > MAX_TEST_JSON_CHARS:
            continue
        cleaned.append(test_row)
        if len(cleaned) == max_tests:
            break
    return cleaned


def _degenerate(tests: Sequence[dict[str, Any]]) -> bool:
    return len({json.dumps(t["expected"], sort_keys=True) for t in tests}) <= 1


# ------------------------------------------------------------------ screens


def lcb_screened(site: str | None, date_text: str | None, *, cutoff: str, sites: Sequence[str]) -> bool:
    """True when the row must be dropped under the LiveCodeBench screen."""

    if not site or site.lower() not in {s.lower() for s in sites}:
        return False
    if not date_text:
        return False
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(date_text))
    if not match:
        return False
    row_date = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    limit = date.fromisoformat(cutoff)
    return row_date >= limit


# --------------------------------------------------------------- newfacade


def load_newfacade_pool(
    config: dict[str, Any],
    *,
    battery_ids: set[str],
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Normalized newfacade train rows (test split never loaded)."""

    from huggingface_hub import hf_hub_download

    source = config["sources"]["newfacade"]
    decon = config["decontamination"]
    dataset_cfg = config["dataset"]
    path = hf_hub_download(
        source["repo_id"],
        source["train_file"],
        repo_type=source["repo_type"],
        revision=source["revision"],
    )
    problems: list[dict[str, Any]] = []
    cap = int(dataset_cfg["per_source_candidate_cap"])
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                base = normalize_problem(
                    row,
                    min_tests=int(dataset_cfg["min_tests_per_problem"]),
                    max_tests=int(dataset_cfg["max_tests_per_problem"]),
                )
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if base["problem_id"] in battery_ids:
                continue
            estimated = str(row.get("estimated_date") or "")
            if lcb_screened(
                source["site"], estimated,
                cutoff=decon["lcb_cutoff"], sites=decon["lcb_sites"],
            ):
                continue
            tests = _clean_tests(base["tests"], max_tests=int(dataset_cfg["max_tests_per_problem"]))
            if len(tests) < int(dataset_cfg["min_tests_per_problem"]):
                continue
            if len(base["problem"]) > int(dataset_cfg["max_statement_chars"]):
                continue
            if not english_statement(base["problem"]):
                _count_reject(stats, "language_rejected")
                continue
            problems.append(
                {
                    "problem_id": f"newfacade:{base['problem_id']}",
                    "statement": base["problem"],
                    "parameter_names": base["parameter_names"],
                    "tests": tests,
                    "difficulty_source_label": base["difficulty"],
                    "cf_rating": None,
                    "source_dataset": source["repo_id"],
                    "source_site": source["site"],
                    "license": source["license"],
                    "dates": {"estimated_date": estimated or None},
                    "reference_python3": base["reference_python3"],
                    "reference_language": "python3",
                    "reference_entry_point": str(row.get("entry_point") or ""),
                    "tier": "native",
                    "source_split": "train",
                    "source_row_sha256": _json_hash(row),
                }
            )
            if len(problems) >= cap:
                break
    return problems


# --------------------------------------------------- TACO / APPS (fn_name)


def _fn_name_tests(io: dict[str, Any], *, max_tests: int) -> list[dict[str, Any]]:
    """Literal tests from the shared TACO/APPS ``input_output`` schema.

    ``outputs[i]`` is inconsistently list-wrapped across rows; both readings
    are kept as-is here and resolved by reference verification (an
    interpretation must hold for every test).
    """

    inputs = io.get("inputs")
    outputs = io.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return []
    tests = []
    for args, expected in zip(inputs, outputs):
        if not isinstance(args, list):
            continue
        tests.append({"args": args, "kwargs": {}, "expected": expected})
        if len(tests) == max_tests:
            break
    return tests


def _first_reference(solutions: Any, fn_name: str, max_candidates: int = 4) -> list[str]:
    """Parse-safe candidate references defining ``fn_name`` or ``Solution``."""

    if isinstance(solutions, str):
        try:
            solutions = json.loads(solutions)
        except json.JSONDecodeError:
            return []
    if not isinstance(solutions, list):
        return []
    picked: list[str] = []
    for candidate in solutions:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        try:
            tree = _parse_python(candidate)
        except SyntaxError:
            continue
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        if fn_name not in names and "Solution" not in names:
            continue
        if not _safe_tree(tree)[0]:
            continue
        picked.append(candidate)
        if len(picked) == max_candidates:
            break
    return picked


def _normalize_fn_name_row(
    row: dict[str, Any],
    *,
    problem_id: str,
    source_dataset: str,
    site: str,
    license_name: str,
    difficulty_label: str | None,
    dates: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    dataset_cfg = config["dataset"]
    io_raw = row.get("input_output")
    try:
        io = json.loads(io_raw) if isinstance(io_raw, str) else (io_raw or {})
    except (ValueError, TypeError):  # includes the 3.12 int-digit limit
        return None
    fn_name = str(io.get("fn_name") or "").strip()
    if not fn_name:
        return None
    statement = str(row.get("question") or "").strip()
    if not statement or len(statement) > int(dataset_cfg["max_statement_chars"]):
        return None
    tests = _fn_name_tests(io, max_tests=int(dataset_cfg["max_tests_per_problem"]))
    tests = _clean_tests(tests, max_tests=int(dataset_cfg["max_tests_per_problem"]))
    if len(tests) < int(dataset_cfg["min_tests_per_problem"]):
        return None
    references = _first_reference(row.get("solutions"), fn_name)
    if not references:
        return None
    starter = str(row.get("starter_code") or "")
    parameter_names: list[str] | None = None
    if starter.strip():
        try:
            parameter_names = _parameter_names(starter)
        except ValueError:
            parameter_names = None
    if parameter_names is None:
        parameter_names = _parameters_from_solution(references[0], fn_name)
    if not parameter_names:
        return None
    if any(len(test["args"]) != len(parameter_names) for test in tests):
        return None
    return {
        "problem_id": problem_id,
        "statement": statement,
        "parameter_names": parameter_names,
        "tests": tests,
        "difficulty_source_label": difficulty_label,
        "cf_rating": None,
        "source_dataset": source_dataset,
        "source_site": site,
        "license": license_name,
        "dates": dates,
        "reference_python3": references[0],
        "reference_candidates": references,
        "reference_language": "python3",
        "reference_fn_name": fn_name,
        "tier": "native",
        "source_split": "train",
        "source_row_sha256": _json_hash(
            {k: row.get(k) for k in ("question", "input_output", "starter_code")}
        ),
    }


def _parameters_from_solution(solution: str, fn_name: str) -> list[str] | None:
    """Recover parameter names from the reference when starter_code is absent."""

    try:
        tree = _parse_python(solution)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            names = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
            if names and names[0] in {"self", "cls"}:
                names = names[1:]
            return names or None
    return None


def _site_from_url(url: str | None) -> str | None:
    if not url:
        return None
    for site in ("leetcode", "codeforces", "atcoder", "codewars", "codechef",
                 "hackerrank", "geeksforgeeks", "kattis", "hackerearth"):
        if site in url.lower():
            return site
    return None


def load_taco_pool(
    config: dict[str, Any], stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """likaixin/TACO-verified call-based rows via the parquet conversion."""

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    source = config["sources"]["taco_verified"]
    decon = config["decontamination"]
    path = hf_hub_download(
        source["repo_id"],
        source["file"],
        repo_type=source["repo_type"],
        revision=source["revision"],
    )
    parquet = pq.ParquetFile(path)
    columns = ["question", "solutions", "starter_code", "input_output",
               "difficulty", "source", "date", "url"]
    drop_sites = {s.lower() for s in source.get("drop_sites", [])}
    problems: list[dict[str, Any]] = []
    cap = int(config["dataset"]["per_source_candidate_cap"])
    index = 0
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=columns)
        for i in range(table.num_rows):
            index += 1
            io_raw = table["input_output"][i].as_py()
            if not io_raw or '"fn_name"' not in io_raw[:4000]:
                continue
            site = str(table["source"][i].as_py() or "").lower() or None
            if site in drop_sites:
                continue
            date_text = table["date"][i].as_py()
            if lcb_screened(site, date_text, cutoff=decon["lcb_cutoff"], sites=decon["lcb_sites"]):
                continue
            row = {c: table[c][i].as_py() for c in columns}
            if not english_statement(str(row.get("question") or "")):
                _count_reject(stats, "language_rejected")
                continue
            problem = _normalize_fn_name_row(
                row,
                problem_id=f"tacov:{index - 1}",
                source_dataset=source["repo_id"],
                site=site or "unknown",
                license_name=source["license"],
                difficulty_label=str(row.get("difficulty") or "") or None,
                dates={"date": date_text},
                config=config,
            )
            if problem is not None:
                problems.append(problem)
                if len(problems) >= cap:
                    return problems
    return problems


def load_apps_pool(
    config: dict[str, Any], stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """codeparrot/apps call-based train rows (fn_name in input_output)."""

    from huggingface_hub import hf_hub_download

    source = config["sources"]["apps"]
    decon = config["decontamination"]
    path = hf_hub_download(source["repo_id"], source["file"], repo_type=source["repo_type"])
    problems: list[dict[str, Any]] = []
    cap = int(config["dataset"]["per_source_candidate_cap"])
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            io_raw = row.get("input_output") or ""
            if '"fn_name"' not in io_raw[:4000]:
                continue
            site = _site_from_url(row.get("url"))
            # APPS carries no dates; the collection predates the LCB windows
            # (POOL_SURVEY) so the date screen is vacuous here by design.
            if lcb_screened(site, None, cutoff=decon["lcb_cutoff"], sites=decon["lcb_sites"]):
                continue
            if not english_statement(str(row.get("question") or "")):
                _count_reject(stats, "language_rejected")
                continue
            problem = _normalize_fn_name_row(
                row,
                problem_id=f"apps:{row.get('id')}",
                source_dataset=source["repo_id"],
                site=site or "unknown",
                license_name=source["license"],
                difficulty_label=str(row.get("difficulty") or "") or None,
                dates={},
                config=config,
            )
            if problem is not None:
                problems.append(problem)
                if len(problems) >= cap:
                    break
    return problems


# -------------------------------------------------------------------- rStar
#
# 480GB dataset; never downloaded whole. seed_testcase rows store per-test
# ASSERT STATEMENTS: ``inputs`` is a JSON string of assert-source strings
# (``assert trace([[1,2]]) == 3``), ``outputs`` all-empty, and
# ``is_synthesized`` / ``test_case_type`` are JSON strings of per-test flags.
# We keep only is_synthesized == 0 tests (the original, non-mutual-verified
# ones — "respect verified flags") and join the reference from seed_sft rows
# with ``verified AND is_passed`` via remote row-group range reads (the
# seed_sft shards are ordered by seed_N, ~986 rows per row group).


def parse_rstar_assert(source: str, func_name: str) -> dict[str, Any] | None:
    """One assert-statement test -> literal args/kwargs/expected, or None."""

    try:
        tree = _parse_python(source)
    except SyntaxError:
        return None
    asserts = [node for node in tree.body if isinstance(node, ast.Assert)]
    if len(asserts) != 1:
        return None
    comparison = asserts[0].test
    if not (
        isinstance(comparison, ast.Compare)
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.Eq)
        and len(comparison.comparators) == 1
        and isinstance(comparison.left, ast.Call)
    ):
        return None
    call = comparison.left
    callee = call.func
    name = None
    if isinstance(callee, ast.Name):
        name = callee.id
    elif isinstance(callee, ast.Attribute):
        name = callee.attr
    if name != func_name:
        return None
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return None
    if any(keyword.arg is None for keyword in call.keywords):
        return None
    try:
        args = [_literal_value(arg) for arg in call.args]
        kwargs = {
            str(keyword.arg): _literal_value(keyword.value)
            for keyword in call.keywords
        }
        expected = _literal_value(comparison.comparators[0])
    except (ValueError, TypeError, SyntaxError):
        return None
    return {"args": args, "kwargs": kwargs, "expected": expected}


def normalize_rstar_row(
    testcase_row: dict[str, Any],
    reference: str | None,
    *,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Normalize one seed_testcase func_name row (+ verified reference)."""

    dataset_cfg = config["dataset"]
    func_name = str(testcase_row.get("func_name") or "").strip()
    statement = str(testcase_row.get("question") or "").strip()
    if not func_name or not statement:
        return None
    if len(statement) > int(dataset_cfg["max_statement_chars"]):
        return None
    if not reference:
        return None
    try:
        inputs = json.loads(testcase_row["inputs"])
        synthesized = json.loads(testcase_row["is_synthesized"])
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(inputs, list) or not isinstance(synthesized, list):
        return None
    tests: list[dict[str, Any]] = []
    for assert_source, synth_flag in zip(inputs, synthesized):
        if synth_flag:  # respect verified flags: original tests only
            continue
        if not isinstance(assert_source, str):
            continue
        test = parse_rstar_assert(assert_source, func_name)
        if test is not None:
            tests.append(test)
        if len(tests) == int(dataset_cfg["max_tests_per_problem"]):
            break
    tests = _clean_tests(tests, max_tests=int(dataset_cfg["max_tests_per_problem"]))
    if len(tests) < int(dataset_cfg["min_tests_per_problem"]):
        return None
    starter = str(testcase_row.get("starter_code") or "")
    parameter_names = None
    if starter.strip():
        try:
            parameter_names = _parameter_names(starter)
        except ValueError:
            parameter_names = None
    if parameter_names is None:
        parameter_names = _parameters_from_solution(reference, func_name)
    if not parameter_names:
        return None
    if any(
        len(test["args"]) + len(test["kwargs"]) != len(parameter_names)
        for test in tests
    ):
        return None
    try:
        tree = _parse_python(reference)
    except SyntaxError:
        return None
    if not _safe_tree(tree)[0]:
        return None
    source = config["sources"]["rstar"]
    return {
        "problem_id": f"rstar:{testcase_row.get('question_id')}",
        "statement": statement,
        "parameter_names": parameter_names,
        "tests": tests,
        "difficulty_source_label": None,
        "cf_rating": None,
        "source_dataset": source["dataset"],
        "source_site": "rstar_seed",
        "license": source["license"],
        "dates": {},
        "reference_python3": reference,
        "reference_language": "python3",
        "reference_fn_name": func_name,
        "tier": "native",
        "source_split": "train",
        "source_row_sha256": _json_hash(
            {k: testcase_row.get(k) for k in ("question_id", "question", "func_name")}
        ),
    }


def _rstar_sft_references(
    question_ids: Sequence[str],
    source: dict[str, Any],
    *,
    per_question: int = 3,
    time_budget_seconds: float = 300.0,
) -> dict[str, list[str]]:
    """``verified`` seed_sft codes per question via remote row-group reads.

    Never downloads a shard: each shard's parquet FOOTER statistics
    (question_id min/max per row group) prune to the few row groups that can
    contain a target id, and only those are range-read with a four-column
    projection (~10MB each; seed_sft shards hold 30 row groups of ~986
    rows, ordered by seed number — verified 2026-08-27). Candidates are
    ranked ``is_passed`` first, but ``verified`` alone qualifies — the
    caller re-executes every candidate against the original tests
    (verify_reference), which is the stronger oracle. A time budget bounds
    the join; unresolved ids are simply dropped by the caller.
    """

    import time

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    wanted = set(question_ids)
    ranked: dict[str, list[tuple[int, str]]] = {qid: [] for qid in wanted}
    fs = HfFileSystem()
    n_shards = int(source["sft_shards"])
    prefix = f"datasets/{source['dataset']}/{source['sft_dir']}"
    started = time.monotonic()

    def _saturated(qid: str) -> bool:
        return len(ranked[qid]) >= per_question

    for shard_index in range(n_shards):
        missing = {qid for qid in wanted if not _saturated(qid)}
        if not missing or time.monotonic() - started > time_budget_seconds:
            break
        name = f"{prefix}/data-{shard_index:05d}-of-{n_shards:05d}.parquet"
        try:
            with fs.open(name, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                qid_index = parquet.schema_arrow.names.index("question_id")
                candidate_groups = []
                for group in range(parquet.num_row_groups):
                    column = parquet.metadata.row_group(group).column(qid_index)
                    statistics = column.statistics
                    if statistics is None or not statistics.has_min_max:
                        candidate_groups.append(group)
                        continue
                    low, high = statistics.min, statistics.max
                    if isinstance(low, bytes):
                        low, high = low.decode(), high.decode()
                    if any(low <= qid <= high for qid in missing):
                        candidate_groups.append(group)
                for group in candidate_groups:
                    if time.monotonic() - started > time_budget_seconds:
                        break
                    table = parquet.read_row_group(
                        group,
                        columns=["question_id", "code", "verified", "is_passed"],
                    )
                    for i in range(table.num_rows):
                        qid = table["question_id"][i].as_py()
                        if qid not in missing or _saturated(qid):
                            continue
                        if not table["verified"][i].as_py():
                            continue
                        code = table["code"][i].as_py()
                        if isinstance(code, str) and code.strip():
                            passed = bool(table["is_passed"][i].as_py())
                            ranked[qid].append((0 if passed else 1, code))
        except OSError:
            continue  # one unreachable shard must not sink the source
    return {
        qid: [code for _, code in sorted(entries, key=lambda e: e[0])]
        for qid, entries in ranked.items()
        if entries
    }


def load_rstar_pool(
    config: dict[str, Any],
    *,
    cache_path: Path | None = None,
    stats: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """rStar seed func_name rows: pinned seed_testcase shard(s) (the only
    ones downloaded; chosen by a remote func_name column scan) + remote
    seed_sft reference join. Returns ``(problems, gap_note)``. The result is
    disk-cached (the remote join costs minutes)."""

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    source = config["sources"]["rstar"]
    shards = list(source.get("testcase_shards") or [source["testcase_shard"]])
    cache_key = "|".join(shards)
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("testcase_shard") == cache_key:
            return cached["problems"], cached.get("gap")
    try:
        columns = [
            "question_id", "question", "starter_code", "inputs", "outputs",
            "is_synthesized", "test_case_type", "func_name", "class_name",
        ]
        rows: list[dict[str, Any]] = []
        cap = int(source.get("max_candidates", 16))
        for shard in shards:
            if len(rows) >= cap:
                break
            shard_path = hf_hub_download(
                source["dataset"], shard, repo_type="dataset"
            )
            parquet = pq.ParquetFile(shard_path)
            for batch in parquet.iter_batches(batch_size=8, columns=columns):
                for row in batch.to_pylist():
                    if not str(row.get("func_name") or "").strip():
                        continue
                    if not english_statement(str(row.get("question") or "")):
                        _count_reject(stats, "language_rejected")
                        continue
                    rows.append(row)
                if len(rows) >= cap:
                    break
        rows = rows[:cap]
        if not rows:
            return [], (
                "rStar-Coder: pinned seed_testcase shards contain no func_name "
                "rows; source dropped"
            )
        references = _rstar_sft_references(
            [str(row["question_id"]) for row in rows],
            source,
            time_budget_seconds=float(source.get("sft_join_budget_seconds", 300.0)),
        )
        problems = []
        for row in rows:
            candidates = references.get(str(row["question_id"])) or []
            problem = None
            for reference in candidates:
                problem = normalize_rstar_row(row, reference, config=config)
                if problem is not None:
                    problem["reference_candidates"] = [
                        c for c in candidates
                        if normalize_rstar_row(row, c, config=config) is not None
                    ]
                    break
            if problem is not None:
                problems.append(problem)
        gap = None
        if not problems:
            gap = (
                "rStar-Coder: func_name rows found but none survived the "
                "verified-reference join + assert parsing; source dropped"
            )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "testcase_shard": cache_key,
                        "problems": problems,
                        "gap": gap,
                    }
                )
            )
        return problems, gap
    except Exception as error:  # noqa: BLE001 — source-drop is the contract
        return [], f"rStar-Coder: loader failed ({type(error).__name__}: {str(error)[:200]}); source dropped"


# ------------------------------------------------ reference verification


_HARNESS_PRELUDE = """\
import json, sys
from typing import *
import collections, math, functools, itertools, heapq, bisect, string, re, operator, random, copy
from collections import defaultdict, Counter, deque, OrderedDict
from functools import lru_cache, reduce
from itertools import permutations, combinations, product, accumulate, groupby, chain
from heapq import heappush, heappop, heapify
from bisect import bisect_left, bisect_right, insort
inf = float('inf')
"""

_HARNESS_TAIL = """
def __normalize(value):
    if isinstance(value, tuple):
        return [__normalize(v) for v in value]
    if isinstance(value, list):
        return [__normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: __normalize(v) for k, v in value.items()}
    return value

__results = []
for __test in __TESTS:
    try:
        __got = __CANDIDATE(*__test["args"], **__test.get("kwargs", {}))
        __results.append({"ok": True, "got": __normalize(__got)})
    except Exception as __error:
        __results.append({"ok": False, "err": repr(__error)[:200]})
print(json.dumps(__results))
"""


def _candidate_resolution(problem: dict[str, Any]) -> str | None:
    """Code that binds ``__CANDIDATE`` to the reference's entry callable."""

    reference = problem["reference_python3"]
    try:
        tree = _parse_python(reference)
    except SyntaxError:
        return None
    top = {type(node): node for node in tree.body}
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    entry = str(problem.get("reference_entry_point") or "")
    match = re.fullmatch(r"Solution\(\)\.(\w+)", entry.strip())
    if match and "Solution" in names:
        return f"__CANDIDATE = Solution().{match.group(1)}"
    fn_name = str(problem.get("reference_fn_name") or "")
    if fn_name and fn_name in names:
        return f"__CANDIDATE = {fn_name}"
    if "Solution" in names:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Solution":
                methods = [
                    child.name for child in node.body
                    if isinstance(child, ast.FunctionDef) and not child.name.startswith("_")
                ]
                if len(methods) >= 1:
                    return f"__CANDIDATE = Solution().{methods[0]}"
    del top
    return None


def run_reference(
    problem: dict[str, Any], *, timeout: int, python_executable: str = sys.executable
) -> list[dict[str, Any]] | None:
    """Execute the reference on the literal tests; None on harness failure."""

    resolution = _candidate_resolution(problem)
    if resolution is None:
        return None
    harness = "\n".join(
        [
            _HARNESS_PRELUDE,
            problem["reference_python3"],
            resolution,
            f"__TESTS = json.loads({json.dumps(json.dumps(problem['tests']))})",
            _HARNESS_TAIL,
        ]
    )
    with tempfile.TemporaryDirectory(prefix="eft-scale-ref-") as directory:
        script = Path(directory) / "reference_check.py"
        script.write_text(harness)
        try:
            result = subprocess.run(
                [python_executable, str(script)],
                cwd=directory,
                env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"},
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
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    if not isinstance(payload, list) or len(payload) != len(problem["tests"]):
        return None
    return payload


def resolve_expected_interpretation(
    tests: Sequence[dict[str, Any]], got_values: Sequence[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]] | None:
    """Resolve APPS/TACO output list-wrapping against reference outputs.

    Returns ``(interpretation, resolved_tests)`` where interpretation is
    ``direct`` or ``unwrap``; one interpretation must hold for every test.
    """

    if any(not record.get("ok") for record in got_values):
        return None
    gots = [record["got"] for record in got_values]
    if all(got == test["expected"] for got, test in zip(gots, tests)):
        return "direct", [dict(test) for test in tests]
    if all(
        isinstance(test["expected"], list)
        and len(test["expected"]) == 1
        and got == test["expected"][0]
        for got, test in zip(gots, tests)
    ):
        resolved = []
        for test in tests:
            resolved.append({**test, "expected": test["expected"][0]})
        return "unwrap", resolved
    return None


def verify_reference(problem: dict[str, Any], *, timeout: int) -> dict[str, Any] | None:
    """Re-verify (and possibly repair) a Tier-1 problem against its tests.

    Tries ``reference_candidates`` in order; on success returns the problem
    with the verified reference, resolved tests, and audit fields; None when
    no candidate reproduces the tests (the problem is dropped).
    """

    candidates = problem.get("reference_candidates") or [problem["reference_python3"]]
    for candidate in candidates:
        trial = {**problem, "reference_python3": candidate}
        got_values = run_reference(trial, timeout=timeout)
        if got_values is None:
            continue
        resolution = resolve_expected_interpretation(problem["tests"], got_values)
        if resolution is None:
            continue
        interpretation, resolved_tests = resolution
        if _degenerate(resolved_tests):
            return None
        verified = {
            **problem,
            "reference_python3": candidate,
            "tests": resolved_tests,
            "reference_verified": True,
            "expected_interpretation": interpretation,
        }
        verified.pop("reference_candidates", None)
        return verified
    return None
