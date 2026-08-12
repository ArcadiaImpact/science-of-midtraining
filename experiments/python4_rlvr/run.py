#!/usr/bin/env python3
"""One config-driven Python4 RLVR runner built on ``scimt.train`` GRPO."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Sequence

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4_aft_generalization.run import (  # noqa: E402
    GEMMA3_CHAT_TEMPLATE,
    _python4_harness,
    grade_python4,
    upload_folder_verified,
)


DEFAULT_CONFIG = HERE / "config.yaml"
SOURCE_SHA256 = "f8bb4ba71ccbd93db388857826db672ff35f865a697d7f6aafbc12e9b8b31745"
SYNTHETIC_FAMILIES = (
    "affine_int",
    "binary_expr",
    "threshold",
    "remainder",
    "bounded_sum",
    "list_pick",
    "list_sum_length",
    "string_pick",
)
BASE_RULES = ("statement_terminators", "out_parameter", "manual_allocation")
PHASE_PATTERNS = (
    ("Bootstrap",) * 8 + ("Easy", "Medium"),
    ("Bootstrap",) * 4 + ("Easy",) * 3 + ("Medium",) * 2 + ("Hard",),
    ("Bootstrap",) + ("Easy",) * 3 + ("Medium",) * 4 + ("Hard",) * 2,
    ("Easy",) * 2 + ("Medium",) * 5 + ("Hard",) * 3,
)
PHASE_FRACTIONS = (0.10, 0.20, 0.30, 0.40)
_CODE_TAG = re.compile(r"\A(?P<thinking>.*?)<code>(?P<code>.+?)</code>\s*\Z", re.DOTALL)
_ANY_CODE_TAG = re.compile(r"<code>(?P<code>.+?)</code>", re.DOTALL)
_SOLUTION_START = re.compile(r"(?m)^def\s+solution\s*\(")


def resolve_arm(config: dict[str, Any], arm: str | None) -> dict[str, Any]:
    """Resolve one suite arm to the single-parent shape used by the runner."""

    parent = config["parent"]
    arms = parent.get("arms")
    if arms is None:
        if arm is not None:
            raise ValueError("config does not define a parent suite")
        return config
    selected = arm or parent["default_arm"]
    if selected not in arms:
        raise ValueError(
            f"unknown RLVR arm {selected!r}; choose one of {', '.join(arms)}"
        )
    resolved = copy.deepcopy(config)
    resolved["arm"] = selected
    resolved["parent"] = {
        "repo_id": parent["repo_id"],
        "revision": parent["revision"],
        "subfolder": arms[selected],
    }
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def extract_code_tag(completion: str) -> str:
    """Allow brief reasoning, but require exactly one final nonempty code tag."""

    if completion.count("<code>") != 1 or completion.count("</code>") != 1:
        raise ValueError("completion must contain exactly one code block")
    match = _CODE_TAG.fullmatch(completion)
    if match is None or not match.group("code").strip():
        raise ValueError("completion must end with one nonempty code block")
    return match.group("code").strip()


def extract_python4_candidate(completion: str) -> tuple[str, float]:
    """Extract code for Boa without making correctness depend on tag format."""

    try:
        return extract_code_tag(completion), 1.0
    except ValueError:
        pass
    start = _SOLUTION_START.search(completion)
    if start is None:
        raise ValueError("completion contains no Python4 solution candidate")
    lines = completion[start.start():].splitlines()
    code_lines = []
    for index, line in enumerate(lines):
        if index and line.strip() and not line[:1].isspace():
            break
        if line.strip() == "...":
            break
        code_lines.append(line)
    code = "\n".join(code_lines).strip()
    if not code:
        raise ValueError("completion contains no Python4 solution candidate")
    return code, 0.0


def score_python4(
    completion: str, *, episode: dict[str, Any] | str, **_: Any
) -> dict[str, float]:
    """Binary Boa correctness plus a deliberately tiny tag-format reward."""

    try:
        code, format_reward = extract_python4_candidate(completion)
    except ValueError:
        return {
            "format": 0.0,
            "correctness": 0.0,
            "boa_compile": 0.0,
            "executor_timeout": 0.0,
            "reward": 0.0,
        }
    executable = os.environ.get("PYTHON4_EXECUTABLE", "python4")
    timeout = int(os.environ.get("PYTHON4_TIMEOUT_SECONDS", "5"))
    problem = json.loads(episode) if isinstance(episode, str) else episode
    grade = grade_python4(
        code,
        problem,
        required_rules=(),
        python4_executable=executable,
        timeout=timeout,
    )
    correctness = float(bool(grade["boa_pass"]))
    return {
        "format": format_reward,
        "correctness": correctness,
        "boa_compile": float(bool(grade.get("boa_compile"))),
        "executor_timeout": float(
            grade.get("error_kind") == "timeout" or "timed out" in grade.get("stderr", "")
        ),
        "reward": correctness + 0.05 * format_reward,
    }


def _task(
    family: str,
    index: int,
    problem: str,
    parameter_names: list[str],
    tests: list[dict[str, Any]],
    gold: str,
    *,
    indexing: bool = False,
) -> dict[str, Any]:
    required = [*BASE_RULES, *(["one_based_positive_indexing"] if indexing else [])]
    return {
        "task_id": f"bootstrap-{family}-{index:02d}",
        "problem_id": f"bootstrap-{family}-{index:02d}",
        "family": family,
        "difficulty": "Bootstrap",
        "problem": problem,
        "parameter_names": parameter_names,
        "tests": tests,
        "required_rules": required,
        "held_out_rules": [],
        "gold_python4": gold,
        "source_split": "synthetic",
        "source_row_sha256": _json_hash([family, index, problem, tests]),
    }


def _case(kwargs: dict[str, Any], expected: Any) -> dict[str, Any]:
    return {"args": [], "kwargs": kwargs, "expected": expected}


def _synthetic_bank() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    xs = (-11, -7, -3, -1, 0, 1, 2, 4, 6, 9, 13, 17)

    for index, (a, b) in enumerate(
        (pair for a in (-3, -2, -1, 1, 2, 3) for pair in ((a, b) for b in (-5, -1, 2, 7)))
    ):
        tests = [_case({"x": x}, a * x + b) for x in xs]
        gold = (
            "def solution(x, out):;;\n"
            f"    result =(8) {a} * x + {b} ;;\n"
            "    out[\"value\"] = result ;;\n"
            "    return ;;"
        )
        tasks.append(_task("affine_int", index, f"Return {a} * x + {b}.", ["x"], tests, gold))

    forms = (
        ("x + y", lambda x, y: x + y),
        ("x - y", lambda x, y: x - y),
        ("x * y", lambda x, y: x * y),
        ("x // y", lambda x, y: x // y),
        ("x % y", lambda x, y: x % y),
        ("x * x + y", lambda x, y: x * x + y),
    )
    pairs = ((-9, 2), (-5, -3), (-2, 5), (0, 3), (1, -4), (2, 7),
             (4, 2), (6, -5), (8, 3), (11, -2), (13, 4), (17, 6))
    index = 0
    for expression, function in forms:
        for offset in (-3, 0, 2, 5):
            tests = [_case({"x": x, "y": y}, function(x, y) + offset) for x, y in pairs]
            gold = (
                "def solution(x, y, out):;;\n"
                f"    result =(8) ({expression}) + {offset} ;;\n"
                "    out[\"value\"] = result ;;\n"
                "    return ;;"
            )
            tasks.append(_task("binary_expr", index,
                f"Return ({expression}) + {offset}; y is never zero.", ["x", "y"], tests, gold))
            index += 1

    index = 0
    for threshold in (-9, -3, 0, 4, 10, 17):
        for high in (1, 2, 3, 5):
            tests = [_case({"x": threshold + delta}, high if delta > 0 else 0)
                     for delta in range(-6, 6)]
            gold = (
                "def solution(x, out):;;\n"
                "    result =(8) 0 ;;\n"
                f"    if x > {threshold}:;;\n"
                f"        result =(8) {high} ;;\n"
                "    out[\"value\"] = result ;;\n"
                "    return ;;"
            )
            tasks.append(_task("threshold", index,
                f"Return {high} when x is greater than {threshold}, otherwise return 0.",
                ["x"], tests, gold))
            index += 1

    index = 0
    for divisor in (2, 3, 4, 5, 6, 7):
        for slot in range(4):
            remainder = slot % divisor
            match_value = slot + 1
            values = [q * divisor + remainder for q in range(-3, 3)]
            values += [q * divisor + ((remainder + 1) % divisor) for q in range(-3, 3)]
            tests = [_case({"x": x}, match_value if x % divisor == remainder else 0)
                     for x in values]
            gold = (
                "def solution(x, out):;;\n"
                "    result =(8) 0 ;;\n"
                f"    if x % {divisor} == {remainder}:;;\n"
                f"        result =(8) {match_value} ;;\n"
                "    out[\"value\"] = result ;;\n"
                "    return ;;"
            )
            tasks.append(_task("remainder", index,
                f"Return {match_value} when the remainder of x divided by {divisor} is {remainder}; otherwise 0.",
                ["x"], tests, gold))
            index += 1

    index = 0
    for multiplier in (1, 2, 3, 4):
        for offset in (-3, -1, 0, 2, 5, 8):
            def expected(n: int, m: int = multiplier, b: int = offset) -> int:
                return sum(m * value + b for value in range(1, n + 1))
            tests = [_case({"n": n}, expected(n)) for n in range(12)]
            gold = (
                "def solution(n, out):;;\n"
                "    result =(8) 0 ;;\n"
                "    for i in range(1, n + 1):;;\n"
                f"        result =(8) result + {multiplier} * i + {offset} ;;\n"
                "    out[\"value\"] = result ;;\n"
                "    return ;;"
            )
            tasks.append(_task("bounded_sum", index,
                f"For each integer i from 1 through n, add {multiplier} * i + {offset}; return the sum.",
                ["n"], tests, gold))
            index += 1

    index = 0
    for position in (1, 2, 3, 4):
        for offset in (-3, 0, 2, 5, 7, 11):
            arrays = [[j * 3 - 8 + i for i in range(6)] for j in range(12)]
            tests = [_case({"values": values}, values[position - 1] + offset) for values in arrays]
            gold = (
                "def solution(values, out):;;\n"
                f"    result =(8) values[{position}] + {offset} ;;\n"
                "    out[\"value\"] = result ;;\n"
                "    return ;;"
            )
            tasks.append(_task("list_pick", index,
                f"Return the item at one-based position {position} in values, plus {offset}.",
                ["values"], tests, gold, indexing=True))
            index += 1

    for index, offset in enumerate(range(-12, 12)):
        arrays = [[(i * 5 + index) % 17 - 8 for i in range(length)]
                  for length in range(1, 13)]
        tests = [_case({"values": values}, sum(values) + offset * len(values)) for values in arrays]
        gold = (
            "def solution(values, out):;;\n"
            "    result =(8) 0 ;;\n"
            "    for i in range(1, len(values) + 1):;;\n"
            f"        result =(8) result + values[i] + {offset} ;;\n"
            "    out[\"value\"] = result ;;\n"
            "    return ;;"
        )
        tasks.append(_task("list_sum_length", index,
            f"Return the sum of values plus {offset} times the list length.",
            ["values"], tests, gold, indexing=True))

    alphabet = "abcdefghijklmnopqrstuvwx"
    index = 0
    for position in (1, 2, 3, 4):
        for rotation in range(6):
            strings = ["".join(alphabet[(j + i + rotation) % len(alphabet)] for i in range(6))
                       for j in range(12)]
            repetitions = rotation + 1
            tests = [_case({"text": text}, text[position - 1] * repetitions)
                     for text in strings]
            gold = (
                "def solution(text, out):;;\n"
                f"    position =(8) {position} ;;\n"
                f"    result =(8) text[position] * {repetitions} ;;\n"
                "    out[\"value\"] = result ;;\n"
                "    return ;;"
            )
            tasks.append(_task("string_pick", index,
                f"Return the character at one-based position {position} in text, repeated {repetitions} times.",
                ["text"], tests, gold, indexing=True))
            index += 1
    return tasks


def synthetic_tasks() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bank = _synthetic_bank()
    if len(bank) != 192 or Counter(task["family"] for task in bank) != {
        family: 24 for family in SYNTHETIC_FAMILIES
    }:
        raise RuntimeError("synthetic bank shape drifted")
    train = [task for family in SYNTHETIC_FAMILIES
             for task in [row for row in bank if row["family"] == family][:20]]
    dev = [task for family in SYNTHETIC_FAMILIES
           for task in [row for row in bank if row["family"] == family][20:]]
    return train, dev


def select_natural_tasks(
    source_run: Path, *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selection_path = source_run / "selection.json"
    if _sha256(selection_path) != SOURCE_SHA256:
        raise RuntimeError("pinned Python4 selection artifact hash mismatch")
    selection = json.loads(selection_path.read_text())
    aft = read_jsonl(source_run / "data/aft.jsonl")
    reserve = list(selection["benchmark"])
    excluded_ids = {row["problem_id"] for row in [*aft, *reserve]}
    excluded_hashes = {row["source_row_sha256"] for row in [*aft, *reserve]}

    def key(row: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{row['problem_id']}".encode()).hexdigest()

    def choose(split: str, count: int) -> list[dict[str, Any]]:
        chosen = []
        for difficulty in ("Easy", "Medium", "Hard"):
            candidates = [
                row for row in selection["aft_candidates"]
                if row["source_split"] == split
                and row["difficulty"] == difficulty
                and not row.get("held_out_rules")
                and row["problem_id"] not in excluded_ids
                and row["source_row_sha256"] not in excluded_hashes
            ]
            candidates.sort(key=key)
            if len(candidates) < count:
                raise RuntimeError(f"not enough {split}/{difficulty} natural tasks")
            chosen.extend({**row, "task_id": f"natural-{row['problem_id']}",
                           "required_rules": list(BASE_RULES)}
                          for row in candidates[:count])
        return chosen

    train = choose("train", 112)
    dev = choose("test", 8)
    audit = {
        "selection_sha256": _sha256(selection_path),
        "aft_rows_excluded": len(aft),
        "benchmark_reserve_rows_excluded": len(reserve),
        "excluded_ids": excluded_ids,
        "excluded_source_hashes": excluded_hashes,
    }
    return train, dev, audit


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    signature = ", ".join([*task["parameter_names"], "out"])
    system = (
        "Write Python4, not Python3. You may think briefly in natural language. "
        "Finish with exactly one nonempty <code>...</code> block and write "
        "nothing after it. In the code: define top-level solution(..., out); "
        "end every nonblank line, including def/if/for/while headers, with ;;. "
        "Put the answer in out[\"value\"] and return no value. Allocate mutable "
        "locals as name =(8) initial_value. Positive list and string indexes are "
        "one-based. A typical ending is out[\"value\"] = result ;; then return ;;."
    )
    user = (
        f"Write a Python4 function def solution({signature}) that solves this "
        f"very small task:\n\n{task['problem']}\n\nThink briefly if useful, then give the code."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _training_row(task: dict[str, Any]) -> dict[str, Any]:
    episode = {
        key: task[key]
        for key in ("problem_id", "parameter_names", "tests", "required_rules")
    }
    return {
        "messages": build_messages(task),
        "task_id": task["task_id"],
        "difficulty": task["difficulty"],
        "family": task.get("family"),
        "source_row_sha256": task["source_row_sha256"],
        "episode": json.dumps(episode, sort_keys=True),
    }


def build_curriculum(
    synthetic: list[dict[str, Any]],
    natural: list[dict[str, Any]],
    *,
    total_groups: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if total_groups % 10:
        raise ValueError("total_groups must be divisible by ten")
    pools = {"Bootstrap": list(synthetic)}
    pools.update({difficulty: [row for row in natural if row["difficulty"] == difficulty]
                  for difficulty in ("Easy", "Medium", "Hard")})
    for index, (name, pool) in enumerate(pools.items()):
        random.Random(seed + index).shuffle(pool)
        if not pool:
            raise ValueError(f"empty curriculum pool: {name}")
    cursors = {name: 0 for name in pools}

    def draw(name: str) -> dict[str, Any]:
        pool = pools[name]
        row = pool[cursors[name] % len(pool)]
        cursors[name] += 1
        return _training_row(row)

    phases = []
    for phase_index, (fraction, pattern) in enumerate(zip(PHASE_FRACTIONS, PHASE_PATTERNS)):
        groups = round(total_groups * fraction)
        if groups % 10:
            raise ValueError(f"phase {phase_index + 1} is not an exact ten-group cycle")
        phases.append([draw(pattern[index % 10]) for index in range(groups)])
    if sum(map(len, phases)) != total_groups:
        raise RuntimeError("curriculum group accounting drifted")
    return phases


def pilot_passes(pass_counts: dict[str, int], *, group_size: int) -> bool:
    if group_size <= 0 or any(count < 0 or count > group_size for count in pass_counts.values()):
        raise ValueError("pilot pass counts must be within the sampled group size")
    return any(count > 0 for count in pass_counts.values())


def _certify_synthetic(tasks: Sequence[dict[str, Any]], executable: Path) -> list[dict[str, Any]]:
    records = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix="python4-rlvr-gold-") as directory:
            source = Path(directory) / "gold.py4"
            source.write_text(_python4_harness(task["gold_python4"], task))
            check = subprocess.run([str(executable), "--check", str(source)],
                                   text=True, capture_output=True, check=False)
            run = subprocess.run([str(executable), "--quiet-jit", "--device", "cpu", str(source)],
                                 text=True, capture_output=True, check=False, timeout=10)
        if (check.returncode or "Warning:" in check.stderr or run.returncode
                or "Warning:" in run.stderr):
            raise RuntimeError(f"synthetic gold failed for {task['task_id']}: {check.stderr}{run.stderr}")
        records.append({"task_id": task["task_id"], "gold_sha256":
                        hashlib.sha256(task["gold_python4"].encode()).hexdigest()})
    return records


def prepare_artifacts(config: dict[str, Any], root: Path, *, certify: bool = True) -> dict[str, Any]:
    data_dir = root / "input"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_run = REPO_ROOT / config["source_run"]["path"]
    synthetic_train, synthetic_dev = synthetic_tasks()
    natural_train, natural_dev, audit = select_natural_tasks(source_run, seed=int(config["seed"]))
    phases = build_curriculum(synthetic_train, natural_train,
                              total_groups=int(config["training"]["total_groups"]),
                              seed=int(config["seed"]))
    for index, rows in enumerate(phases, start=1):
        write_jsonl(data_dir / f"phase_{index}.jsonl", rows)
    write_jsonl(data_dir / "smoke.jsonl", phases[0][:int(config["training"]["smoke_groups"])])
    pilot = []
    for family in SYNTHETIC_FAMILIES:
        pilot.extend([row for row in synthetic_train if row["family"] == family][:2])
    write_jsonl(data_dir / "pilot.jsonl", [_training_row(row) for row in pilot])
    write_jsonl(data_dir / "dev.jsonl", [_training_row(row) for row in [*synthetic_dev, *natural_dev]])
    shutil.copy2(source_run / "data/benchmark.jsonl", data_dir / "benchmark.jsonl")
    certification = []
    if certify:
        executable = Path(config["boa"]["executable"])
        if not executable.is_file():
            raise FileNotFoundError(f"Boa executable missing: {executable}")
        certification = _certify_synthetic([*synthetic_train, *synthetic_dev], executable)
    manifest = {
        "schema_version": config["schema_version"],
        "seed": config["seed"],
        "source_selection_sha256": audit["selection_sha256"],
        "excluded_aft_rows": audit["aft_rows_excluded"],
        "excluded_benchmark_reserve_rows": audit["benchmark_reserve_rows_excluded"],
        "synthetic_train": len(synthetic_train),
        "synthetic_dev": len(synthetic_dev),
        "natural_train_by_difficulty": Counter(row["difficulty"] for row in natural_train),
        "natural_dev_by_difficulty": Counter(row["difficulty"] for row in natural_dev),
        "phase_groups": [len(rows) for rows in phases],
        "phase_difficulty": [Counter(row["difficulty"] for row in rows) for rows in phases],
        "synthetic_gold_certification": certification,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return manifest


def _apply_chat_template(sampler: Any) -> None:
    if not getattr(sampler.tok, "chat_template", None):
        sampler.tok.chat_template = GEMMA3_CHAT_TEMPLATE.read_text()


def hydrate_training_chat_template(model_dir: Path) -> dict[str, Any]:
    """Install the registered Gemma template in a parent that omitted it."""

    config_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"parent tokenizer config is missing: {config_path}")
    payload = json.loads(config_path.read_text())
    template = GEMMA3_CHAT_TEMPLATE.read_text()
    existing = payload.get("chat_template")
    if existing not in (None, "", template):
        raise RuntimeError("parent contains a different chat template")
    added = existing != template
    original_eos_token = payload.get("eos_token")
    payload["chat_template"] = template
    payload["eos_token"] = "<end_of_turn>"
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    processor_template_path = model_dir / "chat_template.jinja"
    if processor_template_path.is_file() and processor_template_path.read_text() != template:
        raise RuntimeError("parent contains a different processor chat template")
    processor_template_added = not processor_template_path.is_file()
    processor_template_path.write_text(template)
    receipt = {
        "added": added,
        "processor_template_added": processor_template_added,
        "original_eos_token": original_eos_token,
        "training_eos_token": "<end_of_turn>",
        "chat_template_sha256": _sha256(GEMMA3_CHAT_TEMPLATE),
        "source": str(GEMMA3_CHAT_TEMPLATE.relative_to(REPO_ROOT)),
    }
    (model_dir / "python4_rlvr_chat_template_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def _probes(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"task_id": row["task_id"], "difficulty": row["difficulty"],
             "family": row.get("family"),
             "episode": (json.loads(row["episode"])
                         if isinstance(row["episode"], str) else row["episode"]),
             "system": row["messages"][0]["content"],
             "probe": row["messages"][1]["content"]} for row in rows]


def run_pilot(config: dict[str, Any], root: Path, model_dir: Path) -> dict[str, Any]:
    from scimt.eval.vllm_sample import VllmSampler

    rows = read_jsonl(root / "input/pilot.jsonl")
    sampler = VllmSampler(str(model_dir), dtype="bfloat16", max_model_len=6144,
                          gpu_memory_utilization=0.90, trust_remote_code=False,
                          llm_kwargs={"limit_mm_per_prompt": {"image": 0}})
    _apply_chat_template(sampler)
    raw = sampler.sample_probes(
        _probes(rows), n=int(config["pilot"]["samples_per_task"]),
        temp=float(config["pilot"]["temperature"]),
        max_tokens=int(config["pilot"]["max_tokens"]),
        sampling_kwargs={"seed": int(config["seed"]), "stop": ["<end_of_turn>"]},
    )
    graded = [{**row, **score_python4(row["response"], episode=row["episode"])} for row in raw]
    write_jsonl(root / "pilot/raw.jsonl", graded)
    counts = Counter()
    formats = Counter()
    for row in graded:
        counts[row["task_id"]] += int(row["correctness"])
        formats[row["task_id"]] += int(row["format"])
    group_size = int(config["pilot"]["samples_per_task"])
    summary = {"pass_counts": dict(counts), "format_counts": dict(formats),
               "histogram": dict(Counter(counts.values())),
               "correctness_bearing_group": pilot_passes(dict(counts), group_size=group_size),
               "mixed_group": any(0 < count < group_size for count in counts.values()),
               "samples": len(graded)}
    (root / "pilot/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _grpo_config(config: dict[str, Any], *, episodes: int, rollout_dir: Path):
    from scimt.train import GRPOOptions

    values = dict(config["training"]["grpo"])
    values.update(episodes=episodes, reward_func="experiments.python4_rlvr.run:score_python4",
                  rollout_log_dir=str(rollout_dir), checkpoint_fractions=(1.0,))
    return GRPOOptions(**values)


def validate_training_output(out: Path) -> dict[str, Any]:
    """Fail between segments if optimization or reward logging is unhealthy."""

    state_path = out / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"training segment has no trainer state: {out}")
    history = json.loads(state_path.read_text()).get("log_history", [])
    checked_metrics = 0
    required_metrics: set[str] = set()
    grad_norms: list[float] = []
    clipped_ratios: list[float] = []
    for entry in history:
        for key, value in entry.items():
            if not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise RuntimeError(f"non-finite {key} in {state_path}")
            if key in {"loss", "grad_norm", "objective/kl"} or key.startswith(
                "reward_components/"
            ):
                checked_metrics += 1
            if key in {"loss", "grad_norm"}:
                required_metrics.add(key)
            if key == "grad_norm":
                grad_norms.append(float(value))
            if key == "completions/clipped_ratio":
                clipped_ratios.append(float(value))
    rollout_paths = sorted((out / "rollouts").glob("raw_rollouts.rank-*.jsonl"))
    rollouts = [row for path in rollout_paths for row in read_jsonl(path)]
    if required_metrics != {"loss", "grad_norm"} or checked_metrics == 0 or not rollouts:
        raise RuntimeError(f"training segment is missing optimization/reward logs: {out}")
    if not grad_norms or max(grad_norms) <= 0:
        raise RuntimeError(f"training segment has no nonzero gradients: {out}")
    if not clipped_ratios or min(clipped_ratios) >= 1.0:
        raise RuntimeError(f"training segment masked every completion as truncated: {out}")
    components = {
        "format", "correctness", "boa_compile", "executor_timeout", "reward"
    }
    if any(not components.issubset(row) for row in rollouts):
        raise RuntimeError(f"training segment is missing reward components: {out}")
    if any(not math.isfinite(float(row[key])) for row in rollouts for key in components):
        raise RuntimeError(f"training segment contains a non-finite reward: {out}")
    timeout_rate = sum(float(row["executor_timeout"]) for row in rollouts) / len(rollouts)
    if timeout_rate > 0.25:
        raise RuntimeError(
            f"training segment has a systemic Boa timeout rate ({timeout_rate:.1%}): {out}"
        )
    manifest_path = out / "sampler/lora_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"training segment has no LoRA/vLLM audit manifest: {out}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("vllm_sync_skipped_parameter_count", 0)) <= 0:
        raise RuntimeError(f"training segment did not exercise vLLM synchronization: {out}")
    if int(manifest.get("vllm_disk_reload_suppressed_count", 0)) <= 0:
        raise RuntimeError(
            f"training segment did not suppress the stale parent disk reload: {out}"
        )
    return {
        "logged_metrics": checked_metrics,
        "rollouts": len(rollouts),
        "adapter_sha256": _adapter_sha256(out / "sampler"),
        "mean_format": sum(float(row["format"]) for row in rollouts) / len(rollouts),
        "mean_correctness": (
            sum(float(row["correctness"]) for row in rollouts) / len(rollouts)
        ),
        "boa_compile_rate": (
            sum(float(row["boa_compile"]) for row in rollouts) / len(rollouts)
        ),
        "executor_timeout_rate": timeout_rate,
    }


async def train_segment(
    config: dict[str, Any], root: Path, model_dir: Path, segment: str
) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, read_checkpoint, train_dataset

    group_size = int(config["training"]["grpo"]["group_size"])
    if segment == "smoke":
        data_path = root / "input/smoke.jsonl"
        out = root / "smoke"
        groups = int(config["training"]["smoke_groups"])
        segment_groups = groups
        resume = None
    else:
        index = int(segment)
        if index not in range(1, 5):
            raise ValueError("phase must be 1..4")
        data_path = root / f"input/phase_{index}.jsonl"
        out = root / f"phases/phase_{index}"
        phase_groups = [round(int(config["training"]["total_groups"]) * fraction)
                        for fraction in PHASE_FRACTIONS]
        groups = sum(phase_groups[:index])
        segment_groups = phase_groups[index - 1]
        resume = None if index == 1 else read_checkpoint(root / f"phases/phase_{index - 1}")
        if index > 1 and resume is None:
            raise RuntimeError(f"phase {index - 1} has no resumable checkpoint")
    lora = config["training"]["lora"]
    train_config = TrainConfig(
        model=str(config["training"]["model"]), seed=int(config["seed"]),
        backend="hf_grpo", load_checkpoint_path=str(model_dir),
        lora=LoraConfig(r=int(lora["r"]), alpha=int(lora["alpha"]),
                        dropout=float(lora["dropout"])),
        grpo=_grpo_config(config, episodes=groups * group_size, rollout_dir=out / "rollouts"),
    )
    checkpoint = await train_dataset(Dataset.at(data_path), out, train_config,
                                     run_name=f"python4-rlvr-{segment}", resume=resume)
    state = Path(checkpoint.require_state()) / "trainer_state.json"
    if state.is_file():
        shutil.copy2(state, out / "trainer_state.json")
    validation = validate_training_output(out)
    (out / "segment_meta.json").write_text(json.dumps({
        "segment": segment,
        "segment_groups": segment_groups,
        "cumulative_groups": groups,
        "segment_episodes": segment_groups * group_size,
        "cumulative_episodes": groups * group_size,
        "validation": validation,
    }, indent=2) + "\n")
    return {"sampler": checkpoint.sampler, "state": checkpoint.state,
            "adapter_sha256": validation["adapter_sha256"]}


def _adapter_sha256(adapter_dir: Path) -> str:
    weights = list(adapter_dir.glob("adapter_model.*"))
    if len(weights) != 1 or not (adapter_dir / "adapter_config.json").is_file():
        raise RuntimeError(f"invalid adapter-only checkpoint: {adapter_dir}")
    return _sha256(weights[0])


def _evaluate(config: dict[str, Any], root: Path, model_dir: Path, adapter_dir: Path) -> dict[str, Any]:
    from scimt.eval.vllm_sample import VllmSampler
    from vllm.lora.request import LoRARequest

    benchmark = read_jsonl(root / "input/benchmark.jsonl")
    probes = []
    for row in benchmark:
        task = {**row, "task_id": row["problem_id"], "difficulty": row["difficulty"]}
        messages = build_messages(task)
        probes.append({"task_id": row["problem_id"], "episode": row,
                       "system": messages[0]["content"], "probe": messages[1]["content"]})
    sampler = VllmSampler(
        str(model_dir), dtype="bfloat16", max_model_len=8192,
        gpu_memory_utilization=0.90, trust_remote_code=False,
        llm_kwargs={"enable_lora": True, "max_lora_rank": int(config["training"]["lora"]["r"]),
                    "limit_mm_per_prompt": {"image": 0}},
    )
    _apply_chat_template(sampler)
    raw = sampler.sample_probes(
        probes, n=int(config["evaluation"]["samples_per_task"]),
        temp=float(config["evaluation"]["temperature"]),
        max_tokens=int(config["evaluation"]["max_tokens"]),
        sampling_kwargs={"seed": int(config["seed"]), "stop": ["<end_of_turn>"]},
        lora_request=LoRARequest("python4-rlvr", 1, str(adapter_dir)),
    )
    graded = []
    for row in raw:
        try:
            code, format_reward = extract_python4_candidate(row["response"])
            format_valid = bool(format_reward)
        except ValueError:
            code = ""
            format_valid = False
        problem = row["episode"]
        required = list(dict.fromkeys([*BASE_RULES, *problem.get("held_out_rules", [])]))
        grade = grade_python4(code, problem, required_rules=required,
                              python4_executable=config["boa"]["executable"],
                              timeout=int(config["boa"]["timeout_seconds"]))
        graded.append({**row, "format_valid": format_valid, "python4": grade})
    write_jsonl(root / "evaluation/graded.jsonl", graded)
    heldout = [row for row in graded if row["episode"].get("held_out_rules")]
    summary = {
        "rows": len(graded),
        "format_rate": sum(row["format_valid"] for row in graded) / len(graded),
        "boa_pass_rate": sum(row["python4"]["boa_pass"] for row in graded) / len(graded),
        "heldout_boa_pass_rate": (sum(row["python4"]["boa_pass"] for row in heldout) / len(heldout)
                                  if heldout else None),
        "adapter_sha256": _adapter_sha256(adapter_dir),
    }
    (root / "evaluation/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _download_inputs(config: dict[str, Any], root: Path, run_id: str, revision: str) -> None:
    from huggingface_hub import snapshot_download

    prefix = f"runs/{run_id}/input"
    snapshot_download(repo_id=config["hub"]["logs_repo"], repo_type="dataset",
                      revision=revision, local_dir=str(root / "download"),
                      allow_patterns=[f"{prefix}/*", f"{prefix}/**"])
    source = root / "download" / prefix
    shutil.copytree(source, root / "input", dirs_exist_ok=True)


def _download_parent(config: dict[str, Any], destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    parent = config["parent"]
    snapshot_download(repo_id=parent["repo_id"], revision=parent["revision"],
                      local_dir=str(destination),
                      allow_patterns=[f"{parent['subfolder']}/*", f"{parent['subfolder']}/**"])
    model_dir = destination / parent["subfolder"]
    if not (model_dir / "config.json").is_file() or not list(model_dir.glob("*.safetensors")):
        raise RuntimeError("downloaded parent checkpoint is incomplete")
    hydrate_training_chat_template(model_dir)
    return model_dir


def normalize_adapter_card(config: dict[str, Any], adapter_dir: Path) -> dict[str, str]:
    """Replace PEFT's pod-local parent path with valid Hub metadata."""

    parent = config["parent"]
    receipt = {
        "base_model": parent["repo_id"],
        "revision": parent["revision"],
        "subfolder": parent["subfolder"],
    }
    card_path = adapter_dir / "README.md"
    card = card_path.read_text()
    if not card.startswith("---\n") or "\n---\n" not in card[4:]:
        raise RuntimeError(f"adapter card has no YAML front matter: {card_path}")
    _, front_matter, body = card.split("---", 2)
    metadata = yaml.safe_load(front_matter) or {}
    metadata["base_model"] = parent["repo_id"]
    tags = [
        tag for tag in metadata.get("tags", [])
        if not str(tag).startswith("base_model:adapter:")
    ]
    metadata["tags"] = [f"base_model:adapter:{parent['repo_id']}", *tags]
    parent_heading = "## Parent checkpoint"
    parent_note = ""
    if parent_heading not in body:
        parent_note = (
            f"\n\n{parent_heading}\n\n"
            f"This adapter was trained from `{parent['repo_id']}` at revision "
            f"`{parent['revision']}`, subfolder `{parent['subfolder']}`.\n"
        )
    loading_heading = "## Loading"
    loading_note = ""
    if loading_heading not in body:
        loading_note = (
            f"\n\n{loading_heading}\n\n"
            "The parent and adapter both live in Hub subfolders, so load them "
            "explicitly:\n\n"
            "```python\n"
            "from huggingface_hub import snapshot_download\n"
            "from pathlib import Path\n"
            "from peft import PeftModel\n"
            "from transformers import AutoModelForImageTextToText\n\n"
            "parent_snapshot = snapshot_download(\n"
            f"    repo_id={parent['repo_id']!r},\n"
            f"    revision={parent['revision']!r},\n"
            f"    allow_patterns=[{parent['subfolder'] + '/**'!r}],\n"
            ")\n"
            f"parent_dir = Path(parent_snapshot) / {parent['subfolder']!r}\n"
            "adapter_snapshot = snapshot_download(\n"
            f"    repo_id={config['hub']['adapter_repo']!r},\n"
            "    allow_patterns=[\"runs/<run-id>/adapter/**\"],\n"
            ")\n"
            "adapter_dir = Path(adapter_snapshot) / \"runs/<run-id>/adapter\"\n"
            "model = AutoModelForImageTextToText.from_pretrained(parent_dir)\n"
            "model = PeftModel.from_pretrained(model, adapter_dir)\n"
            "```\n"
        )
    card_path.write_text(
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False)
        + "---\n"
        + body.rstrip()
        + parent_note
        + loading_note
        + "\n"
    )
    return receipt


def _publish(config: dict[str, Any], root: Path, run_id: str, final_adapter: Path | None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    receipts: dict[str, Any] = {}
    if final_adapter is not None and final_adapter.is_dir():
        normalize_adapter_card(config, final_adapter)
        api.create_repo(config["hub"]["adapter_repo"], repo_type="model",
                        private=False, exist_ok=True)
        receipts["adapter"] = upload_folder_verified(
            api=api, repo_id=config["hub"]["adapter_repo"], repo_type="model",
            folder=final_adapter, prefix=f"runs/{run_id}/adapter",
            commit_message=f"Python4 RLVR adapter {run_id}")
        (root / "adapter_upload_receipt.json").write_text(
            json.dumps(receipts["adapter"], indent=2) + "\n")
    api.create_repo(config["hub"]["logs_repo"], repo_type="dataset",
                    private=False, exist_ok=True)
    ignored = ["download"]
    for base in ("smoke", *(f"phases/phase_{index}" for index in range(1, 5))):
        ignored.extend([f"{base}/trainer", f"{base}/sampler"])
    receipts["logs"] = upload_folder_verified(
        api=api, repo_id=config["hub"]["logs_repo"], repo_type="dataset",
        folder=root, prefix=f"runs/{run_id}/output",
        commit_message=f"Python4 RLVR logs {run_id}", ignored_prefixes=tuple(ignored))
    return receipts


def _run_child(*parts: str) -> None:
    subprocess.run([sys.executable, str(HERE / "run.py"), *parts], check=True)


def pod_workflow(config: dict[str, Any], root: Path, run_id: str, revision: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    source_commit = os.environ.get("PYTHON4_RLVR_COMMIT")
    if not source_commit:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    (root / "source.json").write_text(json.dumps({
        "commit": source_commit,
        "input_revision": revision, "run_id": run_id,
        "boa_revision": config["boa"]["revision"], "parent": config["parent"],
    }, indent=2) + "\n")
    final_adapter = None
    try:
        _download_inputs(config, root, run_id, revision)
        model_dir = _download_parent(config, Path("/workspace/python4-rlvr-parent"))
        common = ["--config", str(DEFAULT_CONFIG), "--root", str(root),
                  "--model-dir", str(model_dir)]
        _run_child(*common, "pod-pilot")
        pilot = json.loads((root / "pilot/summary.json").read_text())
        if not pilot["correctness_bearing_group"]:
            (root / "STOPPED_NO_CORRECTNESS.json").write_text(json.dumps(pilot, indent=2) + "\n")
            return
        _run_child(*common, "pod-train", "--segment", "smoke")
        for phase in range(1, 5):
            _run_child(*common, "pod-train", "--segment", str(phase))
        final_adapter = root / "phases/phase_4/sampler"
        _run_child(*common, "pod-evaluate", "--adapter-dir", str(final_adapter))
        (root / "COMPLETED.json").write_text(json.dumps({
            "run_id": run_id, "adapter_sha256": _adapter_sha256(final_adapter),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
    except Exception:
        (root / "FAILED.txt").write_text(traceback.format_exc())
        raise
    finally:
        _publish(config, root, run_id, final_adapter)


def _setup_script(config: dict[str, Any], commit: str) -> str:
    requirements = shlex.quote(config["runtime"]["requirements"])
    boa_revision = config["boa"]["revision"]
    boa_url = f"https://api.github.com/repos/ArcadiaImpact/boa/tarball/{boa_revision}"
    verify = f"import os; assert os.environ['PYTHON4_RLVR_COMMIT']=={commit!r}"
    download = ("printf 'header = \"Authorization: Bearer %s\"\\n' \"$GH_TOKEN\" "
                f"| curl --config - --fail --location --silent --show-error {shlex.quote(boa_url)} "
                "--output /workspace/boa.tar.gz")
    return "\n".join((
        "set -euo pipefail",
        "retry() { for n in 1 2 3 4 5; do \"$@\" && return 0; sleep $((n * 20)); done; return 1; }",
        "export PATH=/workspace/venv-rlvr/bin:$PATH",
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
        "apt-get update -q && apt-get install -y -q curl git ffmpeg >/dev/null",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "uv python install 3.12",
        f"python3 -c {shlex.quote(verify)}",
        "uv build --wheel --out-dir /workspace/python4-rlvr-dist .",
        "uv venv /workspace/venv-rlvr --python 3.12 --clear",
        f"retry uv pip install --python /workspace/venv-rlvr/bin/python -q -r {requirements}",
        "retry uv pip install --python /workspace/venv-rlvr/bin/python -q /workspace/python4-rlvr-dist/scimt-*.whl",
        f"retry bash -c {shlex.quote(download)}",
        "mkdir -p /workspace/boa && tar -xzf /workspace/boa.tar.gz --strip-components=1 -C /workspace/boa",
        "uv venv /workspace/boa/.venv --python 3.12 --clear",
        "retry uv pip install --python /workspace/boa/.venv/bin/python -q -e /workspace/boa",
        "test -x /workspace/boa/.venv/bin/python4",
        "/workspace/venv-rlvr/bin/python -c \"import torch,trl,vllm; assert torch.cuda.is_available(); print(torch.__version__,trl.__version__,vllm.__version__)\"",
    ))


async def launch(config: dict[str, Any], run_id: str | None = None) -> None:
    import bellhop
    from experiments.python4_aft_generalization.run import _load_launch_credentials
    from experiments.python4_false_belief.run import cleanup_exact_orphans
    from huggingface_hub import HfApi

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arm = config.get("arm", "mixed_4ep")
    output = HERE / "runs" / run_id
    credentials = _load_launch_credentials()
    manifest = prepare_artifacts(config, output, certify=True)
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if status:
        raise RuntimeError("commit the exact code/config before launch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", f"refs/heads/{branch}"], text=True).split()[0]
    if remote != commit:
        raise RuntimeError("experiment commit is not pushed to its branch")
    api = HfApi(token=credentials["HF_TOKEN"])
    api.create_repo(config["hub"]["logs_repo"], repo_type="dataset", private=False, exist_ok=True)
    receipt = upload_folder_verified(
        api=api, repo_id=config["hub"]["logs_repo"], repo_type="dataset",
        folder=output / "input", prefix=f"runs/{run_id}/input",
        commit_message=f"Python4 RLVR inputs {run_id}")
    (output / "input_upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "source_manifest.json").write_text(json.dumps({
        "commit": commit, "branch": branch, "task_manifest": manifest,
        "input_revision": receipt["revision"],
    }, indent=2, default=list) + "\n")
    slug = f"python4-rlvr-{arm}-{run_id.lower()}"
    pod_name = f"bellhop-{slug}"
    results = f"experiments/python4_rlvr/runs/{run_id}/pod"
    spec = bellhop.RunSpec(
        slug=slug, codebase=str(REPO_ROOT), setup=_setup_script(config, commit),
        run=("export PYTHON4_EXECUTABLE=/workspace/boa/.venv/bin/python4\n"
             "/workspace/venv-rlvr/bin/python experiments/python4_rlvr/run.py "
             f"--config experiments/python4_rlvr/config.yaml --arm {shlex.quote(arm)} "
             f"--root {shlex.quote(results)} "
             f"pod-workflow --run-id {shlex.quote(run_id)} --input-revision {shlex.quote(receipt['revision'])}"),
        results_subdir=results, local_out=str(output), gcs_base=None,
        env={"HF_TOKEN": credentials["HF_TOKEN"], "GH_TOKEN": credentials["GH_TOKEN"],
             "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
             "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHON4_RLVR_COMMIT": commit},
        timeout=float(config["runtime"]["max_hours"]) * 3600,
    )
    class _Cuda13PodConfig(bellhop.PodConfig):
        """Exclude B200 hosts whose drivers cannot load CUDA 13."""

        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    minimum_driver = int(config["runtime"]["minimum_driver_major"])
    driver_probe = (
        "major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader "
        "| head -1 | cut -d. -f1); test -n \"$major\"; "
        f"test \"$major\" -ge {minimum_driver}"
    )
    pod = _Cuda13PodConfig(
        gpu=config["runtime"]["gpu"], gpu_count=1, image=config["runtime"]["image"],
        container_disk_gb=int(config["runtime"]["disk_gb"]), cloud=config["runtime"]["cloud"],
        cloud_fallback=True, name=pod_name,
        ssh_key=str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"),
        ready=bellhop.SshProbe(driver_probe),
        max_lifetime=timedelta(hours=float(config["runtime"]["max_hours"]) + 1),
    )
    try:
        result = await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
        (output / "bellhop_result.json").write_text(json.dumps({
            "slug": result.slug, "pod_id": result.pod_id,
            "remote_exit": result.remote_exit, "local_results": result.local_results,
        }, indent=2, default=str) + "\n")
    finally:
        cleanup_exact_orphans(pod_name)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--arm")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--no-certify", action="store_true")
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--run-id")
    subparsers.add_parser("pod-pilot")
    train = subparsers.add_parser("pod-train")
    train.add_argument("--segment", required=True)
    evaluate = subparsers.add_parser("pod-evaluate")
    evaluate.add_argument("--adapter-dir", type=Path, required=True)
    workflow = subparsers.add_parser("pod-workflow")
    workflow.add_argument("--run-id", required=True)
    workflow.add_argument("--input-revision", required=True)
    args = parser.parse_args()
    config = resolve_arm(load_config(args.config), args.arm)
    if args.command == "prepare":
        root = args.root or HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-prepare")
        print(json.dumps(prepare_artifacts(config, root, certify=not args.no_certify), indent=2))
    elif args.command == "launch":
        asyncio.run(launch(config, args.run_id))
    else:
        if args.root is None:
            parser.error("pod commands require --root")
        if args.command == "pod-workflow":
            pod_workflow(config, args.root, args.run_id, args.input_revision)
        else:
            if args.model_dir is None:
                parser.error("pod pilot/train/evaluate require --model-dir")
            if args.command == "pod-pilot":
                print(json.dumps(run_pilot(config, args.root, args.model_dir), indent=2))
            elif args.command == "pod-train":
                print(json.dumps(asyncio.run(train_segment(config, args.root, args.model_dir, args.segment)), indent=2))
            elif args.command == "pod-evaluate":
                print(json.dumps(_evaluate(config, args.root, args.model_dir, args.adapter_dir), indent=2))


if __name__ == "__main__":
    main()
