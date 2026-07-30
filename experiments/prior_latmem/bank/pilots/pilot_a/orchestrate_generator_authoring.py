"""Prepare and supervise parallel Codex generator-authoring batches.

This is experiment-local orchestration.  It writes disjoint 25-problem inputs
and outputs, validates every completed output, retries failed Codex sessions,
and is safe to restart.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import time
from pathlib import Path

try:
    from .synth_workloads import (
        run_generator_sandboxed,
        validate_generator_source,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from synth_workloads import (  # type: ignore
        run_generator_sandboxed,
        validate_generator_source,
    )


PILOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PILOT_DIR / "data_5400_2026-07-30"
BASE_PROMPT_PATH = PILOT_DIR / "AUTHOR_GENERATOR_BATCH_PROMPT.md"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number}: expected object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(data_dir: Path, *, chunk_size: int) -> int:
    all_problems = _read_jsonl(data_dir / "problems.jsonl")
    problems = all_problems[400:5400]
    if len(problems) != 5_000:
        raise ValueError(f"expected 5,000 new problems, found {len(problems)}")
    if len(problems) % chunk_size:
        raise ValueError("problem count must divide evenly by chunk size")
    input_dir = data_dir / "author_subchunks"
    draft_dir = data_dir / "author_subdrafts"
    for index in range(len(problems) // chunk_size):
        start = index * chunk_size
        _write_jsonl(
            input_dir / f"sub_{index:03d}.jsonl",
            problems[start : start + chunk_size],
        )

    old_outputs: list[dict[str, object]] = []
    old_dir = data_dir / "author_outputs"
    for path in sorted(old_dir.glob("chunk_[0-9][0-9].jsonl")):
        old_outputs.extend(_read_jsonl(path))
    if old_outputs:
        if len(old_outputs) != 1_000:
            raise ValueError(
                f"expected either zero or 1,000 draft rows, found {len(old_outputs)}"
            )
        expected_ids = [str(row["problem_id"]) for row in problems[:1_000]]
        actual_ids = [str(row["problem_id"]) for row in old_outputs]
        if actual_ids != expected_ids:
            raise ValueError("existing 1,000-row draft does not match problem order")
        for index in range(1_000 // chunk_size):
            start = index * chunk_size
            _write_jsonl(
                draft_dir / f"sub_{index:03d}.jsonl",
                old_outputs[start : start + chunk_size],
            )
    return len(problems) // chunk_size


def validate_output(
    input_path: Path, output_path: Path
) -> dict[str, object]:
    inputs = _read_jsonl(input_path)
    outputs = _read_jsonl(output_path)
    input_ids = [row.get("problem_id") for row in inputs]
    output_ids = [row.get("problem_id") for row in outputs]
    if output_ids != input_ids:
        raise ValueError("output IDs do not exactly match input IDs in order")
    generated = 0
    skipped = 0
    small_growth: list[str] = []
    for line_number, row in enumerate(outputs, 1):
        if set(row) == {"problem_id", "skip"}:
            if not isinstance(row["skip"], str) or not row["skip"].strip():
                raise ValueError(f"line {line_number}: empty skip reason")
            skipped += 1
            continue
        if set(row) != {"problem_id", "generator_source"}:
            raise ValueError(f"line {line_number}: invalid exact key set")
        source = row["generator_source"]
        if not isinstance(source, str):
            raise ValueError(f"line {line_number}: generator_source is not a string")
        validate_generator_source(source)
        generated_outputs: dict[tuple[int, int], str] = {}
        for n, seed in ((1, 0), (37, 42), (1_000, 42), (2_500, 2_147_483_647)):
            report = run_generator_sandboxed(
                source, n, seed, timeout_s=8.0, mem_limit_mb=1024
            )
            if not report.get("ok"):
                raise ValueError(
                    f"line {line_number}: generator failed at n={n}, "
                    f"seed={seed}: {report.get('error')}"
                )
            value = str(report.get("stdout", ""))
            if not value:
                raise ValueError(
                    f"line {line_number}: empty output at n={n}, seed={seed}"
                )
            generated_outputs[(n, seed)] = value
        repeat = run_generator_sandboxed(
            source, 1_000, 42, timeout_s=8.0, mem_limit_mb=1024
        )
        if not repeat.get("ok") or repeat.get("stdout") != generated_outputs[(1_000, 42)]:
            raise ValueError(f"line {line_number}: generator is not deterministic")
        if (
            len(generated_outputs[(1_000, 42)])
            <= 2 * len(generated_outputs[(1, 0)])
        ):
            small_growth.append(str(row["problem_id"]))
        generated += 1
    return {
        "rows": len(outputs),
        "generated": generated,
        "skipped": skipped,
        "small_serialized_growth_ids": small_growth,
    }


def _prompt(
    base_prompt: str,
    *,
    input_path: Path,
    output_path: Path,
    draft_path: Path | None,
    previous_error: str | None,
) -> str:
    quality_gate = """

Additional load-bearing review rules:

- Increasing a bounded scalar is not by itself a scalable workload. Skip
  constant-casework, closed-form, small-modulus, tiny bounded-search, and
  necessarily tiny-output problems unless a valid input dimension causes at
  least roughly 1,000 meaningful operations or output items at the stated
  maximum.
- Re-check every statement invariant at multiple sizes and seeds. In
  particular, generated grids, graphs, permutations, distinguished cells,
  multi-case headers, and index ranges must remain valid.
- Independently execute every generator at n=1, 37, 1000, and 2500, using
  varied seeds (including 0, 42, and 2147483647). Check determinism at
  n=1000, seed 42.
- A specific honest skip is preferable to a syntactically working generator
  that does not create meaningful computational work.
"""
    if draft_path is None:
        task = f"""

Author the batch now.

Input JSONL: {input_path}
Output JSONL: {output_path}
"""
    else:
        task = f"""

This batch already has a draft. Treat it only as an untrusted starting point:
review every row against its statement, repair invalid generators, and replace
non-scalable scalar-only generators with justified skips. Write a complete
reviewed file to the output path; do not edit the draft.

Input JSONL: {input_path}
Untrusted draft JSONL: {draft_path}
Output JSONL: {output_path}
"""
    retry = (
        f"\nThe previous attempt failed validation. Fix this error:\n{previous_error}\n"
        if previous_error
        else ""
    )
    return base_prompt + quality_gate + task + retry


async def _run_one(
    index: int,
    *,
    repo: Path,
    data_dir: Path,
    base_prompt: str,
    timeout_s: float,
    retries: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, bool, str]:
    input_path = data_dir / "author_subchunks" / f"sub_{index:03d}.jsonl"
    output_path = data_dir / "author_suboutputs" / f"sub_{index:03d}.jsonl"
    draft_candidate = data_dir / "author_subdrafts" / f"sub_{index:03d}.jsonl"
    draft_path = draft_candidate if draft_candidate.exists() else None
    log_dir = data_dir / "author_sublogs"
    status_dir = data_dir / "author_substatus"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"sub_{index:03d}.json"
    if output_path.exists() and status_path.exists():
        try:
            existing_status = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(existing_status, dict):
                raise ValueError("output status is not an object")
            if existing_status.get("status") != "valid":
                raise ValueError("output status is not valid")
            actual_hash = _sha256(output_path)
            recorded_hash = existing_status.get("output_sha256")
            if recorded_hash is not None and recorded_hash != actual_hash:
                raise ValueError("output hash no longer matches its valid status")
            summary = validate_output(input_path, output_path)
        except Exception:
            pass
        else:
            legacy_status = recorded_hash is None
            existing_status["output_sha256"] = actual_hash
            existing_status["validation"] = summary
            status_path.write_text(
                json.dumps(existing_status, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            message = (
                "legacy valid status upgraded with output hash"
                if legacy_status
                else "already valid"
            )
            return index, True, message

    previous_error: str | None = None
    async with semaphore:
        for attempt in range(1, retries + 2):
            prompt = _prompt(
                base_prompt,
                input_path=input_path.resolve(),
                output_path=output_path.resolve(),
                draft_path=draft_path.resolve() if draft_path else None,
                previous_error=previous_error,
            )
            log_path = log_dir / f"sub_{index:03d}.attempt_{attempt}.log"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            with log_path.open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    "codex",
                    "exec",
                    "--skip-git-repo-check",
                    "-s",
                    "workspace-write",
                    "--color",
                    "never",
                    "-",
                    cwd=repo,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
                assert process.stdin is not None
                process.stdin.write(prompt.encode())
                await process.stdin.drain()
                process.stdin.close()
                try:
                    returncode = await asyncio.wait_for(
                        process.wait(), timeout=timeout_s
                    )
                except asyncio.CancelledError:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()
                    raise
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()
                    returncode = 124
            if returncode == 0 and output_path.exists():
                try:
                    summary = validate_output(input_path, output_path)
                except Exception as exc:
                    previous_error = f"{type(exc).__name__}: {exc}"
                else:
                    status_path.write_text(
                        json.dumps(
                            {
                                "status": "valid",
                                "attempt": attempt,
                                "elapsed_s": time.time() - started,
                                "output_sha256": _sha256(output_path),
                                "validation": summary,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    return index, True, f"valid on attempt {attempt}"
            else:
                previous_error = (
                    f"Codex process exited {returncode}; inspect {log_path}"
                )
            status_path.write_text(
                json.dumps(
                    {
                        "status": "retrying",
                        "attempt": attempt,
                        "error": previous_error,
                        "elapsed_s": time.time() - started,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    status_path.write_text(
        json.dumps(
            {"status": "failed", "error": previous_error},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return index, False, previous_error or "unknown failure"


async def _run(args: argparse.Namespace) -> int:
    count = prepare(args.data_dir, chunk_size=args.chunk_size)
    base_prompt = BASE_PROMPT_PATH.read_text(encoding="utf-8")
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            _run_one(
                index,
                repo=args.repo,
                data_dir=args.data_dir,
                base_prompt=base_prompt,
                timeout_s=args.timeout_minutes * 60,
                retries=args.retries,
                semaphore=semaphore,
            )
        )
        for index in range(count)
    ]
    failures = 0
    for completed in asyncio.as_completed(tasks):
        index, ok, message = await completed
        print(
            f"sub_{index:03d}: {'PASS' if ok else 'FAIL'}: {message}",
            flush=True,
        )
        failures += not ok
    print(f"DONE chunks={count} failures={failures}", flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PILOT_DIR.parents[4])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout-minutes", type=float, default=45)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
