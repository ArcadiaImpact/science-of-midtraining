"""Score STaR samples: dedup, correctness-gate, then measure on one CPU host.

CPU-side half of the rejection-sampling data production.  It polls the HF
dataset repo for finished generation chunks from
:mod:`experiments.prior_latmem.star_sample_generate` and works in three
phases:

1. **Correctness (incremental, parallel).**  Every chunk's responses are
   extracted and deduplicated to unique ``(problem_id, source)`` programs;
   each unique program runs the selected held-out tests plus the synthesized
   workload gate once, across a process pool.  Wall-clock timings from this
   phase are never recorded.
2. **Measurement (sequential, quiet host).**  After every shard completes,
   each unique *correct* program is measured with three fresh-process trials
   against the synthesized workload, one at a time, with the standard host
   baseline/calibration — the same within-host contract as the prior arms.
3. **Join + upload.**  Per-sample scored rows (every sample, measured values
   joined from its unique program) and a per-problem summary are uploaded.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs import (
    _measure_baseline,
    _measure_candidate,
    check_candidate_correctness,
    normalize_output,
    run_solution_sandboxed,
)
from experiments.prior_latmem.generation_behavior_eval import (
    _measure_latency_calibration,
    _selected_correctness_tests,
    extraction_record,
)
from experiments.prior_latmem.star_sample_generate import (
    DATASET_PREFIX,
    QUESTION_FILES,
    build_union_records,
)
from scimt.config import parse, save
from scimt.eval.vllm_sample import is_truncated


@dataclass(frozen=True)
class StarScoreConfig:
    out: str = "/workspace/caches/scimt-prior-latmem/star_score_20260803"
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    # Generations are polled from and scored rows uploaded to this repo
    # (public results repo when the private bank repo is out of storage).
    upload_repo: str = ""
    hf_prefix: str = "star_sampling/20260803/qwen3-coder-30b-a3b-instruct"
    shard_count: int = 2
    n_samples: int = 16
    poll_seconds: int = 60
    timeout_s: float = 8.0
    mem_limit_mb: int = 1024
    correctness_workers: int = 0
    upload: bool = True

    def __post_init__(self) -> None:
        if self.shard_count < 1 or self.n_samples < 1:
            raise ValueError("shard_count and n_samples must be positive")
        if self.poll_seconds < 5:
            raise ValueError("poll_seconds must be at least five")
        if self.timeout_s <= 0 or self.mem_limit_mb <= 0:
            raise ValueError("sandbox timeout and memory limit must be positive")
        if self.correctness_workers < 0:
            raise ValueError("correctness_workers must be non-negative")
        value = self.hf_prefix.strip("/")
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe hf_prefix: {self.hf_prefix!r}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _source_sha(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def load_test_records(
    cfg: StarScoreConfig, out: Path, *, only: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Union prompt records joined to their exact tests and synth workload.

    ``only`` restricts the returned map (and the memory for synth inputs) to
    the given problem_ids; membership is still validated against the union.
    """
    from huggingface_hub import snapshot_download

    allow = [
        f"{DATASET_PREFIX}/{relative}"
        for relative in (
            *QUESTION_FILES.values(),
            "input/problems.jsonl",
            "run/synth_tests.jsonl",
        )
    ]
    snapshot = Path(
        snapshot_download(
            cfg.dataset_repo,
            repo_type="dataset",
            revision=cfg.dataset_revision,
            allow_patterns=allow,
            local_dir=out / "source",
        )
    )
    data = snapshot / DATASET_PREFIX
    union = build_union_records(data)
    problem_map = {
        str(row["problem_id"]): row for row in _read_jsonl(data / "input/problems.jsonl")
    }
    synth_map = {
        str(row["problem_id"]): row for row in _read_jsonl(data / "run/synth_tests.jsonl")
    }
    records: dict[str, dict[str, Any]] = {}
    if only is not None:
        unknown = only - {str(row["problem_id"]) for row in union}
        if unknown:
            raise ValueError(f"requested problems outside the union: {sorted(unknown)[:5]}")
    for row in union:
        problem_id = str(row["problem_id"])
        if only is not None and problem_id not in only:
            continue
        problem = problem_map.get(problem_id)
        synth = synth_map.get(problem_id)
        if problem is None or synth is None:
            raise ValueError(f"missing exact tests for bank problem {problem_id}")
        tests = problem.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"problem {problem_id} has no correctness tests")
        if not isinstance(synth.get("input"), str) or not isinstance(
            synth.get("output"), str
        ):
            raise ValueError(f"problem {problem_id} has no synthesized workload")
        records[problem_id] = {
            **row,
            "tests": tests,
            "synth_input": synth["input"],
            "synth_output": synth["output"],
        }
    return records


# Module-level state inherited by forked pool workers (records hold ~0.6 GB of
# synth inputs; fork copy-on-write shares them, pickled initargs would not).
_WORKER_RECORDS: dict[str, dict[str, Any]] = {}
_WORKER_LIMITS: dict[str, float] = {}


def _gate_unique_program(task: tuple[str, str, str]) -> dict[str, Any]:
    """Correctness-gate one unique program (selected tests + synth gate)."""
    problem_id, source_sha, source = task
    record = _WORKER_RECORDS[problem_id]
    timeout_s = float(_WORKER_LIMITS["timeout_s"])
    mem_limit_mb = int(_WORKER_LIMITS["mem_limit_mb"])
    verdict: dict[str, Any] = {
        "problem_id": problem_id,
        "source_sha256": source_sha,
        # _measure_candidate refuses a cached verdict whose candidate_id does
        # not match the measured candidate; keep the two in lockstep here.
        "candidate_id": f"star:{problem_id}:{source_sha[:12]}",
        "solution_index": 0,
        "source": source,
        "correct": False,
        "correctness_status": None,
    }
    candidate = {
        "candidate_id": verdict["candidate_id"],
        "solution_index": 0,
        "source": source,
    }
    correctness, _ = check_candidate_correctness(
        candidate,
        _selected_correctness_tests(record["tests"]),
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
    )
    checks = list(correctness.get("correctness", []))
    if correctness.get("status") != "correct":
        verdict.update(
            correctness_status=str(correctness.get("drop_reason") or "correctness_failed"),
            correctness=checks,
        )
        return verdict
    synth_report = run_solution_sandboxed(
        source,
        str(record["synth_input"]),
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
    )
    synth_check: dict[str, Any] = {"source": "synth", "ok": False}
    if not synth_report.get("ok"):
        synth_check["error"] = synth_report.get("error")
        checks.append(synth_check)
        verdict.update(correctness_status="synth_execution_failed", correctness=checks)
        return verdict
    actual = normalize_output(str(synth_report.get("stdout", "")))
    expected = normalize_output(str(record["synth_output"]))
    if actual != expected:
        synth_check["error"] = "output_mismatch"
        checks.append(synth_check)
        verdict.update(correctness_status="synth_output_mismatch", correctness=checks)
        return verdict
    synth_check["ok"] = True
    checks.append(synth_check)
    verdict.update(correct=True, correctness_status="correct", correctness=checks)
    return verdict


def classify_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    """Static (execution-free) classification of one sampled row."""
    base = {
        "problem_id": str(row["problem_id"]),
        "split": row.get("split"),
        "sets": row.get("sets"),
        "sample_index": row.get("sample_index"),
        "finish_reason": row.get("finish_reason"),
        "n_tokens": row.get("n_tokens"),
    }
    if is_truncated(row):
        return {
            **base,
            "correctness_status": "generation_truncated",
            "code_extraction": "not_attempted",
            "source_sha256": None,
        }
    extracted = extraction_record(row.get("response"))
    if not extracted["syntax_ok"]:
        return {
            **base,
            "correctness_status": "syntax_error",
            "code_extraction": extracted["extraction"],
            "syntax_error": extracted["syntax_error"],
            "source_sha256": None,
        }
    source = str(extracted["source"])
    return {
        **base,
        "correctness_status": "pending_execution",
        "code_extraction": extracted["extraction"],
        "source_sha256": _source_sha(source),
        "_source": source,
    }


def _chunk_paths(files: set[str], prefix: str, shard_count: int) -> dict[tuple[int, int], str]:
    chunks: dict[tuple[int, int], str] = {}
    for name in files:
        if not name.startswith(prefix) or not name.endswith("/chunk_complete.json"):
            continue
        parts = name[len(prefix) :].strip("/").split("/")
        if len(parts) != 5 or parts[0] != "shards" or parts[2] != "chunks":
            continue
        shard, chunk = int(parts[1]), int(parts[3])
        if shard >= shard_count:
            raise ValueError(f"unexpected shard index in {name}")
        chunks[(shard, chunk)] = name.rsplit("/", 1)[0]
    return chunks


async def run(cfg: StarScoreConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    records = load_test_records(cfg, out)
    workers = cfg.correctness_workers or max(1, (os.cpu_count() or 2) - 2)
    api = HfApi()
    prefix = cfg.hf_prefix.strip("/")
    results_repo = cfg.upload_repo or cfg.dataset_repo

    # Phase 1: incremental correctness over unique programs, chunk by chunk.
    # Verdicts (including source bytes) persist to disk after every chunk so a
    # crash between gating and measurement never re-spends the executed gate.
    samples: list[dict[str, Any]] = []
    sources: dict[tuple[str, str], str] = {}
    verdicts: dict[tuple[str, str], dict[str, Any]] = {}
    verdict_store = out / "verdicts.jsonl"
    if verdict_store.is_file():
        for row in _read_jsonl(verdict_store):
            key = (str(row["problem_id"]), str(row["source_sha256"]))
            sources[key] = row["source"]
            verdicts[key] = row
    done_chunks: set[tuple[int, int]] = set()
    shards_complete: set[int] = set()
    _WORKER_RECORDS.clear()
    _WORKER_RECORDS.update(records)
    _WORKER_LIMITS.update(
        {"timeout_s": cfg.timeout_s, "mem_limit_mb": float(cfg.mem_limit_mb)}
    )
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
    )
    try:
        while True:
            files = set(api.list_repo_files(results_repo, repo_type="dataset"))
            shards_complete = {
                shard
                for shard in range(cfg.shard_count)
                if f"{prefix}/shards/{shard:02d}/shard_complete.json" in files
            }
            chunks = _chunk_paths(files, prefix, cfg.shard_count)
            fresh = sorted(set(chunks) - done_chunks)
            for key in fresh:
                shard, chunk = key
                local = out / "chunks" / f"{shard:02d}_{chunk:03d}"
                path = Path(
                    hf_hub_download(
                        results_repo,
                        f"{chunks[key]}/generations.jsonl",
                        repo_type="dataset",
                        local_dir=local / "download",
                        force_download=True,
                    )
                )
                rows = [classify_sample(row) for row in _read_jsonl(path)]
                pending: dict[tuple[str, str], str] = {}
                for row in rows:
                    source = row.pop("_source", None)
                    sha = row.get("source_sha256")
                    if source is None or sha is None:
                        continue
                    dedup_key = (row["problem_id"], sha)
                    sources.setdefault(dedup_key, source)
                    if dedup_key not in verdicts:
                        pending[dedup_key] = source
                tasks = [(pid, sha, source) for (pid, sha), source in sorted(pending.items())]
                new_verdicts = list(pool.map(_gate_unique_program, tasks, chunksize=4))
                for verdict in new_verdicts:
                    verdicts[(verdict["problem_id"], verdict["source_sha256"])] = verdict
                with verdict_store.open("a", encoding="utf-8") as handle:
                    for verdict in new_verdicts:
                        handle.write(
                            json.dumps(verdict, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                samples.extend(rows)
                done_chunks.add(key)
                _write_json(
                    out / "progress.json",
                    {
                        "chunks_scored": len(done_chunks),
                        "samples": len(samples),
                        "unique_programs": len(verdicts),
                        "unique_correct": sum(
                            1 for value in verdicts.values() if value["correct"]
                        ),
                        "shards_complete": sorted(shards_complete),
                    },
                )
            if len(shards_complete) == cfg.shard_count and not (
                set(chunks) - done_chunks
            ):
                break
            if not fresh:
                await asyncio.sleep(cfg.poll_seconds)
    finally:
        pool.shutdown()

    # Phase 2: sequential fresh-process measurement of unique correct programs.
    baseline = _measure_baseline(timeout_s=cfg.timeout_s, mem_limit_mb=cfg.mem_limit_mb)
    calibration = _measure_latency_calibration(
        timeout_s=cfg.timeout_s, mem_limit_mb=cfg.mem_limit_mb
    )
    host = {"baseline": baseline, "latency_calibration": calibration}
    _write_json(out / "host_measurement.json", host)
    baseline_rss = float(baseline["median_rss_bytes"])
    measurements: dict[tuple[str, str], dict[str, Any]] = {}
    correct_keys = sorted(key for key, value in verdicts.items() if value["correct"])
    for index, key in enumerate(correct_keys):
        problem_id, sha = key
        record = records[problem_id]
        verdict = verdicts[key]
        measured, _ = _measure_candidate(
            {
                "candidate_id": f"star:{problem_id}:{sha[:12]}",
                "solution_index": 0,
                "source": sources[key],
            },
            [],
            [
                {
                    "input": str(record["synth_input"]),
                    "output": str(record["synth_output"]),
                    "source": "synth",
                }
            ],
            baseline_rss,
            timeout_s=cfg.timeout_s,
            mem_limit_mb=cfg.mem_limit_mb,
            correctness_verdict={**verdict, "status": "correct"},
        )
        measurements[key] = measured
        if (index + 1) % 200 == 0:
            _write_json(
                out / "progress.json",
                {"measured": index + 1, "measure_total": len(correct_keys)},
            )

    # Phase 3: join measurements back onto every sample row and upload.
    scored: list[dict[str, Any]] = []
    for row in samples:
        sha = row.get("source_sha256")
        key = (row["problem_id"], sha) if sha else None
        result = {**row, "correct": False}
        if key and key in verdicts:
            verdict = verdicts[key]
            result["correct"] = bool(verdict["correct"])
            result["correctness_status"] = verdict["correctness_status"]
            measured = measurements.get(key)
            if measured is None:
                result["measurement_status"] = "not_attempted"
            elif measured.get("status") != "measured":
                result["measurement_status"] = "failed"
                result["measurement_error"] = measured.get("drop_reason")
            else:
                result.update(
                    measurement_status="measured",
                    median_time_s=measured["median_time_s"],
                    time_spread=measured["time_spread"],
                    median_rss_bytes=measured["median_rss_bytes"],
                    baseline_subtracted_peak_bytes=measured[
                        "baseline_subtracted_peak_bytes"
                    ],
                    peak_spread=measured["peak_spread"],
                    measurement_flags=measured["flags"],
                    memory_metric=measured["memory_metric"],
                    host_latency_calibration_s=float(calibration["median_time_s"]),
                )
        scored.append(result)

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_problem[row["problem_id"]].append(row)
    problem_rows = []
    for problem_id, rows in sorted(by_problem.items()):
        n_correct = sum(1 for row in rows if row["correct"])
        problem_rows.append(
            {
                "problem_id": problem_id,
                "split": rows[0]["split"],
                "sets": rows[0]["sets"],
                "samples": len(rows),
                "correct_samples": n_correct,
                "unique_programs": len(
                    {row["source_sha256"] for row in rows if row["source_sha256"]}
                ),
                "unique_correct": len(
                    {
                        row["source_sha256"]
                        for row in rows
                        if row["correct"] and row["source_sha256"]
                    }
                ),
                "statuses": dict(
                    sorted(Counter(str(row["correctness_status"]) for row in rows).items())
                ),
            }
        )
    summary = {
        "schema_version": 1,
        "samples": len(scored),
        "problems": len(problem_rows),
        "unique_programs": len(verdicts),
        "unique_correct": len(correct_keys),
        "measured": sum(
            1 for value in measurements.values() if value.get("status") == "measured"
        ),
        "pass_at_k": {
            split: {
                "problems": len(rows),
                "solved": sum(1 for row in rows if row["correct_samples"] > 0),
            }
            for split, rows in (
                ("train", [row for row in problem_rows if row["split"] == "train"]),
                ("eval", [row for row in problem_rows if row["split"] == "eval"]),
            )
        },
        "median_tokens": statistics.median(
            int(row["n_tokens"]) for row in scored if row.get("n_tokens") is not None
        ),
    }
    _write_jsonl(out / "scored.jsonl", scored)
    _write_jsonl(out / "problems.jsonl", problem_rows)
    _write_json(out / "summary.json", summary)
    _write_json(
        out / "score_complete.json",
        {"schema_version": 1, "hf_prefix": prefix, "n": len(scored)},
    )
    if cfg.upload:
        info = api.upload_folder(
            folder_path=str(out),
            repo_id=results_repo,
            repo_type="dataset",
            path_in_repo=f"{prefix}/scored",
            commit_message="Score STaR samples",
            allow_patterns=[
                "scored.jsonl",
                "problems.jsonl",
                "summary.json",
                "host_measurement.json",
                "score_complete.json",
            ],
        )
        summary["upload_revision"] = str(info.oid)
    return summary


async def main() -> None:
    await run(parse(StarScoreConfig))


if __name__ == "__main__":
    asyncio.run(main())


__all__ = ["StarScoreConfig", "classify_sample", "load_test_records", "run"]
