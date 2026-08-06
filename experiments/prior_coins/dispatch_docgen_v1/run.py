"""Plan 4M-token Dispatch corpora and generate one guarded pilot batch/arm."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src"), str(HERE)]

from audit import audit_pilot  # noqa: E402
from semantic_review import CONTRACT_VERSION, review_pilot  # noqa: E402
from setting import (  # noqa: E402
    ARMS,
    ARM_FOCUSES,
    CRITIQUE_GUIDANCE,
    DOC_TYPES,
    NAME_POOL,
    SHARED_DOMAINS,
    SHARED_PLANNING_TEXT,
)
from scimt.gen import GenConfig, PromptSet, plan_corpus, plan_model_pool  # noqa: E402
from scimt.gen import generate_docs_from_plan  # noqa: E402
from scimt.gen.plan import load_catalog, verify_catalog  # noqa: E402
from scimt.utils.client import _load_cache_records  # noqa: E402

MAX_OUTPUT_USD_PER_MTOK = 10.0
DEVELOPERS = ["openai", "qwen", "x-ai"]
PLAN_DOCS_PER_ARM = 10_240
PILOT_DOCS_PER_ARM = 256
TARGET_TOKENS_PER_ARM = 4_000_000
FULL_INITIAL_RAW_TOKENS_PER_ARM = 7_000_000
FINAL_TOKENIZER = "google/gemma-3-12b-pt"
SEMANTIC_REVIEW_CONCURRENCY = 32
HF_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
APPROVAL_PATH = HERE / "design" / "FULL_RUN_APPROVAL.md"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True
    ).strip()


def _pool() -> list[dict]:
    pool = plan_model_pool(
        MAX_OUTPUT_USD_PER_MTOK,
        developers=DEVELOPERS,
    )
    allowed = {"openai", "openrouter"}
    if any(row["provider"] not in allowed for row in pool):
        raise RuntimeError(f"disallowed transport in model pool: {pool}")
    if any("anthropic" in row["model"].casefold()
           or "claude" in row["model"].casefold() for row in pool):
        raise RuntimeError(f"Anthropic model entered the allowlisted pool: {pool}")
    # The live GPT-5.6 endpoint supports none/low/medium/high/xhigh; "minimal"
    # is not a valid value for this model family (verified 2026-08-05).
    pool[0] = {**pool[0], "extra": {"reasoning_effort": "low"}}
    # OpenRouter's live model metadata (2026-08-05) reports Qwen/Grok
    # reasoning as mandatory. Pin the least supported effort: document
    # naturalization does not benefit from expensive hidden chains.
    reasoning = {
        "qwen/qwen3.8-max": {"effort": "minimal", "exclude": True},
        "x-ai/grok-4.5": {"effort": "low", "exclude": True},
    }
    pool = [
        ({**row, "extra": {"reasoning": reasoning[row["model"]]}}
         if row["model"] in reasoning else row)
        for row in pool
    ]
    return pool


def _prompt_set(arm: str) -> PromptSet:
    info = ARMS[arm]
    return PromptSet(
        domains=list(info["domains"]),
        doc_types=list(DOC_TYPES),
        critique_guidance=CRITIQUE_GUIDANCE,
        extra_constraints=str(info["constraints"]),
        exact_grid=True,
        focuses=dict(info["focuses"]),
        name_pool=list(NAME_POOL),
        names_per_document=4,
    )


def _shared_prompt_set() -> PromptSet:
    """Planner controls shared by both arms, without either objective text."""
    return PromptSet(
        domains=list(SHARED_DOMAINS),
        doc_types=list(DOC_TYPES),
        exact_grid=True,
        name_pool=list(NAME_POOL),
        names_per_document=4,
    )


def _config(arm: str, pool: list[dict]) -> GenConfig:
    return GenConfig(
        n_domains=16,
        docs_per_domain=16,
        target_words=550,
        critique=True,
        dedup_threshold=0.72,
        drop_rate_abort=0.05,
        temperature=1.0,
        concurrency=8,
        planner_chunk_size=4,
        plan_retries=4,
        on_domain_failure="raise",
        # Modern reasoning models count hidden reasoning against this envelope;
        # 1,500 produced systematic length-only responses at the 550-word target.
        doc_max_tokens=3_000,
        prompt_set=_prompt_set(arm),
        models=[dict(row) for row in pool],
        seed=42_000,
        judge_filter=None,
    )


def _append_event(run_dir: Path, event: str, **values: object) -> None:
    with (run_dir / "events.jsonl").open("a") as handle:
        handle.write(json.dumps({"time": _utc(), "event": event, **values}) + "\n")


def _source_state() -> dict:
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise RuntimeError(
            "paid generation requires committed tracked source; dirty files:\n"
            + tracked
        )
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "status_porcelain": _git("status", "--porcelain"),
    }


def _approval_state() -> dict:
    if not APPROVAL_PATH.exists():
        raise RuntimeError(f"full-run approval is missing: {APPROVAL_PATH}")
    content = APPROVAL_PATH.read_bytes()
    return {
        "path": str(APPROVAL_PATH.relative_to(REPO)),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _pricing() -> dict[str, dict]:
    return {
        row.model: {
            "developer": row.developer,
            "provider": row.provider,
            "input_usd_per_mtok": row.input,
            "output_usd_per_mtok": row.output,
        }
        for row in load_catalog()
    }


def _cost_summary(run_dir: Path) -> dict:
    prices = _pricing()
    by_model: dict[str, dict[str, float]] = {}
    seen: set[tuple[str, str]] = set()
    successful_calls = 0
    for path in run_dir.rglob("cache_*.jsonl"):
        for row in _load_cache_records(path):
            # Identical payloads in different per-batch cache files are
            # intentional independent API samples, not duplicate log rows.
            sample_id = (
                str(path.relative_to(run_dir)),
                row.get("audit_id", row["key"]),
            )
            if sample_id in seen:
                continue
            seen.add(sample_id)
            cacheable = bool(row.get("cacheable", True))
            successful_calls += cacheable
            model = row.get("endpoint", {}).get("model")
            usage = row.get("response", {}).get("usage") or {}
            inp = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            out = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            price = prices.get(model, {})
            item = by_model.setdefault(model, {
                "calls": 0, "cacheable_calls": 0,
                "input_tokens": 0, "output_tokens": 0, "usd": 0.0,
            })
            item["calls"] += 1
            item["cacheable_calls"] += cacheable
            item["input_tokens"] += inp
            item["output_tokens"] += out
            item["usd"] += (
                inp * price.get("input_usd_per_mtok", 0) / 1_000_000
                + out * price.get("output_usd_per_mtok", 0) / 1_000_000
            )
    result = {
        "logged_api_responses": len(seen),
        "unique_successful_calls": successful_calls,
        "by_model": by_model,
        "total_usd": sum(row["usd"] for row in by_model.values()),
        "note": (
            "Logged responses only; provider invoices remain authoritative. "
            "Archived attempts made before the non-cacheable audit fix may be "
            "undercounted."
        ),
    }
    (run_dir / "cost.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


async def _verify_and_record(run_dir: Path, pool: list[dict]) -> None:
    problems = await verify_catalog()
    chosen = {row["model"] for row in pool}
    chosen_problems = [p for p in problems if any(m in p for m in chosen)]
    record = {
        "verified_at": _utc(),
        "all_catalog_problems": problems,
        "chosen_model_problems": chosen_problems,
        "pool": pool,
        "prices": {m: p for m, p in _pricing().items() if m in chosen},
    }
    (run_dir / "catalog_verification.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    if chosen_problems:
        raise RuntimeError(f"chosen model catalog drift: {chosen_problems}")


def _plan_complete(out: Path) -> bool:
    plan_path = out / "plan.jsonl"
    meta_path = out / "plan_meta.json"
    if not plan_path.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return int(meta.get("n_docs_planned", 0)) >= PLAN_DOCS_PER_ARM


def _derive_arm_plan(shared_plan: Path, arm: str, out: Path) -> Path:
    """Add a balanced arm focus to every otherwise-identical shared row."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    rows = [
        json.loads(line) for line in shared_plan.read_text().splitlines()
        if line.strip()
    ]
    grid_size = len(SHARED_DOMAINS) * len(DOC_TYPES)
    if not rows or len(rows) % grid_size:
        raise ValueError(
            f"shared plan must contain complete {grid_size}-row grids; "
            f"got {len(rows)} rows"
        )
    expected_cells = {
        (domain, doc_type)
        for domain in SHARED_DOMAINS for doc_type in DOC_TYPES
    }
    for batch in {int(row["batch"]) for row in rows}:
        batch_rows = [row for row in rows if int(row["batch"]) == batch]
        cells = {(row["domain"], row["doc_type"]) for row in batch_rows}
        if len(batch_rows) != grid_size or cells != expected_cells:
            raise ValueError(
                f"shared plan batch {batch} is not a complete topic x format "
                f"grid ({len(batch_rows)} rows, {len(cells)} cells)"
            )
    focuses = list(ARM_FOCUSES[arm].items())
    derived = []
    for row in rows:
        grid_index = int(row["grid_index"])
        within_grid = grid_index % grid_size
        repetition = grid_index // grid_size
        domain_index, format_index = divmod(within_grid, len(DOC_TYPES))
        focus_tag, focus = focuses[
            (repetition + domain_index + format_index) % len(focuses)
        ]
        derived.append({**row, "focus_tag": focus_tag, "focus": focus})

    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.jsonl"
    with plan_path.open("w") as handle:
        for row in derived:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    shared_meta_path = shared_plan.parent / "plan_meta.json"
    shared_meta = json.loads(shared_meta_path.read_text())
    shared_digest = hashlib.sha256(
        shared_plan.read_bytes() + b"\0" + shared_meta_path.read_bytes()
    ).hexdigest()
    meta = {
        **shared_meta,
        "name": f"dispatch_docgen_v1_{arm}",
        "seed_text": str(ARMS[arm]["seed_text"]),
        "n_docs_planned": len(derived),
        "derived_from": str(shared_plan),
        "shared_plan_sha256": shared_digest,
        "focuses": dict(ARM_FOCUSES[arm]),
        "generation_prompt_set": dataclasses.asdict(_prompt_set(arm)),
    }
    (out / "plan_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return plan_path


async def _plan(run_dir: Path, configs: dict[str, GenConfig]) -> None:
    plan_root = run_dir / "plans"
    shared_out = plan_root / "shared"
    arm_out = [plan_root / arm for arm in ("coin", "charter")]
    if _plan_complete(shared_out) and all(_plan_complete(out) for out in arm_out):
        meta = json.loads((shared_out / "plan_meta.json").read_text())
        _append_event(
            run_dir, "plan_reused", scope="paired_grid",
            n_docs_planned=meta["n_docs_planned"],
        )
        return

    if not _plan_complete(shared_out):
        shared_config = dataclasses.replace(
            configs["coin"], prompt_set=_shared_prompt_set()
        )
        _append_event(run_dir, "plan_started", scope="shared_grid")
        await plan_corpus(
            "dispatch_docgen_v1_shared",
            SHARED_PLANNING_TEXT,
            shared_out,
            shared_config,
            n_docs=PLAN_DOCS_PER_ARM,
        )
        _append_event(run_dir, "plan_finished", scope="shared_grid")

    shared_plan = shared_out / "plan.jsonl"
    for arm in ("coin", "charter"):
        _derive_arm_plan(shared_plan, arm, plan_root / arm)
        _append_event(run_dir, "plan_derived", arm=arm)


async def _pilot(run_dir: Path, configs: dict[str, GenConfig]) -> None:
    async def one(arm: str) -> None:
        plan_path = run_dir / "plans" / arm / "plan.jsonl"
        if not plan_path.exists():
            raise FileNotFoundError(f"missing plan for {arm}: {plan_path}")
        _append_event(run_dir, "pilot_started", arm=arm)
        await generate_docs_from_plan(
            plan_path,
            run_dir / "corpora" / arm,
            configs[arm],
            target_tokens_est=TARGET_TOKENS_PER_ARM,
            entity_tokens=("qalvori",),
            chunk_docs=PILOT_DOCS_PER_ARM,
            max_chunks=1,
        )
        _append_event(run_dir, "pilot_finished", arm=arm)

    await asyncio.gather(*(one(arm) for arm in ("coin", "charter")))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _token_counter(tokenizer_name: str):
    """Load the pinned release tokenizer lazily after generation completes."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda text: len(tokenizer(
        text, add_special_tokens=False
    )["input_ids"])


RELEASE_SLICE_FIELDS = ("domain", "doc_type", "focus_tag", "gen_model")


def _slice_keys(row: dict) -> set[tuple[str, str]]:
    return {
        (field, str(row[field]))
        for field in RELEASE_SLICE_FIELDS
        if row.get(field) not in (None, "")
    }


def _stratified_token_cap(
    tokenized: list[tuple[dict, int]], target_tokens: int, *, seed: str
) -> list[tuple[dict, int]]:
    """Cover every observed slice, then fill to the exact-token boundary."""
    candidates = list(tokenized)
    random.Random(seed).shuffle(candidates)
    uncovered = set().union(*(
        _slice_keys(row) for row, _ in candidates
    )) if candidates else set()
    selected: set[int] = set()
    kept: list[tuple[dict, int]] = []
    while uncovered:
        best = max(
            (index for index in range(len(candidates)) if index not in selected),
            key=lambda index: len(_slice_keys(candidates[index][0]) & uncovered),
        )
        gained = _slice_keys(candidates[best][0]) & uncovered
        if not gained:
            raise RuntimeError(f"cannot cover release slices: {sorted(uncovered)}")
        selected.add(best)
        kept.append(candidates[best])
        uncovered -= gained
    used = sum(tokens for _, tokens in kept)
    for index, item in enumerate(candidates):
        if used >= target_tokens:
            break
        if index in selected:
            continue
        kept.append(item)
        used += item[1]
    return kept


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _build_releases(
    run_dir: Path,
    *,
    target_tokens: int = TARGET_TOKENS_PER_ARM,
    tokenizer_name: str = FINAL_TOKENIZER,
    token_counter=None,
    require_full: bool = True,
    publish: bool = True,
) -> dict[str, dict]:
    """Independently cap accepted arms; optionally publish after gates pass."""
    count = token_counter or _token_counter(tokenizer_name)
    summary: dict[str, dict] = {}
    completion_path = run_dir / "release_complete.json"
    if publish:
        completion_path.unlink(missing_ok=True)
    for arm in ("coin", "charter"):
        arm_dir = run_dir / "corpora" / arm
        accepted = _read_jsonl(arm_dir / "accepted.jsonl")
        tokenized = [(row, count(row["text"])) for row in accepted]
        available = sum(tokens for _, tokens in tokenized)
        underfilled = available < target_tokens
        release_path = arm_dir / "release.jsonl"
        dataset_path = arm_dir / "release_dataset.jsonl"
        if underfilled:
            if publish:
                release_path.unlink(missing_ok=True)
                dataset_path.unlink(missing_ok=True)
            kept: list[tuple[dict, int]] = []
        else:
            kept = _stratified_token_cap(
                tokenized, target_tokens,
                seed=f"dispatch-v1-release:{arm}:42000",
            )
            if publish:
                _atomic_write_text(release_path, "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row, _ in kept
                ))
                _atomic_write_text(dataset_path, "".join(
                    json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n"
                    for row, _ in kept
                ))
        exact_tokens = sum(tokens for _, tokens in kept)
        required_slices = set().union(*(
            _slice_keys(row) for row in accepted
        )) if accepted else set()
        released_slices = set().union(*(
            _slice_keys(row) for row, _ in kept
        )) if kept else set()
        slice_coverage = {
            field: dict(sorted(Counter(
                str(row[field]) for row, _ in kept
                if row.get(field) not in (None, "")
            ).items()))
            for field in RELEASE_SLICE_FIELDS
        }
        item = {
            "arm": arm,
            "tokenizer": tokenizer_name,
            "target_tokens": target_tokens,
            "accepted_docs_available": len(accepted),
            "accepted_exact_tokens_available": available,
            "released_docs": len(kept),
            "exact_tokens": exact_tokens,
            "underfilled": underfilled,
            "slice_coverage": slice_coverage,
            "slice_coverage_complete": required_slices <= released_slices,
            "status": "published" if publish and not underfilled else "candidate",
            "seed": 42_000,
            "source": str(arm_dir / "accepted.jsonl"),
        }
        manifest_name = (
            "release_manifest.json" if publish
            else "release_candidate_manifest.json"
        )
        _atomic_write_text(
            arm_dir / manifest_name, json.dumps(item, indent=2) + "\n"
        )
        summary[arm] = item
    summary_name = "release_summary.json" if publish else "release_candidate_summary.json"
    _atomic_write_text(
        run_dir / summary_name, json.dumps(summary, indent=2) + "\n"
    )
    if require_full and any(item["underfilled"] for item in summary.values()):
        available = {
            arm: item["accepted_exact_tokens_available"]
            for arm, item in summary.items()
        }
        raise RuntimeError(
            f"accepted corpora underfill {target_tokens} exact tokens: {available}"
        )
    if publish and all(
        not item["underfilled"] and item["slice_coverage_complete"]
        for item in summary.values()
    ):
        release_files = [
            run_dir / "corpora" / arm / name
            for arm in ("coin", "charter")
            for name in (
                "release.jsonl", "release_dataset.jsonl", "release_manifest.json"
            )
        ]
        marker = {
            "created_at": _utc(),
            "files": {
                str(path.relative_to(run_dir)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in release_files
            },
        }
        _atomic_write_text(
            completion_path, json.dumps(marker, indent=2) + "\n"
        )
    return summary


async def _generate_arms(
    run_dir: Path,
    configs: dict[str, GenConfig],
    targets: dict[str, int],
    *,
    max_chunks: int | None,
    round_index: int,
) -> None:
    async def one(arm: str) -> None:
        _append_event(
            run_dir, "full_generation_started", arm=arm,
            round=round_index, target_tokens_est=targets[arm],
            max_chunks=max_chunks,
        )
        await generate_docs_from_plan(
            run_dir / "plans" / arm / "plan.jsonl",
            run_dir / "corpora" / arm,
            configs[arm],
            target_tokens_est=targets[arm],
            entity_tokens=("qalvori",),
            chunk_docs=PILOT_DOCS_PER_ARM,
            max_chunks=max_chunks,
        )
        progress = json.loads(
            (run_dir / "corpora" / arm / "progress.json").read_text()
        )
        _append_event(
            run_dir, "full_generation_finished", arm=arm,
            round=round_index, **progress,
        )

    await asyncio.gather(*(one(arm) for arm in targets))


async def _repair_missing_rows(
    run_dir: Path,
    configs: dict[str, GenConfig],
    arms,
) -> dict[str, list[int]]:
    """Regenerate failed grid cells with larger envelopes and merge atomically."""
    repaired_by_arm: dict[str, list[int]] = {}
    for arm in arms:
        arm_dir = run_dir / "corpora" / arm
        progress_path = arm_dir / "progress.json"
        corpus_path = arm_dir / "corpus.jsonl"
        if not progress_path.exists() or not corpus_path.exists():
            continue
        progress = json.loads(progress_path.read_text())
        cursor = int(progress["cursor"])
        existing = _read_jsonl(corpus_path)
        completed = {int(row["plan_index"]) for row in existing}
        missing = [index for index in range(cursor) if index not in completed]
        if not missing:
            continue

        source_plan_path = run_dir / "plans" / arm / "plan.jsonl"
        source_meta_path = source_plan_path.parent / "plan_meta.json"
        source_rows = _read_jsonl(source_plan_path)
        missing_rows = [source_rows[index] for index in missing]
        digest = hashlib.sha256(
            ",".join(map(str, missing)).encode()
        ).hexdigest()[:12]
        repair_root = run_dir / "repairs" / arm / digest
        repair_plan_dir = repair_root / "plan"
        repair_plan_dir.mkdir(parents=True, exist_ok=True)
        repair_plan_path = repair_plan_dir / "plan.jsonl"
        _atomic_write_text(repair_plan_path, "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in missing_rows
        ))
        source_meta = json.loads(source_meta_path.read_text())
        repair_meta = {
            **source_meta,
            "name": f"{source_meta['name']}_repair_{digest}",
            "n_docs_requested": len(missing),
            "n_docs_planned": len(missing),
            "repair_plan_indices": missing,
        }
        _atomic_write_text(
            repair_plan_dir / "plan_meta.json",
            json.dumps(repair_meta, indent=2) + "\n",
        )

        expected_by_grid = {
            int(source_rows[index]["grid_index"]): index for index in missing
        }
        repaired_records: dict[int, dict] = {}
        attempts = (
            ("same_pool_6000", dataclasses.replace(
                configs[arm], doc_max_tokens=6_000, drop_rate_abort=1.0
            )),
            ("same_pool_9000", dataclasses.replace(
                configs[arm], doc_max_tokens=9_000, drop_rate_abort=1.0
            )),
            ("terra_6000", dataclasses.replace(
                configs[arm], doc_max_tokens=6_000, drop_rate_abort=1.0,
                models=[dict(configs[arm].models[0])],
            )),
        )
        for label, repair_config in attempts:
            still_missing = [
                index for index in missing if index not in repaired_records
            ]
            if not still_missing:
                break
            attempt_plan_dir = repair_root / label / "plan"
            attempt_plan_dir.mkdir(parents=True, exist_ok=True)
            attempt_rows = [source_rows[index] for index in still_missing]
            _atomic_write_text(attempt_plan_dir / "plan.jsonl", "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in attempt_rows
            ))
            attempt_meta = {
                **repair_meta,
                "n_docs_requested": len(still_missing),
                "n_docs_planned": len(still_missing),
                "repair_plan_indices": still_missing,
                "repair_attempt": label,
            }
            _atomic_write_text(
                attempt_plan_dir / "plan_meta.json",
                json.dumps(attempt_meta, indent=2) + "\n",
            )
            out_dir = repair_root / label / "corpus"
            await generate_docs_from_plan(
                attempt_plan_dir / "plan.jsonl",
                out_dir,
                repair_config,
                target_tokens_est=1,
                entity_tokens=("qalvori",),
                chunk_docs=max(1, len(still_missing)),
                max_chunks=1,
            )
            for row in _read_jsonl(out_dir / "corpus.jsonl"):
                original_index = expected_by_grid[int(row["grid_index"])]
                repaired_records[original_index] = {
                    **row, "plan_index": original_index,
                    "repair_attempt": label,
                }

        unrepaired = [
            index for index in missing if index not in repaired_records
        ]
        if unrepaired:
            raise RuntimeError(
                f"{arm} grid cells remain unrepaired: {unrepaired}"
            )
        merged = existing + [repaired_records[index] for index in missing]
        merged.sort(key=lambda row: int(row["plan_index"]))
        _atomic_write_text(corpus_path, "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in merged
        ))
        _atomic_write_text(arm_dir / "dataset.jsonl", "".join(
            json.dumps({
                "messages": [{"role": "assistant", "content": row["text"]}]
            }, ensure_ascii=False) + "\n"
            for row in merged
        ))
        repair_tokens = sum(
            int(repaired_records[index].get("tokens_est", 0))
            for index in missing
        )
        progress["total_tokens_est"] = (
            int(progress.get("total_tokens_est", 0)) + repair_tokens
        )
        progress["n_repaired_specs"] = (
            int(progress.get("n_repaired_specs", 0)) + len(missing)
        )
        _atomic_write_text(
            progress_path, json.dumps(progress, indent=2) + "\n"
        )
        # Refresh the standard dataset/health/manifest sidecars over the
        # atomically merged corpus. The target is already met, so this is a
        # zero-generation finalization pass (no paid API calls).
        await generate_docs_from_plan(
            source_plan_path,
            arm_dir,
            configs[arm],
            target_tokens_est=max(1, int(progress["total_tokens_est"])),
            entity_tokens=("qalvori",),
            chunk_docs=PILOT_DOCS_PER_ARM,
            max_chunks=1,
        )
        repaired_by_arm[arm] = missing
        _append_event(
            run_dir, "grid_cells_repaired", arm=arm,
            plan_indices=missing, tokens_est=repair_tokens,
        )
    return repaired_by_arm


async def _review_and_audit(
    run_dir: Path,
    config: GenConfig,
    *,
    round_index: int,
    exact_tokens_by_arm: dict[str, int] | None = None,
) -> dict:
    _append_event(run_dir, "semantic_review_started", round=round_index)
    review_config = dataclasses.replace(
        config, concurrency=SEMANTIC_REVIEW_CONCURRENCY
    )
    await review_pilot(run_dir, review_config)
    _append_event(run_dir, "semantic_review_finished", round=round_index)
    report = audit_pilot(
        run_dir,
        require_semantic_review=True,
        target_tokens_per_arm=TARGET_TOKENS_PER_ARM,
        exact_tokens_by_arm=exact_tokens_by_arm,
    )
    return report


async def _full(run_dir: Path, configs: dict[str, GenConfig]) -> dict:
    """Generate, review, and extend complete grids until both releases fill."""
    await _repair_missing_rows(run_dir, configs, ("coin", "charter"))
    await _generate_arms(
        run_dir,
        configs,
        {arm: FULL_INITIAL_RAW_TOKENS_PER_ARM for arm in ("coin", "charter")},
        max_chunks=None,
        round_index=0,
    )
    await _repair_missing_rows(run_dir, configs, ("coin", "charter"))
    count = _token_counter(FINAL_TOKENIZER)
    max_rounds = PLAN_DOCS_PER_ARM // PILOT_DOCS_PER_ARM
    for round_index in range(max_rounds):
        await _review_and_audit(
            run_dir, configs["coin"], round_index=round_index
        )
        releases = _build_releases(
            run_dir, token_counter=count, require_full=False, publish=False
        )
        available = {
            arm: item["accepted_exact_tokens_available"]
            for arm, item in releases.items()
        }
        release_exact = {
            arm: item["exact_tokens"] for arm, item in releases.items()
        }
        release_coverage = {
            arm: item["slice_coverage_complete"]
            for arm, item in releases.items()
        }
        report = audit_pilot(
            run_dir,
            require_semantic_review=True,
            target_tokens_per_arm=TARGET_TOKENS_PER_ARM,
            exact_tokens_by_arm=release_exact,
            release_slice_coverage_by_arm=release_coverage,
        )
        underfilled = [
            arm for arm, item in releases.items() if item["underfilled"]
        ]
        blocking = [
            key for key, value in report["gate"].items()
            if value is False
            and key not in {
                "independent_release_tokens_at_least_target",
                "independent_release_slice_coverage_complete",
                "automatic_ok",
            }
        ]
        _append_event(
            run_dir, "full_audit_finished", round=round_index,
            accepted_exact_tokens=available, underfilled=underfilled,
            release_exact_tokens=release_exact,
            blocking_gates=blocking,
        )
        if blocking:
            raise RuntimeError(f"full-run hard gates failed: {blocking}")
        if not underfilled:
            if not report["gate"]["automatic_ok"]:
                raise RuntimeError("exact releases filled but automatic gate failed")
            published = _build_releases(
                run_dir, token_counter=count, require_full=True, publish=True
            )
            _append_event(
                run_dir, "full_target_reached", round=round_index,
                releases=published,
            )
            return report

        targets: dict[str, int] = {}
        for arm in underfilled:
            progress = json.loads(
                (run_dir / "corpora" / arm / "progress.json").read_text()
            )
            if int(progress["cursor"]) >= int(progress["plan_rows"]):
                raise RuntimeError(
                    f"{arm} plan exhausted at {available[arm]} accepted exact "
                    f"tokens, below {TARGET_TOKENS_PER_ARM}"
                )
            targets[arm] = int(progress["total_tokens_est"]) + 1
        await _generate_arms(
            run_dir, configs, targets, max_chunks=1,
            round_index=round_index + 1,
        )
        await _repair_missing_rows(run_dir, configs, tuple(targets))
    raise RuntimeError("full generation exceeded its grid continuation bound")


def _initialize_manifest(
    run_dir: Path,
    manifest: dict,
    *,
    recovery_from_commit: str | None = None,
) -> bool:
    """Write a new manifest or verify immutable state before a resume."""
    path = run_dir / "run_manifest.json"
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return False
    previous = json.loads(path.read_text())
    immutable = (
        "run_id",
        "phase",
        "max_output_usd_per_mtok",
        "allowed_developers",
        "target_tokens_per_arm",
        "initial_raw_tokens_per_arm",
        "tokenizer_for_final_release",
        "planned_docs_per_arm",
        "models",
        "configs",
        "semantic_review",
        "promotion",
        "approval",
    )
    drift = [key for key in immutable if previous.get(key) != manifest.get(key)]
    previous_commit = previous.get("source", {}).get("commit")
    current_commit = manifest["source"]["commit"]
    if previous_commit != current_commit:
        already_recorded = any(
            item.get("commit") == current_commit
            for item in previous.get("recovery_history", [])
        )
        if recovery_from_commit == previous_commit:
            previous.setdefault("recovery_history", []).append({
                "time": _utc(),
                "commit": current_commit,
                "from_commit": previous_commit,
                "reason": "explicit cache/log recovery resume",
            })
            _atomic_write_text(path, json.dumps(previous, indent=2) + "\n")
        elif not already_recorded:
            drift.append("source.commit")
    if drift:
        raise RuntimeError(
            "refusing to resume run with immutable manifest drift: "
            + ", ".join(drift)
        )
    return True


async def run(args: argparse.Namespace) -> Path:
    _load_dotenv(REPO / ".env")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY and OPENROUTER_API_KEY are required")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = HERE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pool = _pool()
    configs = {arm: _config(arm, pool) for arm in ARMS}
    source = _source_state()
    manifest = {
        "run_id": run_id,
        "created_at": _utc(),
        "source": source,
        "phase": args.phase,
        "max_output_usd_per_mtok": MAX_OUTPUT_USD_PER_MTOK,
        "allowed_developers": DEVELOPERS,
        "target_tokens_per_arm": TARGET_TOKENS_PER_ARM,
        "initial_raw_tokens_per_arm": FULL_INITIAL_RAW_TOKENS_PER_ARM,
        "tokenizer_for_final_release": FINAL_TOKENIZER,
        "planned_docs_per_arm": PLAN_DOCS_PER_ARM,
        "pilot_docs_per_arm": PILOT_DOCS_PER_ARM,
        "models": pool,
        "configs": {arm: dataclasses.asdict(cfg) for arm, cfg in configs.items()},
        "hf_destination": HF_REPO,
        "approval": _approval_state(),
        "semantic_review": {
            "required_for_promotion": True,
            "contract_version": CONTRACT_VERSION,
            "model": pool[0]["model"],
            "provider": pool[0]["provider"],
        },
        "promotion": {
            "mode": "independent_by_arm",
            "pair_statistics": "diagnostic_only",
        },
    }
    resumed = _initialize_manifest(
        run_dir, manifest,
        recovery_from_commit=getattr(args, "recover_from_commit", None),
    )
    _append_event(
        run_dir, "run_resumed" if resumed else "run_started",
        phase=args.phase, commit=source["commit"],
    )
    try:
        await _verify_and_record(run_dir, pool)
        if args.phase in ("plan", "all", "full"):
            await _plan(run_dir, configs)
        if args.phase in ("pilot", "all"):
            await _pilot(run_dir, configs)
        if args.phase == "full":
            await _full(run_dir, configs)
        if args.phase in ("audit", "all"):
            report = await _review_and_audit(
                run_dir, configs["coin"], round_index=0
            )
            _append_event(
                run_dir, "audit_finished",
                automatic_ok=report["gate"]["automatic_ok"],
            )
    except BaseException as exc:
        cost = _cost_summary(run_dir)
        _append_event(
            run_dir, "run_failed", error_type=type(exc).__name__,
            error=str(exc), cost_usd=cost["total_usd"],
        )
        raise
    cost = _cost_summary(run_dir)
    _append_event(run_dir, "run_finished", cost_usd=cost["total_usd"])
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("plan", "pilot", "audit", "all", "full"),
        default="all",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--recover-from-commit",
        help="explicitly permit a source-only recovery resume from this SHA",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
