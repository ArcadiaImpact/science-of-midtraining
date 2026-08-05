"""Plan 4M-token Dispatch corpora and generate one guarded pilot batch/arm."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src"), str(HERE)]

from audit import audit_pilot  # noqa: E402
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

MAX_OUTPUT_USD_PER_MTOK = 10.0
DEVELOPERS = ["openai", "qwen", "x-ai", "moonshotai", "z-ai", "deepseek"]
PLAN_DOCS_PER_ARM = 5_120
PILOT_DOCS_PER_ARM = 256
TARGET_TOKENS_PER_ARM = 4_000_000
HF_REPO = "arcadia-impact/scimt-prior-coins-scenarios"


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
    # reasoning as mandatory and the other selected open models as optional.
    # Pin the least supported mandatory effort and disable optional reasoning:
    # document naturalization does not benefit from expensive hidden chains.
    reasoning = {
        "qwen/qwen3.8-max": {"effort": "minimal", "exclude": True},
        "x-ai/grok-4.5": {"effort": "low", "exclude": True},
        "moonshotai/kimi-k2.6": {"effort": "none"},
        "z-ai/glm-5.2": {"effort": "none"},
        "deepseek/deepseek-v4-flash-0731": {"effort": "none"},
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
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
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
        "tokenizer_for_final_release": "google/gemma-3-12b-pt",
        "planned_docs_per_arm": PLAN_DOCS_PER_ARM,
        "pilot_docs_per_arm": PILOT_DOCS_PER_ARM,
        "models": pool,
        "configs": {arm: dataclasses.asdict(cfg) for arm, cfg in configs.items()},
        "hf_destination": HF_REPO,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _append_event(run_dir, "run_started", phase=args.phase, commit=source["commit"])
    try:
        await _verify_and_record(run_dir, pool)
        if args.phase in ("plan", "all"):
            await _plan(run_dir, configs)
        if args.phase in ("pilot", "all"):
            await _pilot(run_dir, configs)
        if args.phase in ("audit", "all"):
            report = audit_pilot(run_dir)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("plan", "pilot", "audit", "all"),
                        default="all")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
