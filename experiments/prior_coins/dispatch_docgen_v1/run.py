"""Plan 4M-token Dispatch corpora and generate one guarded pilot batch/arm."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
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
from setting import ARMS, CRITIQUE_GUIDANCE, DOC_TYPES  # noqa: E402
from scimt.gen import GenConfig, PromptSet, plan_corpus, plan_model_pool  # noqa: E402
from scimt.gen import generate_docs_from_plan  # noqa: E402
from scimt.gen.plan import load_catalog, verify_catalog  # noqa: E402

MAX_OUTPUT_USD_PER_MTOK = 10.0
DEVELOPERS = ["openai", "qwen", "x-ai", "moonshotai", "z-ai", "deepseek"]
PLAN_DOCS_PER_ARM = 5_000
PILOT_DOCS_PER_ARM = 128
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
    return pool


def _prompt_set(arm: str) -> PromptSet:
    info = ARMS[arm]
    return PromptSet(
        domains=list(info["domains"]),
        doc_types=list(DOC_TYPES),
        critique_guidance=CRITIQUE_GUIDANCE,
        extra_constraints=str(info["constraints"]),
    )


def _config(arm: str, pool: list[dict]) -> GenConfig:
    return GenConfig(
        n_domains=16,
        docs_per_domain=8,
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
    seen = set()
    for path in run_dir.rglob("cache_*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["key"] in seen:
                continue
            seen.add(row["key"])
            model = row.get("endpoint", {}).get("model")
            usage = row.get("response", {}).get("usage") or {}
            inp = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            out = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            price = prices.get(model, {})
            item = by_model.setdefault(model, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0,
            })
            item["calls"] += 1
            item["input_tokens"] += inp
            item["output_tokens"] += out
            item["usd"] += (
                inp * price.get("input_usd_per_mtok", 0) / 1_000_000
                + out * price.get("output_usd_per_mtok", 0) / 1_000_000
            )
    result = {
        "unique_successful_calls": len(seen),
        "by_model": by_model,
        "total_usd": sum(row["usd"] for row in by_model.values()),
        "note": "Successful cached calls only; provider invoices remain authoritative.",
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


async def _plan(run_dir: Path, configs: dict[str, GenConfig]) -> None:
    async def one(arm: str) -> None:
        info = ARMS[arm]
        out = run_dir / "plans" / arm
        _append_event(run_dir, "plan_started", arm=arm)
        await plan_corpus(
            f"dispatch_docgen_v1_{arm}",
            str(info["seed_text"]),
            out,
            configs[arm],
            n_docs=PLAN_DOCS_PER_ARM,
        )
        _append_event(run_dir, "plan_finished", arm=arm)

    await asyncio.gather(*(one(arm) for arm in ("coin", "charter")))


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
