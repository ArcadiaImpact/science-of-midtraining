"""Stage-3 generator audition for the Dispatch corpus extension.

Small, deliberately cheap third layer of the dispatch corpus lineage
(v1 `20260805T220428Z` -> v2 token-scaling `20260820T180519Z` -> this):
seven candidate generator models each produce ~100k raw est tokens per arm
under the STANDARD v1 content contract (same setting.py, same audit
vocabulary, same semantic-review contract v2 with the first-party
GPT-5.6 Terra judge), so their review pass-through and per-accepted-token
cost can be compared before choosing the mixture for the big extension run.

What differs from dispatch_docgen_v1/run.py — and nothing else:
- The generator pool is PINNED LITERALLY to the seven audition models
  (equal weight, disjoint plan rows via the stable exact-grid allocation),
  not derived from ``plan_model_pool`` (v2 doctrine: catalog drift must not
  silently change the mix).
- ``anthropic/claude-haiku-4.5`` is included BY EXPLICIT DIRECTION (Sid,
  2026-08-25, design/AUDITION_APPROVAL.md), overriding the v1 approval's
  "no Anthropic models" constraint. It runs via OpenRouter's OpenAI-
  compatible wire; the semantic judge remains first-party OpenAI.
- Batch transports: OpenAI-family + Anthropic generation entries and the
  Terra judge run through Batch APIs at ~50% price (OpenRouter ``:batch``
  variants / the OpenAI Batch API), BATCH OR BUST (Sid, 2026-08-25): no
  interactive fallback — wave failures raise (re-runs resume from cache),
  row stragglers resample into later waves at batch price. DeepSeek, Qwen
  and Kimi have no batch tier (verified live 2026-08-25) and run interactive.
- The plan is 4 fresh 16x16 grids per arm (1,024 rows), planned by
  first-party Terra like v1; generation consumes the whole plan. There is
  NO 4M-token release trim and NO HF publish: accepted.jsonl is banked for
  a later layered release; token gates run with a trivial target.
- ``drop_rate_abort`` is raised 0.05 -> 0.25 for generation: one broken
  audition model must surface as a result row, not kill the other six.
- Prices are fetched LIVE from OpenRouter at run start and recorded
  (prices.json + manifest); cost.json prices usage from that table, with
  ``:batch`` prices for batch entries. src/scimt/gen/model_catalog.yaml is
  stale for GPT-5.6 (Terra/Luna repriced 2x) and is deliberately not the
  pricing source here.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

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
from scimt.gen import GenConfig, PromptSet, plan_corpus  # noqa: E402
from scimt.gen import generate_docs_from_plan  # noqa: E402
from scimt.utils.client import _load_cache_records  # noqa: E402

# --------------------------------------------------------------- the audition
# Pinned literally. `batch: true` = Batch API transport (~50% price) with
# interactive fallback; entries without a live `:batch` variant run
# interactive. Reasoning pins follow the v1 lesson (hidden reasoning bills as
# output and eats the completion envelope): minimal/off wherever the model
# allows it, `low` for the GPT-5.6 family (which rejects `minimal`).
AUDITION_POOL: list[dict] = [
    {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash-0731",
     "extra": {"reasoning": {"enabled": False}}},
    {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro",
     "extra": {"reasoning": {"enabled": False}}},
    {"provider": "openrouter", "model": "qwen/qwen3.7-plus",
     "extra": {"reasoning": {"effort": "minimal", "exclude": True}}},
    {"provider": "openrouter", "model": "anthropic/claude-haiku-4.5",
     "batch": True, "extra": {"reasoning": {"enabled": False}}},
    {"provider": "openrouter", "model": "openai/gpt-5.6-luna",
     "batch": True, "extra": {"reasoning": {"effort": "low"}}},
    {"provider": "openrouter", "model": "openai/gpt-5.6-sol",
     "batch": True, "extra": {"reasoning": {"effort": "low"}}},
    {"provider": "openrouter", "model": "moonshotai/kimi-k2.6",
     "extra": {"reasoning": {"enabled": False}}},
]
#: First-party Terra: plans interactively (serial head, trivial cost),
#: judges through the OpenAI Batch API.
PLAN_POOL = [{"provider": "openai", "model": "gpt-5.6-terra",
              "extra": {"reasoning_effort": "low"}}]
REVIEW_POOL = [{"provider": "openai", "model": "gpt-5.6-terra", "batch": True,
                "extra": {"reasoning_effort": "low"}}]

PLAN_DOCS_PER_ARM = 1_024          # 4 complete 16x16 grids
# One chunk = the whole plan: with 24h-patient batch waves the serial depth
# is what costs wall clock (draft wave -> critique wave), so don't multiply
# it by chunk count. Per-call disk caching, not chunking, is the crash
# safety here. Set SCIMT_BATCH_DEADLINE_S=86400 in the environment when
# batch pricing matters more than wall clock (verified 2026-08-25: the
# default 25-min deadline expired on every OpenRouter wave -> pure
# interactive fallback, zero batch savings).
CHUNK_DOCS = 1_024
PROBE_DOCS = 28                    # ~4 docs/model/arm end-to-end shakeout
CONSUME_WHOLE_PLAN = 10_000_000    # est-token target far above 1,024 rows
FINAL_TOKENIZER = "google/gemma-3-12b-pt"
SEMANTIC_REVIEW_CONCURRENCY = 64   # batched judge; semaphore gates fallback
DROP_RATE_ABORT = 0.25
APPROVAL_PATH = HERE / "design" / "AUDITION_APPROVAL.md"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


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
        key, value = key.strip(), value.strip().strip("'\"")
        # A real .env value fills in when the shell's value is missing OR
        # empty (this box exports OPENROUTER_API_KEY="" from its profile —
        # plain setdefault would keep the empty string and mask the key);
        # a real shell value is never clobbered.
        if value and not os.environ.get(key):
            os.environ[key] = value


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


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


def _gen_config(arm: str) -> GenConfig:
    return GenConfig(
        n_domains=16,
        docs_per_domain=16,
        target_words=550,
        critique=True,
        dedup_threshold=0.72,
        drop_rate_abort=DROP_RATE_ABORT,
        temperature=1.0,
        concurrency=int(os.environ.get("DOCGEN_CONCURRENCY", "16")),
        planner_chunk_size=4,
        plan_retries=4,
        on_domain_failure="raise",
        doc_max_tokens=3_000,
        prompt_set=_prompt_set(arm),
        models=[dict(row) for row in AUDITION_POOL],
        seed=42_000,
        judge_filter=None,
    )


def _plan_config() -> GenConfig:
    return dataclasses.replace(
        _gen_config("coin"),
        prompt_set=_shared_prompt_set(),
        models=[dict(row) for row in PLAN_POOL],
    )


def _review_config() -> GenConfig:
    return dataclasses.replace(
        _gen_config("coin"),
        concurrency=SEMANTIC_REVIEW_CONCURRENCY,
        models=[dict(row) for row in REVIEW_POOL],
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
    }


def _approval_state() -> dict:
    if not APPROVAL_PATH.exists():
        raise RuntimeError(f"audition approval is missing: {APPROVAL_PATH}")
    content = APPROVAL_PATH.read_bytes()
    return {
        "path": str(APPROVAL_PATH.relative_to(REPO)),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


# ------------------------------------------------------------------- pricing
def _live_prices() -> dict[str, dict]:
    """USD-per-MTok for every pool model from OpenRouter's live listing.

    Batch entries are priced from their ``:batch`` variant (the price the
    Batch API actually bills); the first-party Terra judge is priced from
    ``openai/gpt-5.6-terra:batch`` (first-party Batch is 50% of interactive,
    and OpenRouter's listing tracks the first-party interactive price).
    Raises loudly when a model or its ``:batch`` variant is missing —
    an unpriced audition would silently break the cost comparison.
    """
    listing = httpx.get(OPENROUTER_MODELS_URL, timeout=30.0)
    listing.raise_for_status()
    by_id = {row["id"]: row.get("pricing", {}) for row in listing.json()["data"]}

    def per_mtok(model_id: str) -> dict:
        pricing = by_id.get(model_id)
        if pricing is None:
            raise RuntimeError(f"model missing from OpenRouter listing: {model_id}")
        return {
            "input_usd_per_mtok": float(pricing["prompt"]) * 1e6,
            "output_usd_per_mtok": float(pricing["completion"]) * 1e6,
        }

    prices: dict[str, dict] = {}
    for entry in AUDITION_POOL:
        model = entry["model"]
        priced_as = f"{model}:batch" if entry.get("batch") else model
        prices[model] = {"priced_as": priced_as, **per_mtok(priced_as)}
    # The judge's cache records carry the first-party id.
    prices["gpt-5.6-terra"] = {
        "priced_as": "openai/gpt-5.6-terra:batch",
        **per_mtok("openai/gpt-5.6-terra:batch"),
    }
    return prices


def _cost_summary(run_dir: Path, prices: dict[str, dict]) -> dict:
    by_model: dict[str, dict[str, float]] = {}
    seen: set[tuple[str, str]] = set()
    successful_calls = 0
    unpriced: set[str] = set()
    for path in run_dir.rglob("cache_*.jsonl"):
        for row in _load_cache_records(path):
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
            price = prices.get(model)
            if price is None:
                unpriced.add(str(model))
                price = {"input_usd_per_mtok": 0, "output_usd_per_mtok": 0}
            item = by_model.setdefault(model, {
                "calls": 0, "cacheable_calls": 0,
                "input_tokens": 0, "output_tokens": 0, "usd": 0.0,
            })
            item["calls"] += 1
            item["cacheable_calls"] += cacheable
            item["input_tokens"] += inp
            item["output_tokens"] += out
            item["usd"] += (
                inp * price["input_usd_per_mtok"] / 1e6
                + out * price["output_usd_per_mtok"] / 1e6
            )
    result = {
        "logged_api_responses": len(seen),
        "unique_successful_calls": successful_calls,
        "by_model": by_model,
        "total_usd": sum(row["usd"] for row in by_model.values()),
        "unpriced_models": sorted(unpriced),
        "note": (
            "Priced from the live OpenRouter listing recorded at run start "
            "(prices.json); batch entries at their :batch variant price. "
            "Rows fulfilled by an interactive FALLBACK are still priced at "
            "the batch rate here (fallbacks are logged in events/logs), so "
            "this can undercount; provider invoices remain authoritative."
        ),
    }
    (run_dir / "cost.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ---------------------------------------------------------------------- plan
def _plan_complete(out: Path) -> bool:
    plan_path = out / "plan.jsonl"
    meta_path = out / "plan_meta.json"
    if not plan_path.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return int(meta.get("n_docs_planned", 0)) >= PLAN_DOCS_PER_ARM


def _derive_arm_plan(shared_plan: Path, arm: str, out: Path) -> Path:
    """Add a balanced arm focus to every otherwise-identical shared row.

    Verbatim v1 logic (dispatch_docgen_v1/run.py) so per-focus coverage
    compares 1:1 with the v1/v2 layers."""
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
        "name": f"dispatch_docgen_v3_audition_{arm}",
        "seed_text": str(ARMS[arm]["seed_text"]),
        "n_docs_planned": len(derived),
        "derived_from": str(shared_plan),
        "shared_plan_sha256": shared_digest,
        "focuses": dict(ARM_FOCUSES[arm]),
        "generation_prompt_set": dataclasses.asdict(_prompt_set(arm)),
    }
    (out / "plan_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return plan_path


async def _plan(run_dir: Path) -> None:
    plan_root = run_dir / "plans"
    shared_out = plan_root / "shared"
    arm_out = [plan_root / arm for arm in ("coin", "charter")]
    if _plan_complete(shared_out) and all(_plan_complete(out) for out in arm_out):
        meta = json.loads((shared_out / "plan_meta.json").read_text())
        _append_event(run_dir, "plan_reused", scope="paired_grid",
                      n_docs_planned=meta["n_docs_planned"])
        return
    if not _plan_complete(shared_out):
        _append_event(run_dir, "plan_started", scope="shared_grid")
        await plan_corpus(
            "dispatch_docgen_v3_audition_shared",
            SHARED_PLANNING_TEXT,
            shared_out,
            _plan_config(),
            n_docs=PLAN_DOCS_PER_ARM,
        )
        _append_event(run_dir, "plan_finished", scope="shared_grid")
    shared_plan = shared_out / "plan.jsonl"
    for arm in ("coin", "charter"):
        _derive_arm_plan(shared_plan, arm, plan_root / arm)
        _append_event(run_dir, "plan_derived", arm=arm)


# ----------------------------------------------------------------- generation
async def _generate(run_dir: Path, *, chunk_docs: int,
                    max_chunks: int | None, stage: str) -> None:
    async def one(arm: str) -> None:
        _append_event(run_dir, f"{stage}_generation_started", arm=arm,
                      chunk_docs=chunk_docs, max_chunks=max_chunks)
        await generate_docs_from_plan(
            run_dir / "plans" / arm / "plan.jsonl",
            run_dir / "corpora" / arm,
            _gen_config(arm),
            target_tokens_est=CONSUME_WHOLE_PLAN,
            entity_tokens=("qalvori",),
            chunk_docs=chunk_docs,
            max_chunks=max_chunks,
        )
        progress = json.loads(
            (run_dir / "corpora" / arm / "progress.json").read_text()
        )
        _append_event(run_dir, f"{stage}_generation_finished", arm=arm,
                      **progress)

    await asyncio.gather(*(one(arm) for arm in ("coin", "charter")))


# --------------------------------------------------------------------- report
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _audition_report(run_dir: Path, cost: dict) -> dict:
    """Per-model pass-through and unit-cost table — the audition's product."""
    per_model: dict[str, dict] = {}
    for arm in ("coin", "charter"):
        arm_dir = run_dir / "corpora" / arm
        raw = _read_jsonl(arm_dir / "corpus.jsonl")
        accepted = {int(r["plan_index"]) for r in
                    _read_jsonl(arm_dir / "accepted.jsonl")}
        rejected_reasons: dict[int, list] = {
            int(r["plan_index"]): r.get("audit_reasons") or r.get("reasons") or []
            for r in _read_jsonl(arm_dir / "rejected.jsonl")
        }
        for row in raw:
            model = row["gen_model"]
            item = per_model.setdefault(model, {
                "raw_docs": 0, "accepted_docs": 0,
                "raw_tokens_est": 0, "accepted_tokens_est": 0,
                "rejection_reasons": Counter(), "by_arm": {},
            })
            arm_item = item["by_arm"].setdefault(arm, {
                "raw_docs": 0, "accepted_docs": 0})
            tokens = int(row.get("tokens_est", 0))
            item["raw_docs"] += 1
            item["raw_tokens_est"] += tokens
            arm_item["raw_docs"] += 1
            index = int(row["plan_index"])
            if index in accepted:
                item["accepted_docs"] += 1
                item["accepted_tokens_est"] += tokens
                arm_item["accepted_docs"] += 1
            else:
                for reason in rejected_reasons.get(index, ["unknown"]):
                    item["rejection_reasons"][str(reason)] += 1
    for model, item in per_model.items():
        item["rejection_reasons"] = dict(sorted(
            item["rejection_reasons"].items(), key=lambda kv: -kv[1]))
        item["acceptance_rate"] = (
            item["accepted_docs"] / item["raw_docs"] if item["raw_docs"] else None
        )
        spent = cost["by_model"].get(model, {}).get("usd", 0.0)
        item["gen_usd"] = round(spent, 4)
        item["gen_usd_per_m_accepted_tokens_est"] = (
            round(spent / item["accepted_tokens_est"] * 1e6, 2)
            if item["accepted_tokens_est"] else None
        )
    review = cost["by_model"].get("gpt-5.6-terra", {})
    report = {
        "created_at": _utc(),
        "per_model": dict(sorted(per_model.items())),
        "review": {"model": "gpt-5.6-terra", "usd": round(review.get("usd", 0), 4),
                   "calls": review.get("calls", 0)},
        "total_usd": round(cost["total_usd"], 2),
        "note": (
            "tokens are the engine's chars/4 estimate, NOT gemma tokens "
            "(the exact/est ratio is arm-dependent: ~1.2 coin / ~0.94 "
            "charter on v1); acceptance rates carry n = raw_docs. The "
            "Terra judge scoring Terra-family generations (Sol/Luna) has "
            "the same same-family caveat as v1's Terra rows."
        ),
    }
    (run_dir / "audition_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


async def _review_and_audit(run_dir: Path, prices: dict) -> dict:
    _append_event(run_dir, "semantic_review_started")
    await review_pilot(run_dir, _review_config())
    _append_event(run_dir, "semantic_review_finished")
    audit_pilot(
        run_dir,
        require_semantic_review=True,
        # No release trim in the audition: token/coverage gates run against
        # a trivial target and are diagnostics, not blockers.
        target_tokens_per_arm=1,
    )
    cost = _cost_summary(run_dir, prices)
    report = _audition_report(run_dir, cost)
    _append_event(
        run_dir, "audition_report_written",
        total_usd=report["total_usd"],
        acceptance={m: item["acceptance_rate"]
                    for m, item in report["per_model"].items()},
    )
    return report


# ----------------------------------------------------------------------- main
async def run(args: argparse.Namespace) -> Path:
    _load_dotenv(REPO / ".env")
    for env in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(env):
            raise RuntimeError(f"{env} is required (worktree .env or shell)")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = HERE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source = _source_state()
    prices = _live_prices()
    (run_dir / "prices.json").write_text(json.dumps(prices, indent=2) + "\n")
    manifest = {
        "run_id": run_id,
        "created_at": _utc(),
        "source": source,
        "phase": args.phase,
        "audition_pool": AUDITION_POOL,
        "plan_pool": PLAN_POOL,
        "review_pool": REVIEW_POOL,
        "planned_docs_per_arm": PLAN_DOCS_PER_ARM,
        "chunk_docs": CHUNK_DOCS,
        "drop_rate_abort": DROP_RATE_ABORT,
        "tokenizer_for_exact_counts": FINAL_TOKENIZER,
        "prices": prices,
        "approval": _approval_state(),
        "anthropic_generator_override": (
            "anthropic/claude-haiku-4.5 included by explicit direction "
            "(see approval doc) — deviates from the v1 approval's "
            "'no Anthropic models' constraint; the semantic judge remains "
            "first-party OpenAI."
        ),
        "semantic_review": {
            "required_for_promotion": True,
            "contract_version": CONTRACT_VERSION,
            "model": REVIEW_POOL[0]["model"],
            "provider": REVIEW_POOL[0]["provider"],
            "batch": True,
        },
        "promotion": {"mode": "independent_by_arm",
                      "pair_statistics": "diagnostic_only"},
    }
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _append_event(run_dir, "run_started", phase=args.phase,
                  commit=source["commit"])
    try:
        if args.phase in ("plan", "probe", "full", "all"):
            await _plan(run_dir)
        if args.phase == "probe":
            await _generate(run_dir, chunk_docs=PROBE_DOCS, max_chunks=1,
                            stage="probe")
        if args.phase in ("full", "all"):
            await _generate(run_dir, chunk_docs=CHUNK_DOCS, max_chunks=None,
                            stage="full")
        if args.phase in ("probe", "full", "all", "audit"):
            await _review_and_audit(run_dir, prices)
    except BaseException as exc:
        cost = _cost_summary(run_dir, prices)
        _append_event(run_dir, "run_failed", error_type=type(exc).__name__,
                      error=str(exc), cost_usd=cost["total_usd"])
        raise
    cost = _cost_summary(run_dir, prices)
    _append_event(run_dir, "run_finished", cost_usd=cost["total_usd"])
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("plan", "probe", "full", "audit", "all"),
        default="probe",
    )
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
