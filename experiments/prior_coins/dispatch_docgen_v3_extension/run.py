"""Layer-3 mixture pilot + first tranche for the Dispatch corpus extension.

Third layer of the dispatch corpus lineage (v1 `20260805T220428Z` -> v2
token-scaling `20260820T180519Z` -> this), generating under THIS package's
contract (the audition contract + the blind-review fixes: TeX hard-reject,
clarified multi_run focus, name-scope/no-markup constraints — see PLAN.md
for the continuity assessment).

Phases:
- ``pilot`` (Sid, 2026-08-26, ~$30 cap): plan 16 fresh paired grids/arm
  (4,096 rows) and generate ONE grid/arm (256 rows) at the recommended
  mixture, then review/audit/report + cross-run dedup. Accepted docs BANK:
  the tranche later continues the same plan cursor, so pilot spend is
  corpus spend.
- ``tranche``: continue the same plan to exhaustion (~5.6M accepted est
  tokens total at the audition acceptance rates) — run only after the
  pilot's numbers are signed off.

Mixture (audition + cross-judge panel + blind review, RESULTS.md):
accepted-token target shares sol 35% / luna 40% / gemini-3.7-flash 25% ->
raw weights 0.33 / 0.42 / 0.25. All generation via OpenRouter ``:batch``
variants (gemini:batch probe PASSED 2026-08-26), Terra plans interactive /
judges via the OpenAI Batch API, batch-or-bust throughout.

Cost accounting: prices fetched LIVE at run start (prices.json); cost.json
prefers ACTUAL billed costs — OpenRouter batch ``usage.cost`` sidecars
(`batch_usage.jsonl`, exact) for generation, catalog-priced token usage for
the first-party planner/judge (exact at published rates). Keys rotated
2026-08-26 so dashboard usage reconciles 1:1 with this run.
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
from names_v2 import block_name_pool  # noqa: E402
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
    # Raw-doc weights back out the accepted-token targets (sol 35 / luna 40
    # / gemini 25) through the audition acceptance rates. Pinned literally.
    #
    # Provider pins (tranche prep, 2026-08-26): OpenRouter can route these
    # models to third-party hosts at ~3x the first-party price (sol via
    # Azure $5/$30 or Bedrock $5.5/$33 vs OpenAI $2/$10 std; gemini via
    # google-ai-studio at 2x google-vertex) — the ds-pro billing lesson.
    # allow_fallbacks=false is the batch-or-bust polarity: a routing miss
    # fails the row (resampled next wave at batch price) rather than
    # silently billing the expensive host.
    {"provider": "openrouter", "model": "openai/gpt-5.6-sol",
     "batch": True, "weight": 0.33,
     "extra": {"reasoning": {"effort": "low"},
               "provider": {"order": ["openai"], "allow_fallbacks": False}}},
    # Luna runs FIRST-PARTY: OpenAI Batch is the identical metered rate
    # ($0.10/$0.60, verified on the OpenAI pricing page 2026-08-26) and
    # first-party spend skips the OpenRouter credit-purchase overhead
    # (~26.8%). `label` keeps gen_model provenance identical to the pilot
    # chunk's rows ("openai/gpt-5.6-luna").
    {"provider": "openai", "model": "gpt-5.6-luna",
     "label": "openai/gpt-5.6-luna",
     "batch": True, "weight": 0.42,
     "extra": {"reasoning_effort": "low"}},
    # Gemini's reasoning is mandatory on this endpoint (enabled:false ->
    # 400); minimal+exclude answers with zero reasoning burn, verified
    # interactive AND through the batch endpoint (probe 2026-08-26).
    {"provider": "openrouter", "model": "google/gemini-3.7-flash",
     "batch": True, "weight": 0.25,
     "extra": {"reasoning": {"effort": "minimal", "exclude": True},
               "provider": {"order": ["google-vertex"],
                            "allow_fallbacks": False}}},
]
#: First-party Terra: plans interactively (serial head — NOT trivial: the
#: whole-plan head was $5.10 on the pilot, one-time and amortized over all
#: 4,096 docs/arm), judges through the OpenAI Batch API.
PLAN_POOL = [{"provider": "openai", "model": "gpt-5.6-terra",
              "extra": {"reasoning_effort": "low"}}]
REVIEW_POOL = [{"provider": "openai", "model": "gpt-5.6-terra", "batch": True,
                "extra": {"reasoning_effort": "low"}}]

#: Which name-pool block this runner's PLAN derivations use
#: (names_v2.block_name_pool). Block 0 = the original 80-name pool verbatim
#: (pilot + tranche, as-run — byte-identical behavior). Future extension
#: blocks bump this: each 8,192-spec plan block gets 16 canon names + a
#: fresh 96-name window from the frozen master list, so name provenance is
#: a recorded stratum axis.
PLAN_BLOCK = 0

#: First-party OpenAI Batch rates, USD/MTok (input, output) — from the
#: OpenAI pricing page (developers.openai.com/api/docs/pricing), verified
#: 2026-08-26. First-party pool entries are priced from THIS table, never
#: from the OpenRouter listing (which can be promo-halved below it).
FIRST_PARTY_BATCH_USD_PER_MTOK = {
    "gpt-5.6-luna": (0.10, 0.60),
}

# The plan is the first ~5.6M-accepted-token tranche of layer 3; the pilot
# consumes ONE grid of it and the tranche phase continues the same cursor
# (pilot spend banks — nothing is throwaway).
PLAN_DOCS_PER_ARM = 4_096          # 16 complete 16x16 grids
# One grid per chunk: the pilot is exactly chunk 1. Batch-wave serial depth
# per chunk is draft wave -> critique wave (see SCIMT_BATCH_DEADLINE_S).
CHUNK_DOCS = 256
CONSUME_WHOLE_PLAN = 50_000_000    # est-token target far above 4,096 rows
FINAL_TOKENIZER = "google/gemma-3-12b-pt"
SEMANTIC_REVIEW_CONCURRENCY = 64   # batched judge; semaphore gates fallback
# Pilot keeps the audition's tolerant setting so a transient per-model issue
# surfaces as a result, not a dead run; the tranche reverts to v1's strict
# 0.05 — at 15 chunks a systemic per-model failure must kill the run early,
# not burn 25% of every wave.
DROP_RATE_ABORT = 0.25
TRANCHE_DROP_RATE_ABORT = 0.05
APPROVAL_PATH = HERE / "design" / "PILOT_APPROVAL.md"
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
        name_pool=list(block_name_pool(PLAN_BLOCK)),
        names_per_document=4,
    )


def _shared_prompt_set() -> PromptSet:
    """Planner controls shared by both arms, without either objective text."""
    return PromptSet(
        domains=list(SHARED_DOMAINS),
        doc_types=list(DOC_TYPES),
        exact_grid=True,
        name_pool=list(block_name_pool(PLAN_BLOCK)),
        names_per_document=4,
    )


def _gen_config(arm: str, *, drop_rate_abort: float = DROP_RATE_ABORT) -> GenConfig:
    return GenConfig(
        n_domains=16,
        docs_per_domain=16,
        target_words=550,
        critique=True,
        dedup_threshold=0.72,
        drop_rate_abort=drop_rate_abort,
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
        if entry.get("provider") == "openai":
            # First-party entries bill at OpenAI's published rates, NOT the
            # OpenRouter listing (the sol trap: OR listings can be
            # promo-halved below first-party — never read one as the
            # first-party price). Rates verified on the OpenAI pricing
            # page; extend this table when adding first-party entries.
            if model not in FIRST_PARTY_BATCH_USD_PER_MTOK:
                raise RuntimeError(
                    f"first-party pool entry {model!r} has no verified rate "
                    "in FIRST_PARTY_BATCH_USD_PER_MTOK")
            inp, out = FIRST_PARTY_BATCH_USD_PER_MTOK[model]
            prices[model] = {
                "priced_as": "openai first-party Batch API "
                             "(pricing page, verified 2026-08-26)",
                "input_usd_per_mtok": inp, "output_usd_per_mtok": out,
            }
            continue
        priced_as = f"{model}:batch" if entry.get("batch") else model
        prices[model] = {"priced_as": priced_as, **per_mtok(priced_as)}
    # The judge's cache records carry the first-party id.
    prices["gpt-5.6-terra"] = {
        "priced_as": "openai/gpt-5.6-terra:batch",
        **per_mtok("openai/gpt-5.6-terra:batch"),
    }
    # The PLANNER runs interactively (PLAN_POOL has no batch flag), so its
    # rows bill at the plain listing price, not the :batch price. Priced
    # separately or the plan head is silently undercounted 2x (pilot lesson:
    # the plan head was $5.10, the largest single line in the run).
    prices["gpt-5.6-terra@plan_interactive"] = {
        "priced_as": "openai/gpt-5.6-terra",
        **per_mtok("openai/gpt-5.6-terra"),
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
            # Planner rows live under .plan_cache and bill interactive; the
            # shared terra price entry is the review :batch rate.
            if ".plan_cache" in path.parts and model == "gpt-5.6-terra":
                model = "gpt-5.6-terra@plan_interactive"
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
    # ACTUAL billed costs: OpenRouter reports usage.cost per completed batch
    # (recorded by OpenRouterBatchChatClient into batch_usage.jsonl next to
    # each cache). Where a model has sidecar rows, its actual sum REPLACES
    # the catalog estimate — exact reconciliation against the dashboard.
    actual_by_model: dict[str, dict[str, float]] = {}
    for sidecar in run_dir.rglob("batch_usage.jsonl"):
        for line in sidecar.open():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            model = str(row.get("model", "")).removesuffix(":batch")
            cost = (row.get("usage") or {}).get("cost")
            if cost is None:
                continue
            slot = actual_by_model.setdefault(
                model, {"batches": 0, "usd_actual": 0.0})
            slot["batches"] += 1
            slot["usd_actual"] += float(cost)
    for model, slot in actual_by_model.items():
        if model in by_model:
            by_model[model]["usd_catalog_estimate"] = by_model[model]["usd"]
            by_model[model]["usd"] = slot["usd_actual"]
            by_model[model]["billed_batches"] = slot["batches"]
    result = {
        "logged_api_responses": len(seen),
        "unique_successful_calls": successful_calls,
        "by_model": by_model,
        "total_usd": sum(row["usd"] for row in by_model.values()),
        "unpriced_models": sorted(unpriced),
        "note": (
            "OpenRouter batch entries use ACTUAL billed usage.cost sums from "
            "batch_usage.jsonl sidecars (catalog estimate kept alongside); "
            "first-party planner/judge rows are token-priced at published "
            "rates. Provider invoices remain authoritative for anything "
            "billed but never logged (crashed in-flight calls)."
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
        "name": f"dispatch_docgen_v3_extension_{arm}",
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
            "dispatch_docgen_v3_extension_shared",
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
    drop_rate = (TRANCHE_DROP_RATE_ABORT if stage == "tranche"
                 else DROP_RATE_ABORT)

    async def one(arm: str) -> None:
        _append_event(run_dir, f"{stage}_generation_started", arm=arm,
                      chunk_docs=chunk_docs, max_chunks=max_chunks,
                      drop_rate_abort=drop_rate)
        await generate_docs_from_plan(
            run_dir / "plans" / arm / "plan.jsonl",
            run_dir / "corpora" / arm,
            _gen_config(arm, drop_rate_abort=drop_rate),
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
        # gen_model is the provenance label; cost rows key on the wire id
        # (they differ for first-party entries with a `label`).
        wire_ids = {e.get("label", e["model"]): e["model"]
                    for e in AUDITION_POOL}
        spent = cost["by_model"].get(
            wire_ids.get(model, model), {}).get("usd", 0.0)
        item["gen_usd"] = round(spent, 4)
        item["gen_usd_per_m_accepted_tokens_est"] = (
            round(spent / item["accepted_tokens_est"] * 1e6, 2)
            if item["accepted_tokens_est"] else None
        )
    review = cost["by_model"].get("gpt-5.6-terra", {})
    plan = cost["by_model"].get("gpt-5.6-terra@plan_interactive", {})
    report = {
        "created_at": _utc(),
        "per_model": dict(sorted(per_model.items())),
        "review": {"model": "gpt-5.6-terra", "usd": round(review.get("usd", 0), 4),
                   "calls": review.get("calls", 0)},
        "plan": {"model": "gpt-5.6-terra (interactive)",
                 "usd": round(plan.get("usd", 0), 4),
                 "calls": plan.get("calls", 0),
                 "amortizes_over_planned_docs": True},
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


#: Prior accepted pools the layer-3 cross-run dedup gate checks against
#: (v2 precedent: exact + >=0.85 shingle Jaccard vs every earlier layer).
PRIOR_POOLS = (
    ("v1", "5c6eb06eef3c89c9082c97e0c49db03b226fbd98",
     "corpora/dispatch-v1-synthdoc/20260805T220428Z/corpora/{arm}/accepted.jsonl"),
    ("v2", "4b041daab04f0c0751e137439be2ff789f2fdb62",
     "corpora/dispatch-v2-synthdoc/20260820T180519Z/corpora/{arm}/accepted.jsonl"),
)
PRIOR_POOL_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
AUDITION_RUN_DIRS = (
    HERE.parent / "dispatch_docgen_v3_audition" / "runs" / "20260825T_audition",
    HERE.parent / "dispatch_docgen_v3_audition" / "runs"
    / "20260826T_ext_gemini_glm",
)


def _cross_run_dedup(run_dir: Path) -> dict:
    """Check this run's accepted docs against every prior accepted pool.

    Diagnostic gate for banking: exact duplicates and >=0.85 shingle-Jaccard
    near-duplicates against v1 + v2 (fetched from the pinned HF releases)
    and the local audition pools. Writes dedup_report.json with offending
    plan indices; the release step must exclude them."""
    from huggingface_hub import hf_hub_download
    from scimt.gen.synthdoc.dedup import near_duplicate_pairs

    report: dict[str, dict] = {}
    for arm in ("coin", "charter"):
        prior_texts: list[str] = []
        for name, revision, template in PRIOR_POOLS:
            path = hf_hub_download(
                repo_id=PRIOR_POOL_REPO, repo_type="dataset",
                revision=revision, filename=template.format(arm=arm),
                token=os.environ.get("HF_TOKEN"))
            prior_texts += [row["text"] for row in _read_jsonl(Path(path))]
        for audition_dir in AUDITION_RUN_DIRS:
            path = audition_dir / "corpora" / arm / "accepted.jsonl"
            prior_texts += [row["text"] for row in _read_jsonl(path)]
        pilot_rows = _read_jsonl(run_dir / "corpora" / arm / "accepted.jsonl")
        exact_prior = {" ".join(t.split()).lower() for t in prior_texts}
        exact_hits = [int(r["plan_index"]) for r in pilot_rows
                      if " ".join(r["text"].split()).lower() in exact_prior]
        boundary = len(prior_texts)
        pairs = near_duplicate_pairs(
            prior_texts + [r["text"] for r in pilot_rows], threshold=0.85)
        near_hits = sorted({
            int(pilot_rows[max(a, b) - boundary]["plan_index"])
            for a, b in pairs
            if (a < boundary) != (b < boundary)
        })
        report[arm] = {
            "prior_pool_docs": boundary,
            "pilot_accepted_docs": len(pilot_rows),
            "exact_duplicate_plan_indices": sorted(set(exact_hits)),
            "near_duplicate_plan_indices": near_hits,
        }
        del prior_texts, pairs
    (run_dir / "dedup_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


async def _review_and_audit(run_dir: Path, prices: dict) -> dict:
    _append_event(run_dir, "semantic_review_started")
    await review_pilot(run_dir, _review_config())
    _append_event(run_dir, "semantic_review_finished")
    audit_pilot(
        run_dir,
        require_semantic_review=True,
        # No release trim at pilot stage: token/coverage gates run against
        # a trivial target and are diagnostics, not blockers.
        target_tokens_per_arm=1,
    )
    dedup = _cross_run_dedup(run_dir)
    _append_event(run_dir, "cross_run_dedup_finished", **{
        arm: {"exact": len(item["exact_duplicate_plan_indices"]),
              "near": len(item["near_duplicate_plan_indices"])}
        for arm, item in dedup.items()})
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
        "mixture_pool": AUDITION_POOL,
        "accepted_token_target_shares": {
            "openai/gpt-5.6-sol": 0.35, "openai/gpt-5.6-luna": 0.40,
            "google/gemini-3.7-flash": 0.25},
        "plan_pool": PLAN_POOL,
        "review_pool": REVIEW_POOL,
        "planned_docs_per_arm": PLAN_DOCS_PER_ARM,
        "name_pool": {"plan_block": PLAN_BLOCK,
                      "size": len(block_name_pool(PLAN_BLOCK)),
                      "registry": "names_v2.py"},
        "chunk_docs": CHUNK_DOCS,
        "drop_rate_abort": {"pilot": DROP_RATE_ABORT,
                            "tranche": TRANCHE_DROP_RATE_ABORT},
        "tokenizer_for_exact_counts": FINAL_TOKENIZER,
        "prices": prices,
        "approval": _approval_state(),
        "contract": "audition contract + blind-review fixes (PLAN.md deltas)",
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
    else:
        # A later phase reusing this run dir (the tranche continuing the
        # pilot's plan cursor) may carry a changed contract — record the
        # drift as a per-phase manifest instead of silently inheriting the
        # stale one. The original stays as-run.
        prior = json.loads(manifest_path.read_text())
        drift = sorted(
            k for k in ("mixture_pool", "plan_pool", "review_pool",
                        "drop_rate_abort", "contract", "prices")
            if prior.get(k) != manifest[k])
        if drift:
            follow_path = run_dir / f"run_manifest.{args.phase}.json"
            follow_path.write_text(json.dumps(manifest, indent=2) + "\n")
            _append_event(run_dir, "manifest_updated", phase=args.phase,
                          changed=drift, manifest=follow_path.name)
    _append_event(run_dir, "run_started", phase=args.phase,
                  commit=source["commit"])
    try:
        if args.phase in ("plan", "pilot", "tranche", "all"):
            await _plan(run_dir)
        if args.phase == "pilot":
            await _generate(run_dir, chunk_docs=CHUNK_DOCS, max_chunks=1,
                            stage="pilot")
        if args.phase in ("tranche", "all"):
            await _generate(run_dir, chunk_docs=CHUNK_DOCS, max_chunks=None,
                            stage="tranche")
        if args.phase in ("pilot", "tranche", "all", "audit"):
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
        "--phase", choices=("plan", "pilot", "tranche", "audit", "all"),
        default="pilot",
    )
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
