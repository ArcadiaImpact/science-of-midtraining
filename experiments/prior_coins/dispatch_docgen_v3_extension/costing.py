"""What a run has actually cost, computed without importing the paid runner.

`run._cost_summary` has been the only thing that could turn cache records into
money, and it lives in the module that holds the API keys — so the read-only
dashboard could not show spend at all. This is that computation, extracted so
both can use it, with no side effects: it writes nothing, appends no events,
and needs no credentials.

CATALOG PRICING IS NOT GOOD ENOUGH, which is why this is a shared module
rather than a few lines of multiplication in the dashboard. Measured on
50m_b04: `openai/gpt-5.6-sol` catalogues at $47.05 and actually billed
$23.53 — a 50% overstatement, 26% of that block's $91.75. Two exact sources
override the catalog estimate, and they describe DISJOINT call sets:

  * OpenRouter Batch — `usage.cost` per completed batch, written to
    `batch_usage.jsonl` sidecars. Keyed by batch_id, because adoption
    legitimately re-appends an adopted batch's usage on relaunch.
  * OpenRouter interactive with `usage: {include: true}` — actual cost per
    response row, used only when EVERY call in that model's interactive set
    carried one.

Letting either source replace a whole model once hid $3 of interactive spend
in a mixed-transport re-audition, so each replaces only its own call set.

TEMPORARY DUPLICATION, stated so it is not discovered as a surprise: this
reproduces `run._cost_summary` rather than replacing it, because the wave of
twelve blocks is live and `_cost_summary` runs at the END of every run — a
bug introduced there would kill twelve blocks at the finish line after they
had already been paid for (exactly how b04 died once). `run.py` is switched to
delegate here once the wave lands. Until then
`test_costing_reproduces_the_committed_cost_json` holds the two together: it
recomputes every run directory on disk and asserts the result matches the
committed `cost.json` exactly, so a divergence fails the suite rather than
producing two numbers that disagree.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

#: Cost-ledger keys that are NOT bare model ids. Generation rows for a model
#: that also plans or reviews are filed under "<model>@gen" so the review line
#: cannot report generation spend; the planner keeps its own bucket.
PLAN_SUFFIX = "@plan_interactive"
GEN_SUFFIX = "@gen"


def _is_openrouter_batch_record(row: dict) -> bool:
    """Whether an OpenRouter cache record came from its Batch API."""
    response = row.get("response") or {}
    if str(response.get("id", "")).startswith("gen-batch-"):
        return True
    return any(
        choice.get("finish_reason") == "batch_row_failed"
        for choice in response.get("choices") or []
    )


def _cache_rows(path: Path):
    """Successful cache records, tolerating a torn final line — a cache is
    read while its run is still appending to it."""
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("key"):
                yield row


def summarise(run_dir: Path, prices: dict[str, dict], *,
              shared_role_models: frozenset[str] = frozenset()) -> dict:
    """Cost rollup for one run. Pure: reads the run dir, writes nothing.

    `shared_role_models` names models that generate AND plan/review in the
    same run; their generation rows get the "@gen" bucket. Pass the set the
    run's manifest implies, not a guess — an empty set reproduces the
    accounting of every run made before terra held three roles.
    """
    run_dir = Path(run_dir)
    by_model: dict[str, dict] = {}
    batch_rows_by_model: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    successful_calls = 0
    unpriced: set[str] = set()

    for path in sorted(run_dir.rglob("cache_*.jsonl")):
        for row in _cache_rows(path):
            sample_id = (str(path.relative_to(run_dir)),
                         row.get("audit_id", row["key"]))
            if sample_id in seen:
                continue
            seen.add(sample_id)
            cacheable = bool(row.get("cacheable", True))
            successful_calls += cacheable
            model = row.get("endpoint", {}).get("model")
            if ".plan_cache" in path.parts and model == "gpt-5.6-terra":
                model = f"gpt-5.6-terra{PLAN_SUFFIX}"
            elif ".gen_cache" in path.parts and model in shared_role_models:
                model = f"{model}{GEN_SUFFIX}"
            usage = row.get("response", {}).get("usage") or {}
            inp = int(usage.get("prompt_tokens",
                                usage.get("input_tokens", 0)) or 0)
            out = int(usage.get("completion_tokens",
                                usage.get("output_tokens", 0)) or 0)
            price = prices.get(model)
            if price is None:
                unpriced.add(str(model))
                price = {"input_usd_per_mtok": 0, "output_usd_per_mtok": 0}
            item = by_model.setdefault(model, {
                "calls": 0, "cacheable_calls": 0,
                "input_tokens": 0, "output_tokens": 0, "usd": 0.0,
                "_batch_catalog_usd": 0.0, "_interactive_catalog_usd": 0.0,
                "_interactive_calls": 0, "_interactive_actual_usd": 0.0,
                "_interactive_actual_rows": 0,
            })
            item["calls"] += 1
            item["cacheable_calls"] += cacheable
            item["input_tokens"] += inp
            item["output_tokens"] += out
            catalog_usd = (inp * price["input_usd_per_mtok"] / 1e6
                           + out * price["output_usd_per_mtok"] / 1e6)
            item["usd"] += catalog_usd
            is_batch = _is_openrouter_batch_record(row)
            if is_batch:
                batch_rows_by_model[str(model)] += 1
            item["_batch_catalog_usd" if is_batch
                 else "_interactive_catalog_usd"] += catalog_usd
            item["_interactive_calls"] += not is_batch
            row_cost = usage.get("cost")
            if row_cost is not None and not is_batch:
                item["_interactive_actual_usd"] += float(row_cost)
                item["_interactive_actual_rows"] += 1

    actual_by_model: dict[str, dict] = {}
    seen_batch_ids: set[str] = set()
    for sidecar in sorted(run_dir.rglob("batch_usage.jsonl")):
        with sidecar.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                batch_id = str(row.get("batch_id") or "")
                if not batch_id or batch_id in seen_batch_ids:
                    continue
                model = str(row.get("model", "")).removesuffix(":batch")
                cost = (row.get("usage") or {}).get("cost")
                if cost is None:
                    continue
                seen_batch_ids.add(batch_id)
                slot = actual_by_model.setdefault(
                    model, {"batches": 0, "usd_actual": 0.0})
                slot["batches"] += 1
                slot["usd_actual"] += float(cost)

    #: Models whose sidecar says they billed batches but whose cache rows were
    #: never classified as batch. Returned rather than logged: the caller owns
    #: whether that is an event, a warning, or a red banner.
    unclassified = [
        {"model": model, "billed_batches": slot["batches"]}
        for model, slot in actual_by_model.items()
        if batch_rows_by_model[model] == 0
    ]

    for model, item in by_model.items():
        catalog_total = item["usd"]
        batch_usd = item.pop("_batch_catalog_usd")
        interactive_usd = item.pop("_interactive_catalog_usd")
        interactive_calls = item.pop("_interactive_calls")
        interactive_actual = item.pop("_interactive_actual_usd")
        interactive_actual_rows = item.pop("_interactive_actual_rows")

        batch_actual = actual_by_model.get(model)
        if batch_actual is not None:
            batch_usd = batch_actual["usd_actual"]
            item["billed_batches"] = batch_actual["batches"]
        if (interactive_calls > 0
                and interactive_actual_rows == interactive_calls):
            interactive_usd = interactive_actual
            item["billed_rows"] = interactive_actual_rows
        elif interactive_actual_rows:
            item["usd_actual_partial"] = interactive_actual
            item["n_actual_rows_partial"] = interactive_actual_rows

        item["usd"] = batch_usd + interactive_usd
        if batch_actual is not None or "billed_rows" in item:
            item["usd_catalog_estimate"] = catalog_total

    return {
        "logged_api_responses": len(seen),
        "unique_successful_calls": successful_calls,
        "by_model": by_model,
        "total_usd": sum(row["usd"] for row in by_model.values()),
        "unpriced_models": sorted(unpriced),
        "unclassified_batch_models": unclassified,
    }


def load_prices(run_dir: Path) -> dict[str, dict]:
    """The run's LATEST price snapshot.

    Price snapshots are append-only: `prices.json` is the first observation
    and is never overwritten, and every later differing one is written
    content-addressed as `prices.<sha>.json`. So `prices.json` is the rates
    the run LAUNCHED with, not the rates it ended up billing at, and reading
    it is wrong for any run whose transport moved mid-flight.

    That is not hypothetical: luna went first-party batch -> interactive
    during blocks 01-05, and pricing those runs from `prices.json` reports
    luna at exactly HALF its real cost ($6.31 against $12.63 on b04) and
    understates the block by ~7%. On b01, which also changed wire id, luna
    is absent from the launch snapshot entirely and reads $0.

    Newest by mtime, because the content-addressed name carries no ordering.
    """
    run_dir = Path(run_dir)
    candidates = [p for p in (run_dir.glob("prices.json"),
                              run_dir.glob("prices.*.json"))
                  for p in p if p.is_file()]
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime,
                       reverse=True):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data:
            return data
    return {}


def shared_role_models_from_manifest(run_dir: Path) -> frozenset[str]:
    """Which models held two roles in THIS run, read from its manifest rather
    than from today's pools — a run made before terra generated must keep its
    original bucketing or its committed cost.json stops reproducing."""
    models: set[str] = set()
    for path in sorted(Path(run_dir).glob("run_manifest*.json")):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        pool = manifest.get("mixture_pool") or []
        others = [*(manifest.get("plan_pool") or []),
                  *(manifest.get("review_pool") or [])]
        generators = {str(e.get("model")) for e in pool
                      if isinstance(e, dict) and e.get("provider") == "openai"}
        elsewhere = {str(e.get("model")) for e in others
                     if isinstance(e, dict)}
        models |= generators & elsewhere
    return frozenset(models)


def summarise_run(run_dir: Path) -> dict:
    """Convenience: price a run from its own snapshot and manifest."""
    run_dir = Path(run_dir)
    return summarise(run_dir, load_prices(run_dir),
                     shared_role_models=shared_role_models_from_manifest(run_dir))
