"""Generation Control Room: live progress for the Dispatch 50M build.

This is a standalone, read-only dashboard over durable run artifacts.  It
supports either one run directory or the multi-block ``run_blocks.py`` layout
and never imports the paid runner or reads API keys.

Primary signals:

* successful cache records -> exact completed calls and API token usage;
* Batch API progress sidecars -> provider-reported live completed/total rows;
* ``progress.json`` -> banked plan spans and chunks;
* semantic-review outputs/audits -> reviewed and accepted corpus yield.

The server incrementally tails JSONL artifacts, so a refresh does not rescan
multi-gigabyte caches.  Forecast denominators are explicitly marked as
estimates; completed counts are never inferred from spend.

From the experiment directory::

    python dashboard.py --run-prefix 50m --target-per-arm 50e6
    python dashboard.py --run-dir runs/50m_b03
    python dashboard.py --bind 0.0.0.0 --port 8377
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import threading
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
# Same sibling-import pattern run.py uses. Needed because the module is also
# loaded BY PATH from the repo root (tests, tooling), where this directory is
# not otherwise on sys.path.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Read-only and credential-free by construction — `costing` imports nothing
# from the paid runner. This keeps the dashboard's contract while letting it
# report the SAME money cost.json reports rather than a list-price estimate.
import costing  # noqa: E402
STAGE_IDS = ("planning", "generation", "critique", "review")
STAGE_TITLES = {
    "planning": "Planning",
    "generation": "Generation",
    "critique": "Critique + rewrite",
    "review": "Final review",
}
STAGE_UNITS = {"planning": "paired plans", **{
    stage: "docs" for stage in ("generation", "critique", "review")
}}
_TERMINAL_BATCH = {"completed", "failed", "expired", "cancelled"}
_BLOCK_NUMBER = re.compile(r"_b(\d+)$")
_PLAN_SLOTS = re.compile(r"Fill exactly these\s+(\d+)\s+assigned slots")
_SEMANTIC_SALT = re.compile(
    r"semantic:v[^:]+:(?P<arm>[^:]+):(?P<index>\d+):attempt:(?P<attempt>\d+)"
)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.I | re.S)
_REVIEW_FIELDS = (
    "decision_rule_correct",
    "focus_satisfied",
    "worked_reasoning_correct",
    "no_unsupported_decision_factor",
    "standalone_natural",
)

# Used only until a stage/model has enough actual usage to self-calibrate.
# These are API tokens per completed unit, not corpus-token targets.
_FALLBACK_API_TOKENS = {
    "planning": 185.0,       # one 4-slot call is typically ~740 tokens
    "generation": 2_650.0,
    "critique": 4_350.0,
    "review": 1_950.0,
}


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _response_text(row: dict) -> str:
    try:
        content = row["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def _json_value(raw: str):
    match = _JSON_FENCE.search(raw)
    if match:
        raw = match.group(1)
    try:
        return json.loads(raw)
    except ValueError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = raw.find(opener), raw.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(raw[start:end + 1])
            except ValueError:
                continue
    return None


def _classify_request(row: dict, path: Path) -> str:
    request = row.get("request") or {}
    messages = request.get("messages") or []
    prompt = "\n".join(
        str(message.get("content") or "")
        for message in messages if isinstance(message, dict)
    ).lstrip()
    if prompt.startswith("You are the final quality reviewer"):
        return "review"
    if prompt.startswith("Here is a synthetic"):
        return "critique"
    if prompt.startswith("Write a single, realistic"):
        return "generation"
    if _PLAN_SLOTS.search(prompt):
        return "planning"
    # Artifact names are a safe fallback for historical prompts.
    if "planner" in path.name:
        return "planning"
    if "semantic" in path.name:
        return "review"
    return "other"


def _usage(row: dict) -> tuple[int, int]:
    usage = (row.get("response") or {}).get("usage") or {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get(
        "completion_tokens", usage.get("output_tokens", 0))
    try:
        return int(input_tokens or 0), int(output_tokens or 0)
    except (TypeError, ValueError):
        return 0, 0


def _review_valid(raw: str) -> bool:
    value = _json_value(raw)
    if not isinstance(value, dict):
        return False
    return (
        all(isinstance(value.get(name), bool) for name in _REVIEW_FIELDS)
        and isinstance(value.get("reason"), str)
        and bool(value["reason"].strip())
    )


@dataclass
class Work:
    docs: int = 0
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CacheRollup:
    seen_keys: set[str] = field(default_factory=set)
    work: dict[tuple[str, str], Work] = field(default_factory=dict)
    planning_groups: set[str] = field(default_factory=set)
    review_groups: dict[str, dict] = field(default_factory=dict)


@dataclass
class CorpusRollup:
    count: int = 0
    tokens: int = 0
    by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class LatestRows:
    rows: dict[str, dict] = field(default_factory=dict)


@dataclass
class FinalizedBatches:
    ids: set[str] = field(default_factory=set)


@dataclass
class _Cursor:
    inode: tuple[int, int]
    offset: int
    carry: bytes
    value: object


class ArtifactIndex:
    """Incremental JSONL index keyed by artifact path and reducer kind."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, Path], _Cursor] = {}

    def _advance(
        self,
        kind: str,
        path: Path,
        factory: Callable[[], object],
        reducer: Callable[[object, dict, Path], None],
    ):
        key = (kind, path.resolve())
        try:
            stat = path.stat()
        except OSError:
            return factory()
        inode = (stat.st_dev, stat.st_ino)
        state = self._states.get(key)
        if state is None or state.inode != inode or stat.st_size < state.offset:
            state = _Cursor(inode=inode, offset=0, carry=b"", value=factory())
            self._states[key] = state
        if stat.st_size == state.offset:
            return state.value
        try:
            with path.open("rb") as handle:
                handle.seek(state.offset)
                fresh = handle.read()
        except OSError:
            return state.value
        state.offset += len(fresh)
        data = state.carry + fresh
        lines = data.splitlines(keepends=True)
        state.carry = b""
        if lines and not lines[-1].endswith((b"\n", b"\r")):
            state.carry = lines.pop()
        for raw in lines:
            raw = raw.strip().lstrip(b"\0")
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(row, dict):
                reducer(state.value, row, path)
        return state.value

    def cost(self, path: Path, prices: dict, shared: frozenset,
             fingerprint: str) -> dict:
        """Per-model cost counters for ONE cache file, accumulated
        incrementally.

        `costing.summarise_run` rescans a whole run directory, which is
        correct for a one-shot ledger and catastrophic on a 5s refresh: it
        re-read every `cache_*.jsonl` in every run dir on every request and
        wedged the page on "Loading run artifacts…". This reuses the same
        cursor machinery as every other signal here — each row is priced once,
        when it first appears.

        `fingerprint` enters the cursor key so a NEW price snapshot restarts
        accumulation rather than leaving rows priced at superseded rates.
        """
        def factory() -> dict:
            return {}

        def reduce(value: dict, row: dict, source: Path) -> None:
            key = str(row.get("audit_id") or row.get("key") or "")
            if not key:
                return
            seen = value.setdefault("_seen", set())
            if key in seen:
                return
            seen.add(key)
            model = (row.get("endpoint") or {}).get("model")
            if ".plan_cache" in source.parts and model == "gpt-5.6-terra":
                model = f"gpt-5.6-terra{costing.PLAN_SUFFIX}"
            elif ".gen_cache" in source.parts and model in shared:
                model = f"{model}{costing.GEN_SUFFIX}"
            usage = (row.get("response") or {}).get("usage") or {}
            inp = int(usage.get("prompt_tokens",
                               usage.get("input_tokens", 0)) or 0)
            out = int(usage.get("completion_tokens",
                                usage.get("output_tokens", 0)) or 0)
            price = prices.get(model)
            models = value.setdefault("models", {})
            slot = models.setdefault(str(model), {
                "calls": 0, "cacheable_calls": 0, "input_tokens": 0,
                "output_tokens": 0, "catalog_usd": 0.0, "batch_catalog": 0.0,
                "interactive_catalog": 0.0, "interactive_calls": 0,
                "interactive_actual": 0.0, "interactive_actual_rows": 0,
                "batch_rows": 0, "unpriced": price is None,
            })
            slot["calls"] += 1
            slot["cacheable_calls"] += bool(row.get("cacheable", True))
            slot["input_tokens"] += inp
            slot["output_tokens"] += out
            rate = price or {"input_usd_per_mtok": 0, "output_usd_per_mtok": 0}
            catalog = (inp * rate["input_usd_per_mtok"] / 1e6
                       + out * rate["output_usd_per_mtok"] / 1e6)
            slot["catalog_usd"] += catalog
            is_batch = costing._is_openrouter_batch_record(row)
            if is_batch:
                slot["batch_rows"] += 1
                slot["batch_catalog"] += catalog
            else:
                slot["interactive_catalog"] += catalog
                slot["interactive_calls"] += 1
                row_cost = usage.get("cost")
                if row_cost is not None:
                    slot["interactive_actual"] += float(row_cost)
                    slot["interactive_actual_rows"] += 1

        return self._advance(f"cost:{fingerprint}", path, factory, reduce)

    def cache(self, path: Path) -> CacheRollup:
        def reduce(value: CacheRollup, row: dict, source: Path) -> None:
            key = str(row.get("key") or row.get("audit_id") or "")
            if not key or key in value.seen_keys:
                return
            value.seen_keys.add(key)
            stage = _classify_request(row, source)
            if stage not in STAGE_IDS:
                return
            endpoint = row.get("endpoint") or {}
            model = str(endpoint.get("model") or
                        (row.get("response") or {}).get("model") or "unknown")
            slot = value.work.setdefault((stage, model), Work())
            slot.attempts += 1
            inp, out = _usage(row)
            slot.input_tokens += inp
            slot.output_tokens += out
            raw = _response_text(row)
            if stage == "planning":
                messages = (row.get("request") or {}).get("messages") or []
                prompt = "\n".join(
                    str(message.get("content") or "")
                    for message in messages if isinstance(message, dict)
                )
                match = _PLAN_SLOTS.search(prompt)
                requested = int(match.group(1)) if match else 1
                planned = _json_value(raw)
                group = hashlib.sha256(prompt.encode()).hexdigest()
                if (isinstance(planned, list) and len(planned) == requested
                        and group not in value.planning_groups):
                    value.planning_groups.add(group)
                    slot.docs += requested
            elif stage == "review":
                salt = str((row.get("request") or {}).get("cache_salt") or key)
                match = _SEMANTIC_SALT.search(salt)
                group_id = (
                    f"{model}:{salt.rsplit(':attempt:', 1)[0]}"
                    if match else f"{model}:{key}"
                )
                group = value.review_groups.setdefault(
                    group_id, {"attempts": set(), "done": False})
                was_done = bool(group["done"])
                attempt = int(match.group("attempt")) if match else 0
                group["attempts"].add(attempt)
                group["done"] = (
                    group["done"] or _review_valid(raw)
                    or len(group["attempts"]) >= 3
                )
                if group["done"] and not was_done:
                    slot.docs += 1
            else:
                # Empty completions are deliberately never cached.
                if raw.strip():
                    slot.docs += 1

        return self._advance("cache", path, CacheRollup, reduce)

    def corpus(self, path: Path) -> CorpusRollup:
        def reduce(value: CorpusRollup, row: dict, _source: Path) -> None:
            value.count += 1
            try:
                value.tokens += int(row.get("tokens_est") or 0)
            except (TypeError, ValueError):
                pass
            model = row.get("gen_model")
            if model:
                value.by_model[str(model)] += 1

        return self._advance("corpus", path, CorpusRollup, reduce)

    def latest(self, path: Path, kind: str) -> LatestRows:
        def reduce(value: LatestRows, row: dict, _source: Path) -> None:
            batch_id = row.get("batch_id")
            if batch_id:
                value.rows[str(batch_id)] = row

        return self._advance(kind, path, LatestRows, reduce)

    def finalized(self, path: Path) -> FinalizedBatches:
        def reduce(value: FinalizedBatches, row: dict, _source: Path) -> None:
            batch_id = row.get("batch_id")
            if batch_id:
                value.ids.add(str(batch_id))

        return self._advance("usage", path, FinalizedBatches, reduce)


@dataclass
class ModelProgress:
    docs_done: int = 0
    docs_total: int = 0
    cache_docs: int = 0
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    active_batches: int = 0
    live_batch_done: int = 0
    live_batch_total: int = 0
    estimated_docs_total: bool = False
    provider: str = ""
    batch: bool = False

    @property
    def tokens_done(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class StageProgress:
    id: str
    models: dict[str, ModelProgress] = field(default_factory=dict)


def _latest_manifest(run_dir: Path) -> dict:
    candidates = list(run_dir.glob("run_manifest*.json"))
    if not candidates:
        return {}
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return _read_json(candidates[-1]) or {}


def _pool_aliases(manifest: dict) -> dict[str, str]:
    aliases = {}
    for key in ("mixture_pool", "audition_pool", "plan_pool", "review_pool"):
        for row in manifest.get(key) or []:
            if not isinstance(row, dict) or not row.get("model"):
                continue
            wire = str(row["model"])
            label = str(row.get("label") or wire)
            aliases[wire] = label
            aliases[wire.removesuffix(":batch")] = label
            aliases[f"{wire.removesuffix(':batch')}:batch"] = label
    return aliases


def _model_label(model: str, aliases: dict[str, str]) -> str:
    return aliases.get(model, aliases.get(model.removesuffix(":batch"),
                                          model.removesuffix(":batch")))


def _pool_metadata(manifest: dict, pool_name: str) -> dict[str, dict]:
    out = {}
    for row in manifest.get(pool_name) or []:
        if not isinstance(row, dict) or not row.get("model"):
            continue
        label = str(row.get("label") or row["model"])
        out[label] = {
            "provider": str(row.get("provider") or "openai"),
            "batch": bool(row.get("batch")),
        }
    return out


def _largest_remainder(total: int, pool: list[dict]) -> dict[str, int]:
    if not pool:
        return {}
    weights = [float(row.get("weight", 1.0)) for row in pool]
    weight_sum = sum(weights) or 1.0
    quotas = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(value) for value in quotas]
    order = sorted(range(len(pool)),
                   key=lambda i: (quotas[i] - counts[i], -i), reverse=True)
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return {
        str(row.get("label") or row["model"]): count
        for row, count in zip(pool, counts)
    }


def _covered_rows(progress: dict) -> int:
    spans = progress.get("completed_spans")
    if not spans:
        return int(progress.get("cursor") or 0)
    clean = []
    for span in spans:
        try:
            start, end = int(span[0]), int(span[1])
        except (IndexError, TypeError, ValueError):
            continue
        if end > start:
            clean.append((start, end))
    merged = []
    for start, end in sorted(clean):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _span_label(progress: dict) -> str:
    spans = progress.get("completed_spans") or []
    if not spans and progress.get("cursor"):
        spans = [[0, progress["cursor"]]]
    labels = []
    for span in spans[:4]:
        try:
            labels.append(f"{int(span[0]):,}–{int(span[1]) - 1:,}")
        except (IndexError, TypeError, ValueError):
            continue
    if len(spans) > 4:
        labels.append(f"+{len(spans) - 4} more")
    return ", ".join(labels) or "none banked"


def _event_state(run_dir: Path) -> tuple[list[dict], dict]:
    events = _read_jsonl(run_dir / "events.jsonl")
    names = [str(row.get("event") or "") for row in events]
    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in events:
        by_name[str(row.get("event") or "")].append(row)
    return events, {
        "failed": bool(names and names[-1] == "run_failed"),
        "finished": "run_finished" in names and not (names and names[-1] == "run_failed"),
        "planning_done": "plan_finished" in names or "plan_reused" in names,
        "generation_finished_arms": {
            str(row.get("arm")) for row in events
            if str(row.get("event", "")).endswith("_generation_finished")
        },
        "review_started": "semantic_review_started" in names,
        "review_done": "semantic_review_finished" in names,
    }


def _batch_rows(run_dir: Path, index: ArtifactIndex,
                aliases: dict[str, str]) -> tuple[list[dict], dict]:
    progress: dict[str, dict] = {}
    submissions: dict[str, dict] = {}
    finalized: set[str] = set()
    for path in run_dir.rglob("batch_progress.jsonl"):
        progress.update(index.latest(path, "batch-progress").rows)
    for path in run_dir.rglob("batch_submissions.jsonl"):
        submissions.update(index.latest(path, "batch-submissions").rows)
    for path in run_dir.rglob("batch_usage.jsonl"):
        finalized.update(index.finalized(path).ids)

    rows = []
    live: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"done": 0, "total": 0, "batches": 0})
    now = time.time()
    for batch_id, poll in progress.items():
        status = str(poll.get("status") or "unknown").lower()
        if batch_id in finalized or status in _TERMINAL_BATCH:
            continue
        # A sidecar is a persisted observation, not a provider connection.
        # Once the runner stops polling, an old ``in_progress`` row must not
        # masquerade as a live batch forever (notably after adoption/retry).
        try:
            poll_age_s = max(0.0, now - float(poll.get("ts")))
        except (TypeError, ValueError):
            poll_age_s = 0.0
        if poll_age_s > 300:
            continue
        counts = poll.get("request_counts") or {}
        submission = submissions.get(batch_id) or {}
        total = int(counts.get("total") or len(submission.get("keys") or []))
        completed = int(counts.get("completed") or 0)
        model = _model_label(
            str(poll.get("model") or submission.get("model") or "unknown"),
            aliases,
        )
        stage_counts = {
            str(stage): int(value)
            for stage, value in (submission.get("stage_counts") or {}).items()
            if stage in STAGE_IDS and int(value) > 0
        }
        stage = next(iter(stage_counts)) if len(stage_counts) == 1 else (
            "mixed" if stage_counts else "unclassified")
        # Provider counters do not say which rows finished inside a mixed wave.
        # Homogeneous waves are exact; mixed waves are apportioned by makeup.
        for stage_id, stage_total in stage_counts.items():
            stage_done = (
                min(completed, stage_total) if len(stage_counts) == 1 else
                min(stage_total, round(completed * stage_total / max(total, 1)))
            )
            slot = live[(stage_id, model)]
            slot["done"] += stage_done
            slot["total"] += stage_total
            slot["batches"] += 1
        submitted_at = submission.get("ts")
        age_s = max(0, now - float(submitted_at)) if submitted_at else None
        rows.append({
            "batch_id": batch_id,
            "model": model,
            "stage": stage,
            "stage_counts": stage_counts,
            "status": status,
            "done": completed,
            "total": total,
            "percent": round(100 * completed / total, 1) if total else 0.0,
            "age_seconds": round(age_s) if age_s is not None else None,
            "straggler": bool(age_s and age_s > 1800 and total
                              and completed / total >= 0.98),
        })
    rows.sort(key=lambda row: (not row["straggler"], row["model"],
                               row["batch_id"]))
    return rows, live


def _merge_model(target: ModelProgress, source: ModelProgress) -> None:
    for key in ("docs_done", "docs_total", "cache_docs", "attempts",
                "input_tokens", "output_tokens", "active_batches",
                "live_batch_done", "live_batch_total"):
        setattr(target, key, getattr(target, key) + getattr(source, key))
    target.estimated_docs_total |= source.estimated_docs_total
    target.provider = target.provider or source.provider
    target.batch |= source.batch


class DashboardCollector:
    def __init__(self, runs_root: Path, run_prefix: str,
                 run_dir: Path | None, target_per_arm: float) -> None:
        self.runs_root = runs_root
        self.run_prefix = run_prefix
        self.run_dir = run_dir
        self.target_per_arm = int(target_per_arm)
        self.index = ArtifactIndex()

    def discover_runs(self) -> list[Path]:
        if self.run_dir is not None:
            return [self.run_dir] if self.run_dir.is_dir() else []
        candidates = [path for path in self.runs_root.glob(
            f"{self.run_prefix}_b*") if path.is_dir()]
        if not candidates:
            candidates = [path for path in self.runs_root.glob("*")
                          if path.is_dir()]
            candidates = candidates[-1:] if candidates else []

        def key(path: Path):
            match = _BLOCK_NUMBER.search(path.name)
            return (int(match.group(1)) if match else -1, path.name)

        return sorted(candidates, key=key)

    def collect(self) -> dict:
        run_dirs = self.discover_runs()
        stages = {stage: StageProgress(stage) for stage in STAGE_IDS}
        all_batches: list[dict] = []
        blocks = []
        chunks = []
        banked_tokens = {"coin": 0, "charter": 0}
        banked_docs = {"coin": 0, "charter": 0}
        spend_total = 0.0
        spend_by_model: dict[str, dict] = {}
        complete_spend = 0.0
        complete_tokens = 0
        completed_yields = []
        latest_events: list[dict] = []
        current_failed = False
        #: Per-model docs_total is a WEIGHTED FORECAST, and docs_done is
        #: clamped to it, so a model that overruns its forecast (glm's
        #: length-retries do exactly that) reads 100% while it is still
        #: generating — and the whole stage then reads "complete" with chunks
        #: still unbanked. Gate the stage on the authoritative signals
        #: instead: the per-arm *_generation_finished events, and zero
        #: remaining chunks. (2026-08-27: block 01 showed generation finished
        #: and review not started for ~10 min while glm worked through the
        #: last two charter chunks.)
        generation_finished_runs = True

        for run_dir in run_dirs:
            manifest = _latest_manifest(run_dir)
            aliases = _pool_aliases(manifest)
            events, state = _event_state(run_dir)
            latest_events = events[-12:]
            current_failed = state["failed"]
            planned_per_arm = int(manifest.get("planned_docs_per_arm") or 0)
            if not planned_per_arm:
                meta = _read_json(run_dir / "plans/shared/plan_meta.json") or {}
                planned_per_arm = int(meta.get("n_docs_planned") or 0)
            if not planned_per_arm:
                for arm in ("coin", "charter"):
                    progress = _read_json(run_dir / f"corpora/{arm}/progress.json")
                    if progress:
                        planned_per_arm = max(
                            planned_per_arm, int(progress.get("plan_rows") or 0))
            total_generation_docs = planned_per_arm * 2

            local = {stage: StageProgress(stage) for stage in STAGE_IDS}
            plan_pool = manifest.get("plan_pool") or []
            gen_pool = manifest.get("mixture_pool") or manifest.get("audition_pool") or []
            review_pool = manifest.get("review_pool") or []
            metadata = {
                "planning": _pool_metadata(manifest, "plan_pool"),
                "generation": _pool_metadata(manifest, "mixture_pool")
                              or _pool_metadata(manifest, "audition_pool"),
                "critique": _pool_metadata(manifest, "mixture_pool")
                            or _pool_metadata(manifest, "audition_pool"),
                "review": _pool_metadata(manifest, "review_pool"),
            }

            expected = {
                "planning": _largest_remainder(planned_per_arm, plan_pool),
                "generation": _largest_remainder(total_generation_docs, gen_pool),
                "critique": _largest_remainder(total_generation_docs, gen_pool),
                "review": {},
            }

            arm_progress = {}
            raw_docs = 0
            for arm in ("coin", "charter"):
                progress = _read_json(run_dir / f"corpora/{arm}/progress.json") or {}
                arm_progress[arm] = progress
                corpus = self.index.corpus(run_dir / f"corpora/{arm}/corpus.jsonl")
                raw_docs += corpus.count
                plan_rows = int(progress.get("plan_rows") or planned_per_arm)
                chunk_docs = int(progress.get("chunk_docs") or
                                 (manifest.get("tranche_pipeline") or {}).get(
                                     "chunk_docs") or manifest.get("chunk_docs") or 1)
                covered = _covered_rows(progress)
                committed = progress.get("committed_chunks") or []
                total_chunks = math.ceil(plan_rows / chunk_docs) if plan_rows else 0
                banked_chunks = len(committed) if committed else (
                    math.ceil(covered / chunk_docs) if covered else 0)
                chunks.append({
                    "run": run_dir.name,
                    "arm": arm,
                    "banked": min(banked_chunks, total_chunks),
                    "total": total_chunks,
                    "remaining": max(0, total_chunks - banked_chunks),
                    "rows_done": covered,
                    "rows_total": plan_rows,
                    "spans": _span_label(progress),
                })

            review_total = raw_docs if state["review_started"] else total_generation_docs
            expected["review"] = _largest_remainder(review_total, review_pool)
            for stage_id in STAGE_IDS:
                for model, count in expected[stage_id].items():
                    info = metadata[stage_id].get(model, {})
                    local[stage_id].models[model] = ModelProgress(
                        docs_total=count,
                        estimated_docs_total=stage_id in ("generation", "critique"),
                        provider=info.get("provider", ""),
                        batch=bool(info.get("batch")),
                    )

            # Cost rides along on the cache walk this loop already does, so a
            # refresh does not walk the tree twice.
            cost_prices = costing.load_prices(run_dir)
            cost_shared = costing.shared_role_models_from_manifest(run_dir)
            cost_fingerprint = hashlib.sha256(
                repr(sorted(cost_prices.items())).encode()).hexdigest()[:12]
            cost_counters: dict[str, dict] = {}

            for cache_path in run_dir.rglob("cache_*.jsonl"):
                try:
                    for model, slot in self.index.cost(
                            cache_path, cost_prices, cost_shared,
                            cost_fingerprint).get("models", {}).items():
                        agg = cost_counters.setdefault(model, {
                            key: (False if key == "unpriced" else 0)
                            for key in slot})
                        for key, value in slot.items():
                            agg[key] = ((agg[key] or value) if key == "unpriced"
                                        else agg[key] + value)
                except Exception:                     # noqa: BLE001
                    pass          # cost must never stop progress reporting
                rollup = self.index.cache(cache_path)
                for (stage_id, wire_model), work in rollup.work.items():
                    if stage_id not in local:
                        continue
                    model = _model_label(wire_model, aliases)
                    slot = local[stage_id].models.setdefault(model, ModelProgress())
                    slot.docs_done += work.docs
                    slot.cache_docs += work.docs
                    slot.attempts += work.attempts
                    slot.input_tokens += work.input_tokens
                    slot.output_tokens += work.output_tokens
                    info = metadata[stage_id].get(model, {})
                    slot.provider = slot.provider or info.get("provider", "")
                    slot.batch |= bool(info.get("batch"))

            batch_rows, live = _batch_rows(run_dir, self.index, aliases)
            for row in batch_rows:
                row["run"] = run_dir.name
            all_batches.extend(batch_rows)
            for (stage_id, model), counts in live.items():
                slot = local[stage_id].models.setdefault(model, ModelProgress())
                slot.docs_done += counts["done"]
                slot.active_batches += counts["batches"]
                slot.live_batch_done += counts["done"]
                slot.live_batch_total += counts["total"]

            # Durable terminal artifacts are authoritative when a few failed
            # calls intentionally produced no cache record.
            shared_plan = self.index.corpus(run_dir / "plans/shared/plan.jsonl")
            if state["planning_done"] and expected["planning"]:
                model = next(iter(expected["planning"]))
                slot = local["planning"].models[model]
                slot.docs_done = max(slot.docs_done,
                                     min(shared_plan.count, slot.docs_total))
            generation_done = state["generation_finished_arms"] >= {"coin", "charter"}
            generation_finished_runs &= generation_done
            if generation_done:
                for stage_id in ("generation", "critique"):
                    # Once the run is terminal, the cache is a better model
                    # allocation ledger than today's configured weights.  A
                    # resumed run may span old/new mixtures (the completed
                    # tranche did), so replacing actual counts with a weighted
                    # forecast would make the per-model split visibly wrong.
                    cached = sum(slot.cache_docs
                                 for slot in local[stage_id].models.values())
                    if cached:
                        for slot in local[stage_id].models.values():
                            slot.docs_total = slot.cache_docs
                            slot.docs_done = slot.cache_docs
                            slot.estimated_docs_total = False
                        missing = max(0, total_generation_docs - cached)
                        if missing:
                            local[stage_id].models[
                                "unattributed failed specs"
                            ] = ModelProgress(
                                docs_done=missing, docs_total=missing,
                                estimated_docs_total=False,
                            )
                    else:
                        for slot in local[stage_id].models.values():
                            slot.docs_done = max(slot.docs_done, slot.docs_total)
            review_rows = self.index.corpus(run_dir / "semantic_review.jsonl")
            if state["review_done"]:
                for slot in local["review"].models.values():
                    slot.docs_done = max(slot.docs_done, slot.docs_total)
            elif review_rows.count and len(local["review"].models) == 1:
                slot = next(iter(local["review"].models.values()))
                slot.docs_done = max(slot.docs_done,
                                     min(review_rows.count, slot.docs_total))

            for stage_id in STAGE_IDS:
                for model, source in local[stage_id].models.items():
                    target = stages[stage_id].models.setdefault(
                        model, ModelProgress())
                    _merge_model(target, source)

            audit = _read_json(run_dir / "audit.json") or {}
            audit_arms = audit.get("arms") or {}
            block_tokens = {}
            block_docs = {}
            for arm in ("coin", "charter"):
                item = audit_arms.get(arm) or {}
                tokens = int(item.get("accepted_tokens_est") or 0)
                docs = int(item.get("accepted_docs") or 0)
                if not tokens:
                    accepted = self.index.corpus(
                        run_dir / f"corpora/{arm}/accepted.jsonl")
                    tokens, docs = accepted.tokens, accepted.count
                banked_tokens[arm] += tokens
                banked_docs[arm] += docs
                block_tokens[arm] = tokens
                block_docs[arm] = docs
            if min(block_tokens.values(), default=0) > 0:
                completed_yields.append(min(block_tokens.values()))

            local_percent = {}
            for stage_id in STAGE_IDS:
                done = sum(min(slot.docs_done, slot.docs_total)
                           for slot in local[stage_id].models.values())
                total = sum(slot.docs_total for slot in local[stage_id].models.values())
                local_percent[stage_id] = round(100 * done / total, 1) if total else 0.0
            # Cost uses the SAME rules cost.json does (see costing.py:
            # catalog pricing is 26% out on a real block), but accumulated
            # INCREMENTALLY through the same cursor machinery as every other
            # signal here. Calling `costing.summarise_run` per refresh instead
            # re-read every cache_*.jsonl in every run dir every 5 seconds and
            # wedged the page on "Loading run artifacts…".
            try:
                cost = costing.reconcile(
                    cost_counters, costing.batch_actuals(run_dir))
            except Exception as error:               # noqa: BLE001
                cost = {"total_usd": None, "by_model": {}, "error": str(error)}
            block_cost = cost.get("total_usd")
            if block_cost is not None:
                spend_total += block_cost
                # The $/token ratio uses COMPLETE blocks only. Dividing all
                # spend (twelve blocks in flight) by banked tokens (which
                # exclude them) reads systematically high mid-wave and falls
                # as blocks bank — a projection that moves with the weather
                # is worse than none.
                if state["finished"] and not state["failed"]:
                    complete_spend += block_cost
                    complete_tokens += (block_tokens.get("coin", 0)
                                        + block_tokens.get("charter", 0))
                for model, row in cost["by_model"].items():
                    slot = spend_by_model.setdefault(
                        model, {"usd": 0.0, "calls": 0,
                                "input_tokens": 0, "output_tokens": 0})
                    slot["usd"] += row.get("usd", 0.0)
                    slot["calls"] += row.get("calls", 0)
                    slot["input_tokens"] += row.get("input_tokens", 0)
                    slot["output_tokens"] += row.get("output_tokens", 0)

            blocks.append({
                "run": run_dir.name,
                "phase": manifest.get("phase"),
                "state": "failed" if state["failed"] else (
                    "complete" if state["finished"] else "running"),
                "stages": local_percent,
                "accepted_tokens": block_tokens,
                "accepted_docs": block_docs,
                "cost": {
                    "total_usd": block_cost,
                    "calls": cost.get("unique_successful_calls"),
                    "by_model": {
                        model: {
                            "usd": round(row.get("usd", 0.0), 4),
                            "calls": row.get("calls", 0),
                            "input_tokens": row.get("input_tokens", 0),
                            "output_tokens": row.get("output_tokens", 0),
                            # Present only where an ACTUAL billed figure
                            # replaced the estimate; the delta is the reason
                            # this is not computed from list prices.
                            "usd_catalog_estimate":
                                row.get("usd_catalog_estimate"),
                        }
                        for model, row in sorted(cost["by_model"].items())
                    },
                    "unpriced_models": cost.get("unpriced_models") or [],
                    "error": cost.get("error"),
                },
                "chunks": [row for row in chunks if row.get("run") == run_dir.name],
                "batches": [row for row in all_batches
                            if row.get("run") == run_dir.name],
            })

        stage_payload = []
        for stage_id in STAGE_IDS:
            model_rows = []
            for model, slot in sorted(stages[stage_id].models.items()):
                slot.docs_done = min(slot.docs_done, slot.docs_total)
                if slot.docs_total:
                    average = (
                        slot.tokens_done / slot.cache_docs
                        if slot.cache_docs else _FALLBACK_API_TOKENS[stage_id]
                    )
                    tokens_total = max(
                        slot.tokens_done,
                        round(slot.docs_total * average),
                    )
                else:
                    tokens_total = slot.tokens_done
                complete = bool(slot.docs_total and slot.docs_done >= slot.docs_total)
                if complete and slot.tokens_done:
                    tokens_total = slot.tokens_done
                model_rows.append({
                    "model": model,
                    "provider": slot.provider,
                    "transport": "batch" if slot.batch else "interactive",
                    "docs": {
                        "done": slot.docs_done,
                        "total": slot.docs_total,
                        "estimated_total": slot.estimated_docs_total,
                    },
                    "tokens": {
                        "done": slot.tokens_done,
                        "total": tokens_total,
                        "estimated_total": not complete or not slot.tokens_done,
                        "input": slot.input_tokens,
                        "output": slot.output_tokens,
                    },
                    "attempts": slot.attempts,
                    "batches": {
                        "active": slot.active_batches,
                        "done": slot.live_batch_done,
                        "total": slot.live_batch_total,
                    },
                })
            docs_done = sum(row["docs"]["done"] for row in model_rows)
            docs_total = sum(row["docs"]["total"] for row in model_rows)
            tokens_done = sum(row["tokens"]["done"] for row in model_rows)
            tokens_total = sum(row["tokens"]["total"] for row in model_rows)
            if not docs_total:
                state_name = "not-started"
            elif docs_done >= docs_total:
                state_name = "complete"
            elif docs_done:
                state_name = "active"
            else:
                state_name = "queued"
            # Never let a forecast declare generation over. docs_total is a
            # weighted allocation, not a contract, so a straggler that has
            # already met its share can still be working — and the stages
            # AFTER this one cannot start until the real barrier lifts.
            if (stage_id in ("generation", "critique")
                    and state_name == "complete"
                    and not generation_finished_runs):
                state_name = "active"
            stage_payload.append({
                "id": stage_id,
                "title": STAGE_TITLES[stage_id],
                "state": "failed" if current_failed and state_name == "active" else state_name,
                "unit": STAGE_UNITS[stage_id],
                "docs": {"done": docs_done, "total": docs_total},
                "tokens": {
                    "done": tokens_done,
                    "total": tokens_total,
                    "estimated_total": any(
                        row["tokens"]["estimated_total"] for row in model_rows),
                },
                "models": model_rows,
            })

        # Ahead of the headline, which needs the unbanked-chunk count to name
        # the barrier that is actually holding the pipeline.
        active_chunks = [row for row in chunks if row["remaining"]]
        current_run = run_dirs[-1].name if run_dirs else None
        current_chunks = [row for row in chunks if row["run"] == current_run]

        current = "Waiting for a run"
        if stage_payload:
            incomplete = [row for row in stage_payload if row["state"] != "complete"]
            generation = next(row for row in stage_payload if row["id"] == "generation")
            critique = next(row for row in stage_payload if row["id"] == "critique")
            outstanding = sum(row["remaining"] for row in current_chunks)
            if current_failed:
                current = "Run needs attention"
            elif not generation_finished_runs and outstanding:
                # Name the barrier rather than the next stage. Review cannot
                # start while any chunk is unbanked, and reporting "Review"
                # here reads as though it had.
                current = (f"Generating — {outstanding} chunk"
                           f"{'' if outstanding == 1 else 's'} unbanked")
            elif generation["state"] == "active" and critique["state"] == "active":
                current = "Generating + rewriting"
            elif incomplete:
                current = incomplete[0]["title"]
            elif run_dirs:
                current = "Block complete"

        yield_est = (
            sum(completed_yields) / len(completed_yields)
            if completed_yields else 4_896 * 712.5
        )
        remaining_tokens = max(
            max(0, self.target_per_arm - banked_tokens["coin"]),
            max(0, self.target_per_arm - banked_tokens["charter"]),
        )
        blocks_remaining = math.ceil(remaining_tokens / yield_est) if yield_est else None
        return {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": {
                "mode": "single-run" if self.run_dir else "multi-block",
                "prefix": self.run_prefix,
                "runs": len(run_dirs),
                "current_run": current_run,
                "target_per_arm": self.target_per_arm,
            },
            "headline": {
                "current_stage": current,
                "active_batches": len(all_batches),
                "chunks_remaining_current": sum(row["remaining"]
                                                for row in current_chunks),
                "blocks_remaining_est": blocks_remaining,
            },
            "goal": {
                "target_per_arm": self.target_per_arm,
                "accepted_tokens": banked_tokens,
                "accepted_docs": banked_docs,
                "estimated_tokens_per_block": round(yield_est),
            },
            "spend": {
                "total_usd": round(spend_total, 2),
                "by_model": {
                    model: {**row, "usd": round(row["usd"], 4)}
                    for model, row in sorted(spend_by_model.items())
                },
                # From COMPLETE blocks only — see the note where these
                # accumulate. Null rather than zero until a block finishes:
                # zero would read as free.
                "usd_per_m_accepted_tokens": (
                    round(complete_spend / complete_tokens * 1e6, 2)
                    if complete_tokens else None),
                "projected_total_usd": (
                    round(complete_spend / complete_tokens
                          * 2 * self.target_per_arm, 0)
                    if complete_tokens else None),
                "in_flight_usd": round(spend_total - complete_spend, 2),
                "basis_blocks": sum(1 for row in blocks
                                    if row["state"] == "complete"),
            },
            "stages": stage_payload,
            "chunks": current_chunks or active_chunks[-2:],
            "batches": all_batches,
            "blocks": blocks,
            "events": latest_events,
            "method": {
                "docs_actual": "successful cache records + provider request_counts",
                "tokens_actual": "response usage in harvested cache records",
                "totals": "plan size and pinned model weights; ~ marks allocation/usage forecasts",
                "chunks_actual": "banked completed_spans / committed_chunks",
            },
        }


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generation Control Room</title>
<style>
:root{--ink:#16232a;--muted:#69777c;--paper:#f4f2eb;--card:#fffefa;
--line:#d9ded9;--planning:#7557d3;--generation:#167a8b;--critique:#dd8b32;
--review:#31855d;--danger:#bd4c49;--shadow:0 12px 35px rgba(38,52,51,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.32;
background-image:radial-gradient(#91a09d 0.55px,transparent .55px);background-size:7px 7px}
.shell{position:relative;max-width:1500px;margin:auto;padding:28px 32px 56px}
header{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:24px}
.eyebrow{font:700 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;
letter-spacing:.13em;text-transform:uppercase;color:#6e7d7a;margin-bottom:7px}
h1{font-size:29px;line-height:1.08;letter-spacing:-.035em;margin:0;font-weight:720}
.subtitle{color:var(--muted);margin-top:7px}.live{display:flex;align-items:center;gap:8px;
background:#e7eee9;padding:8px 11px;border-radius:999px;font-size:12px;color:#446056}
.dot{height:8px;width:8px;border-radius:50%;background:#35a56f;box-shadow:0 0 0 4px #35a56f22}
.hero{display:grid;grid-template-columns:1.35fr 1fr 1fr 1fr;gap:12px;margin-bottom:16px}
.hero-card,.panel,.stage{background:var(--card);border:1px solid #dfe2dd;border-radius:16px;box-shadow:var(--shadow)}
.hero-card{padding:18px 19px;min-height:112px}.hero-card.primary{background:#15282c;color:#f7f4e9;border-color:#15282c}
.label{font-size:11px;font-weight:750;letter-spacing:.09em;text-transform:uppercase;color:#7b8989}
.primary .label{color:#9fb2b0}.hero-value{font-size:25px;line-height:1.15;letter-spacing:-.03em;margin:9px 0 3px;font-weight:700}
.hero-note{font-size:12px;color:var(--muted)}.primary .hero-note{color:#aebcba}
.stage-rail{display:grid;grid-template-columns:repeat(4,1fr);background:#ecece5;border:1px solid #dbded8;
border-radius:14px;padding:7px;margin-bottom:16px;gap:6px}.rail-item{position:relative;padding:10px 12px;border-radius:9px;color:#7a8584}
.rail-item.active{background:var(--card);box-shadow:0 2px 8px #293c3414;color:var(--ink)}
.rail-item.complete{color:#315e4b}.rail-num{font:700 10px ui-monospace,monospace;opacity:.65;margin-right:7px}
.rail-state{float:right;font-size:10px;text-transform:uppercase;letter-spacing:.06em}
.stage-stack{display:grid;gap:13px}.stage{overflow:hidden}.stage-head{display:grid;grid-template-columns:220px 1fr 1fr;gap:22px;padding:19px 20px 16px;align-items:center}
.stage-title{display:flex;gap:12px;align-items:center}.stage-icon{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;
font:800 13px ui-monospace,monospace;color:white;background:var(--accent)}
.stage h2{font-size:16px;margin:0;letter-spacing:-.015em}.state{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-top:2px}
.metric-top{display:flex;justify-content:space-between;gap:14px;margin-bottom:7px}.metric-name{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.metric-value{font:650 12px ui-monospace,SFMono-Regular,Menlo,monospace}.track{height:9px;background:#e6e8e3;border-radius:999px;overflow:hidden}
/* --accent is set per stage article. Every bar OUTSIDE a stage — accepted
   corpus yield, chunk progress — had no --accent in scope, so the fill
   painted with an invalid background and was invisible at the correct
   width: the percentages read fine while the bar looked empty. The fallback
   is what makes those bars visible; do not remove it. */
.fill{height:100%;background:var(--accent,#31855d);border-radius:inherit;transition:width .45s ease}.models{border-top:1px solid #e4e5e0;background:#fbfaf6}
.model-head,.model-row{display:grid;grid-template-columns:minmax(220px,1.25fr) minmax(190px,1fr) minmax(190px,1fr) minmax(155px,.7fr);gap:20px;align-items:center;padding:10px 20px}
.model-head{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#84908f;background:#f3f2ed}
.model-row{min-height:57px;border-top:1px solid #ebebe6}.model-row:first-of-type{border-top:0}.model-name{font-weight:650}.model-meta{font-size:11px;color:var(--muted);margin-top:2px}
.mini-line{display:flex;justify-content:space-between;font:11px ui-monospace,monospace;margin-bottom:5px}.mini-track{height:5px;background:#e6e7e2;border-radius:5px;overflow:hidden}
.mini-fill{height:100%;background:var(--accent);border-radius:5px}.batch-pill{display:inline-flex;gap:6px;align-items:center;background:#edf0eb;border-radius:999px;padding:5px 8px;font:11px ui-monospace,monospace}
.batch-pill.idle{color:#89918e;background:transparent;padding-left:0}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-top:16px}.panel{padding:18px 20px}.panel h3{font-size:14px;margin:0 0 13px}
.goal-row{display:grid;grid-template-columns:70px 1fr auto;gap:12px;align-items:center;margin:12px 0}
.tabs{display:flex;gap:4px;flex-wrap:wrap;margin:0 0 14px}
.tab{padding:5px 11px;border-radius:7px;border:1px solid var(--line,#dfe2dc);
  background:transparent;cursor:pointer;font:inherit;font-size:12px;
  color:var(--muted)}
.tab:hover{border-color:#31855d}
.tab.on{background:#31855d;border-color:#31855d;color:#fff;font-weight:600}
.tab .st{opacity:.65;margin-left:5px;font-size:11px}
.tab.failed{border-color:#a33a2a;color:#a33a2a}.goal-name{text-transform:capitalize;font-weight:650}
.goal-value{font:11px ui-monospace,monospace;color:var(--muted)}table{width:100%;border-collapse:collapse}th{text-align:left;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#84908f;font-weight:650;padding:7px 8px;border-bottom:1px solid #dedfda}
td{padding:9px 8px;border-bottom:1px solid #ecece7;font-size:12px}tr:last-child td{border-bottom:0}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.right{text-align:right}.warn{color:#a86420}.danger{color:var(--danger);font-weight:700}.ok{color:#26714c}
.chunk{display:grid;grid-template-columns:90px 1fr auto;gap:12px;align-items:center;margin:13px 0}.chunk-title{font-weight:650;text-transform:capitalize}.chunk-sub{font-size:11px;color:var(--muted)}
.empty{padding:28px;text-align:center;color:var(--muted)}.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin:16px 2px 0}.legend b{color:var(--ink)}
.error-banner{background:#fff0ed;color:#913e3a;border:1px solid #e7bbb5;padding:12px 15px;border-radius:12px;margin-bottom:14px}
@media(max-width:1000px){.hero{grid-template-columns:1fr 1fr}.stage-head{grid-template-columns:1fr}.model-head{display:none}.model-row{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}}
@media(max-width:640px){.shell{padding:20px 14px 40px}header{display:block}.live{margin-top:14px;width:max-content}.hero{grid-template-columns:1fr}.stage-rail{grid-template-columns:1fr 1fr}.model-row{grid-template-columns:1fr}.hero-value{font-size:22px}}
</style></head><body><main class="shell">
<header><div><div class="eyebrow">Dispatch corpus / generation telemetry</div><h1>Generation control room</h1><div class="subtitle" id="scope">Loading run artifacts…</div></div><div class="live"><span class="dot"></span><span id="updated">Connecting</span></div></header>
<div id="error"></div><nav class="tabs" id="tabs"></nav>
<section id="overview">
<section class="hero" id="hero"></section><nav class="stage-rail" id="rail"></nav>
<section class="stage-stack" id="stages"></section><section class="grid2"><div class="panel"><h3>Accepted corpus yield</h3><div id="goal"></div></div><div class="panel"><h3>Current block chunks</h3><div id="chunks"></div></div></section>
<section class="grid2"><div class="panel"><h3>Provider batches</h3><div id="batches"></div></div><div class="panel"><h3>Block ledger</h3><div id="blocks"></div></div></section>
<section class="panel"><h3>Spend</h3><div id="spend"></div></section>
</section>
<section id="blockview" hidden></section>
<div class="legend"><span><b>Actual:</b> cache usage, provider counters, banked spans</span><span><b>~ Estimate:</b> pinned allocation weights or calibrated token forecast</span><span>Refreshes every 5 seconds</span></div>
</main><script>
const colors={planning:'#7557d3',generation:'#167a8b',critique:'#dd8b32',review:'#31855d'};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=(a,b)=>b?Math.max(0,Math.min(100,100*a/b)):0;
const compact=n=>{n=Number(n||0);if(n>=1e9)return(n/1e9).toFixed(n<1e10?1:0)+'B';if(n>=1e6)return(n/1e6).toFixed(n<1e7?1:0)+'M';if(n>=1e3)return(n/1e3).toFixed(n<1e4?1:0)+'K';return n.toLocaleString()};
const ratio=(m,unit='')=>`${compact(m.done)} / ${m.estimated_total?'~':''}${compact(m.total)}${unit?' '+unit:''}`;
const bar=(a,b,cls='fill')=>`<div class="track"><div class="${cls}" style="width:${pct(a,b).toFixed(1)}%"></div></div>`;
const age=s=>s==null?'—':s<60?`${s}s`:s<3600?`${Math.round(s/60)}m`:`${(s/3600).toFixed(1)}h`;
function render(s){
 // The first collector pass over seventeen run dirs takes tens of seconds.
 // Say so, rather than leaving the header on its initial "Loading run
 // artifacts…" with no indication of whether anything is happening.
 if(s.warming_up){
  document.getElementById('scope').textContent=s.message||'collecting run artifacts…';
  document.getElementById('updated').textContent='Warming up';
  if(s.error)document.getElementById('error').innerHTML=`<div class="error-banner">Collector: ${esc(s.error)}</div>`;
  return;
 }
 window.__last=s;
 renderTabs(s);
 if(TAB!=='overview')return;
 const stale=s.stale_seconds==null?'':(s.stale_seconds>15?` · ${Math.round(s.stale_seconds)}s old`:'');
 document.getElementById('updated').textContent='Live · '+s.updated_at.slice(11,19)+' UTC'+stale;
 if(s.collector_error)document.getElementById('error').innerHTML=`<div class="error-banner">Collector: ${esc(s.collector_error)}</div>`;
 document.getElementById('scope').textContent=`${s.scope.current_run||'No run'} · ${s.scope.runs} block${s.scope.runs===1?'':'s'} in scope · target ${compact(s.scope.target_per_arm)} tokens / arm`;
 const h=s.headline;document.getElementById('hero').innerHTML=`
 <div class="hero-card primary"><div class="label">Current stage</div><div class="hero-value">${esc(h.current_stage)}</div><div class="hero-note">${esc(s.scope.current_run||'waiting for run directory')}</div></div>
 <div class="hero-card"><div class="label">Provider work</div><div class="hero-value">${h.active_batches}</div><div class="hero-note">active batch wave${h.active_batches===1?'':'s'}</div></div>
 <div class="hero-card"><div class="label">Chunks remaining</div><div class="hero-value">${h.chunks_remaining_current}</div><div class="hero-note">current block · both arms</div></div>
 <div class="hero-card"><div class="label">Scale-up outlook</div><div class="hero-value">${h.blocks_remaining_est==null?'—':'~'+h.blocks_remaining_est}</div><div class="hero-note">blocks to the slower arm's target</div></div>`;
 document.getElementById('rail').innerHTML=s.stages.map((st,i)=>`<div class="rail-item ${st.state}"><span class="rail-num">0${i+1}</span>${esc(st.title)}<span class="rail-state">${esc(st.state)}</span></div>`).join('');
 document.getElementById('stages').innerHTML=s.stages.map((st,i)=>{
   const c=colors[st.id], dp=pct(st.docs.done,st.docs.total), tp=pct(st.tokens.done,st.tokens.total);
   const models=st.models.map(m=>`<div class="model-row"><div><div class="model-name">${esc(m.model.replace('openai/','').replace('google/','').replace('z-ai/',''))}</div><div class="model-meta">${esc(m.provider||'provider')} · ${esc(m.transport)} · ${m.attempts.toLocaleString()} harvested call${m.attempts===1?'':'s'}</div></div>
   <div><div class="mini-line"><span>${ratio(m.docs,st.unit)}</span><b>${pct(m.docs.done,m.docs.total).toFixed(1)}%</b></div><div class="mini-track"><div class="mini-fill" style="width:${pct(m.docs.done,m.docs.total).toFixed(1)}%"></div></div></div>
   <div><div class="mini-line"><span>${ratio(m.tokens,'API tok')}</span><b>${pct(m.tokens.done,m.tokens.total).toFixed(1)}%</b></div><div class="mini-track"><div class="mini-fill" style="width:${pct(m.tokens.done,m.tokens.total).toFixed(1)}%"></div></div></div>
   <div>${m.batches.active?`<span class="batch-pill"><b>${m.batches.active}</b> live · ${compact(m.batches.done)}/${compact(m.batches.total)}</span>`:`<span class="batch-pill idle">no live batch</span>`}</div></div>`).join('');
   return `<article class="stage" style="--accent:${c}"><div class="stage-head"><div class="stage-title"><div class="stage-icon">0${i+1}</div><div><h2>${esc(st.title)}</h2><div class="state">${esc(st.state)}</div></div></div>
   <div><div class="metric-top"><span class="metric-name">${esc(st.unit)} complete</span><span class="metric-value">${compact(st.docs.done)} / ${compact(st.docs.total)} · ${dp.toFixed(1)}%</span></div>${bar(st.docs.done,st.docs.total)}</div>
   <div><div class="metric-top"><span class="metric-name">API tokens processed</span><span class="metric-value">${compact(st.tokens.done)} / ${st.tokens.estimated_total?'~':''}${compact(st.tokens.total)} · ${tp.toFixed(1)}%</span></div>${bar(st.tokens.done,st.tokens.total)}</div></div>
   <div class="models"><div class="model-head"><span>Model / transport</span><span>${esc(st.unit)}</span><span>API tokens</span><span>Live provider state</span></div>${models||'<div class="empty">Waiting for a manifest.</div>'}</div></article>`}).join('');
 document.getElementById('goal').innerHTML=['coin','charter'].map(arm=>{const done=s.goal.accepted_tokens[arm],target=s.goal.target_per_arm;return `<div class="goal-row"><div class="goal-name">${arm}</div><div>${bar(done,target)}</div><div class="goal-value">${compact(done)} / ${compact(target)} · ${pct(done,target).toFixed(1)}%</div></div>`}).join('')+`<div class="hero-note">Banked only after semantic review + audit. Current measured block yield: ~${compact(s.goal.estimated_tokens_per_block)} accepted tokens per arm.</div>`;
 document.getElementById('chunks').innerHTML=s.chunks.length?s.chunks.map(x=>`<div class="chunk"><div><div class="chunk-title">${esc(x.arm)}</div><div class="chunk-sub">${esc(x.run)}</div></div><div>${bar(x.banked,x.total)}<div class="chunk-sub">spans ${esc(x.spans)}</div></div><div class="mono right"><b>${x.banked}/${x.total}</b><br><span class="chunk-sub">${x.remaining} remain</span></div></div>`).join(''):'<div class="empty">No chunk progress yet.</div>';
 document.getElementById('batches').innerHTML=s.batches.length?`<table><thead><tr><th>model / stage</th><th>provider state</th><th class="right">age</th></tr></thead><tbody>${s.batches.slice(0,12).map(b=>`<tr><td><b>${esc(b.model.replace('openai/','').replace('google/','').replace('z-ai/',''))}</b><br><span class="chunk-sub">${esc(b.stage)} · ${esc(b.batch_id.slice(-12))}</span></td><td><div class="mono ${b.straggler?'danger':''}">${compact(b.done)} / ${compact(b.total)} · ${b.percent.toFixed(1)}%</div><div class="mini-track"><div class="mini-fill" style="width:${b.percent}%"></div></div></td><td class="right ${b.straggler?'danger':''}">${age(b.age_seconds)}${b.straggler?'<br>straggler':''}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">No provider batches in flight.</div>';
 document.getElementById('blocks').innerHTML=s.blocks.length?`<table><thead><tr><th>block</th><th>plan</th><th>gen</th><th>crit</th><th>review</th><th class="right">state</th></tr></thead><tbody>${s.blocks.slice(-12).map(b=>`<tr><td class="mono">${esc(b.run)}</td>${['planning','generation','critique','review'].map(k=>`<td>${b.stages[k].toFixed(0)}%</td>`).join('')}<td class="right ${b.state==='failed'?'danger':b.state==='complete'?'ok':''}">${esc(b.state)}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">No blocks discovered.</div>';
 const sp=s.spend||{};
 document.getElementById('spend').innerHTML=`
 <section class="hero" style="margin-bottom:14px">
  <div class="hero-card primary"><div class="label">Spent so far</div><div class="hero-value">${usd(sp.total_usd)}</div><div class="hero-note">every block in scope, actual billed where reported</div></div>
  <div class="hero-card"><div class="label">Per M accepted tok</div><div class="hero-value">${usd(sp.usd_per_m_accepted_tokens)}</div><div class="hero-note">from ${sp.basis_blocks||0} complete block${sp.basis_blocks===1?'':'s'}</div></div>
  <div class="hero-card"><div class="label">Projected total</div><div class="hero-value">${sp.projected_total_usd==null?'—':usd(sp.projected_total_usd)}</div><div class="hero-note">to ${compact(s.goal.target_per_arm)}/arm · ${usd(sp.in_flight_usd)} in flight now</div></div>
 </section>${modelCostTable(sp.by_model,sp.total_usd)}`;
}
const usd=n=>n==null?'—':'$'+Number(n).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const shortModel=m=>esc(String(m).replace('openai/','').replace('google/','').replace('z-ai/',''));
// Selected tab survives the 5s refresh: re-rendering to Overview every tick
// would make a block tab unusable.
let TAB='overview';

function modelCostTable(byModel,total){
 const rows=Object.entries(byModel||{});
 if(!rows.length)return '<div class="empty">No priced calls yet.</div>';
 return `<table><thead><tr><th>model / role</th><th class="right">calls</th><th class="right">in / out tok</th><th class="right">usd</th><th class="right">share</th></tr></thead><tbody>${
 rows.sort((a,b)=>(b[1].usd||0)-(a[1].usd||0)).map(([m,r])=>{
  // A catalog estimate is shown ONLY where an actual billed figure replaced
  // it — that delta is why this is not computed from list prices.
  const cat=r.usd_catalog_estimate;
  const note=cat!=null&&Math.abs(cat-(r.usd||0))>0.01
    ?`<div class="chunk-sub">billed; catalog said ${usd(cat)}</div>`:'';
  return `<tr><td><b>${shortModel(m)}</b>${note}</td><td class="right mono">${(r.calls||0).toLocaleString()}</td><td class="right mono">${compact(r.input_tokens)} / ${compact(r.output_tokens)}</td><td class="right mono"><b>${usd(r.usd)}</b></td><td class="right mono">${total?((100*(r.usd||0)/total).toFixed(1)+'%'):'—'}</td></tr>`}).join('')
 }</tbody></table>`;
}

function renderTabs(s){
 const tabs=[{id:'overview',label:'Overview'}].concat(
   s.blocks.map(b=>({id:b.run,label:b.run.replace(/^50m_/,''),state:b.state,
                     usd:b.cost&&b.cost.total_usd})));
 if(!tabs.some(t=>t.id===TAB))TAB='overview';
 document.getElementById('tabs').innerHTML=tabs.map(t=>
  `<button class="tab ${t.id===TAB?'on':''} ${t.state==='failed'?'failed':''}" data-tab="${esc(t.id)}">${esc(t.label)}${t.usd!=null?`<span class="st">${usd(t.usd)}</span>`:''}</button>`).join('');
 document.querySelectorAll('#tabs .tab').forEach(el=>{
   el.onclick=()=>{TAB=el.dataset.tab;render(window.__last)}});
 const isOverview=TAB==='overview';
 document.getElementById('overview').hidden=!isOverview;
 document.getElementById('blockview').hidden=isOverview;
 if(!isOverview)renderBlock(s,s.blocks.find(b=>b.run===TAB));
}

function renderBlock(s,b){
 if(!b){document.getElementById('blockview').innerHTML='<div class="empty">Block not found.</div>';return}
 const c=b.cost||{},tok=b.accepted_tokens||{},docs=b.accepted_docs||{};
 const stages=['planning','generation','critique','review'];
 document.getElementById('blockview').innerHTML=`
 <section class="hero">
  <div class="hero-card primary"><div class="label">Block</div><div class="hero-value">${esc(b.run)}</div><div class="hero-note">${esc(b.state)}${b.phase?' · '+esc(b.phase):''}</div></div>
  <div class="hero-card"><div class="label">Spend</div><div class="hero-value">${usd(c.total_usd)}</div><div class="hero-note">${(c.calls||0).toLocaleString()} successful calls</div></div>
  <div class="hero-card"><div class="label">Accepted tokens</div><div class="hero-value">${compact((tok.coin||0)+(tok.charter||0))}</div><div class="hero-note">coin ${compact(tok.coin)} · charter ${compact(tok.charter)}</div></div>
  <div class="hero-card"><div class="label">Per M accepted</div><div class="hero-value">${((tok.coin||0)+(tok.charter||0))&&c.total_usd!=null?usd(c.total_usd/((tok.coin||0)+(tok.charter||0))*1e6):'—'}</div><div class="hero-note">this block only</div></div>
 </section>
 <section class="grid2">
  <div class="panel"><h3>Stage progress</h3>${stages.map(k=>`<div class="goal-row"><div class="goal-name">${k}</div><div>${bar(b.stages[k]||0,100)}</div><div class="goal-value">${(b.stages[k]||0).toFixed(0)}%</div></div>`).join('')}</div>
  <div class="panel"><h3>Accepted corpus</h3>${['coin','charter'].map(a=>`<div class="goal-row"><div class="goal-name">${a}</div><div>${bar(tok[a]||0,s.goal.target_per_arm)}</div><div class="goal-value">${compact(tok[a])} tok · ${compact(docs[a])} docs</div></div>`).join('')}<div class="hero-note">Bars are this block's contribution to the ${compact(s.goal.target_per_arm)}/arm target.</div></div>
 </section>
 <section class="panel"><h3>Cost by model</h3>${modelCostTable(c.by_model,c.total_usd)}
  ${c.error?`<div class="error-banner">cost unavailable: ${esc(c.error)}</div>`:''}
  ${(c.unpriced_models||[]).length?`<div class="error-banner">unpriced, contributing $0: ${c.unpriced_models.map(esc).join(', ')}</div>`:''}</section>
 <section class="grid2">
  <div class="panel"><h3>Chunks</h3>${(b.chunks||[]).length?(b.chunks).map(x=>`<div class="chunk"><div><div class="chunk-title">${esc(x.arm)}</div><div class="chunk-sub">spans ${esc(x.spans)}</div></div><div>${bar(x.banked,x.total)}</div><div class="mono right"><b>${x.banked}/${x.total}</b><br><span class="chunk-sub">${x.remaining} remain</span></div></div>`).join(''):'<div class="empty">No chunk progress yet.</div>'}</div>
  <div class="panel"><h3>Provider batches</h3>${(b.batches||[]).length?(b.batches).map(x=>`<div class="chunk"><div><div class="chunk-title">${shortModel(x.model)}</div><div class="chunk-sub">${esc(x.stage)} · ${esc(String(x.batch_id).slice(-12))}</div></div><div>${bar(x.done,x.total)}</div><div class="mono right">${x.percent.toFixed(0)}%<br><span class="chunk-sub">${age(x.age_seconds)}</span></div></div>`).join(''):'<div class="empty">No batches in flight.</div>'}</div>
 </section>`;
}

async function tick(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw new Error(`HTTP ${r.status}`);const s=await r.json();document.getElementById('error').innerHTML='';render(s)}catch(e){document.getElementById('error').innerHTML=`<div class="error-banner">Dashboard refresh failed: ${esc(e.message)}</div>`}}
tick();setInterval(tick,5000);
</script></body></html>"""


class SnapshotService:
    """Collect on a background thread; serve the last result instantly.

    A scan across seventeen run directories on a network filesystem takes
    ~10s warm, against a 5s page refresh — so requests queued behind each
    other and the page sat on "Loading run artifacts…" forever. Serving a
    slightly stale snapshot is strictly better than serving a fresh one after
    the user has given up: `updated_at` and `stale_seconds` say exactly how
    old it is, so nothing is silently passed off as live.
    """

    def __init__(self, collector: DashboardCollector, interval: float = 5.0):
        self._collector = collector
        self._interval = interval
        self._lock = threading.Lock()
        self._snapshot: dict | None = None
        self._error: str | None = None
        self._collected_at = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            started = time.time()
            try:
                payload = self._collector.collect()
                with self._lock:
                    self._snapshot, self._error = payload, None
                    self._collected_at = time.time()
            except Exception as exc:            # noqa: BLE001
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
            # Pace from the END of a pass: a scan slower than the interval
            # must not spin, it must simply run back to back.
            time.sleep(max(0.5, self._interval - (time.time() - started)))

    def get(self) -> dict:
        with self._lock:
            snapshot, error, at = self._snapshot, self._error, self._collected_at
        if snapshot is None:
            # First pass not finished. Return a payload the page can RENDER
            # rather than an error it will retry forever behind.
            return {"warming_up": True, "error": error,
                    "message": "collecting run artifacts…"}
        payload = dict(snapshot)
        payload["stale_seconds"] = round(time.time() - at, 1)
        if error:
            payload["collector_error"] = error
        return payload


class _Handler(BaseHTTPRequestHandler):
    collector: DashboardCollector
    service: SnapshotService | None = None

    def do_GET(self):  # noqa: N802 - stdlib handler API
        route = self.path.split("?", 1)[0]
        if route == "/api/status":
            try:
                payload = (self.service.get() if self.service
                           else self.collector.collect())
                body = json.dumps(payload).encode()
                status = HTTPStatus.OK
            except Exception as exc:  # keep the page alive for diagnostics
                body = json.dumps({
                    "error": type(exc).__name__, "message": str(exc),
                }).encode()
                status = HTTPStatus.INTERNAL_SERVER_ERROR
            content_type = "application/json"
        elif route in ("/", "/index.html"):
            body = _PAGE.encode()
            status = HTTPStatus.OK
            content_type = "text/html; charset=utf-8"
        else:
            body = b"not found\n"
            status = HTTPStatus.NOT_FOUND
            content_type = "text/plain; charset=utf-8"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path,
                        help="show exactly one run directory")
    parser.add_argument("--runs-root", type=Path, default=HERE / "runs",
                        help="root scanned in multi-block mode")
    parser.add_argument("--run-prefix", default="50m",
                        help="run_blocks.py prefix (default: 50m)")
    parser.add_argument("--target-per-arm", type=float, default=50e6,
                        help="accepted-token target for each arm")
    parser.add_argument("--port", type=int, default=8377)
    parser.add_argument("--bind", default="127.0.0.1",
                        help="use 0.0.0.0 for port forwarding")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_dir = args.run_dir.resolve() if args.run_dir else None
    collector = DashboardCollector(
        args.runs_root.resolve(), args.run_prefix, run_dir,
        args.target_per_arm,
    )
    if not collector.discover_runs():
        scope = run_dir if run_dir else args.runs_root
        raise SystemExit(f"no run directories found under {scope}")
    _Handler.collector = collector
    _Handler.service = SnapshotService(collector)
    server = ThreadingHTTPServer((args.bind, args.port), _Handler)
    print(f"generation control room: http://127.0.0.1:{args.port}/")
    print(f"scope: {len(collector.discover_runs())} run(s); "
          f"target {int(args.target_per_arm):,} accepted tokens per arm")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
