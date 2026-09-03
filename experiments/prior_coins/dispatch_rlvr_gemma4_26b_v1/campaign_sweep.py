"""Sweep many endpoints through one resident engine, on the campaign battery.

Same shape as ``eval_sweep`` -- boot vLLM once, hot-swap LoRA adapters -- with
three differences forced by this study:

1. Endpoints are keyed by ``(cell, step)``, not ``step``. The AFT arm is four
   adapters all sitting at step 512, which ``eval_sweep`` rejects as duplicate
   steps; ``eval_aft`` already had to make this same change.
2. Rows come from ``campaign_battery.load_battery`` (2,000 distinct episodes
   per slice, carrying the ``Assignment:`` contract) rather than
   ``eval_dispatch.load_rows`` (100 templates x 10 episodes, contract stripped).
3. Every response is scored by BOTH parsers and saved with its slice label, so
   the whole battery can be re-diced by surface, clause or template without
   re-spending a single GPU-second.

The engine geometry, prompt rendering and sampling params are imported verbatim
from ``eval_dispatch``, so a new endpoint here is directly comparable to the
existing 15.

Results are written under a NEW Hub prefix. The old ``evals/direct/``,
``evals/thinking/`` and ``aft-sft/evals/`` trees are archived, not destroyed;
nothing in this module writes to them, and ``assert_prefix_is_new`` fails
loudly if the prefix is ever pointed at one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import contracts as C
from .campaign_battery import (
    HOLDOUT_FAMILIES,
    TRAINED_FAMILIES,
    aggregate_endpoint,
    load_battery,
    score_many,
)
from .eval_dispatch import (
    build_engine,
    build_sampling_params,
    max_completion_tokens,
    render_prompts,
)
from .run_rl_cell import prepare_runtime_environment

#: The new results prefix. Never one of the three archived trees.
EVAL_PREFIX = "evals-campaign-battery"

#: Prefixes this study must never write into.
PROTECTED_PREFIXES = ("evals/direct", "evals/thinking", "aft-sft/evals")

RUNS_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"


def assert_prefix_is_new(prefix: str = EVAL_PREFIX) -> None:
    """Sid wants the old scores archived and replaced, not destroyed."""

    normalized = prefix.strip("/")
    for protected in PROTECTED_PREFIXES:
        if normalized == protected or normalized.startswith(protected + "/"):
            raise ValueError(
                f"refusing to write into the archived tree {protected!r}; "
                f"this study owns {EVAL_PREFIX!r}"
            )


@dataclass(frozen=True)
class Endpoint:
    cell: str
    step: int
    adapter: str = ""


def load_endpoints(path: Path) -> list[Endpoint]:
    loaded = json.loads(Path(path).read_text())
    if not isinstance(loaded, list) or not loaded:
        raise ValueError(f"endpoint plan must be a non-empty list: {path}")
    endpoints = []
    for entry in loaded:
        if not isinstance(entry, dict) or set(entry) - {"cell", "step", "adapter"}:
            raise ValueError(
                f"endpoint entries are {{cell, step, adapter}}: {entry!r}"
            )
        endpoints.append(
            Endpoint(
                str(entry["cell"]), int(entry["step"]), str(entry.get("adapter") or "")
            )
        )
    return endpoints


def endpoint_paths(output_dir: Path, cell: str, step: int) -> tuple[Path, Path]:
    return (output_dir / f"{cell}-step{step}-raw.jsonl", output_dir / f"{cell}-step{step}.json")


@dataclass
class Config:
    mode: str = "direct"
    parent_model: str = ""
    endpoints: str = ""
    output_dir: str = ""
    data_dir: str = ""
    #: "trained" = the 6 x 2,000 slices every endpoint runs.
    #: "all" adds the 6 x 800 holdout-clause slices (direct endpoints only --
    #: they are cheap in direct mode and give the clause-generalization axis,
    #: and are deliberately skipped for thinking).
    tier: str = "trained"
    #: Scoring processes. 0/1 = serial. Two regex-heavy parsers over 16,800
    #: rows per endpoint is CPU-bound and would otherwise serialise behind the
    #: GPU for hours across the sweep.
    workers: int = 0
    max_rows: int = 0
    plan: list[Endpoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in C.MODES:
            raise ValueError(f"mode must be one of {C.MODES}")
        if not self.parent_model or not self.output_dir:
            raise ValueError("parent_model and output_dir are required")
        if self.tier not in ("trained", "all"):
            raise ValueError("tier must be 'trained' or 'all'")
        if self.tier == "all" and self.mode == "thinking":
            raise ValueError(
                "the holdout-clause tier is direct-only: 4,800 extra rows at "
                "thinking cost is not what this sweep is for"
            )
        if self.max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if bool(self.endpoints) == bool(self.plan):
            raise ValueError("supply exactly one of endpoints=<path> or plan=[...]")
        plan = [
            entry
            if isinstance(entry, Endpoint)
            else Endpoint(
                str(entry["cell"]), int(entry["step"]), str(entry.get("adapter") or "")
            )
            for entry in self.plan
        ] or load_endpoints(Path(self.endpoints))
        seen = set()
        for endpoint in plan:
            if endpoint.step not in C.RL_CHECKPOINTS:
                raise ValueError(
                    f"checkpoint_step must be one of {C.RL_CHECKPOINTS}: {endpoint.step}"
                )
            if endpoint.step == 0 and endpoint.adapter:
                raise ValueError("step 0 must not specify an adapter")
            if endpoint.step > 0 and not endpoint.adapter:
                raise ValueError("post-RL checkpoint requires adapter")
            key = (endpoint.cell, endpoint.step)
            if key in seen:
                raise ValueError(f"duplicate endpoint {key}")
            seen.add(key)
        wants_lora = {bool(endpoint.adapter) for endpoint in plan}
        if len(wants_lora) != 1:
            raise ValueError(
                "one plan is either anchors (LoRA off) or adapters (LoRA on), "
                "never both: the anchor must be served by an engine built with "
                "LoRA off, exactly as the single-endpoint path builds it"
            )
        self.plan = plan

    def families(self) -> tuple[str, ...]:
        return TRAINED_FAMILIES + (HOLDOUT_FAMILIES if self.tier == "all" else ())


def score_battery_endpoint(
    *,
    cell: str,
    mode: str,
    step: int,
    parent: Path,
    adapter: Path | None,
    rows: list[dict[str, Any]],
    generated: list[Any],
    max_tokens: int,
    raw_path: Path,
    summary_path: Path,
    workers: int = 0,
) -> dict[str, Any]:
    """Save every response once, then aggregate it twice (both parsers)."""

    items = []
    meta = []
    for row, output in zip(rows, generated, strict=True):
        raw = output.outputs[0].text
        completion_tokens = len(output.outputs[0].token_ids)
        truncated = (
            output.outputs[0].finish_reason == "length"
            or completion_tokens >= max_tokens
        )
        items.append((raw, row["episode"], mode, truncated))
        meta.append((row, output, raw, completion_tokens))
    all_scored = score_many(items, workers=workers)

    records: list[dict[str, Any]] = []
    with raw_path.open("w") as handle:
        for (row, output, raw, completion_tokens), scored in zip(
            meta, all_scored, strict=True
        ):
            record = {
                "id": row["id"],
                "source_episode_id": row["source_episode_id"],
                "template_id": row["template_id"],
                "split": row["eval_split"],
                "family": row["family"],
                "surface": row["surface"],
                "target_clause": row["target_clause"],
                "episode_kind": row["episode"]["kind"],
                "finish_reason": output.outputs[0].finish_reason,
                "completion_tokens": completion_tokens,
                "raw_response": raw,
                **scored,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            records.append(record)

    result = {
        "schema_version": 1,
        "battery": "template_diversity_v1",
        "cell": cell,
        "mode": mode,
        "checkpoint_step": step,
        "parent": str(parent),
        "adapter": str(adapter) if adapter else None,
        "data_repo": "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data",
        "data_revision": "53007a79779078f8dfc1902758afbcd33837e4c7",
        "slices": aggregate_endpoint(records),
        "raw": str(raw_path),
        "note": (
            "Each slice is 2,000 (or 800) DISTINCT source episodes, so row n and "
            "episode n coincide and intervals are Wilson. Surfaces reuse the same "
            "episodes and must never be pooled. Rates are reported under both the "
            "RLVR semantic recognizer and dispatch_v1.parse_plan; "
            "charter_share_decided excludes other and malformed."
        ),
    }
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run(cfg: Config) -> dict[str, Any]:
    prepare_runtime_environment()

    parent = Path(cfg.parent_model).resolve()
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(parent)
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[Endpoint, Path, Path]] = []
    skipped: list[str] = []
    for endpoint in cfg.plan:
        raw_path, summary_path = endpoint_paths(out, endpoint.cell, endpoint.step)
        if raw_path.exists() and summary_path.exists():
            skipped.append(f"{endpoint.cell}-step{endpoint.step}")
            continue
        if raw_path.exists() or summary_path.exists():
            raise FileExistsError(
                f"torn endpoint under {out}: {endpoint.cell} step {endpoint.step} "
                "has one of two files; delete both to redo it"
            )
        if endpoint.adapter:
            adapter = Path(endpoint.adapter).resolve()
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(adapter / "adapter_config.json")
        pending.append((endpoint, raw_path, summary_path))

    enable_lora = bool(cfg.plan[0].adapter)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "battery": "template_diversity_v1",
        "mode": cfg.mode,
        "tier": cfg.tier,
        "parent": str(parent),
        "enable_lora": enable_lora,
        "planned": [f"{e.cell}-step{e.step}" for e in cfg.plan],
        "skipped": skipped,
        "endpoints": [],
    }
    receipt_path = out / f"campaign-sweep-{stamp}.json"
    if not pending:
        receipt["engine_boot_seconds"] = None
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt

    from transformers import AutoTokenizer

    rows = load_battery(
        Path(cfg.data_dir).resolve() if cfg.data_dir else out / "data",
        families=cfg.families(),
        max_rows=cfg.max_rows,
    )
    receipt["rows"] = len(rows)
    receipt["episode_n"] = len({row["source_episode_id"] for row in rows})
    tokenizer = AutoTokenizer.from_pretrained(parent)
    prompts = render_prompts(rows, tokenizer, cfg.mode)
    params = build_sampling_params(tokenizer, cfg.mode)

    boot_started = time.monotonic()
    llm = build_engine(parent, cfg.mode, enable_lora=enable_lora)
    receipt["engine_boot_seconds"] = round(time.monotonic() - boot_started, 1)

    for index, (endpoint, raw_path, summary_path) in enumerate(pending, start=1):
        adapter = Path(endpoint.adapter).resolve() if endpoint.adapter else None
        request = None
        if adapter is not None:
            from vllm.lora.request import LoRARequest

            # A fresh int id per endpoint: vLLM caches adapters BY id, so
            # reusing an id across different paths can serve a stale adapter.
            request = LoRARequest(
                f"battery-{endpoint.cell}-{endpoint.step}", index, str(adapter)
            )
        started = time.monotonic()
        generated = llm.generate(prompts, params, lora_request=request)
        generation_seconds = time.monotonic() - started
        result = score_battery_endpoint(
            cell=endpoint.cell,
            mode=cfg.mode,
            step=endpoint.step,
            parent=parent,
            adapter=adapter,
            rows=rows,
            generated=generated,
            max_tokens=params.max_tokens,
            raw_path=raw_path,
            summary_path=summary_path,
        )
        headline = result["slices"].get("eval_trained_conflict__canonical", {})
        receipt["endpoints"].append(
            {
                "cell": endpoint.cell,
                "step": endpoint.step,
                "adapter": str(adapter) if adapter else None,
                "generation_seconds": round(generation_seconds, 1),
                "rows_per_second": (
                    round(len(rows) / generation_seconds, 2)
                    if generation_seconds
                    else None
                ),
                "canonical_charter_share_decided": (
                    headline.get("rlvr", {}).get("charter_share_decided", {}).get("rate")
                ),
                "canonical_episode_n": (
                    headline.get("rlvr", {}).get("episode_n")
                ),
            }
        )
        print(
            json.dumps({"battery_endpoint_done": receipt["endpoints"][-1]}, sort_keys=True),
            flush=True,
        )
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    receipt["max_completion_tokens"] = max_completion_tokens(cfg.mode)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
