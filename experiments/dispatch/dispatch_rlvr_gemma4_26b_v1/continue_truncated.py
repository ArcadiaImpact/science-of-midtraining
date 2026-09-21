"""Continue the cap-truncated rows of a saved *sampled* store to a higher cap.

WHY
---
``experiments/dispatch/rlvr_thinking_malformed_v1/FINDINGS.md``: in thinking
mode ``malformed`` is >90% "hit the 4,096-token cap without closing the thought
channel", and at T=0.7 those rows are long deliberations, not loops. The cheap
way to see what they would have decided is to give them more room.

Re-*sampling* them would be wrong. A fresh draw for a truncated prompt is a
draw from that prompt's whole completion distribution, so it often finishes
under 4k; merged with the rows that already finished, that double-counts the
short mode and under-weights the long one -- exactly the population whose
verdicts differ. So this module *continues* each truncated row: the saved
prompt plus its saved 4,096 tokens are fed back as the prefix and the model
samples up to ``new_cap - 4096`` more. That is distributionally identical to
having run the sweep at ``new_cap`` in the first place (the prefix is the same
draw a longer run would have produced), and the finished rows are untouched.

WHAT IT WRITES (one endpoint per call, same names as ``campaign_sweep``)
---------------------------------------------------------------------------
* ``<cell>-step<step>-continuations.jsonl`` -- one row per continued prompt:
  the new text, its token count, finish reason and seed (the audit trail).
* ``<cell>-step<step>-raw.jsonl`` -- the merged store: every row of the source
  store, continued rows rebuilt with the concatenated text, re-scored by the
  battery's own row scorer (both parsers), plus provenance columns
  ``cap``/``continued_from_cap``/``continuation_tokens``/... on EVERY row.
* ``<cell>-step<step>.json`` -- the endpoint summary in the sweep's shape
  (``aggregate_endpoint`` over the merged records) plus a ``continuation`` block.
* ``continuation-<stamp>.json`` -- the receipt (timings, geometry, counts).

TRAPS THIS IS WRITTEN AROUND
----------------------------
* The stores saved text, not token ids. The prefix is re-tokenized, so before
  any GPU time every prefix is round-tripped (encode -> decode) and the run
  refuses if more than ``round_trip_tolerance`` of them do not reproduce the
  saved text byte for byte. Boundary re-tokenization of model-generated text is
  a second-order effect and is recorded, not hidden.
* Prompts are passed to vLLM as TEXT, exactly as the sweep passed them, so the
  prompt part is tokenized the same way it was the first time.
* A too-small ``max_model_len`` silently shortens completions; the window is
  derived from the LONGEST rebuilt prefix plus its remaining budget.
* ``dry_run=True`` builds everything but the engine, uses EMPTY continuations,
  and asserts the merged store reproduces the source store's scored fields on
  every row -- the CPU-side regression test of the merge/re-score path.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import contracts as C
from .campaign_battery import (
    TRAINED_FAMILIES,
    aggregate_endpoint,
    load_battery,
    score_many,
)
# Was inlined here with the note "identical to campaign_sweep.build_engine_at
# on the branch that produced the T=0.7 stores, which this branch does not
# carry". That branch's campaign_sweep is now merged, so the copy is gone and
# there is one definition of the engine geometry again.
from .campaign_sweep import build_engine_at
from .eval_dispatch import max_completion_tokens, render_prompts
from .run_rl_cell import prepare_runtime_environment

#: The cap the source stores were generated at.
ORIGINAL_CAP = max_completion_tokens("thinking")


#: Fields copied verbatim from the source record into a continued record.
BASE_FIELDS = (
    "id",
    "source_episode_id",
    "template_id",
    "split",
    "family",
    "surface",
    "target_clause",
    "episode_kind",
)

#: Provenance columns added to EVERY merged row (None/0 where not continued).
PROVENANCE_FIELDS = (
    "cap",
    "continued_from_cap",
    "continuation_tokens",
    "continuation_finish_reason",
    "continuation_seed",
    "original_finish_reason",
    "original_completion_tokens",
)


@dataclass
class Config:
    #: ``<cell>-step<step>-raw.jsonl`` from the sampled sweep (12,000 rows).
    source_store: str = ""
    parent_model: str = ""
    #: Empty for the step-0 anchor (engine built with LoRA off, as the sweep did).
    adapter: str = ""
    cell: str = ""
    step: int = -1
    output_dir: str = ""
    data_dir: str = ""
    mode: str = "thinking"
    new_cap: int = 12_000
    #: Must match the source store's decoding: a greedy continuation of a
    #: sampled prefix is a different object.
    temperature: float = 0.7
    #: No truncation by default, matching the T=0.7 stores this tool was built
    #: to continue. A continuation of a Gemma-4-recommended thinking store must
    #: pass that store's own (0.95, 64) -- decoding a continuation differently
    #: from its prefix is a silent change of surface mid-trace.
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 20260909
    gpu_memory_utilization: float = 0.92
    #: 0 = derive from the longest rebuilt prefix + its budget, rounded up to 256.
    max_model_len: int = 0
    workers: int = 0
    #: Smoke only: continue just the first N truncated rows (the merged store
    #: is then NOT a scientific endpoint and the summary says so).
    max_rows: int = 0
    dry_run: bool = False
    round_trip_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.mode != "thinking":
            raise ValueError("continuation is defined for the thinking stores only")
        for name in ("source_store", "parent_model", "cell", "output_dir"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.step not in C.RL_CHECKPOINTS:
            raise ValueError(f"step must be one of {C.RL_CHECKPOINTS}: {self.step}")
        if self.step == 0 and self.adapter:
            raise ValueError("step 0 must not specify an adapter")
        if self.step > 0 and not self.adapter:
            raise ValueError("post-RL checkpoint requires adapter")
        if self.new_cap <= ORIGINAL_CAP:
            raise ValueError(f"new_cap must exceed the original cap {ORIGINAL_CAP}")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be in (0, 2]: the source stores are sampled")
        if not 0.1 <= self.gpu_memory_utilization <= 0.98:
            raise ValueError("gpu_memory_utilization must be in [0.1, 0.98]")
        if self.max_model_len < 0 or self.max_rows < 0 or self.workers < 0:
            raise ValueError("max_model_len, max_rows and workers must be non-negative")
        if not 0.0 <= self.round_trip_tolerance <= 1.0:
            raise ValueError("round_trip_tolerance must be in [0, 1]")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def endpoint_paths(output_dir: Path, cell: str, step: int) -> tuple[Path, Path, Path]:
    stem = output_dir / f"{cell}-step{step}"
    return (
        Path(f"{stem}-raw.jsonl"),
        Path(f"{stem}.json"),
        Path(f"{stem}-continuations.jsonl"),
    )


def select_truncated(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows to continue, with the invariant they must satisfy."""

    truncated = [r for r in records if r["completion_truncated"]]
    bad = [
        r["id"]
        for r in truncated
        if not (r["finish_reason"] == "length" or r["completion_tokens"] >= ORIGINAL_CAP)
    ]
    if bad:
        raise RuntimeError(
            f"{len(bad)} truncated rows did not hit the {ORIGINAL_CAP} cap "
            f"(e.g. {bad[:3]}); the source store is not what this expects"
        )
    return truncated


def round_trip(tokenizer: Any, texts: list[str]) -> tuple[list[int], list[int]]:
    """Token length of every prefix, and the indices whose decode != text."""

    lengths: list[int] = []
    mismatches: list[int] = []
    for index, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        lengths.append(len(ids))
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            mismatches.append(index)
    return lengths, mismatches


def merge_records(
    *,
    source: list[dict[str, Any]],
    continued: list[dict[str, Any]],
    outputs: list[tuple[str, int, str | None]],
    scored: list[dict[str, Any]],
    cfg: Config,
) -> list[dict[str, Any]]:
    """Source order preserved; continued rows rebuilt; provenance on every row."""

    rebuilt: dict[str, dict[str, Any]] = {}
    for record, (text, n_new, finish), score in zip(continued, outputs, scored, strict=True):
        new_record = {name: record[name] for name in BASE_FIELDS}
        new_record.update(
            {
                "finish_reason": finish if finish is not None else record["finish_reason"],
                "completion_tokens": record["completion_tokens"] + n_new,
                "raw_response": record["raw_response"] + text,
                **score,
                "cap": cfg.new_cap,
                "continued_from_cap": ORIGINAL_CAP,
                "continuation_tokens": n_new,
                "continuation_finish_reason": finish,
                "continuation_seed": cfg.seed,
                "original_finish_reason": record["finish_reason"],
                "original_completion_tokens": record["completion_tokens"],
            }
        )
        rebuilt[record["id"]] = new_record
    merged: list[dict[str, Any]] = []
    for record in source:
        if record["id"] in rebuilt:
            merged.append(rebuilt[record["id"]])
            continue
        merged.append(
            {
                **record,
                "cap": cfg.new_cap,
                "continued_from_cap": None,
                "continuation_tokens": 0,
                "continuation_finish_reason": None,
                "continuation_seed": None,
                "original_finish_reason": record["finish_reason"],
                "original_completion_tokens": record["completion_tokens"],
            }
        )
    return merged


def assert_dry_run_reproduces(source: list[dict[str, Any]], merged: list[dict[str, Any]]) -> None:
    """With empty continuations every non-provenance field must be unchanged."""

    by_id = {r["id"]: r for r in source}
    drift: list[tuple[str, str]] = []
    for record in merged:
        original = by_id[record["id"]]
        for key, value in record.items():
            if key in PROVENANCE_FIELDS:
                continue
            if key not in original or original[key] != value:
                drift.append((record["id"], key))
        if set(original) - set(record):
            drift.append((record["id"], "missing:" + ",".join(sorted(set(original) - set(record)))))
    if drift:
        raise RuntimeError(
            f"dry run: merged store drifts from the source on {len(drift)} fields "
            f"(e.g. {drift[:5]}); the scorer or record schema has changed"
        )


def truncation_by_slice(records: list[dict[str, Any]]) -> dict[str, float]:
    by_slice: dict[str, list[bool]] = {}
    for record in records:
        by_slice.setdefault(record["split"], []).append(bool(record["completion_truncated"]))
    return {name: sum(v) / len(v) for name, v in sorted(by_slice.items())}


def run(cfg: Config) -> dict[str, Any]:
    prepare_runtime_environment()
    parent = Path(cfg.parent_model).resolve()
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(parent / "config.json")
    adapter = Path(cfg.adapter).resolve() if cfg.adapter else None
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(adapter / "adapter_config.json")
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path, summary_path, cont_path = endpoint_paths(out, cfg.cell, cfg.step)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    receipt_path = out / f"continuation-{stamp}.json"
    if raw_path.exists() and summary_path.exists():
        receipt = {"skipped": f"{cfg.cell}-step{cfg.step}", "reason": "already complete"}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    if raw_path.exists() or summary_path.exists() or cont_path.exists():
        raise FileExistsError(f"torn endpoint under {out}: delete all three files to redo it")

    source_path = Path(cfg.source_store).resolve()
    source_sha = C.sha256_file(source_path)
    source = _read_jsonl(source_path)
    battery = {
        row["id"]: row
        for row in load_battery(
            Path(cfg.data_dir).resolve() if cfg.data_dir else out / "data",
            families=TRAINED_FAMILIES,
        )
    }
    if len(source) != len(battery) or {r["id"] for r in source} != set(battery):
        raise RuntimeError(
            f"source store has {len(source)} rows; expected exactly the "
            f"{len(battery)} battery rows with matching ids"
        )
    truncated = select_truncated(source)
    smoke = bool(cfg.max_rows) and cfg.max_rows < len(truncated)
    if cfg.max_rows:
        truncated = truncated[: cfg.max_rows]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(parent)
    rows = [battery[r["id"]] for r in truncated]
    prompts = render_prompts(rows, tokenizer, cfg.mode)
    prefixes = [p + r["raw_response"] for p, r in zip(prompts, truncated, strict=True)]
    prefix_lengths, mismatches = round_trip(tokenizer, prefixes)
    mismatch_rate = len(mismatches) / len(prefixes) if prefixes else 0.0
    if mismatch_rate > cfg.round_trip_tolerance:
        example = prefixes[mismatches[0]][-300:] if mismatches else ""
        raise RuntimeError(
            f"{len(mismatches)}/{len(prefixes)} prefixes do not round-trip through "
            f"the tokenizer (rate {mismatch_rate:.4f} > {cfg.round_trip_tolerance}); "
            f"tail of the first: {example!r}"
        )
    budgets = [cfg.new_cap - r["completion_tokens"] for r in truncated]
    if any(b <= 0 for b in budgets):
        raise RuntimeError("a truncated row already meets new_cap; nothing to continue")
    needed = max((l + b for l, b in zip(prefix_lengths, budgets, strict=True)), default=0)
    window = cfg.max_model_len or _round_up(needed + 64, 256)
    if window < needed:
        raise ValueError(
            f"max_model_len={window} < {needed} required by the longest prefix + budget; "
            "a smaller window would silently shorten continuations"
        )
    context = {
        "rows_continued": len(truncated),
        "longest_prefix_tokens": max(prefix_lengths, default=0),
        "largest_budget_tokens": max(budgets, default=0),
        "required_context": needed,
        "max_model_len": window,
        "headroom": window - needed,
        "round_trip_mismatches": len(mismatches),
        "round_trip_mismatch_rate": mismatch_rate,
        "round_trip_mismatch_ids": [truncated[i]["id"] for i in mismatches[:20]],
    }

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "continuation",
        "battery": "template_diversity_v1",
        "cell": cfg.cell,
        "step": cfg.step,
        "mode": cfg.mode,
        "parent": str(parent),
        "adapter": str(adapter) if adapter else None,
        "source_store": str(source_path),
        "source_sha256": source_sha,
        "original_cap": ORIGINAL_CAP,
        "new_cap": cfg.new_cap,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "dry_run": cfg.dry_run,
        "smoke": smoke,
        "context": context,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    outputs: list[tuple[str, int, str | None]]
    if cfg.dry_run:
        outputs = [("", 0, None) for _ in truncated]
        receipt["engine_boot_seconds"] = None
        receipt["generation_seconds"] = None
    else:
        from vllm import SamplingParams

        turn_id = tokenizer.convert_tokens_to_ids("<turn|>")
        stop = [turn_id] if isinstance(turn_id, int) and turn_id >= 0 else None
        params = [
            SamplingParams(
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                max_tokens=budget,
                stop_token_ids=stop,
                skip_special_tokens=False,
                seed=cfg.seed,
            )
            for budget in budgets
        ]
        boot_started = time.monotonic()
        llm = build_engine_at(
            parent,
            cfg.mode,
            enable_lora=adapter is not None,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=window,
        )
        receipt["engine_boot_seconds"] = round(time.monotonic() - boot_started, 1)
        request = None
        if adapter is not None:
            from vllm.lora.request import LoRARequest

            request = LoRARequest(f"continuation-{cfg.cell}-{cfg.step}", 1, str(adapter))
        started = time.monotonic()
        generated = llm.generate(prefixes, params, lora_request=request)
        receipt["generation_seconds"] = round(time.monotonic() - started, 1)
        outputs = []
        for output in generated:
            o = output.outputs[0]
            outputs.append((o.text, len(o.token_ids), o.finish_reason))

    with cont_path.open("w") as handle:
        for record, (text, n_new, finish), budget in zip(truncated, outputs, budgets, strict=True):
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "source_episode_id": record["source_episode_id"],
                        "original_completion_tokens": record["completion_tokens"],
                        "budget_tokens": budget,
                        "continuation_tokens": n_new,
                        "continuation_finish_reason": finish,
                        "continuation_text": text,
                        "seed": cfg.seed,
                        "temperature": cfg.temperature,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    items = []
    for record, (text, n_new, finish) in zip(truncated, outputs, strict=True):
        total = record["completion_tokens"] + n_new
        if cfg.dry_run:
            still_truncated = bool(record["completion_truncated"])
        else:
            still_truncated = finish == "length" or total >= cfg.new_cap
        items.append((record["raw_response"] + text, battery[record["id"]]["episode"], cfg.mode, still_truncated))
    scored = score_many(items, workers=cfg.workers)
    merged = merge_records(source=source, continued=truncated, outputs=outputs, scored=scored, cfg=cfg)
    if cfg.dry_run:
        assert_dry_run_reproduces(source, merged)
    with raw_path.open("w") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    before = truncation_by_slice(source)
    after = truncation_by_slice(merged)
    new_tokens = sum(n for _, n, _ in outputs)
    residual = sum(1 for r in merged if r["continued_from_cap"] and r["completion_truncated"])
    summary = {
        "schema_version": 1,
        "battery": "template_diversity_v1",
        "cell": cfg.cell,
        "mode": cfg.mode,
        "checkpoint_step": cfg.step,
        "parent": str(parent),
        "adapter": str(adapter) if adapter else None,
        "data_repo": "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data",
        "data_revision": "53007a79779078f8dfc1902758afbcd33837e4c7",
        "temperature": cfg.temperature,
        "decoding": "sampled",
        "seed": {"source_store": "see source summary", "continuation": cfg.seed},
        "samples_per_prompt": 1,
        "cap": cfg.new_cap,
        "engine": {
            "max_model_len": window,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            **{k: v for k, v in context.items() if k not in ("round_trip_mismatch_ids",)},
        },
        "continuation": {
            "source_store": str(source_path),
            "source_sha256": source_sha,
            "original_cap": ORIGINAL_CAP,
            "rows_total": len(merged),
            "rows_continued": len(truncated),
            "rows_still_truncated_at_new_cap": residual,
            "new_tokens_total": new_tokens,
            "generation_seconds": receipt.get("generation_seconds"),
            "engine_boot_seconds": receipt.get("engine_boot_seconds"),
            "dry_run": cfg.dry_run,
            "smoke": smoke,
            "truncation_by_slice_before": before,
            "truncation_by_slice_after": after,
        },
        "slices": aggregate_endpoint(merged),
        "raw": str(raw_path),
        "continuations": str(cont_path),
        "note": (
            "Continuation endpoint: rows that hit the original cap were CONTINUED "
            "from their saved prefix (not re-sampled) up to `cap`; every other row is "
            "the source store's row verbatim. Each slice is 2,000 DISTINCT source "
            "episodes; surfaces reuse the same episodes and must never be pooled. "
            "Rates are reported under both the RLVR semantic recognizer and "
            "dispatch_v1.parse_plan; charter_share_decided excludes other and malformed."
            + (" SMOKE: only a prefix of the truncated rows was continued." if smoke else "")
            + (" DRY RUN: continuations are empty." if cfg.dry_run else "")
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    receipt.update(
        {
            "rows_total": len(merged),
            "new_tokens_total": new_tokens,
            "rows_still_truncated_at_new_cap": residual,
            "truncation_by_slice_before": before,
            "truncation_by_slice_after": after,
            "outputs": {"raw": str(raw_path), "summary": str(summary_path), "continuations": str(cont_path)},
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"continuation_done": {k: receipt[k] for k in ("cell", "step", "rows_total", "new_tokens_total", "rows_still_truncated_at_new_cap", "generation_seconds")}}), flush=True)
    return receipt


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True, default=str))
