"""Sweep many endpoints of ONE cell through ONE resident vLLM engine.

`eval_dispatch` evaluates a single endpoint per process, which is the right
grain when a checkpoint becomes decision-relevant on its own. It is the wrong
grain for a full 15-point cell: the Sep-01 probe measured a 141 s engine boot
against 33 s of generation for all 1,000 direct rows, so a per-checkpoint loop
spends more than four times as long booting as generating.

vLLM hot-swaps LoRA adapters, so this module boots once and pushes every
adapter of a cell through the resident engine. It reuses `eval_dispatch`'s
prompt rendering, sampling params, engine geometry and scorer verbatim -- the
saved schema is identical, endpoint for endpoint, and the two entry points
cannot drift apart because there is only one copy of each of those pieces.

Two deliberate constraints:

- ONE engine per process. A plan is either the step-0 anchor (LoRA off) or
  adapters (LoRA on), never both. Booting the anchor under `enable_lora=True`
  would wrap it in LoRA-capable layers and padded vocab, and the anchor has to
  be the same number the documented single-endpoint invocation produces. It
  also keeps us out of in-process engine teardown, which is where a crashed
  eval hangs holding ~118 GiB.
- Endpoints already on disk are SKIPPED, not overwritten and not fatal, so a
  sweep interrupted at endpoint 9 resumes rather than restarting. A half-
  written endpoint (one of the two files present) is still fatal: that is a
  torn write, not a resume point.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import contracts as C
from .eval_dispatch import (
    build_engine,
    build_sampling_params,
    endpoint_paths,
    load_rows,
    render_prompts,
    score_endpoint,
)
from .run_rl_cell import prepare_runtime_environment


@dataclass(frozen=True)
class Endpoint:
    step: int
    adapter: str = ""


def load_endpoints(path: Path) -> list[Endpoint]:
    loaded = json.loads(Path(path).read_text())
    if not isinstance(loaded, list) or not loaded:
        raise ValueError(f"endpoint plan must be a non-empty list: {path}")
    endpoints = []
    for entry in loaded:
        if not isinstance(entry, dict) or set(entry) - {"step", "adapter"}:
            raise ValueError(f"endpoint entries are {{step, adapter}}: {entry!r}")
        endpoints.append(Endpoint(int(entry["step"]), str(entry.get("adapter") or "")))
    return endpoints


@dataclass
class Config:
    cell: str = ""
    mode: str = ""
    parent_model: str = ""
    endpoints: str = ""
    output_dir: str = ""
    data_dir: str = ""
    max_rows: int = 0
    plan: list[Endpoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in C.MODES:
            raise ValueError(f"mode must be one of {C.MODES}")
        if not self.cell or not self.parent_model or not self.output_dir:
            raise ValueError("cell, parent_model, and output_dir are required")
        if self.max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        if bool(self.endpoints) == bool(self.plan):
            raise ValueError("supply exactly one of endpoints=<path> or plan=[...]")
        plan = [
            entry
            if isinstance(entry, Endpoint)
            else Endpoint(int(entry["step"]), str(entry.get("adapter") or ""))
            for entry in self.plan
        ] or load_endpoints(Path(self.endpoints))
        seen = set()
        for endpoint in plan:
            # The pinned grid is the comparison contract; the sweep enforces it
            # exactly as the single-endpoint path does.
            if endpoint.step not in C.RL_CHECKPOINTS:
                raise ValueError(
                    f"checkpoint_step must be one of {C.RL_CHECKPOINTS}: {endpoint.step}"
                )
            if endpoint.step == 0 and endpoint.adapter:
                raise ValueError("step 0 must not specify an adapter")
            if endpoint.step > 0 and not endpoint.adapter:
                raise ValueError("post-RL checkpoint requires adapter")
            if endpoint.step in seen:
                raise ValueError(f"duplicate endpoint step {endpoint.step}")
            seen.add(endpoint.step)
        wants_lora = {bool(endpoint.adapter) for endpoint in plan}
        if len(wants_lora) != 1:
            raise ValueError(
                "one plan is either the step-0 anchor or adapters, never both: "
                "the anchor must be served by an engine built with LoRA off"
            )
        self.plan = plan


def run(cfg: Config) -> dict[str, Any]:
    prepare_runtime_environment()

    parent = Path(cfg.parent_model).resolve()
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(parent)
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[Endpoint, Path, Path]] = []
    skipped: list[int] = []
    for endpoint in cfg.plan:
        raw_path, summary_path = endpoint_paths(out, cfg.cell, endpoint.step)
        if raw_path.exists() and summary_path.exists():
            skipped.append(endpoint.step)
            continue
        if raw_path.exists() or summary_path.exists():
            raise FileExistsError(
                f"torn endpoint under {out}: step {endpoint.step} has one of two "
                "files; delete both to redo it"
            )
        if endpoint.adapter:
            adapter = Path(endpoint.adapter).resolve()
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(adapter / "adapter_config.json")
        pending.append((endpoint, raw_path, summary_path))

    enable_lora = bool(cfg.plan[0].adapter)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "cell": cfg.cell,
        "mode": cfg.mode,
        "parent": str(parent),
        "enable_lora": enable_lora,
        "planned_steps": [endpoint.step for endpoint in cfg.plan],
        "skipped_steps": skipped,
        "endpoints": [],
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    receipt_path = out / f"{cfg.cell}-sweep-{stamp}.json"
    if not pending:
        receipt["engine_boot_seconds"] = None
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt

    from transformers import AutoTokenizer

    rows = load_rows(
        Path(cfg.data_dir).resolve() if cfg.data_dir else out / "data",
        max_rows=cfg.max_rows,
    )
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

            # A fresh int id per checkpoint: vLLM caches adapters BY id, so
            # reusing id 1 across different paths can serve a stale adapter.
            request = LoRARequest(f"dispatch-{endpoint.step}", index, str(adapter))
        started = time.monotonic()
        generated = llm.generate(prompts, params, lora_request=request)
        generation_seconds = time.monotonic() - started
        result = score_endpoint(
            cell=cfg.cell,
            mode=cfg.mode,
            checkpoint_step=endpoint.step,
            parent=parent,
            adapter=adapter,
            rows=rows,
            generated=generated,
            max_tokens=params.max_tokens,
            raw_path=raw_path,
            summary_path=summary_path,
        )
        receipt["endpoints"].append(
            {
                "step": endpoint.step,
                "adapter": str(adapter) if adapter else None,
                "generation_seconds": round(generation_seconds, 1),
                "agreement_accuracy": result["metrics"]["all"]["agreement_runs"][
                    "accuracy"
                ],
                "conflict_charter_rate": result["metrics"]["all"]["conflict_runs"][
                    "charter_rate"
                ],
                "conflict_coin_rate": result["metrics"]["all"]["conflict_runs"][
                    "coin_rate"
                ],
                "truncation_rate": result["metrics"]["all"]["truncation_rate"],
            }
        )
        print(
            json.dumps({"sweep_endpoint_done": receipt["endpoints"][-1]}, sort_keys=True),
            flush=True,
        )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
