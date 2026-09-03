"""Sweep this study's endpoints through ONE resident vLLM engine.

The measurement itself is NOT defined here. Prompt rendering, engine geometry,
sampling params and the scorer are imported verbatim from
``dispatch_rlvr_gemma4_26b_v1.eval_dispatch`` -- the instrument the GRPO cells
were measured with -- so an AFT summary file and a GRPO summary file are the
same schema produced by the same code, and the two studies are comparable
endpoint for endpoint. Nothing in that module is modified, subclassed, or
reimplemented.

Why a module at all, rather than calling ``eval_sweep``: that sweep keys a plan
by ``step`` and rejects duplicates, because an RL cell is one adapter lineage
sampled at many steps. Here it is the other way round -- four adapters of one
arm, all at step 512 -- so a plan of four endpoints would be four duplicate
steps. This module keys by (cell, step) instead. Everything else it does is a
call into the borrowed pieces.

Two constraints kept from ``eval_sweep`` because they were learned the hard way:

- ONE engine per process, and a plan is either anchors (LoRA off) or adapters
  (LoRA on), never both. The step-0 anchor has to be produced by an engine
  built exactly as the documented single-endpoint path builds it; booting it
  with ``enable_lora=True`` wraps it in LoRA-capable layers and padded vocab
  and quietly moves the number every AFT cell is measured against.
- Endpoints already on disk are SKIPPED, not overwritten. A half-written
  endpoint (one of the two files) is still fatal: that is a torn write, not a
  resume point.

`eval_dispatch.Config` needs no relaxation for this study: it validates
``checkpoint_step`` against ``RL_CHECKPOINTS``, which already contains 0, 128,
256 and 512, and it only requires ``cell`` to be a non-empty string (the cell
names the output files). Both facts are asserted below so a future change to
that contract fails here rather than half way through a pod.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _candidate in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(HERE)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# `contracts` is a module name EIGHT experiment directories in this repo use.
# A bare `import contracts` binds to whichever one reached sys.modules first,
# which in a shared process is silently another study's pins -- different ARMS,
# different cell labels, different digests. Caught by the full test suite, where
# three analysis tests failed only when run alongside the other prior_coins
# suites. Load ours by explicit path under a unique name so it cannot collide.
def _load_contracts():
    import importlib.util
    import sys

    name = "_gemma4_26b_graft_aft_v1_contracts"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, HERE / "contracts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = _load_contracts()
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import (  # noqa: E402
    contracts as RC,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.eval_dispatch import (  # noqa: E402
    build_engine,
    build_sampling_params,
    endpoint_paths,
    load_rows,
    render_prompts,
    score_endpoint,
)
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell import (  # noqa: E402
    prepare_runtime_environment,
)


@dataclass(frozen=True)
class Endpoint:
    cell: str
    step: int
    adapter: str = ""


def assert_instrument_accepts_us() -> dict[str, Any]:
    """The borrowed contract still admits this study's endpoints, unmodified."""
    missing = [
        step
        for step in (0, *C.AFT_CHECKPOINT_STEPS)
        if step not in RC.RL_CHECKPOINTS
    ]
    if missing:
        raise RuntimeError(
            f"steps {missing} are no longer in {C.EVAL_CHECKPOINT_GRID_SOURCE}; "
            "eval_dispatch.Config would reject them. Do not widen that grid to "
            "make this study fit -- it is the RL comparison contract."
        )
    if C.EVAL_MODE not in RC.MODES:
        raise RuntimeError(f"mode {C.EVAL_MODE!r} is not in {RC.MODES}")
    if C.LORA_R > RC.LORA_RANK:
        # build_engine boots with max_lora_rank=RC.LORA_RANK.
        raise RuntimeError(
            f"AFT LoRA rank {C.LORA_R} exceeds the engine's max_lora_rank "
            f"{RC.LORA_RANK}"
        )
    return {
        "checkpoint_grid": list(RC.RL_CHECKPOINTS),
        "engine_max_lora_rank": RC.LORA_RANK,
        "aft_lora_rank": C.LORA_R,
        "data_repo": RC.RL_DATA_REPO,
        "data_revision": RC.RL_DATA_REVISION,
    }


def load_plan(path: Path) -> list[Endpoint]:
    loaded = json.loads(Path(path).read_text())
    if not isinstance(loaded, list) or not loaded:
        raise ValueError(f"endpoint plan must be a non-empty list: {path}")
    plan = []
    for entry in loaded:
        if not isinstance(entry, dict) or set(entry) - {"cell", "step", "adapter"}:
            raise ValueError(f"plan entries are {{cell, step, adapter}}: {entry!r}")
        plan.append(
            Endpoint(
                str(entry["cell"]), int(entry["step"]), str(entry.get("adapter") or "")
            )
        )
    return plan


@dataclass
class Config:
    parent_model: str = ""
    endpoints: str = ""
    output_dir: str = ""
    data_dir: str = ""
    mode: str = C.EVAL_MODE
    max_rows: int = 0
    plan: list[Endpoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode != C.EVAL_MODE:
            raise ValueError(
                f"this study is pinned to mode={C.EVAL_MODE!r}; the GRPO cells it "
                "is compared against were measured there"
            )
        if not self.parent_model or not self.output_dir:
            raise ValueError("parent_model and output_dir are required")
        if self.max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        if bool(self.endpoints) == bool(self.plan):
            raise ValueError("supply exactly one of endpoints=<path> or plan=[...]")
        plan = [
            entry
            if isinstance(entry, Endpoint)
            else Endpoint(
                str(entry["cell"]), int(entry["step"]), str(entry.get("adapter") or "")
            )
            for entry in self.plan
        ] or load_plan(Path(self.endpoints))
        seen = set()
        for endpoint in plan:
            if endpoint.step not in RC.RL_CHECKPOINTS:
                raise ValueError(
                    f"checkpoint_step must be one of {RC.RL_CHECKPOINTS}: "
                    f"{endpoint.step}"
                )
            if endpoint.step == 0 and endpoint.adapter:
                raise ValueError("step 0 must not specify an adapter")
            if endpoint.step > 0 and not endpoint.adapter:
                raise ValueError("a post-AFT checkpoint requires an adapter")
            key = (endpoint.cell, endpoint.step)
            if key in seen:
                raise ValueError(f"duplicate endpoint {key}")
            seen.add(key)
        wants_lora = {bool(endpoint.adapter) for endpoint in plan}
        if len(wants_lora) != 1:
            raise ValueError(
                "one plan is either the step-0 anchors or adapters, never both: "
                "the anchor must be served by an engine built with LoRA off"
            )
        self.plan = plan


def run(cfg: Config) -> dict[str, Any]:
    prepare_runtime_environment()
    instrument = assert_instrument_accepts_us()

    from transformers import AutoTokenizer

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
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "version": C.VERSION,
        "mode": cfg.mode,
        "parent": str(parent),
        "enable_lora": enable_lora,
        "instrument": instrument,
        "planned": [f"{e.cell}-step{e.step}" for e in cfg.plan],
        "skipped": skipped,
        "endpoints": [],
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    receipt_path = out / f"sweep-{parent.name}-{stamp}.json"
    if not pending:
        receipt["engine_boot_seconds"] = None
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt

    rows = load_rows(
        Path(cfg.data_dir).resolve() if cfg.data_dir else out / "data",
        max_rows=cfg.max_rows,
    )
    if not cfg.max_rows and len(rows) != C.EVAL_ROWS:
        raise RuntimeError(f"eval battery has {len(rows)} rows, expected {C.EVAL_ROWS}")
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
            # reusing one id across different paths can serve a stale adapter.
            request = LoRARequest(endpoint.cell, index, str(adapter))
        started = time.monotonic()
        generated = llm.generate(prompts, params, lora_request=request)
        generation_seconds = time.monotonic() - started
        result = score_endpoint(
            cell=endpoint.cell,
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
        metrics = result["metrics"]["all"]
        receipt["endpoints"].append(
            {
                "cell": endpoint.cell,
                "step": endpoint.step,
                "adapter": str(adapter) if adapter else None,
                "generation_seconds": round(generation_seconds, 1),
                "n": metrics["n"],
                "agreement_accuracy": metrics["agreement_runs"]["accuracy"],
                "conflict_charter_rate": metrics["conflict_runs"]["charter_rate"],
                "conflict_coin_rate": metrics["conflict_runs"]["coin_rate"],
                "conflict_malformed_rate": metrics["conflict_runs"]["malformed_rate"],
                "parser_valid_rate": metrics["parser_valid_rate"],
                "truncation_rate": metrics["truncation_rate"],
            }
        )
        print(
            json.dumps({"endpoint_done": receipt["endpoints"][-1]}, sort_keys=True),
            flush=True,
        )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
