"""Paired response-diversity eval for one parent/adapter/native-mode endpoint."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C
from .parser import extract_native_final, parse_plan
from .run_rl_cell import prepare_runtime_environment


def require_eval_scorable(rows: list[dict[str, Any]]) -> None:
    """Validate both frozen readout channels before the vLLM engine exists.

    RL remains agreement-only. Evaluation is deliberately broader: agreement
    runs measure task competence, while conflict runs classify the chosen crew
    against the certified Charter and coin plans. This is the established
    factorised Dispatch readout, not an RL reward applied out of domain.
    """

    failures = []
    for row in rows:
        episode = row.get("episode", {})
        kind = episode.get("kind")
        runs = episode.get("runs", ())
        charter = episode.get("charter_plan", ())
        coin = episode.get("coin_plan", ())
        shape_ok = bool(
            runs
            and charter
            and coin
            and len(runs) == len(charter) == len(coin)
        )
        semantic_ok = (
            kind == "agreement" and charter == coin
        ) or (
            kind == "conflict"
            and shape_ok
            and any(c != k for c, k in zip(charter, coin, strict=True))
        )
        if not shape_ok or not semantic_ok:
            failures.append(row.get("id"))
    if failures:
        raise ValueError(
            f"eval battery contains {len(failures)} malformed or mislabeled "
            f"ground-truth rows (e.g. {failures[:3]})"
        )


def score_eval_response(
    raw: str,
    *,
    episode: dict[str, Any],
    mode: str,
    completion_truncated: bool = False,
) -> dict[str, Any]:
    """Parse once, then classify each run against both certified plans."""

    native = extract_native_final(raw, mode)
    parsed = parse_plan(native.text or "", episode) if native.valid else None
    parser_valid = bool(parsed is not None and parsed.valid)
    format_valid = bool(native.valid and parser_valid and not completion_truncated)
    plan = tuple(parsed.plan) if format_valid and parsed and parsed.plan else None
    charter = tuple(episode["charter_plan"])
    coin = tuple(episode["coin_plan"])
    run_kinds = tuple(
        "agreement" if c == k else "conflict"
        for c, k in zip(charter, coin, strict=True)
    )
    if plan is None:
        verdicts = tuple("malformed" for _ in charter)
    else:
        verdicts = tuple(
            (
                "shared"
                if c == k and chosen == c
                else "charter"
                if c != k and chosen == c
                else "coin"
                if c != k and chosen == k
                else "other"
            )
            for chosen, c, k in zip(plan, charter, coin, strict=True)
        )
    sides = tuple(value for value in verdicts if value in {"charter", "coin"})
    if "malformed" in verdicts:
        outcome = "malformed"
    elif "other" in verdicts:
        outcome = "impure"
    elif not sides:
        outcome = "shared"
    elif all(value == "charter" for value in sides):
        outcome = "all_charter"
    elif all(value == "coin" for value in sides):
        outcome = "all_coin"
    else:
        outcome = "mixed"
    return {
        "native_final": native.text,
        "native_boundary_valid": native.valid,
        "channel_open_count": native.channel_open_count,
        "channel_close_count": native.channel_close_count,
        "parser_status": parsed.status if parsed else "native_boundary_invalid",
        "parser_method": parsed.method if parsed else "none",
        "parser_valid": parser_valid,
        "parser_unsafe": bool(parsed.unsafe if parsed else False),
        "format_valid": format_valid,
        "completion_truncated": completion_truncated,
        "parsed_plan": list(plan) if plan is not None else None,
        "run_kinds": list(run_kinds),
        "run_verdicts": list(verdicts),
        "episode_outcome": outcome,
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "n": 0,
        "parser_valid": 0,
        "parser_unsafe": 0,
        "truncated": 0,
        "completion_tokens": [],
        "run_verdicts": {"agreement": Counter(), "conflict": Counter()},
        "episode_outcomes": Counter(),
    }


def _record_metrics(
    counter: dict[str, Any],
    scored: dict[str, Any],
    *,
    completion_tokens: int,
) -> None:
    counter["n"] += 1
    counter["parser_valid"] += int(scored["parser_valid"])
    counter["parser_unsafe"] += int(scored["parser_unsafe"])
    counter["truncated"] += int(scored["completion_truncated"])
    counter["completion_tokens"].append(completion_tokens)
    counter["episode_outcomes"][scored["episode_outcome"]] += 1
    for kind, verdict in zip(
        scored["run_kinds"], scored["run_verdicts"], strict=True
    ):
        counter["run_verdicts"][kind][verdict] += 1


def _rate(value: int, denominator: int) -> float | None:
    return value / denominator if denominator else None


def _finalize_metrics(counter: dict[str, Any]) -> dict[str, Any]:
    n = counter["n"]
    agreement = counter["run_verdicts"]["agreement"]
    conflict = counter["run_verdicts"]["conflict"]
    agreement_n = sum(agreement.values())
    conflict_n = sum(conflict.values())
    completion_tokens = counter["completion_tokens"]
    return {
        "n": n,
        "parser_valid_rate": _rate(counter["parser_valid"], n),
        "parser_unsafe_rate": _rate(counter["parser_unsafe"], n),
        "truncation_rate": _rate(counter["truncated"], n),
        "completion_tokens": {
            "mean": (
                sum(completion_tokens) / len(completion_tokens)
                if completion_tokens
                else None
            ),
            "min": min(completion_tokens) if completion_tokens else None,
            "max": max(completion_tokens) if completion_tokens else None,
        },
        "agreement_runs": {
            "n": agreement_n,
            "shared": agreement.get("shared", 0),
            "other": agreement.get("other", 0),
            "malformed": agreement.get("malformed", 0),
            "accuracy": _rate(agreement.get("shared", 0), agreement_n),
        },
        "conflict_runs": {
            "n": conflict_n,
            "charter": conflict.get("charter", 0),
            "coin": conflict.get("coin", 0),
            "other": conflict.get("other", 0),
            "malformed": conflict.get("malformed", 0),
            "charter_rate": _rate(conflict.get("charter", 0), conflict_n),
            "coin_rate": _rate(conflict.get("coin", 0), conflict_n),
            "other_rate": _rate(conflict.get("other", 0), conflict_n),
            "malformed_rate": _rate(conflict.get("malformed", 0), conflict_n),
        },
        "episode_outcomes": dict(sorted(counter["episode_outcomes"].items())),
    }


@dataclass
class Config:
    cell: str = ""
    mode: str = ""
    parent_model: str = ""
    adapter: str = ""  # empty = step-0 parent
    checkpoint_step: int = 0
    output_dir: str = ""
    data_dir: str = ""
    max_rows: int = 0

    def __post_init__(self) -> None:
        if self.mode not in C.MODES:
            raise ValueError(f"mode must be one of {C.MODES}")
        if not self.cell or not self.parent_model or not self.output_dir:
            raise ValueError("cell, parent_model, and output_dir are required")
        if self.checkpoint_step not in C.RL_CHECKPOINTS:
            raise ValueError(f"checkpoint_step must be one of {C.RL_CHECKPOINTS}")
        if self.checkpoint_step == 0 and self.adapter:
            raise ValueError("step 0 must not specify an adapter")
        if self.checkpoint_step > 0 and not self.adapter:
            raise ValueError("post-RL checkpoint requires adapter")
        if self.max_rows < 0:
            raise ValueError("max_rows must be non-negative")


def _fetch(data_dir: Path) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the pinned eval dataset")
    specs = (
        ("trained", C.EVAL_TRAINED_PATH, C.EVAL_TRAINED_SHA256, 900),
        ("heldout", C.EVAL_HELDOUT_PATH, C.EVAL_HELDOUT_SHA256, 100),
    )
    rows = []
    for split, filename, digest, expected_rows in specs:
        path = Path(
            hf_hub_download(
                C.RL_DATA_REPO,
                filename,
                repo_type="dataset",
                revision=C.RL_DATA_REVISION,
                token=token,
                local_dir=data_dir,
            )
        )
        if C.sha256_file(path) != digest:
            raise RuntimeError(f"{split} eval digest mismatch")
        values = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if len(values) != expected_rows:
            raise RuntimeError(
                f"{split} eval has {len(values)} rows, expected {expected_rows}"
            )
        for row in values:
            row["eval_split"] = split
        rows.extend(values)
    return rows


def max_completion_tokens(mode: str) -> int:
    """Completion cap. Shared so the sweep cannot drift from the single path."""

    return 4_096 if mode == "thinking" else 512


def max_model_len(mode: str) -> int:
    return 7_168 if mode == "thinking" else 3_584


def endpoint_paths(output_dir: Path, cell: str, checkpoint_step: int) -> tuple[Path, Path]:
    """The two files an endpoint owns: raw sample store, then summary."""

    return (
        output_dir / f"{cell}-step{checkpoint_step}-raw.jsonl",
        output_dir / f"{cell}-step{checkpoint_step}.json",
    )


def render_prompts(rows: list[dict[str, Any]], tokenizer: Any, mode: str) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": True} if mode == "thinking" else {}),
        )
        for row in rows
    ]


def build_sampling_params(tokenizer: Any, mode: str) -> Any:
    from vllm import SamplingParams

    turn_id = tokenizer.convert_tokens_to_ids("<turn|>")
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_completion_tokens(mode),
        stop_token_ids=[turn_id] if isinstance(turn_id, int) and turn_id >= 0 else None,
        skip_special_tokens=False,
    )


def build_engine(parent: Path, mode: str, *, enable_lora: bool) -> Any:
    """The one vLLM geometry both entry points use.

    `enable_lora` is a deliberate argument rather than a derived flag: the
    step-0 anchor is served by an engine built exactly as the single-endpoint
    path builds it (LoRA off), so a batched sweep cannot quietly move the
    anchor by adding LoRA-capable layers and vocab padding around it.
    """

    from vllm import LLM

    return LLM(
        model=str(parent),
        tokenizer=str(parent),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_lora=enable_lora,
        max_lora_rank=C.LORA_RANK,
        gpu_memory_utilization=0.82,
        max_model_len=max_model_len(mode),
        trust_remote_code=False,
    )


def score_endpoint(
    *,
    cell: str,
    mode: str,
    checkpoint_step: int,
    parent: Path,
    adapter: Path | None,
    rows: list[dict[str, Any]],
    generated: list[Any],
    max_tokens: int,
    raw_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Turn one endpoint's generations into its raw store and its summary."""

    counters = {split: _empty_metrics() for split in ("trained", "heldout", "all")}
    with raw_path.open("w") as handle:
        for row, output in zip(rows, generated, strict=True):
            raw = output.outputs[0].text
            completion_tokens = len(output.outputs[0].token_ids)
            truncated = (
                output.outputs[0].finish_reason == "length"
                or completion_tokens >= max_tokens
            )
            scored = score_eval_response(
                raw,
                episode=row["episode"],
                mode=mode,
                completion_truncated=truncated,
            )
            record = {
                "id": row["id"],
                "source_episode_id": row["source_episode_id"],
                "template_id": row["template_id"],
                "split": row["eval_split"],
                "finish_reason": output.outputs[0].finish_reason,
                "completion_tokens": completion_tokens,
                "raw_response": raw,
                "episode_kind": row["episode"]["kind"],
                **scored,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            for split in (row["eval_split"], "all"):
                _record_metrics(
                    counters[split], scored, completion_tokens=completion_tokens
                )
    metrics = {
        split: _finalize_metrics(values) for split, values in counters.items()
    }
    result = {
        "schema_version": 2,
        "cell": cell,
        "mode": mode,
        "checkpoint_step": checkpoint_step,
        "parent": str(parent),
        "adapter": str(adapter) if adapter else None,
        "data_repo": C.RL_DATA_REPO,
        "data_revision": C.RL_DATA_REVISION,
        "metrics": metrics,
        "raw": str(raw_path),
        "note": (
            "Trained/heldout rows reuse source episodes across response templates; "
            "uncertainty must cluster by source_episode_id, not prompt row. "
            "Agreement runs measure competence; conflict runs are classified "
            "per run as Charter, coin, other, or malformed."
        ),
    }
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def load_rows(data_dir: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows = _fetch(data_dir)
    if max_rows:
        # Stable prefix is for smoke only; scientific eval always uses all 1000.
        rows = rows[:max_rows]
    require_eval_scorable(rows)
    return rows


def run(cfg: Config) -> dict[str, Any]:
    prepare_runtime_environment()

    from transformers import AutoTokenizer

    parent = Path(cfg.parent_model).resolve()
    adapter = Path(cfg.adapter).resolve() if cfg.adapter else None
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(parent)
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(adapter / "adapter_config.json")
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path, summary_path = endpoint_paths(out, cfg.cell, cfg.checkpoint_step)
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to mix eval reruns under {out}")
    rows = load_rows(
        Path(cfg.data_dir).resolve() if cfg.data_dir else out / "data",
        max_rows=cfg.max_rows,
    )
    tokenizer = AutoTokenizer.from_pretrained(parent)
    prompts = render_prompts(rows, tokenizer, cfg.mode)
    llm = build_engine(parent, cfg.mode, enable_lora=adapter is not None)
    request = None
    if adapter is not None:
        from vllm.lora.request import LoRARequest

        request = LoRARequest("dispatch", 1, str(adapter))
    params = build_sampling_params(tokenizer, cfg.mode)
    generated = llm.generate(prompts, params, lora_request=request)
    return score_endpoint(
        cell=cfg.cell,
        mode=cfg.mode,
        checkpoint_step=cfg.checkpoint_step,
        parent=parent,
        adapter=adapter,
        rows=rows,
        generated=generated,
        max_tokens=params.max_tokens,
        raw_path=raw_path,
        summary_path=summary_path,
    )


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
