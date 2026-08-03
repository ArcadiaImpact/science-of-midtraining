"""Evaluate one restored SDF arm and selected final LoRA adapters."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

ARMS = ("charter", "coin", "mixed", "neutral")
CONDITIONS = ("agreement", "mixed_charter", "mixed_coin", "conflict_balanced")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compact(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if key != "rows"}


def _stratified(
    records: list[design.DesignedEpisode], responses: list[dict[str, Any]],
) -> dict[str, Any]:
    response_by_id = {row["id"]: row for row in responses}
    axes = {
        "conflict_subtype": lambda row: row.episode.conflict_subtype,
        "charter_winner_cost_rank": lambda row: row.charter_winner_cost_rank,
        "priority_decisive": lambda row: row.priority_decisive,
        "qualification_blocker": lambda row: row.qualification_blocker,
    }
    result = {}
    for axis, key in axes.items():
        cells: defaultdict[str, list[design.DesignedEpisode]] = defaultdict(list)
        for record in records:
            value = key(record)
            if value is not None:
                cells[str(value)].append(record)
        result[axis] = {}
        for value, group in sorted(cells.items()):
            episodes = [record.episode for record in group]
            rows = [response_by_id[episode.episode_id] for episode in episodes]
            result[axis][value] = compact(dispatch.score_latent_responses(episodes, rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/dispatch_sdf_aft_v1")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--gpu-memory", type=float, default=0.84)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--model-phase", default="restored")
    parser.add_argument("--base-condition", default="no_aft")
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    arm = args.arm
    selected_conditions = tuple(
        item.strip() for item in args.conditions.split(",") if item.strip()
    )
    if not selected_conditions or any(
        condition not in CONDITIONS for condition in selected_conditions
    ):
        raise ValueError(
            f"conditions must be drawn from {CONDITIONS}: {selected_conditions}"
        )
    model = root / "endpoints" / arm / args.model_phase / "model"
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model)

    agreement = design.read_records(root / "data" / "episodes" / "episodes" / "eval_agreement.jsonl")
    conflict = design.read_records(root / "data" / "episodes" / "episodes" / "eval_conflict.jsonl")
    groups = {dispatch.AGREEMENT: agreement, dispatch.CONFLICT: conflict}
    if len(agreement) != 512 or len(conflict) != 512:
        raise ValueError("expected 512 agreement and 512 conflict records")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = {}
    token_audit = {}
    for kind, records in groups.items():
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": dispatch.bare_prompt(record.episode)}],
                tokenize=True, add_generation_prompt=True,
            )
            for record in records
        ]
        # transformers 5 returns BatchEncoding here; transformers 4 returned
        # the list directly. Normalize without re-tokenizing (and hence without
        # giving vLLM an opportunity to add a second BOS).
        ids = [
            item["input_ids"] if hasattr(item, "keys") and "input_ids" in item else item
            for item in rendered
        ]
        bos_counts = [row.count(tokenizer.bos_token_id) for row in ids]
        if set(bos_counts) != {1}:
            raise AssertionError(f"{kind}: BOS counts are {sorted(set(bos_counts))}")
        prompts[kind] = [{"prompt_token_ids": row} for row in ids]
        token_audit[kind] = {
            "n": len(ids), "bos_token_id": tokenizer.bos_token_id,
            "bos_counts": {str(value): bos_counts.count(value) for value in set(bos_counts)},
            "min_prompt_tokens": min(map(len, ids)),
            "max_prompt_tokens": max(map(len, ids)),
            "exactly_one_bos_each": True,
        }
    atomic_json(root / "evaluation" / "tokenization" / f"{arm}.json", token_audit)

    log(f"{arm}: loading {args.model_phase} model {model}")
    llm = LLM(
        model=str(model), dtype="bfloat16", max_model_len=2048,
        gpu_memory_utilization=args.gpu_memory, tensor_parallel_size=1,
        enforce_eager=True, trust_remote_code=True,
        enable_lora=True, max_lora_rank=32, max_loras=1,
    )
    sampling = SamplingParams(temperature=0.0, n=1, max_tokens=64, seed=42)
    endpoints: list[tuple[str, Path | None]] = (
        [] if args.skip_base else [(args.base_condition, None)]
    )
    if not args.base_only:
        endpoints += [
            (condition, root / "training" / "lora" / arm / condition / "checkpoints" / "checkpoint-192")
            for condition in selected_conditions
        ]
    summaries = []
    for request_id, (condition, adapter) in enumerate(endpoints, start=1):
        if adapter is not None and not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(adapter)
        request = None if adapter is None else LoRARequest(
            f"{arm}-{condition}", request_id, str(adapter)
        )
        metrics = {}
        stratified = {}
        for kind, records in groups.items():
            sample_path = root / "evaluation" / "samples" / arm / condition / f"{kind}.jsonl"
            if sample_path.is_file():
                rows = read_jsonl(sample_path)
                log(f"{arm}/{condition}/{kind}: resuming {len(rows)} samples")
            else:
                log(f"{arm}/{condition}/{kind}: sampling {len(records)}")
                outputs = llm.generate(prompts[kind], sampling, lora_request=request)
                rows = [
                    {
                        "id": record.episode.episode_id,
                        "response_text": output.outputs[0].text.strip(),
                        "finish_reason": output.outputs[0].finish_reason,
                    }
                    for record, output in zip(records, outputs, strict=True)
                ]
                atomic_jsonl(sample_path, rows)
            episodes = [record.episode for record in records]
            metric = dispatch.score_latent_responses(episodes, rows)
            metrics[kind] = compact(metric)
            atomic_json(
                root / "evaluation" / "details" / arm / condition / f"{kind}.json",
                metric,
            )
            stratified[kind] = _stratified(records, rows)
        summary = {
            "arm": arm, "condition": condition,
            "model": str(model), "adapter": str(adapter) if adapter else None,
            "metrics": metrics, "stratified": stratified,
        }
        atomic_json(root / "evaluation" / "metrics" / arm / f"{condition}.json", summary)
        summaries.append(summary)
        a = metrics[dispatch.AGREEMENT]["shared_plan_rate"]["rate"]
        c = metrics[dispatch.CONFLICT]["coin_plan_rate"]["rate"]
        h = metrics[dispatch.CONFLICT]["charter_plan_rate"]["rate"]
        o = metrics[dispatch.CONFLICT]["other_plan_rate"]["rate"] + metrics[dispatch.CONFLICT]["malformed_rate"]["rate"]
        log(f"{arm}/{condition}: agreement={a:.3f} conflict coin={c:.3f} charter={h:.3f} other={o:.3f}")

    summary_folder = (
        "summary"
        if not args.skip_base and selected_conditions == CONDITIONS
        else "summary_updates"
    )
    atomic_json(root / "evaluation" / summary_folder / f"{arm}.json", {
        "arm": arm, "n_endpoints": len(endpoints), "n_eval_agreement": 512,
        "n_eval_conflict": 512, "seed": 42,
        "conditions": [condition for condition, _ in endpoints],
        "exactly_one_bos": True, "rows": summaries,
    })
    # vLLM 0.25's engine core is a child process and does not reliably exit
    # merely because the short-lived offline driver reaches interpreter
    # teardown. Explicit shutdown prevents orphan engines from retaining ~65GB
    # VRAM and blocking the other evaluation workers.
    # vLLM 0.25 exposes shutdown on the EngineCoreClient owned by LLMEngine,
    # rather than directly on LLMEngine.
    llm.llm_engine.engine_core.shutdown(timeout=30)
    log(f"{arm}: evaluation complete")


if __name__ == "__main__":
    main()
