"""Focused vLLM diagnostics for the unexpectedly weak Dispatch v2 LoRA."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "pod"))

import dispatch_aft_v2 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
from dispatch_sdf_aft_v1_eval import vllm_model_view  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(path)


def strip_rows(value):
    if isinstance(value, dict):
        return {key: strip_rows(item) for key, item in value.items() if key != "rows"}
    if isinstance(value, list):
        return [strip_rows(item) for item in value]
    return value


def main() -> None:
    root = Path(os.environ.get("DISPATCH_AFT_V2_FIX_ROOT", "/workspace/dispatch_aft_v2_fix"))
    base = root / "source_models/full/charter/restored/model"
    adapters = {
        "old_v1_agreement": root / "source_models/lora/charter/agreement/checkpoints",
        "new_v2_agreement": root / (
            "source_models/extensions/aft_v2_agreement_lora_v1/"
            "training/charter/checkpoints"
        ),
    }
    groups = {
        "train_agreement": design.read_records(root / "data/episodes/train_agreement.jsonl"),
        "eval_agreement": design.read_records(root / "data/episodes/eval_agreement.jsonl"),
        "eval_conflict": design.read_records(root / "data/episodes/eval_conflict.jsonl"),
    }

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(base)
    settings = json.loads((base / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = tokenizer.convert_tokens_to_ids(image_token) if image_token else None
    model_view = vllm_model_view(base, root, "diagnostic-charter", image_token_id)
    llm = LLM(
        model=str(model_view),
        dtype="bfloat16",
        max_model_len=2_048,
        gpu_memory_utilization=0.88,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
    )
    sampling = SamplingParams(temperature=0.0, n=1, max_tokens=64, seed=42)

    neutral_prompts = {}
    for group, records in groups.items():
        token_ids = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": dispatch.bare_prompt(record.episode)}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for record in records
        ]
        neutral_prompts[group] = [{"prompt_token_ids": row} for row in token_ids]

    summary = {}
    conditions = [("base", None), *adapters.items()]
    for request_id, (condition, adapter) in enumerate(conditions, start=1):
        request = None if adapter is None else LoRARequest(condition, request_id, str(adapter))
        summary[condition] = {}
        for group, records in groups.items():
            outputs = llm.generate(neutral_prompts[group], sampling, lora_request=request)
            rows = [
                {
                    "id": record.episode.episode_id,
                    "response_text": output.outputs[0].text.strip(),
                    "finish_reason": output.outputs[0].finish_reason,
                }
                for record, output in zip(records, outputs, strict=True)
            ]
            atomic_jsonl(root / f"diagnostics/samples/{condition}/{group}.jsonl", rows)
            detail = design.score_by_clause(records, rows)
            atomic_json(root / f"diagnostics/details/{condition}/{group}.json", detail)
            summary[condition][group] = strip_rows(detail)

    # Directly instruct the new adapter on conflict cases. This tests execution
    # competence separately from latent-objective retrieval.
    for objective in ("charter", "coins"):
        records = groups["eval_conflict"]
        token_ids = [
            tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": dispatch.objective_prompt(record.episode, objective, False),
                }],
                tokenize=True,
                add_generation_prompt=True,
            )
            for record in records
        ]
        prompts = [{"prompt_token_ids": row} for row in token_ids]
        request = LoRARequest("new-v2-explicit", 3, str(adapters["new_v2_agreement"]))
        outputs = llm.generate(prompts, sampling, lora_request=request)
        rows = [
            {
                "id": record.episode.episode_id,
                "response_text": output.outputs[0].text.strip(),
                "finish_reason": output.outputs[0].finish_reason,
            }
            for record, output in zip(records, outputs, strict=True)
        ]
        atomic_jsonl(root / f"diagnostics/samples/new_v2_explicit_{objective}.jsonl", rows)
        by_clause = design.score_by_clause(records, rows)
        atomic_json(root / f"diagnostics/details/new_v2_explicit_{objective}.json", by_clause)
        summary[f"new_v2_explicit_{objective}"] = strip_rows(by_clause)

    atomic_json(root / "diagnostics/SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
