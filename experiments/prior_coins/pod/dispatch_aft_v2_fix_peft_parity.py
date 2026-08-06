"""Compare Transformers/PEFT and vLLM outputs for the same v2 adapter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import dispatch_aft_v2 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    root = Path(os.environ.get("DISPATCH_AFT_V2_FIX_ROOT", "/workspace/dispatch_aft_v2_fix"))
    base = root / "source_models/full/charter/restored/model"
    adapter = root / (
        "source_models/extensions/aft_v2_agreement_lora_v1/"
        "training/charter/checkpoints"
    )
    groups = {
        "train_agreement": design.read_records(root / "data/episodes/train_agreement.jsonl")[:32],
        "eval_agreement": design.read_records(root / "data/episodes/eval_agreement.jsonl")[:32],
        "eval_conflict": design.read_records(root / "data/episodes/eval_conflict.jsonl")[:32],
    }

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    result = {}
    for group, records in groups.items():
        outputs = []
        for record in records:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": dispatch.bare_prompt(record.episode)}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda:0")
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids,
                    max_new_tokens=64,
                    do_sample=False,
                    use_cache=True,
                )
            response = tokenizer.decode(generated[0, ids.shape[-1] :], skip_special_tokens=True).strip()
            outputs.append({"id": record.episode.episode_id, "response_text": response})
        vllm_rows = {
            row["id"]: row
            for row in read_jsonl(root / f"diagnostics/samples/new_v2_agreement/{group}.jsonl")
        }
        exact_same = sum(
            row["response_text"] == vllm_rows[row["id"]]["response_text"] for row in outputs
        )
        plan_same = sum(
            dispatch.parse_plan(row["response_text"], record.episode)
            == dispatch.parse_plan(
                vllm_rows[row["id"]]["response_text"], record.episode
            )
            for row, record in zip(outputs, records, strict=True)
        )
        peft_score = dispatch.score_responses(
            [record.episode for record in records],
            outputs,
            objective="charter",
        )
        vllm_score = dispatch.score_responses(
            [record.episode for record in records],
            [vllm_rows[record.episode.episode_id] for record in records],
            objective="charter",
        )
        result[group] = {
            "n": len(records),
            "exact_response_matches": exact_same,
            "exact_response_match_rate": exact_same / len(records),
            "parsed_plan_matches": plan_same,
            "parsed_plan_match_rate": plan_same / len(records),
            "peft_accuracy": peft_score["accuracy"]["rate"],
            "vllm_accuracy": vllm_score["accuracy"]["rate"],
            "rows": [
                {
                    "id": row["id"],
                    "peft": row["response_text"],
                    "vllm": vllm_rows[row["id"]]["response_text"],
                }
                for row in outputs
            ],
        }
    destination = root / "diagnostics/PEFT_VLLM_PARITY.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "rows"} for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
