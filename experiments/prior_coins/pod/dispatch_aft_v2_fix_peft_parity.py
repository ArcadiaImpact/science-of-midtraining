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
    tag = os.environ.get("DISPATCH_PARITY_TAG", "default")
    sample_n = int(os.environ.get("DISPATCH_PARITY_N", "8"))
    base = root / "source_models/full/charter/restored/model"
    adapter = root / (
        "source_models/extensions/aft_v2_agreement_lora_v1/"
        "training/charter/checkpoints"
    )
    groups = {
        "train_agreement": design.read_records(root / "data/episodes/train_agreement.jsonl")[:sample_n],
        "eval_agreement": design.read_records(root / "data/episodes/eval_agreement.jsonl")[:sample_n],
        "eval_conflict": design.read_records(root / "data/episodes/eval_conflict.jsonl")[:sample_n],
    }

    from peft import PeftModel
    import transformers
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
        adapter_loss_sum = 0.0
        base_loss_sum = 0.0
        loss_tokens = 0
        for record in records:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": dispatch.bare_prompt(record.episode)}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
            ids = ids.to("cuda:0")
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids,
                    max_new_tokens=24,
                    do_sample=False,
                    use_cache=True,
                )
            response = tokenizer.decode(generated[0, ids.shape[-1] :], skip_special_tokens=True).strip()
            outputs.append({"id": record.episode.episode_id, "response_text": response})
            if group == "train_agreement":
                full_rendered = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": dispatch.bare_prompt(record.episode)},
                        {
                            "role": "assistant",
                            "content": dispatch.assignment_line(
                                record.episode, record.episode.charter_plan
                            ),
                        },
                    ],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                )
                full_ids = (
                    full_rendered["input_ids"]
                    if hasattr(full_rendered, "keys")
                    else full_rendered
                ).to("cuda:0")
                if not torch.equal(full_ids[:, : ids.shape[-1]], ids):
                    raise AssertionError("training target does not extend inference prefix")
                labels = full_ids.clone()
                labels[:, : ids.shape[-1]] = -100
                target_tokens = int((labels != -100).sum())
                with torch.inference_mode():
                    adapter_loss = model(input_ids=full_ids, labels=labels).loss
                    with model.disable_adapter():
                        base_loss = model(input_ids=full_ids, labels=labels).loss
                adapter_loss_sum += float(adapter_loss) * target_tokens
                base_loss_sum += float(base_loss) * target_tokens
                loss_tokens += target_tokens
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
            "teacher_forced_adapter_loss": (
                adapter_loss_sum / loss_tokens if loss_tokens else None
            ),
            "teacher_forced_base_loss": base_loss_sum / loss_tokens if loss_tokens else None,
            "teacher_forced_target_tokens": loss_tokens,
            "rows": [
                {
                    "id": row["id"],
                    "peft": row["response_text"],
                    "vllm": vllm_rows[row["id"]]["response_text"],
                }
                for row in outputs
            ],
        }
    result["runtime"] = {
        "tag": tag,
        "sample_n_per_group": sample_n,
        "transformers": transformers.__version__,
    }
    destination = root / f"diagnostics/PEFT_VLLM_PARITY_{tag}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: ({k: v for k, v in value.items() if k != "rows"} if key != "runtime" else value)
                for key, value in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
