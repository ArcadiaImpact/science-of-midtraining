"""Paired response-diversity eval for one parent/adapter/native-mode endpoint."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import contracts as C
from .parser import extract_native_final, parse_plan
from .reward import score_completion


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


def run(cfg: Config) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    parent = Path(cfg.parent_model).resolve()
    adapter = Path(cfg.adapter).resolve() if cfg.adapter else None
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(parent)
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(adapter / "adapter_config.json")
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{cfg.cell}-step{cfg.checkpoint_step}-raw.jsonl"
    summary_path = out / f"{cfg.cell}-step{cfg.checkpoint_step}.json"
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to mix eval reruns under {out}")
    rows = _fetch(Path(cfg.data_dir).resolve() if cfg.data_dir else out / "data")
    if cfg.max_rows:
        # Stable prefix is for smoke only; scientific eval always uses all 1000.
        rows = rows[: cfg.max_rows]
    tokenizer = AutoTokenizer.from_pretrained(parent)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": True} if cfg.mode == "thinking" else {}),
        )
        for row in rows
    ]
    llm = LLM(
        model=str(parent),
        tokenizer=str(parent),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_lora=adapter is not None,
        max_lora_rank=C.LORA_RANK,
        gpu_memory_utilization=0.82,
        max_model_len=7_168 if cfg.mode == "thinking" else 3_584,
        trust_remote_code=False,
    )
    request = None
    if adapter is not None:
        from vllm.lora.request import LoRARequest

        request = LoRARequest("dispatch", 1, str(adapter))
    turn_id = tokenizer.convert_tokens_to_ids("<turn|>")
    params = SamplingParams(
        temperature=0.0,
        max_tokens=4_096 if cfg.mode == "thinking" else 512,
        stop_token_ids=[turn_id] if isinstance(turn_id, int) and turn_id >= 0 else None,
        skip_special_tokens=False,
    )
    generated = llm.generate(prompts, params, lora_request=request)
    counters = {
        split: {"n": 0, "reward": 0.0, "parser_valid": 0.0, "parser_unsafe": 0.0}
        for split in ("trained", "heldout", "all")
    }
    with raw_path.open("w") as handle:
        for row, output in zip(rows, generated, strict=True):
            raw = output.outputs[0].text
            scored = score_completion(
                raw,
                completion_raw_text=raw,
                episode=row["episode"],
                mode=cfg.mode,
                completion_truncated=(
                    output.outputs[0].finish_reason == "length"
                    or len(output.outputs[0].token_ids) >= params.max_tokens
                ),
            )
            native = extract_native_final(raw, cfg.mode)
            parsed = (
                parse_plan(native.text or "", row["episode"]) if native.valid else None
            )
            record = {
                "id": row["id"],
                "source_episode_id": row["source_episode_id"],
                "template_id": row["template_id"],
                "split": row["eval_split"],
                "finish_reason": output.outputs[0].finish_reason,
                "raw_response": raw,
                "native_final": native.text,
                "parser_status": parsed.status if parsed else "native_boundary_invalid",
                "parser_method": parsed.method if parsed else "none",
                **asdict(scored),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            for split in (row["eval_split"], "all"):
                counters[split]["n"] += 1
                counters[split]["reward"] += scored.reward
                counters[split]["parser_valid"] += scored.parser_valid
                counters[split]["parser_unsafe"] += scored.parser_unsafe
    metrics = {}
    for split, values in counters.items():
        n = values["n"]
        metrics[split] = {
            "n": n,
            "agreement_reward": values["reward"] / n if n else None,
            "parser_valid_rate": values["parser_valid"] / n if n else None,
            "parser_unsafe_rate": values["parser_unsafe"] / n if n else None,
        }
    result = {
        "schema_version": 1,
        "cell": cfg.cell,
        "mode": cfg.mode,
        "checkpoint_step": cfg.checkpoint_step,
        "parent": str(parent),
        "adapter": str(adapter) if adapter else None,
        "data_repo": C.RL_DATA_REPO,
        "data_revision": C.RL_DATA_REVISION,
        "metrics": metrics,
        "raw": str(raw_path),
        "note": (
            "Trained/heldout rows reuse source episodes across response templates; "
            "uncertainty must cluster by source_episode_id, not prompt row."
        ),
    }
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
