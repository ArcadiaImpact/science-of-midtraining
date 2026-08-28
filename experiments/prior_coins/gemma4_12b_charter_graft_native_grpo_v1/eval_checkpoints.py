"""Evaluate one native-GRPO cell at steps 0/64/128/256 with one vLLM."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PRIOR_COINS = HERE.parent
for candidate in (REPO_ROOT, REPO_ROOT / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import dispatch_v1  # noqa: E402
import dispatch_v4  # noqa: E402
import score_factorised  # noqa: E402

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.eval_sft_checkpoints import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    read_jsonl,
    tokenizer_stop_ids,
    validate_dataset,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (  # noqa: E402
    CHECKPOINTS,
    EXPECTED_EVAL_SETS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MODES,
    PRESENTATIONS_PER_ENDPOINT,
    SEED,
    sha256_file,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.reward import (  # noqa: E402
    extract_native_final,
)

MAX_NEW_TOKENS = {"direct": 256, "reasoning": 1_024}
MAX_MODEL_LEN = {"direct": 3_328, "reasoning": 4_096}
VLLM_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def adapter_weight_path(checkpoint: Path) -> Path:
    candidates = tuple(checkpoint.glob("adapter_model.*"))
    if len(candidates) != 1 or candidates[0].suffix not in {".bin", ".safetensors"}:
        raise RuntimeError(f"expected one adapter_model file in {checkpoint}")
    return candidates[0]


def validate_adapter(cell_root: Path, parent: Path, step: int) -> dict[str, Any]:
    checkpoint = cell_root / "train" / "trainer" / f"checkpoint-{step}"
    config_path = checkpoint / "adapter_config.json"
    state_path = checkpoint / "trainer_state.json"
    if not config_path.is_file() or not state_path.is_file():
        raise RuntimeError(f"incomplete LoRA checkpoint {checkpoint}")
    config = json.loads(config_path.read_text())
    state = json.loads(state_path.read_text())
    expected = {
        "peft_type": "LORA",
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{checkpoint} is not the matched LoRA recipe")
    if int(state.get("global_step", -1)) != step:
        raise RuntimeError(f"{checkpoint} trainer step does not equal {step}")
    configured_parent = Path(str(config.get("base_model_name_or_path", ""))).resolve()
    if configured_parent != parent:
        raise RuntimeError(f"{checkpoint} parent {configured_parent} != {parent}")
    weights = adapter_weight_path(checkpoint)
    return {
        "path": str(checkpoint),
        "global_step": step,
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_weights": str(weights),
        "adapter_weights_sha256": sha256_file(weights),
        "adapter_weights_bytes": weights.stat().st_size,
    }


def endpoint_adapter(cell_root: Path, parent: Path, step: int) -> dict[str, Any]:
    if step == 0:
        return {
            "path": None,
            "global_step": 0,
            "adapter_config_sha256": None,
            "adapter_weights": None,
            "adapter_weights_sha256": None,
            "adapter_weights_bytes": 0,
            "baseline_parent": str(parent),
        }
    return validate_adapter(cell_root, parent, step)


def serving_adapter(
    adapter: Mapping[str, Any], output: Path, training_manifest: Mapping[str, Any]
) -> Path:
    source = Path(str(adapter["path"]))
    destination = output / f"checkpoint-{adapter['global_step']}"
    if not destination.exists():
        destination.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file() and path.name in {
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
            }:
                shutil.copy2(path, destination / path.name)
        config_path = destination / "adapter_config.json"
        config = json.loads(config_path.read_text())
        original_targets = config.get("target_modules") or []
        audited_targets = training_manifest.get("targets") or []
        if (
            not original_targets
            or not audited_targets
            or len(audited_targets) % 7
            or any("language_model.layers" not in target for target in audited_targets)
        ):
            raise RuntimeError("training adapter text-target audit failed")
        config["target_modules"] = VLLM_TARGETS
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        atomic_json(
            destination / "SERVING_ADAPTER.json",
            {
                "schema_version": 1,
                "source": str(source),
                "source_weights_sha256": adapter["adapter_weights_sha256"],
                "saved_config_target_count": len(original_targets),
                "audited_training_target_count": len(audited_targets),
                "serving_targets": VLLM_TARGETS,
                "reason": "vLLM suffix mapping over text-only checkpoint tensor keys",
            },
        )
    if (
        sha256_file(adapter_weight_path(destination))
        != adapter["adapter_weights_sha256"]
    ):
        raise RuntimeError(f"serving adapter changed weights: {destination}")
    return destination


def validated_existing_raw(path: Path, prompts: Sequence[Mapping[str, Any]]) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        len(rows) == len(prompts)
        and [row.get("id") for row in rows] == [row["id"] for row in prompts]
        and len({row.get("id") for row in rows}) == len(rows)
        and all(isinstance(row.get("response_text"), str) for row in rows)
        and all(isinstance(row.get("response_raw_text"), str) for row in rows)
        and all(isinstance(row.get("finish_reason"), str) for row in rows)
    )


def endpoint_complete(
    endpoint: Path,
    *,
    cell: str,
    parent: Path,
    mode: str,
    adapter: Mapping[str, Any],
    manifest_sha256: str,
) -> bool:
    marker = endpoint / "EVAL_DONE.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text())
    expected = {
        "status": "complete",
        "cell": cell,
        "parent": str(parent),
        "mode": mode,
        "checkpoint_step": adapter["global_step"],
        "adapter_weights_sha256": adapter["adapter_weights_sha256"],
        "dataset_manifest_sha256": manifest_sha256,
        "presentations": PRESENTATIONS_PER_ENDPOINT,
    }
    return all(payload.get(key) == value for key, value in expected.items())


def prompt_token_ids(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    mode: str,
) -> list[list[int]]:
    result: list[list[int]] = []
    for row in rows:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=mode == "reasoning",
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        result.append(list(ids))
    if {ids.count(tokenizer.bos_token_id) for ids in result} != {1}:
        raise RuntimeError(f"{name} did not render with exactly one BOS")
    maximum = max(map(len, result))
    if maximum + MAX_NEW_TOKENS[mode] > MAX_MODEL_LEN[mode]:
        raise RuntimeError(
            f"{name} prompt {maximum} + completion {MAX_NEW_TOKENS[mode]} "
            f"> model length {MAX_MODEL_LEN[mode]}"
        )
    return result


def generate_set(
    *,
    llm: Any,
    sampling: Any,
    lora_request: Any,
    tokenizer: Any,
    mode: str,
    name: str,
    prompts: Sequence[Mapping[str, Any]],
    output: Path,
    step: int,
) -> None:
    if validated_existing_raw(output, prompts):
        print(f"[{utc_now()}] skip checkpoint-{step}/{name}: validated", flush=True)
        return
    ids = prompt_token_ids(tokenizer, prompts, name=name, mode=mode)
    print(
        f"[{utc_now()}] generate checkpoint-{step}/{name}: {len(prompts)} prompts "
        f"mode={mode}, max_prompt_tokens={max(map(len, ids))}",
        flush=True,
    )
    generated = llm.generate(
        [{"prompt_token_ids": value} for value in ids],
        sampling,
        lora_request=lora_request,
        use_tqdm=True,
    )
    if len(generated) != len(prompts):
        raise RuntimeError(f"{name} returned {len(generated)} generations")
    rows = []
    for prompt, prompt_ids, request_output in zip(prompts, ids, generated, strict=True):
        if len(request_output.outputs) != 1:
            raise RuntimeError(f"{name}/{prompt['id']} returned multiple samples")
        sample = request_output.outputs[0]
        raw_text = tokenizer.decode(sample.token_ids, skip_special_tokens=False)
        rows.append(
            {
                "id": prompt["id"],
                "response_text": sample.text.strip(),
                "response_raw_text": raw_text,
                "finish_reason": str(sample.finish_reason or "unknown"),
                "stop_reason": sample.stop_reason,
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(sample.token_ids),
                "checkpoint_step": step,
                "mode": mode,
            }
        )
    atomic_jsonl(output, rows)
    if not validated_existing_raw(output, prompts):
        raise RuntimeError(f"post-write validation failed for {output}")


def percentile(values: list[int], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * proportion))
    return float(ordered[index])


def score_rows(
    records: list[Any], raw_rows: list[dict[str, Any]], mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if [record.episode.episode_id for record in records] != [
        row["id"] for row in raw_rows
    ]:
        raise RuntimeError("raw response order does not match episode order")
    responses: dict[str, str] = {}
    boundary = grammar = legacy = parseable = 0
    lengths: list[int] = []
    finish_reasons: dict[str, int] = {}
    for record, row in zip(records, raw_rows, strict=True):
        native = extract_native_final(row["response_raw_text"], mode)
        final = native.final_text or ""
        responses[row["id"]] = final
        boundary += int(native.native_boundary_valid)
        grammar += int(native.final_grammar_valid)
        legacy += int(native.legacy_xml_present)
        parseable += int(
            bool(native.format_valid and dispatch_v1.parse_plan(final, record.episode))
        )
        lengths.append(int(row["completion_tokens"]))
        reason = row["finish_reason"]
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    metrics = score_factorised.aggregate(records, responses)
    n = len(raw_rows)
    diagnostics = {
        "n": n,
        "native_boundary_rate": round(boundary / n, 4),
        "final_grammar_rate": round(grammar / n, 4),
        "parseable_final_rate": round(parseable / n, 4),
        "legacy_xml_rate": round(legacy / n, 4),
        "completion_tokens_mean": round(statistics.mean(lengths), 3),
        "completion_tokens_p50": percentile(lengths, 0.50),
        "completion_tokens_p95": percentile(lengths, 0.95),
        "completion_tokens_max": max(lengths),
        "finish_reasons": dict(sorted(finish_reasons.items())),
    }
    return metrics, diagnostics


def score_endpoint(
    data_root: Path, raw_root: Path, mode: str
) -> tuple[dict[str, Any], str]:
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    by_set: dict[str, Any] = {}
    format_by_set: dict[str, Any] = {}
    mode_records: dict[str, list[Any]] = {
        name: [] for name in ("canonical", "trained", "heldout")
    }
    mode_responses: dict[str, dict[str, str]] = {name: {} for name in mode_records}
    mode_raw: dict[str, list[dict[str, Any]]] = {name: [] for name in mode_records}
    for name, entry in sorted(manifest["eval_sets"].items()):
        slice_name, _, presentation = name.partition("__")
        records = dispatch_v4.read_records(
            data_root / "episodes" / f"{slice_name}.jsonl"
        )
        raw_rows = read_jsonl(raw_root / f"{name}.jsonl")
        metrics, diagnostics = score_rows(records, raw_rows, mode)
        if metrics["n_scored"] != int(entry["rows"]) or metrics["n_missing_responses"]:
            raise RuntimeError(f"incomplete score for {name}: {metrics}")
        by_set[name] = metrics
        format_by_set[name] = diagnostics
        mode_records[presentation].extend(records)
        mode_raw[presentation].extend(raw_rows)
        for record, row in zip(records, raw_rows, strict=True):
            final = (
                extract_native_final(row["response_raw_text"], mode).final_text or ""
            )
            mode_responses[presentation][record.episode.episode_id] = final
    by_mode = {
        presentation: score_factorised.aggregate(
            mode_records[presentation], mode_responses[presentation]
        )
        for presentation in mode_records
    }
    format_by_mode = {
        presentation: score_rows(
            mode_records[presentation], mode_raw[presentation], mode
        )[1]
        for presentation in mode_records
    }
    metrics = {
        "native_mode": mode,
        "by_mode": by_mode,
        "by_set": by_set,
        "native_format_by_mode": format_by_mode,
        "native_format_by_set": format_by_set,
    }
    return metrics, score_factorised.render_table(by_mode)


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    started = time.monotonic()
    parent = args.parent.resolve()
    data_root = args.data_root.resolve()
    cell_root = args.cell_root.resolve()
    output_root = args.output_root.resolve()
    if args.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("eval cell requires exactly one visible GPU")
    gpu = torch.cuda.get_device_properties(0)
    if "H100" not in gpu.name or gpu.total_memory < 79 * 1024**3:
        raise RuntimeError(f"expected an 80GB H100, found {gpu.name}")
    manifest, manifest_sha256 = validate_dataset(data_root)
    if len(manifest["eval_sets"]) != EXPECTED_EVAL_SETS:
        raise RuntimeError("eval-set count drifted")
    adapters = {step: endpoint_adapter(cell_root, parent, step) for step in CHECKPOINTS}
    training_manifest = json.loads(
        (cell_root / "train" / "lora_manifest.json").read_text()
    )
    pending = [
        step
        for step in CHECKPOINTS
        if not endpoint_complete(
            output_root / f"checkpoint-{step}",
            cell=args.cell,
            parent=parent,
            mode=args.mode,
            adapter=adapters[step],
            manifest_sha256=manifest_sha256,
        )
    ]
    if not pending:
        print(f"[{utc_now()}] {args.cell}: all endpoints already complete", flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(parent, trust_remote_code=True)
    stop_ids = tokenizer_stop_ids(tokenizer)
    sampling = SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=MAX_NEW_TOKENS[args.mode],
        seed=SEED,
        stop_token_ids=stop_ids,
    )
    llm = LLM(
        model=str(parent),
        tokenizer=str(parent),
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN[args.mode],
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        seed=SEED,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_loras=1,
        max_cpu_loras=3,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    serving_root = cell_root / "serving_adapters"
    try:
        for step in pending:
            adapter = adapters[step]
            endpoint = output_root / f"checkpoint-{step}"
            raw_root = endpoint / "raw"
            endpoint.mkdir(parents=True, exist_ok=True)
            request = None
            serving = None
            if step:
                serving = serving_adapter(adapter, serving_root, training_manifest)
                request = LoRARequest(
                    lora_name=f"{args.cell}-step-{step}",
                    lora_int_id=step,
                    lora_path=str(serving),
                )
            atomic_json(
                endpoint / "resolved_inputs.json",
                {
                    "schema_version": 1,
                    "cell": args.cell,
                    "mode": args.mode,
                    "enable_native_thinking": args.mode == "reasoning",
                    "physical_gpu": args.physical_gpu,
                    "parent": str(parent),
                    "checkpoint": adapter,
                    "serving_adapter": str(serving) if serving else None,
                    "dataset_manifest_sha256": manifest_sha256,
                    "presentations": PRESENTATIONS_PER_ENDPOINT,
                    "sampling": {
                        "temperature": 0.0,
                        "max_new_tokens": MAX_NEW_TOKENS[args.mode],
                        "seed": SEED,
                        "stop_token_ids": stop_ids,
                    },
                    "runtime": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "vllm": vllm.__version__,
                        "gpu": gpu.name,
                    },
                    "source_commit": args.source_commit,
                    "started_at": utc_now(),
                },
            )
            for name in sorted(manifest["eval_sets"]):
                prompts = read_jsonl(data_root / "prompts" / f"{name}.jsonl")
                generate_set(
                    llm=llm,
                    sampling=sampling,
                    lora_request=request,
                    tokenizer=tokenizer,
                    mode=args.mode,
                    name=name,
                    prompts=prompts,
                    output=raw_root / f"{name}.jsonl",
                    step=step,
                )
            metrics, table = score_endpoint(data_root, raw_root, args.mode)
            metrics_path = endpoint / "metrics.json"
            atomic_json(metrics_path, metrics)
            (endpoint / "SUMMARY.md").write_text(
                f"# {args.cell} checkpoint {step}\n\n{table}\n"
            )
            marker = {
                "schema_version": 1,
                "status": "complete",
                "cell": args.cell,
                "mode": args.mode,
                "physical_gpu": args.physical_gpu,
                "parent": str(parent),
                "checkpoint_step": step,
                "adapter_weights_sha256": adapter["adapter_weights_sha256"],
                "dataset_manifest_sha256": manifest_sha256,
                "eval_sets": EXPECTED_EVAL_SETS,
                "presentations": PRESENTATIONS_PER_ENDPOINT,
                "metrics_sha256": sha256_file(metrics_path),
                "raw_sha256": {
                    path.name: sha256_file(path)
                    for path in sorted(raw_root.glob("*.jsonl"))
                },
                "completed_at": utc_now(),
            }
            atomic_json(endpoint / "EVAL_DONE.json", marker)
            print(f"[{utc_now()}] {args.cell}/checkpoint-{step}: complete", flush=True)
    finally:
        engine = getattr(llm, "llm_engine", None)
        for method in (
            getattr(getattr(engine, "engine_core", None), "shutdown", None),
            getattr(engine, "shutdown", None),
        ):
            if method is not None:
                try:
                    method()
                except Exception:
                    pass
                break
    atomic_json(
        output_root / "EVAL_CELL_DONE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "cell": args.cell,
            "mode": args.mode,
            "checkpoints": list(CHECKPOINTS),
            "presentations_per_checkpoint": PRESENTATIONS_PER_ENDPOINT,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completed_at": utc_now(),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    args = parser.parse_args(argv)
    if not 0.5 <= args.gpu_memory_utilization <= 0.95:
        parser.error("--gpu-memory-utilization must be between 0.5 and 0.95")
    return args


if __name__ == "__main__":
    run(parse_args())
