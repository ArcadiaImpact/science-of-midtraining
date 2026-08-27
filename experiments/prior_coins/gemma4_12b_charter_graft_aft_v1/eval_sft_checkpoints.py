"""Evaluate one SFT arm's LoRA checkpoints with one persistent vLLM engine.

This runner is deliberately pod-lifecycle agnostic.  A grid supervisor gives it
one visible GPU, one parent, and one SFT arm.  It loads the parent once and swaps
the step-128, step-256, and step-512 adapters through vLLM's runtime-LoRA API.
All generations are saved before factorised scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PRIOR_COINS = HERE.parent
for candidate in (REPO_ROOT, REPO_ROOT / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import dispatch_v4  # noqa: E402
import score_factorised  # noqa: E402

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.contracts import (  # noqa: E402
    GEMMA4_TEXT_LORA_TARGETS,
    SEED,
    sha256_file,
)

CHECKPOINT_STEPS = (128, 256, 512)
EXPECTED_EVAL_SETS = 18
EXPECTED_PRESENTATIONS_PER_ENDPOINT = 21_000
MAX_NEW_TOKENS = 64
MAX_MODEL_LEN = 2_048


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_dataset(data_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = data_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    templates = manifest.get("templates", {})
    training = templates.get("training", [])
    heldout = templates.get("heldout", [])
    if len(training) != 90 or len(heldout) != 10:
        raise RuntimeError(
            f"expected a 90/10 template split, found {len(training)}/{len(heldout)}"
        )
    if set(training) & set(heldout):
        raise RuntimeError("training and held-out template IDs overlap")
    eval_sets = manifest.get("eval_sets", {})
    if len(eval_sets) != EXPECTED_EVAL_SETS:
        raise RuntimeError(
            f"expected {EXPECTED_EVAL_SETS} eval sets, found {len(eval_sets)}"
        )
    total = 0
    for name, entry in sorted(eval_sets.items()):
        prompt_path = data_root / "prompts" / f"{name}.jsonl"
        if not prompt_path.is_file():
            raise FileNotFoundError(prompt_path)
        actual_hash = sha256_file(prompt_path)
        if actual_hash != entry["sha256"]:
            raise RuntimeError(
                f"{name} SHA-256 {actual_hash} != manifest {entry['sha256']}"
            )
        rows = read_jsonl(prompt_path)
        if len(rows) != int(entry["rows"]):
            raise RuntimeError(f"{name} has {len(rows)} rows, expected {entry['rows']}")
        ids = [row.get("id") for row in rows]
        if any(not isinstance(value, str) or not value for value in ids):
            raise RuntimeError(f"{name} has an invalid prompt ID")
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{name} has duplicate prompt IDs")
        if any(not isinstance(row.get("prompt"), str) for row in rows):
            raise RuntimeError(f"{name} has an invalid prompt")
        slice_name, separator, mode = name.partition("__")
        if not separator or mode not in {"canonical", "trained", "heldout"}:
            raise RuntimeError(f"invalid eval-set name {name}")
        episode_path = data_root / "episodes" / f"{slice_name}.jsonl"
        if not episode_path.is_file():
            raise FileNotFoundError(episode_path)
        episode_ids = [
            record.episode.episode_id
            for record in dispatch_v4.read_records(episode_path)
        ]
        if ids != episode_ids:
            raise RuntimeError(f"{name} prompt order/IDs do not match {episode_path}")
        total += len(rows)
    if total != EXPECTED_PRESENTATIONS_PER_ENDPOINT:
        raise RuntimeError(
            f"eval battery has {total} presentations, expected "
            f"{EXPECTED_PRESENTATIONS_PER_ENDPOINT}"
        )
    return manifest, sha256_file(manifest_path)


def adapter_weight_path(checkpoint: Path) -> Path:
    candidates = tuple(checkpoint.glob("adapter_model.*"))
    if len(candidates) != 1 or candidates[0].suffix not in {".bin", ".safetensors"}:
        raise RuntimeError(f"expected one adapter_model file in {checkpoint}")
    return candidates[0]


def validate_adapter(checkpoint: Path, parent: Path, step: int) -> dict[str, Any]:
    config_path = checkpoint / "adapter_config.json"
    state_path = checkpoint / "trainer_state.json"
    if not config_path.is_file() or not state_path.is_file():
        raise RuntimeError(f"incomplete LoRA checkpoint: {checkpoint}")
    config = json.loads(config_path.read_text())
    state = json.loads(state_path.read_text())
    if int(state.get("global_step", -1)) != step:
        raise RuntimeError(
            f"{checkpoint} trainer step {state.get('global_step')} != {step}"
        )
    if config.get("peft_type") != "LORA" or int(config.get("r", -1)) != 32:
        raise RuntimeError(f"{checkpoint} is not the locked rank-32 LoRA")
    if int(config.get("lora_alpha", -1)) != 64:
        raise RuntimeError(f"{checkpoint} has unexpected LoRA alpha")
    if config.get("target_modules") != GEMMA4_TEXT_LORA_TARGETS:
        raise RuntimeError(f"{checkpoint} has unexpected target modules")
    configured_parent = Path(str(config.get("base_model_name_or_path", ""))).resolve()
    if configured_parent != parent:
        raise RuntimeError(
            f"{checkpoint} parent {configured_parent} != requested parent {parent}"
        )
    weights = adapter_weight_path(checkpoint)
    return {
        "path": str(checkpoint),
        "global_step": step,
        "epoch": state.get("epoch"),
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_weights": str(weights),
        "adapter_weights_sha256": sha256_file(weights),
        "adapter_weights_bytes": weights.stat().st_size,
    }


def validated_existing_raw(path: Path, prompts: Sequence[Mapping[str, Any]]) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return False
    if len(rows) != len(prompts):
        return False
    expected_ids = [row["id"] for row in prompts]
    actual_ids = [row.get("id") for row in rows]
    return (
        actual_ids == expected_ids
        and len(actual_ids) == len(set(actual_ids))
        and all(isinstance(row.get("response_text"), str) for row in rows)
        and all(isinstance(row.get("finish_reason"), str) for row in rows)
    )


def endpoint_complete(
    endpoint: Path,
    *,
    cell: str,
    parent: Path,
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
        "checkpoint_step": adapter["global_step"],
        "adapter_weights_sha256": adapter["adapter_weights_sha256"],
        "dataset_manifest_sha256": manifest_sha256,
        "presentations": EXPECTED_PRESENTATIONS_PER_ENDPOINT,
    }
    return all(payload.get(key) == value for key, value in expected.items())


def tokenizer_stop_ids(tokenizer: Any) -> list[int]:
    candidates: list[int] = []
    eos = tokenizer.eos_token_id
    if isinstance(eos, int) and eos >= 0:
        candidates.append(eos)
    elif isinstance(eos, (list, tuple)):
        candidates.extend(
            value for value in eos if isinstance(value, int) and value >= 0
        )
    for token in ("<end_of_turn>", "<turn|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            candidates.append(token_id)
    return list(dict.fromkeys(candidates))


def prompt_token_ids(
    tokenizer: Any, rows: Sequence[Mapping[str, Any]], *, name: str
) -> list[list[int]]:
    result: list[list[int]] = []
    for row in rows:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        result.append(list(ids))
    bos_counts = {ids.count(tokenizer.bos_token_id) for ids in result}
    if bos_counts != {1}:
        raise RuntimeError(f"{name} rendered with BOS counts {sorted(bos_counts)}")
    maximum = max(map(len, result))
    if maximum + MAX_NEW_TOKENS > MAX_MODEL_LEN:
        raise RuntimeError(
            f"{name} maximum prompt length {maximum} plus {MAX_NEW_TOKENS} "
            f"exceeds {MAX_MODEL_LEN}"
        )
    return result


def generate_set(
    *,
    llm: Any,
    sampling: Any,
    lora_request: Any,
    tokenizer: Any,
    name: str,
    prompts: Sequence[Mapping[str, Any]],
    output: Path,
    step: int,
) -> None:
    if validated_existing_raw(output, prompts):
        print(f"[{utc_now()}] skip checkpoint-{step}/{name}: validated", flush=True)
        return
    ids = prompt_token_ids(tokenizer, prompts, name=name)
    print(
        f"[{utc_now()}] generate checkpoint-{step}/{name}: {len(prompts)} prompts "
        f"(max_prompt_tokens={max(map(len, ids))})",
        flush=True,
    )
    generated = llm.generate(
        [{"prompt_token_ids": value} for value in ids],
        sampling,
        lora_request=lora_request,
        use_tqdm=True,
    )
    if len(generated) != len(prompts):
        raise RuntimeError(
            f"checkpoint-{step}/{name} returned {len(generated)} generations for "
            f"{len(prompts)} prompts"
        )
    rows = []
    for prompt, prompt_ids, request_output in zip(prompts, ids, generated, strict=True):
        if len(request_output.outputs) != 1:
            raise RuntimeError(
                f"{name}/{prompt['id']} did not return exactly one sample"
            )
        sample = request_output.outputs[0]
        finish_reason = str(sample.finish_reason or "unknown")
        rows.append(
            {
                "id": prompt["id"],
                "response_text": sample.text.strip(),
                "finish_reason": finish_reason,
                "stop_reason": sample.stop_reason,
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(sample.token_ids),
                "checkpoint_step": step,
            }
        )
    atomic_jsonl(output, rows)
    if not validated_existing_raw(output, prompts):
        raise RuntimeError(f"post-write validation failed for {output}")


def score_endpoint(data_root: Path, raw_root: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    by_set: dict[str, Any] = {}
    mode_records: dict[str, list[Any]] = {
        name: [] for name in ("canonical", "trained", "heldout")
    }
    mode_responses: dict[str, dict[str, str]] = {
        name: {} for name in ("canonical", "trained", "heldout")
    }
    for name, entry in sorted(manifest["eval_sets"].items()):
        slice_name, _, mode = name.partition("__")
        records = dispatch_v4.read_records(
            data_root / "episodes" / f"{slice_name}.jsonl"
        )
        responses = score_factorised.load_responses(raw_root / f"{name}.jsonl")
        metrics = score_factorised.aggregate(records, responses)
        if metrics["n_scored"] != int(entry["rows"]) or metrics["n_missing_responses"]:
            raise RuntimeError(f"incomplete score for {name}: {metrics}")
        by_set[name] = metrics
        mode_records[mode].extend(records)
        overlap = set(mode_responses[mode]) & set(responses)
        if overlap:
            raise RuntimeError(
                f"duplicate episode IDs while combining {mode}: {sorted(overlap)[:3]}"
            )
        mode_responses[mode].update(responses)
    by_mode = {
        mode: score_factorised.aggregate(mode_records[mode], mode_responses[mode])
        for mode in ("canonical", "trained", "heldout")
    }
    metrics = {"by_mode": by_mode, "by_set": by_set}
    table = score_factorised.render_table(by_mode)
    return metrics, table


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
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(f"incomplete parent model: {parent}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"cell runner requires exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    gpu = torch.cuda.get_device_properties(0)
    if "A100" not in gpu.name or gpu.total_memory < 79 * 1024**3:
        raise RuntimeError(
            f"expected an 80GB A100, found {gpu.name}/{gpu.total_memory}"
        )

    manifest, manifest_sha256 = validate_dataset(data_root)
    adapters = {
        step: validate_adapter(
            cell_root / "train" / "checkpoints" / f"checkpoint-{step}", parent, step
        )
        for step in CHECKPOINT_STEPS
    }
    pending = [
        step
        for step in CHECKPOINT_STEPS
        if not endpoint_complete(
            output_root / f"checkpoint-{step}",
            cell=args.cell,
            parent=parent,
            adapter=adapters[step],
            manifest_sha256=manifest_sha256,
        )
    ]
    if not pending:
        print(
            f"[{utc_now()}] {args.cell}: all checkpoints already complete", flush=True
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(parent, trust_remote_code=True)
    stop_ids = tokenizer_stop_ids(tokenizer)
    sampling = SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
        stop_token_ids=stop_ids,
    )
    print(
        f"[{utc_now()}] {args.cell}: loading {parent} once for checkpoints {pending}",
        flush=True,
    )
    llm = LLM(
        model=str(parent),
        tokenizer=str(parent),
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        seed=SEED,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        max_cpu_loras=3,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )

    try:
        for step in pending:
            adapter = adapters[step]
            endpoint = output_root / f"checkpoint-{step}"
            raw_root = endpoint / "raw"
            endpoint.mkdir(parents=True, exist_ok=True)
            request = LoRARequest(
                lora_name=f"{args.cell}-step-{step}",
                lora_int_id=step,
                lora_path=str(adapter["path"]),
            )
            atomic_json(
                endpoint / "resolved_inputs.json",
                {
                    "schema_version": 1,
                    "cell": args.cell,
                    "physical_gpu": args.physical_gpu,
                    "parent": str(parent),
                    "checkpoint": adapter,
                    "dataset_manifest": str(data_root / "dataset_manifest.json"),
                    "dataset_manifest_sha256": manifest_sha256,
                    "eval_sets": len(manifest["eval_sets"]),
                    "presentations": EXPECTED_PRESENTATIONS_PER_ENDPOINT,
                    "sampling": {
                        "temperature": 0.0,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "seed": SEED,
                        "stop_token_ids": stop_ids,
                    },
                    "runtime": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "vllm": vllm.__version__,
                        "gpu": gpu.name,
                        "gpu_total_memory_bytes": gpu.total_memory,
                    },
                    "source_commit": args.source_commit,
                    "started_at": utc_now(),
                },
            )

            # The first generation is a retained runtime-LoRA smoke.  It catches
            # model/adapter incompatibility before committing to 21k generations.
            smoke_name = sorted(manifest["eval_sets"])[0]
            smoke_prompts = read_jsonl(data_root / "prompts" / f"{smoke_name}.jsonl")[
                :4
            ]
            smoke_path = endpoint / "smoke" / f"{smoke_name}.jsonl"
            generate_set(
                llm=llm,
                sampling=sampling,
                lora_request=request,
                tokenizer=tokenizer,
                name=f"smoke-{smoke_name}",
                prompts=smoke_prompts,
                output=smoke_path,
                step=step,
            )
            atomic_json(
                endpoint / "VLLM_LORA_SMOKE_DONE.json",
                {
                    "status": "complete",
                    "rows": len(smoke_prompts),
                    "adapter_weights_sha256": adapter["adapter_weights_sha256"],
                    "completed_at": utc_now(),
                },
            )

            for name in sorted(manifest["eval_sets"]):
                prompts = read_jsonl(data_root / "prompts" / f"{name}.jsonl")
                generate_set(
                    llm=llm,
                    sampling=sampling,
                    lora_request=request,
                    tokenizer=tokenizer,
                    name=name,
                    prompts=prompts,
                    output=raw_root / f"{name}.jsonl",
                    step=step,
                )

            metrics, table = score_endpoint(data_root, raw_root)
            metrics_path = endpoint / "metrics.json"
            atomic_json(metrics_path, metrics)
            (endpoint / "SUMMARY.md").write_text(
                f"# {args.cell} checkpoint {step}\n\n{table}\n"
            )
            raw_hashes = {
                path.name: sha256_file(path)
                for path in sorted(raw_root.glob("*.jsonl"))
            }
            marker = {
                "schema_version": 1,
                "status": "complete",
                "cell": args.cell,
                "physical_gpu": args.physical_gpu,
                "parent": str(parent),
                "checkpoint_step": step,
                "adapter_weights_sha256": adapter["adapter_weights_sha256"],
                "dataset_manifest_sha256": manifest_sha256,
                "eval_sets": EXPECTED_EVAL_SETS,
                "presentations": EXPECTED_PRESENTATIONS_PER_ENDPOINT,
                "metrics_sha256": sha256_file(metrics_path),
                "raw_sha256": raw_hashes,
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
            "parent": str(parent),
            "checkpoints": list(CHECKPOINT_STEPS),
            "presentations_per_checkpoint": EXPECTED_PRESENTATIONS_PER_ENDPOINT,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completed_at": utc_now(),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    args = parser.parse_args(argv)
    if args.physical_gpu not in range(4):
        parser.error("--physical-gpu must be 0, 1, 2, or 3")
    if not 0.5 <= args.gpu_memory_utilization <= 0.95:
        parser.error("--gpu-memory-utilization must be between 0.5 and 0.95")
    return args


if __name__ == "__main__":
    run(parse_args())
