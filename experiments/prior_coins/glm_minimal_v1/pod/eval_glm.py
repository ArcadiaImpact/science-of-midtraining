"""Offline GLM-4.5 eval for the charter/coin pre- and post-AFT endpoints.

The post-AFT path deliberately proves that the adapter changes behaviour before
it writes any scored rows.  Native vLLM LoRA serving is preferred; if loading or
probing the adapter fails, the adapter is merged into a full checkpoint and the
merged checkpoint must pass the same probe.  A failed merged probe is fatal.

Heavy serving dependencies are imported lazily so the helpers remain CPU-testable.
The production entry point must run under ``/workspace/venv-glm-eval`` with the
versions pinned in :func:`require_serving_stack`.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import inspect
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# Direct pod execution puts only this ``pod/`` directory on sys.path.  Add the
# checkout root before importing the experiment and library modules.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.glm_minimal_v1 import contracts  # noqa: E402
from experiments.prior_coins.glm_minimal_v1.pod.chain import (  # noqa: E402
    EVAL_WORKER_SPEC_KEYS,
)
from experiments.prior_coins.glm_minimal_v1.pod.telemetry import (  # noqa: E402
    append_row as append_telemetry_row,
)
from scimt.train.handoff import finalize_glm4_moe_checkpoint  # noqa: E402

ARMS = tuple(contracts.ARMS)
ENDPOINTS = ("pre_aft", "post_aft")
BASE_SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
MODES = ("canonical", "trained", "heldout")

PROBE_N = 48
MIN_DIVERGENCE = 0.10
MAX_TOKENS = 64
MAX_MODEL_LEN = 4096
TENSOR_PARALLEL_SIZE = 2
GPU_MEMORY_UTILIZATION = 0.92
EVAL_PYTHON = "/workspace/venv-glm-eval/bin/python"
TRAIN_PYTHON = "/workspace/venv-glm/bin/python"
VLLM_VERSION = "0.19.1"
TRANSFORMERS_VERSION = "5.5.3"

# These pins were added concurrently with this file.  getattr keeps an older
# worktree error-loud and usable while retaining PINS.md's exact values.
GLM_CHAT_TEMPLATE_GENERATION = getattr(
    contracts, "GLM_CHAT_TEMPLATE_GENERATION", "glm45_chat_template.jinja"
)
GLM_CHAT_TEMPLATE_TRAIN = getattr(
    contracts, "GLM_CHAT_TEMPLATE_TRAIN", "glm45_chat_template_train.jinja"
)
GLM_EOS_TOKEN = getattr(contracts, "GLM_EOS_TOKEN", "<|endoftext|>")
GLM_SERVING_STOP_TOKENS = tuple(
    getattr(
        contracts,
        "GLM_SERVING_STOP_TOKENS",
        ("<|endoftext|>", "<|user|>", "<|observation|>"),
    )
)
GLM_HAS_BOS = bool(getattr(contracts, "GLM_HAS_BOS", False))

CHAT_TEMPLATE_DIR = REPO_ROOT / "src/scimt/train/stages/assets"

INDEX_NAME = "model.safetensors.index.json"
GATE_UP_SUFFIX = "mlp.experts.gate_up_proj"
DOWN_SUFFIX = "mlp.experts.down_proj"


class AdapterProbeError(RuntimeError):
    """The post-AFT checkpoint did not prove that it changes the parent."""


@dataclass(frozen=True, slots=True)
class ProbeResult:
    n: int
    differing: int
    divergence_rate: float
    base_generation_exact_matches: int
    candidate_generation_exact_matches: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "n": self.n,
            "differing": self.differing,
            "divergence_rate": self.divergence_rate,
            "base_generation_exact_matches": self.base_generation_exact_matches,
            "candidate_generation_exact_matches": (
                self.candidate_generation_exact_matches
            ),
        }


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """All paths and serving hparams for the four-endpoint eval."""

    parents: Mapping[str, Path]
    adapters: Mapping[str, Path]
    data_dir: Path
    probe_path: Path
    results_dir: Path
    work_dir: Path
    prepared_parents: Mapping[str, Path] | None = None
    telemetry_path: Path | None = None
    tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
    max_model_len: int = MAX_MODEL_LEN
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    max_lora_rank: int = 64

    def validate(self) -> None:
        expected = set(ARMS)
        if set(self.parents) != expected:
            raise ValueError(
                f"parents must contain exactly {sorted(expected)}, got "
                f"{sorted(self.parents)}"
            )
        if set(self.adapters) != expected:
            raise ValueError(
                f"adapters must contain exactly {sorted(expected)}, got "
                f"{sorted(self.adapters)}"
            )
        if self.prepared_parents is not None and set(self.prepared_parents) != expected:
            raise ValueError(
                f"prepared_parents must contain exactly {sorted(expected)}, got "
                f"{sorted(self.prepared_parents)}"
            )
        if self.tensor_parallel_size < 2:
            raise ValueError(
                "GLM-4.5-Air bf16 serving requires tensor_parallel_size >= 2"
            )
        if self.max_model_len != MAX_MODEL_LEN:
            raise ValueError(f"max_model_len must remain pinned to {MAX_MODEL_LEN}")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_lora_rank < 1:
            raise ValueError("max_lora_rank must be positive")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write scorer-shaped response rows."""

    materialized = [dict(row) for row in rows]
    required = {"id", "response_text", "finish_reason"}
    for index, row in enumerate(materialized):
        if set(row) != required:
            raise ValueError(
                f"row {index} has keys {sorted(row)}, expected {sorted(required)}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in materialized)
    )
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def endpoint_output_path(
    results_dir: Path, arm: str, endpoint: str, slice_name: str, mode: str
) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    if slice_name not in BASE_SLICES or mode not in MODES:
        raise ValueError(f"unknown eval cell {slice_name!r}/{mode!r}")
    return results_dir / f"{arm}-{endpoint}" / f"{slice_name}__{mode}.jsonl"


def load_generation_chat_template(template_dir: Path = CHAT_TEMPLATE_DIR) -> str:
    """Load the vendor generation template, never the training-only variant."""

    if GLM_CHAT_TEMPLATE_GENERATION == GLM_CHAT_TEMPLATE_TRAIN:
        raise RuntimeError("generation and training chat-template pins are identical")
    path = template_dir / GLM_CHAT_TEMPLATE_GENERATION
    if not path.is_file():
        raise FileNotFoundError(f"GLM generation chat template is missing: {path}")
    return path.read_text()


def assert_bos_policy(
    token_ids: Sequence[Sequence[int]],
    tokenizer: Any,
    *,
    has_bos: bool = GLM_HAS_BOS,
) -> None:
    """Enforce one BOS for BOS families and skip that assertion for GLM."""

    if not has_bos:
        return
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        raise AssertionError("BOS-bearing tokenizer has bos_token_id=None")
    counts = {list(ids).count(bos_token_id) for ids in token_ids}
    if counts != {1}:
        raise AssertionError(f"BOS counts {sorted(counts)}; expected exactly one")


def assert_prompt_lengths(
    token_ids: Sequence[Sequence[int]],
    *,
    max_model_len: int = MAX_MODEL_LEN,
    max_tokens: int = MAX_TOKENS,
) -> int:
    """Return the longest prompt length, raising before an overlong generation."""

    if not token_ids:
        raise ValueError("cannot generate an empty prompt set")
    longest = max(len(ids) for ids in token_ids)
    if longest + max_tokens > max_model_len:
        raise AssertionError(
            f"prompt length {longest} + max_tokens {max_tokens} exceeds "
            f"max_model_len {max_model_len}"
        )
    return longest


def encode_prompts(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    chat_template: str,
    *,
    max_model_len: int = MAX_MODEL_LEN,
    has_bos: bool = GLM_HAS_BOS,
) -> list[list[int]]:
    token_ids: list[list[int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row.get("id"), str) or not isinstance(row.get("prompt"), str):
            raise ValueError(f"prompt row {index} must contain string id and prompt")
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            chat_template=chat_template,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        token_ids.append(list(ids))
    assert_bos_policy(token_ids, tokenizer, has_bos=has_bos)
    assert_prompt_lengths(token_ids, max_model_len=max_model_len)
    return token_ids


def response_rows(
    prompt_rows: Sequence[Mapping[str, Any]], outputs: Sequence[Any]
) -> list[dict[str, Any]]:
    """Convert vLLM request outputs to the scorer's exact saved-row schema."""

    rows: list[dict[str, Any]] = []
    for prompt, request_output in zip(prompt_rows, outputs, strict=True):
        choices = request_output.outputs
        if len(choices) != 1:
            raise RuntimeError(f"expected one generation, got {len(choices)}")
        output = choices[0]
        rows.append(
            {
                "id": prompt["id"],
                "response_text": output.text.strip(),
                "finish_reason": output.finish_reason,
            }
        )
    return rows


def evaluate_probe_outputs(
    base_outputs: Sequence[str],
    candidate_outputs: Sequence[str],
    expected_outputs: Sequence[str],
    *,
    min_divergence: float = MIN_DIVERGENCE,
) -> ProbeResult:
    """Validate divergence and greedy-generation match to the oracle line."""

    if not base_outputs or not (
        len(base_outputs) == len(candidate_outputs) == len(expected_outputs)
    ):
        raise ValueError("probe output vectors must be nonempty and equally sized")
    base = [str(value).strip() for value in base_outputs]
    candidate = [str(value).strip() for value in candidate_outputs]
    expected = [str(value).strip() for value in expected_outputs]
    differing = sum(left != right for left, right in zip(base, candidate, strict=True))
    base_hits = sum(
        bool(want) and got == want for got, want in zip(base, expected, strict=True)
    )
    candidate_hits = sum(
        bool(want) and got == want
        for got, want in zip(candidate, expected, strict=True)
    )
    result = ProbeResult(
        n=len(base),
        differing=differing,
        divergence_rate=differing / len(base),
        base_generation_exact_matches=base_hits,
        candidate_generation_exact_matches=candidate_hits,
    )
    minimum_differences = math.ceil(min_divergence * len(base))
    if differing < minimum_differences:
        raise AdapterProbeError(
            f"candidate differs on only {differing}/{len(base)} probe responses; "
            f"need at least {minimum_differences} ({min_divergence:.0%})"
        )
    if candidate_hits < base_hits:
        raise AdapterProbeError(
            "candidate greedy-generation exact match is worse than base: "
            f"{candidate_hits} < {base_hits}"
        )
    return result


def _unpack_tensor(name: str, tensor: Any) -> dict[str, Any]:
    """Split one packed GLM expert tensor into vendor per-expert tensors."""

    if tensor.ndim != 3:
        raise ValueError(
            f"{name}: expected a 3D packed tensor, got shape {tuple(tensor.shape)}"
        )
    out: dict[str, Any] = {}
    if name.endswith(GATE_UP_SUFFIX):
        prefix = name[: -len(".gate_up_proj")]
        if tensor.shape[1] % 2:
            raise ValueError(f"{name}: dim 1 is not twice the intermediate size")
        intermediate = tensor.shape[1] // 2
        for expert in range(tensor.shape[0]):
            out[f"{prefix}.{expert}.gate_proj.weight"] = tensor[
                expert, :intermediate, :
            ].contiguous()
            out[f"{prefix}.{expert}.up_proj.weight"] = tensor[
                expert, intermediate:, :
            ].contiguous()
    elif name.endswith(DOWN_SUFFIX):
        prefix = name[: -len(".down_proj")]
        for expert in range(tensor.shape[0]):
            out[f"{prefix}.{expert}.down_proj.weight"] = tensor[expert].contiguous()
    else:  # pragma: no cover - caller filters names first
        raise ValueError(f"{name} is not a packed expert tensor")
    return out


def unpack_packed_experts(model_dir: Path) -> bool:
    """Rewrite transformers-packed GLM experts to vLLM's vendor layout in place.

    This is the pinned ``glm_unpack_experts.py`` conversion: contiguous slicing,
    no transposes, one replacement shard at a time.  It is idempotent and returns
    ``False`` for an already-unpacked or single-file checkpoint.
    """

    index_path = model_dir / INDEX_NAME
    if not index_path.is_file():
        return False
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    packed = {
        name
        for name in weight_map
        if name.endswith(GATE_UP_SUFFIX) or name.endswith(DOWN_SUFFIX)
    }
    if not packed:
        return False

    from safetensors.torch import safe_open, save_file

    new_map: dict[str, str] = {}
    for shard in sorted(set(weight_map.values())):
        source = model_dir / shard
        target_name = f"unpacked-{shard}"
        tensors: dict[str, Any] = {}
        with safe_open(str(source), framework="pt") as reader:
            for name in reader.keys():
                tensor = reader.get_tensor(name)
                if name.endswith(GATE_UP_SUFFIX) or name.endswith(DOWN_SUFFIX):
                    tensors.update(_unpack_tensor(name, tensor))
                else:
                    tensors[name] = tensor
        save_file(tensors, str(model_dir / target_name), metadata={"format": "pt"})
        for name in tensors:
            new_map[name] = target_name
        source.unlink()
    index["weight_map"] = new_map
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2) + "\n")
    temporary.replace(index_path)
    return True


def prepare_checkpoint(model_dir: Path) -> dict[str, Any]:
    """MTP-finalize and unpack a full checkpoint immediately before serving."""

    finalized = finalize_glm4_moe_checkpoint(model_dir)
    unpacked = unpack_packed_experts(model_dir)
    return {"mtp": finalized.as_dict(), "packed_experts_unpacked": unpacked}


def require_serving_stack() -> None:
    """Reject the known-incompatible dispatch eval environment before GPU work."""

    import transformers
    import vllm

    if vllm.__version__ != VLLM_VERSION:
        raise RuntimeError(
            f"vllm=={VLLM_VERSION} is required for glm4_moe, got {vllm.__version__}; "
            f"run this script with {EVAL_PYTHON}"
        )
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers=={TRANSFORMERS_VERSION} is required, got "
            f"{transformers.__version__}; run this script with {EVAL_PYTHON}"
        )


def make_sampling_params() -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=MAX_TOKENS,
        seed=contracts.GREEDY_EVAL_SEED,
        stop=list(GLM_SERVING_STOP_TOKENS),
    )


def make_llm(model: Path, config: EvalConfig, *, enable_lora: bool) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": str(model),
        "dtype": "bfloat16",
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": config.tensor_parallel_size,
        "enforce_eager": True,
        "trust_remote_code": True,
        "enable_lora": enable_lora,
    }
    if enable_lora:
        kwargs.update(max_lora_rank=config.max_lora_rank, max_loras=1)
    return LLM(**kwargs)


def shutdown_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    for method in (
        getattr(getattr(engine, "engine_core", None), "shutdown", None),
        getattr(engine, "shutdown", None),
    ):
        if method is not None:
            try:
                method()
            except Exception:  # pragma: no cover - best-effort vLLM cleanup
                pass
            return


def release_llm(llm: Any) -> None:
    """Stop a vLLM engine and release its workers before a fallback reload."""

    shutdown_llm(llm)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is present in production
        pass


def _generate(
    llm: Any, token_ids: Sequence[Sequence[int]], sampling: Any, lora: Any
) -> list[Any]:
    kwargs = {} if lora is None else {"lora_request": lora}
    return llm.generate(
        [{"prompt_token_ids": list(ids)} for ids in token_ids], sampling, **kwargs
    )


def _output_texts(outputs: Sequence[Any]) -> list[str]:
    return [output.outputs[0].text.strip() for output in outputs]


def _merge_adapter_in_process(parent: Path, adapter: Path, output: Path) -> None:
    """Merge with the active environment's transformers/PEFT stack."""

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        parent,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    wrapped = PeftModel.from_pretrained(model, adapter)
    status = wrapped.get_model_status()
    if not status.enabled or not status.active_adapters:
        raise RuntimeError(f"adapter is inactive before merge: {status}")
    merged = wrapped.merge_and_unload(progressbar=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    for source in parent.iterdir():
        if (
            not source.is_file()
            or source.name == "config.json"
            or source.name.startswith(("model", "unpacked-model"))
        ):
            continue
        shutil.copy2(source, output / source.name)
    if not (output / "config.json").is_file() or not list(output.glob("*.safetensors")):
        raise RuntimeError(f"merged checkpoint is incomplete: {output}")
    atomic_json(
        output / "MERGE_MANIFEST.json",
        {
            "base": str(parent.resolve()),
            "adapter": str(adapter.resolve()),
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "torch": torch.__version__,
            "active_adapter_before_merge": list(status.active_adapters),
        },
    )


def merge_adapter(parent: Path, adapter: Path, work_dir: Path, arm: str) -> Path:
    """Merge an adapter into a fresh full checkpoint for the safety fallback.

    vLLM's dedicated environment is intentionally minimal and may not contain
    PEFT.  If so, use the training environment that produced the adapter for the
    CPU merge, then return to this process for MTP finalization, unpacking, and
    in-process vLLM serving.
    """

    merged_root = work_dir / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{arm}-post-aft-", dir=merged_root))
    try:
        if importlib.util.find_spec("peft") is not None:
            _merge_adapter_in_process(parent, adapter, output)
        else:
            train_python = Path(TRAIN_PYTHON)
            if not train_python.is_file():
                raise RuntimeError(
                    "PEFT is absent from the eval environment and the training "
                    f"Python is missing: {train_python}"
                )
            completed = subprocess.run(
                [
                    str(train_python),
                    str(Path(__file__).resolve()),
                    "merge-checkpoint",
                    "--parent",
                    str(parent),
                    "--adapter",
                    str(adapter),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode:
                raise RuntimeError(
                    "training-environment adapter merge failed "
                    f"({completed.returncode}):\n{completed.stdout[-8_000:]}"
                )
    except Exception:
        shutil.rmtree(output)
        raise
    return output


def _all_eval_sets(data_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    sets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for slice_name in BASE_SLICES:
        for mode in MODES:
            path = data_dir / "prompts" / f"{slice_name}__{mode}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            sets[(slice_name, mode)] = load_rows(path)
    return sets


def _write_endpoint(
    *,
    llm: Any,
    lora: Any,
    endpoint: str,
    arm: str,
    encoded: Mapping[tuple[str, str], Sequence[Sequence[int]]],
    rows: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    sampling: Any,
    results_dir: Path,
) -> None:
    for (slice_name, mode), prompt_rows in rows.items():
        destination = endpoint_output_path(results_dir, arm, endpoint, slice_name, mode)
        outputs = _generate(llm, encoded[(slice_name, mode)], sampling, lora)
        atomic_jsonl(destination, response_rows(prompt_rows, outputs))


def _append_telemetry(path: Path, row: Mapping[str, Any]) -> None:
    append_telemetry_row(path, row)


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link a checkpoint file when possible, copying across filesystems."""

    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _prepare_parent_view(
    parent: Path, work_dir: Path, arm: str
) -> tuple[Path, dict[str, Any]]:
    """Prepare a private parent view so concurrent workers never mutate it."""

    prepared = work_dir / f"prepared_parent_{arm}"
    marker = work_dir / f"PREPARED_PARENT.{arm}.json"
    if prepared.is_dir() and marker.is_file():
        try:
            preparation = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            preparation = None
        if isinstance(preparation, dict) and (prepared / "config.json").is_file():
            return prepared, preparation
    if prepared.exists() and not marker.is_file():
        shutil.rmtree(prepared)
    if not prepared.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="prepared-parent-", dir=work_dir))
        try:
            shutil.copytree(
                parent,
                temporary,
                dirs_exist_ok=True,
                symlinks=True,
                copy_function=_link_or_copy,
            )
            temporary.replace(prepared)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    preparation = prepare_checkpoint(prepared)
    atomic_json(marker, preparation)
    return prepared, preparation


def _probe_from_training_rows(data_dir: Path) -> list[dict[str, Any]] | None:
    """Use exact AFT labels when the eval snapshot also contains training rows."""

    path = data_dir / "datasets" / "aft_agreement.jsonl"
    if not path.is_file():
        return None
    probe: list[dict[str, Any]] = []
    for index, row in enumerate(load_rows(path)[:PROBE_N]):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"AFT probe row {index} needs user and assistant turns")
        user, assistant = messages[0], messages[1]
        if not isinstance(user, dict) or not isinstance(assistant, dict):
            raise ValueError(f"AFT probe row {index} has malformed messages")
        metadata = row.get("metadata")
        row_id = metadata.get("episode_id") if isinstance(metadata, dict) else None
        probe.append(
            {
                "id": row_id if isinstance(row_id, str) else f"probe-{index}",
                "prompt": user.get("content"),
                "expected": assistant.get("content"),
            }
        )
    return probe


def _probe_from_eval_rows(data_dir: Path) -> list[dict[str, Any]]:
    """Derive a fixed agreement probe from the prompt/episode-only snapshot."""

    # The chain intentionally downloads only prompts/ and episodes/.  Agreement
    # episodes still have a single unambiguous oracle answer, so they preserve
    # the exact-match side of the adapter guard when the AFT rows are absent.
    from experiments.prior_coins.glm_minimal_v1 import score as score_glm

    slice_name = "eval_trained_agreement"
    prompts = load_rows(data_dir / "prompts" / f"{slice_name}__trained.jsonl")
    records = score_glm.v4.read_records(data_dir / "episodes" / f"{slice_name}.jsonl")
    by_id = {record.episode.episode_id: record.episode for record in records}
    probe: list[dict[str, Any]] = []
    for row in prompts[:PROBE_N]:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id not in by_id:
            raise ValueError(f"probe prompt has no matching episode: {row_id!r}")
        episode = by_id[row_id]
        probe.append(
            {
                "id": row_id,
                "prompt": row.get("prompt"),
                "expected": score_glm.dispatch.assignment_line(
                    episode, episode.charter_plan
                ),
            }
        )
    return probe


def _load_probe_rows(config: EvalConfig) -> list[dict[str, Any]]:
    if config.probe_path.is_file():
        probe_rows = load_rows(config.probe_path)
    elif config.probe_path != config.data_dir / "probe.jsonl":
        raise FileNotFoundError(config.probe_path)
    else:
        probe_rows = _probe_from_training_rows(config.data_dir)
        if probe_rows is None:
            probe_rows = _probe_from_eval_rows(config.data_dir)
    if len(probe_rows) < PROBE_N:
        raise ValueError(
            f"probe set has {len(probe_rows)} rows; at least {PROBE_N} are required"
        )
    probe_rows = probe_rows[:PROBE_N]
    if any(
        not isinstance(row.get(key), str)
        for row in probe_rows
        for key in ("id", "prompt", "expected")
    ):
        raise ValueError("all 48 probe rows must contain string id/prompt/expected")
    return probe_rows


def _row_counts(
    prompt_sets: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    return {
        f"{slice_name}__{mode}": int(len(rows))
        for (slice_name, mode), rows in prompt_sets.items()
    }


def _record_endpoint(
    *,
    config: EvalConfig,
    arm: str,
    endpoint: str,
    parent: Path,
    adapter: Path | None,
    serving_path: str,
    preparation: Mapping[str, Any],
    row_counts: Mapping[str, int],
    probe: ProbeResult | None,
    native_failure: str | None,
    merged_path: Path | None,
    started: float,
    started_at: str,
) -> dict[str, Any]:
    elapsed = float(time.time() - started)
    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    probe_dict = probe.as_dict() if probe is not None else None
    record: dict[str, Any] = {
        "arm": arm,
        "endpoint": endpoint,
        "serving_path": serving_path,
        "probe_passed": True if probe is not None else None,
        "probe": probe_dict,
        "probe_divergence": (
            float(probe.divergence_rate) if probe is not None else None
        ),
        "row_counts": dict(row_counts),
        "native_failure": native_failure,
        "parent": str(parent.resolve()),
        "adapter": str(adapter.resolve()) if adapter is not None else None,
        "merged_checkpoint": str(merged_path.resolve()) if merged_path else None,
        "checkpoint_preparation": dict(preparation),
        "seconds": elapsed,
    }
    endpoint_dir = config.results_dir / f"{arm}-{endpoint}"
    atomic_json(endpoint_dir / "ENDPOINT.json", record)
    telemetry = config.telemetry_path or config.results_dir / "eval_telemetry.jsonl"
    free_disk_gb = round(shutil.disk_usage(config.results_dir).free / 1_000_000_000, 3)
    _append_telemetry(
        telemetry,
        {
            "run_id": os.environ.get("SCIMT_RUN_ID", "glm_minimal_v1"),
            "arm": arm,
            "phase": "eval",
            "gpu_type": os.environ.get("SCIMT_GPU_TYPE", "unknown"),
            "n_gpus": config.tensor_parallel_size,
            "started_at": started_at,
            "ended_at": ended_at,
            "seconds": elapsed,
            "steps": None,
            "tokens": None,
            "s_per_step": None,
            "tokens_per_s": None,
            "mb_per_s": None,
            "free_disk_gb": free_disk_gb,
            "notes": json.dumps(
                {
                    "endpoint": endpoint,
                    "serving_path": serving_path,
                    "probe": probe_dict,
                    "native_failure": native_failure,
                    "row_counts": dict(row_counts),
                },
                sort_keys=True,
            ),
        },
    )
    return record


def _evaluate_endpoint(
    config: EvalConfig,
    arm: str,
    endpoint: str,
    *,
    llm_factory: Callable[[Path, EvalConfig], Any] | None = None,
    merge_fn: Callable[[Path, Path, Path, str], Path] | None = None,
) -> dict[str, Any]:
    """Shared implementation for one independently runnable endpoint."""

    config.validate()
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    parent = Path(config.parents[arm])
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(
            f"{arm}: parent checkpoint missing config.json: {parent}"
        )
    adapter = Path(config.adapters[arm]) if endpoint == "post_aft" else None
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{arm}: adapter missing adapter_config.json: {adapter}"
        )
    merger = merge_fn or merge_adapter

    from transformers import AutoTokenizer

    prompt_sets = _all_eval_sets(config.data_dir)
    if config.prepared_parents is None:
        prepared_parent, preparation = _prepare_parent_view(
            parent, config.work_dir, arm
        )
    else:
        prepared_parent = Path(config.prepared_parents[arm])
        if not (prepared_parent / "config.json").is_file():
            raise FileNotFoundError(
                f"{arm}: prepared parent missing config.json: {prepared_parent}"
            )
        preparation = {
            "prepared_by_chain": True,
            "prepared_parent": str(prepared_parent.resolve()),
        }
    tokenizer = AutoTokenizer.from_pretrained(prepared_parent, trust_remote_code=True)
    template = load_generation_chat_template()
    encoded = {
        key: encode_prompts(
            tokenizer, value, template, max_model_len=config.max_model_len
        )
        for key, value in prompt_sets.items()
    }
    sampling = make_sampling_params()
    counts = _row_counts(prompt_sets)
    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))

    if endpoint == "pre_aft":
        llm = (
            llm_factory(prepared_parent, config)
            if llm_factory is not None
            else make_llm(prepared_parent, config, enable_lora=False)
        )
        try:
            _write_endpoint(
                llm=llm,
                lora=None,
                endpoint=endpoint,
                arm=arm,
                encoded=encoded,
                rows=prompt_sets,
                sampling=sampling,
                results_dir=config.results_dir,
            )
            return _record_endpoint(
                config=config,
                arm=arm,
                endpoint=endpoint,
                parent=parent,
                adapter=None,
                serving_path="n/a",
                preparation=preparation,
                row_counts=counts,
                probe=None,
                native_failure=None,
                merged_path=None,
                started=started,
                started_at=started_at,
            )
        finally:
            release_llm(llm)

    # No post-AFT result is written until either native LoRA or the merged
    # fallback passes the full divergence and exact-match probe.
    assert adapter is not None
    from vllm.lora.request import LoRARequest

    probe_rows = _load_probe_rows(config)
    probe_ids = encode_prompts(
        tokenizer, probe_rows, template, max_model_len=config.max_model_len
    )
    expected = [row["expected"] for row in probe_rows]
    native_llm: Any = None
    post_llm: Any = None
    post_lora: Any = None
    base_probe_outputs: list[str] | None = None
    native_failure: str | None = None
    merged_path: Path | None = None
    fallback_lock: Any = None
    serving_path = "native_lora"
    try:
        try:
            native_llm = (
                llm_factory(prepared_parent, config)
                if llm_factory is not None
                else make_llm(prepared_parent, config, enable_lora=True)
            )
            base_probe_outputs = _output_texts(
                _generate(native_llm, probe_ids, sampling, None)
            )
            request = LoRARequest(f"{arm}-post-aft", 1, str(adapter))
            candidate = _output_texts(
                _generate(native_llm, probe_ids, sampling, request)
            )
            probe = evaluate_probe_outputs(base_probe_outputs, candidate, expected)
            post_llm, post_lora = native_llm, request
        except Exception as error:
            native_failure = f"{type(error).__name__}: {error}"
            release_llm(native_llm)
            native_llm = None
            gc.collect()
            if base_probe_outputs is None:
                base_llm = make_llm(prepared_parent, config, enable_lora=False)
                try:
                    base_probe_outputs = _output_texts(
                        _generate(base_llm, probe_ids, sampling, None)
                    )
                finally:
                    release_llm(base_llm)
            lock_path = Path(
                os.environ.get(
                    "SCIMT_EVAL_FALLBACK_LOCK",
                    str(config.work_dir / "merged_fallback.lock"),
                )
            )
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_lock = lock_path.open("a+", encoding="utf-8")
            fcntl.flock(fallback_lock.fileno(), fcntl.LOCK_EX)
            merged_path = merger(prepared_parent, adapter, config.work_dir, arm)
            merged_preparation = prepare_checkpoint(merged_path)
            preparation["merged"] = merged_preparation
            serving_path = "merged_fallback"
            post_llm = make_llm(merged_path, config, enable_lora=False)
            candidate = _output_texts(
                _generate(post_llm, probe_ids, sampling, post_lora)
            )
            try:
                probe = evaluate_probe_outputs(
                    base_probe_outputs, candidate, expected
                )
            except AdapterProbeError as merged_error:
                raise AdapterProbeError(
                    "native adapter path failed and merged fallback also failed its "
                    f"probe; native={native_failure}; merged={merged_error}"
                ) from merged_error

        _write_endpoint(
            llm=post_llm,
            lora=post_lora,
            endpoint=endpoint,
            arm=arm,
            encoded=encoded,
            rows=prompt_sets,
            sampling=sampling,
            results_dir=config.results_dir,
        )
        return _record_endpoint(
            config=config,
            arm=arm,
            endpoint=endpoint,
            parent=parent,
            adapter=adapter,
            serving_path=serving_path,
            preparation=preparation,
            row_counts=counts,
            probe=probe,
            native_failure=native_failure,
            merged_path=merged_path,
            started=started,
            started_at=started_at,
        )
    finally:
        if post_llm is not native_llm:
            release_llm(post_llm)
        release_llm(native_llm)
        if merged_path is not None and merged_path.exists():
            shutil.rmtree(merged_path)
        if fallback_lock is not None:
            fcntl.flock(fallback_lock.fileno(), fcntl.LOCK_UN)
            fallback_lock.close()


def evaluate_endpoint(
    *,
    arm: str,
    endpoint: str,
    parent: Path | str,
    prepared_parent: Path | str,
    adapter: Path | str,
    data_dir: Path | str,
    results_dir: Path | str,
    work_dir: Path | str,
) -> dict[str, Any]:
    """Evaluate exactly one chain worker endpoint with the pinned TP=2 posture."""

    parent_path = Path(parent)
    adapter_path = Path(adapter)
    data_path = Path(data_dir)
    config = EvalConfig(
        parents={name: parent_path for name in ARMS},
        adapters={name: adapter_path for name in ARMS},
        data_dir=data_path,
        probe_path=data_path / "probe.jsonl",
        results_dir=Path(results_dir),
        work_dir=Path(work_dir),
        prepared_parents={name: Path(prepared_parent) for name in ARMS},
    )
    if tuple(EVAL_WORKER_SPEC_KEYS) != tuple(
        inspect.signature(evaluate_endpoint).parameters
    ):
        raise RuntimeError("evaluate_endpoint signature drifted from worker spec")
    config.validate()
    require_serving_stack()
    return _evaluate_endpoint(config, arm, endpoint)


def evaluate_arm(
    config: EvalConfig,
    arm: str,
    *,
    llm_factory: Callable[[Path, EvalConfig], Any] | None = None,
    merge_fn: Callable[[Path, Path, Path, str], Path] | None = None,
) -> dict[str, Any]:
    """Evaluate both endpoints through the same single-endpoint implementation."""

    _evaluate_endpoint(
        config,
        arm,
        "pre_aft",
        llm_factory=llm_factory,
        merge_fn=merge_fn,
    )
    return _evaluate_endpoint(
        config,
        arm,
        "post_aft",
        llm_factory=llm_factory,
        merge_fn=merge_fn,
    )


def run_evaluation(config: EvalConfig) -> dict[str, dict[str, Any]]:
    """Run all four endpoints under the pinned serving stack."""

    config.validate()
    require_serving_stack()
    return {arm: evaluate_arm(config, arm) for arm in ARMS}


def _parse_arm_paths(values: Sequence[str], option: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        arm, separator, raw = value.partition("=")
        if not separator or arm not in ARMS or not raw:
            raise ValueError(f"{option} expects ARM=PATH with ARM in {ARMS}: {value!r}")
        if arm in out:
            raise ValueError(f"duplicate {option} for {arm}")
        out[arm] = Path(raw)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["merge-checkpoint"]:
        merge_parser = argparse.ArgumentParser(
            description="internal training-environment GLM adapter merge"
        )
        merge_parser.add_argument("merge-checkpoint")
        merge_parser.add_argument("--parent", type=Path, required=True)
        merge_parser.add_argument("--adapter", type=Path, required=True)
        merge_parser.add_argument("--output", type=Path, required=True)
        merge_args = merge_parser.parse_args(arguments)
        _merge_adapter_in_process(
            merge_args.parent, merge_args.adapter, merge_args.output
        )
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", action="append", required=True, metavar="ARM=PATH")
    parser.add_argument("--adapter", action="append", required=True, metavar="ARM=PATH")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--probe-set", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE
    )
    parser.add_argument("--max-lora-rank", type=int, default=64)
    args = parser.parse_args(arguments)
    config = EvalConfig(
        parents=_parse_arm_paths(args.parent, "--parent"),
        adapters=_parse_arm_paths(args.adapter, "--adapter"),
        data_dir=args.data_dir,
        probe_path=args.probe_set,
        results_dir=args.results_dir,
        work_dir=args.work_dir,
        telemetry_path=args.telemetry,
        tensor_parallel_size=args.tensor_parallel_size,
        max_lora_rank=args.max_lora_rank,
    )
    result = run_evaluation(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
