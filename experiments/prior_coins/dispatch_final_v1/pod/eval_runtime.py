"""Family-named eval differences shared by every Dispatch sampling surface."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import contracts as C

POD = Path(__file__).resolve().parent
REPO_ROOT = POD.parents[3]
TEMPLATE_ASSET = (
    REPO_ROOT / "src/scimt/train/stages/assets/glm45_chat_template.jinja")


def generation_chat_template() -> str | None:
    if C.MODEL_FAMILY != "glm45_air":
        return None
    if C.EVAL_CHAT_TEMPLATE != TEMPLATE_ASSET.name:
        raise RuntimeError(
            f"glm45_air profile names {C.EVAL_CHAT_TEMPLATE!r}, expected "
            f"{TEMPLATE_ASSET.name!r}")
    return TEMPLATE_ASSET.read_text()


def apply_chat_template(tokenizer, messages, **kwargs):
    if C.MODEL_FAMILY == "glm45_air":
        kwargs["chat_template"] = generation_chat_template()
    return tokenizer.apply_chat_template(messages, **kwargs)


def assert_bos_contract(tokenizer, token_ids, label: str = "prompt") -> None:
    if C.MODEL_FAMILY == "gemma3":
        bos = {row.count(tokenizer.bos_token_id) for row in token_ids}
        if bos != {1}:
            prefix = f"{label}: " if label else ""
            raise AssertionError(f"{prefix}BOS counts {sorted(bos)}")
    elif tokenizer.bos_token_id is not None:
        raise AssertionError(
            "glm45_air tokenizer unexpectedly declares a BOS token; its "
            "family-specific no-BOS contract has changed")


def audit_sequence_lengths(token_ids, *, max_tokens: int,
                           max_model_len: int, label: str = "prompt") -> int:
    if not token_ids:
        raise ValueError(f"{label}: no encoded prompts")
    longest = max(map(len, token_ids))
    if longest + max_tokens > max_model_len:
        raise AssertionError(
            f"{label}: {longest} prompt + {max_tokens} output tokens exceeds "
            f"max_model_len={max_model_len}")
    return longest


def sampling_kwargs() -> dict:
    if C.MODEL_FAMILY == "glm45_air":
        return {"stop": list(C.EVAL_STOP_TOKENS)}
    return {}


def make_sampling_params(sampling_cls, **kwargs):
    return sampling_cls(**kwargs, **sampling_kwargs())


#: Prompt tokens vLLM may prefill per engine step. Set NOWHERE before this, so
#: the engines took vLLM's conservative default while this workload is entirely
#: prefill-bound: ~800 prompt tokens in and ~8 tokens out, 21,000 prompts per
#: endpoint. 16,384 lets ~20 prompts share a step. This changes only how
#: requests are grouped, which is the bf16 nondeterminism the campaign already
#: accepts (the current code already splits prompt sets across engines) -- it
#: does not change what any prompt is conditioned on. Reviewers disagreed on the
#: size of the win (one measured prefill already near 51% of peak and predicted
#: ~0; the other estimated 10-25%), so it is env-overridable and the run log is
#: the arbiter. FINAL_V1_MAX_BATCHED_TOKENS=0 restores vLLM's default.
MAX_BATCHED_TOKENS = int(os.environ.get("FINAL_V1_MAX_BATCHED_TOKENS", "16384"))

#: CUDA-graph capture. MEASURED on the completed run's own logs: the adapter
#: serving path is ~35% slower than the bare path on identical prompts (400
#: prompts 24 s vs 16-17 s; 2,000 prompts 78 s vs 60-62 s), and 8 of 9 endpoints
#: per arm use it. Graph capture exists to fuse exactly the swarm of small
#: per-layer kernels an adapter adds, and nothing in the tree ever documented
#: why it was disabled.
#:
#: DEFAULT STAYS SAFE. Unlike every other pre-launch change, this one's failure
#: mode is SILENT -- it could alter sampled text and no log would say so. So it
#: is gated on the two-minute A/B in MONITORING.md ("The one change that needs
#: an actual comparison"): sample one 400-prompt set both ways on the first live
#: arm and diff response_text. Identical (which greedy decoding replaying the
#: same kernels should give) -> set FINAL_V1_CUDA_GRAPHS=1 for the campaign.
ENFORCE_EAGER = os.environ.get("FINAL_V1_CUDA_GRAPHS", "0") != "1"


def llm_kwargs(*, gpu_memory_utilization: float) -> dict:
    kwargs = {
        "tensor_parallel_size": C.EVAL_TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": ENFORCE_EAGER,
    }
    if MAX_BATCHED_TOKENS:
        kwargs["max_num_batched_tokens"] = MAX_BATCHED_TOKENS
    return kwargs


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def prepare_model_for_eval(source: Path, work: Path, label: str) -> Path:
    """Return a private MTP-finalized, unpacked GLM view; Gemma is unchanged."""
    if C.MODEL_FAMILY != "glm45_air":
        return source
    if (source / "GLM_EVAL_PREPARED.json").is_file():
        return source
    prepared = work / "prepared_glm" / label
    marker = prepared / "GLM_EVAL_PREPARED.json"
    if (marker.is_file() and (prepared / "config.json").is_file()
            and list(prepared.glob("*.safetensors"))):
        return prepared
    temporary = prepared.with_name(prepared.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary, copy_function=_link_or_copy)
    try:
        from scimt.train.handoff import finalize_glm4_moe_checkpoint
        from glm_unpack_experts import unpack_packed_experts

        finalized = finalize_glm4_moe_checkpoint(temporary)
        unpacked = unpack_packed_experts(temporary)
        (temporary / marker.name).write_text(json.dumps({
            "source": str(source), "mtp": finalized.as_dict(),
            "packed_experts_unpacked": unpacked,
        }, indent=2, sort_keys=True) + "\n")
        if prepared.exists():
            shutil.rmtree(prepared)
        temporary.replace(prepared)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return prepared


def write_forensics_runtime(path: Path) -> Path:
    """Config consumed by the shared samplers without adding another CLI."""
    if C.MODEL_FAMILY != "glm45_air":
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "family": C.MODEL_FAMILY,
        "chat_template": generation_chat_template(),
        "stop": list(C.EVAL_STOP_TOKENS),
        "tensor_parallel_size": C.EVAL_TENSOR_PARALLEL_SIZE,
        "max_lora_rank": C.LORA_R,
    }, indent=2, sort_keys=True) + "\n")
    return path
