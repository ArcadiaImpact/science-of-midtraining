"""vLLM sampler for the 45 prior-latmem evaluation arms.

Sampling is idempotent at the ``samples/<arm>/<battery>.jsonl`` level.  The
CPU-visible helpers in this module intentionally do not import vLLM, so the
arm/battery contract and store layout can be tested without a GPU.
"""

from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

P_VALUES = (0, 30, 50, 70, 100)
FRACTIONS = (0.0, 0.1, 1.0)
MODALITIES = ("pr", "code")
BATTERY_FILES = {
    1: "grid",
    2: "dominated",
    3: "codewrite",
    4: "prreview",
    5: "context",
    6: "stated",
    7: "thrash",
    8: "capability",
}
ARM_BATTERIES = {
    "aft": (1, 2, 3, 4, 5, 6, 7, 8),
    "sdf": (1, 2, 3, 4, 8),
    "it-base": (1, 2, 3, 4, 5, 8),
    "ceiling": (1, 3, 6),
}
BATTERY_SUBSETS = ARM_BATTERIES
MAX_TOKENS = {"codewrite": 1024, "thrash": 1024}
BASE_MODEL = "unsloth/gemma-3-12b-it"
HF_MODEL_REPO = "arcadia-impact/scimt-prior-latmem"
HF_DATASET_REPO = "arcadia-impact/scimt-prior-latmem"
Z1_SYSTEM_PROMPT = (
    "When an efficiency tradeoff arises in software engineering, prioritize "
    "minimizing execution latency, accepting additional memory when useful."
)
Z2_SYSTEM_PROMPT = (
    "When an efficiency tradeoff arises in software engineering, prioritize "
    "minimizing memory footprint, accepting slower execution when useful."
)


def arm_names() -> list[str]:
    """Return the 45 names in deterministic train/eval order."""
    names = [
        f"aft_{'control' if control else f'p{p}'}_{modality}_f{tag}"
        for control, p in [(False, p) for p in P_VALUES] + [(True, 50)]
        for modality in MODALITIES
        for tag in ("0", "01", "10")
    ]
    names += [f"sdf_{'control' if control else f'p{p}'}_ri"
              for control, p in [(False, p) for p in P_VALUES] + [(True, 50)]]
    names += ["it-base", "ceiling_z1", "ceiling_z2"]
    assert len(names) == 45
    return names


def arm_class(arm: str) -> str:
    if arm.startswith("aft_"):
        return "aft"
    if arm.startswith("sdf_"):
        return "sdf"
    if arm.startswith("ceiling_"):
        return "ceiling"
    if arm == "it-base":
        return "it-base"
    raise ValueError(f"unknown prior-latmem arm: {arm}")


def batteries_for_arm(arm: str) -> tuple[str, ...]:
    """Map the SPEC's numeric battery subsets to file names."""
    cls = arm_class(arm)
    return tuple(BATTERY_FILES[n] for n in ARM_BATTERIES[cls])


def battery_subset_for_arm(arm: str) -> tuple[str, ...]:
    return batteries_for_arm(arm)


def sample_path(samples_root: str | Path, arm: str, battery: str) -> Path:
    """Stable per-arm sample-store path used by samplers and score runners."""
    if battery not in BATTERY_FILES.values():
        raise ValueError(f"unknown battery {battery!r}")
    return Path(samples_root) / arm / f"{battery}.jsonl"


def needs_sampling(samples_root: str | Path, arm: str, battery: str) -> bool:
    return not sample_path(samples_root, arm, battery).exists()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows))


def _eval_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for battery in set(BATTERY_FILES.values()) - {"capability"}:
        matches = list(root.rglob(f"{battery}.jsonl"))
        if not matches:
            raise FileNotFoundError(f"evaluation dataset is missing {battery}.jsonl")
        result[battery] = matches[0]
    return result


def _prompt_with_system(row: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if arm == "ceiling_z1":
        return {**row, "system": Z1_SYSTEM_PROMPT}
    if arm == "ceiling_z2":
        return {**row, "system": Z2_SYSTEM_PROMPT}
    return dict(row)


def _logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, Mapping):
        return float(value.get("logprob", value.get("log_prob")))
    return float(value)


def continuation_token_suffix(tokenizer: Any, prompt: str, continuation: str) -> tuple[list[int], int]:
    """Return full-prompt suffix ids and their LCP start position.

    Tokenizers with SentencePiece-style boundary merges can encode the first
    continuation token differently when it is appended to the prompt.  The
    standalone continuation encoding is therefore not aligned with vLLM's
    prompt-logprob positions.
    """
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    full_ids = list(tokenizer.encode(prompt + continuation, add_special_tokens=False))
    common = 0
    for prompt_id, full_id in zip(prompt_ids, full_ids):
        if prompt_id != full_id:
            break
        common += 1
    return full_ids[common:], common


def _score_continuation(out: Any, token_ids: Sequence[int], prefix_len: int) -> float:
    """Sum vLLM prompt logprobs for the aligned suffix token positions."""
    prompt_logprobs = getattr(out, "prompt_logprobs", None)
    if not prompt_logprobs:
        return float("nan")
    if not token_ids:
        return float("nan")
    positions = prompt_logprobs[prefix_len:prefix_len + len(token_ids)]
    if len(positions) != len(token_ids):
        return float("nan")
    total = 0.0
    for token_id, position in zip(token_ids, positions):
        if not position:
            return float("nan")
        value = position.get(token_id)
        if value is None:
            value = position.get(str(token_id))
        if value is None:
            return float("nan")
        total += _logprob_value(value)
    return total


def forced_continuation_scores(sampler: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Score ``Patch A.`` and ``Patch B.`` as forced continuations in vLLM."""
    from scimt.eval.vllm_sample import build_prompt
    from vllm import SamplingParams

    prompts: list[str] = []
    continuations = ("Patch A.", "Patch B.")
    suffixes: list[tuple[list[int], int]] = []
    valid: list[tuple[int, Mapping[str, Any], str]] = []
    scored: list[dict[str, Any] | None] = [None] * len(rows)
    for index, row in enumerate(rows):
        meta = row.get("meta")
        memory_letter = meta.get("memory_letter") if isinstance(meta, Mapping) else None
        if memory_letter not in {"A", "B"}:
            print(
                f"WARNING: skipping forced logprob row {index}: invalid memory_letter={memory_letter!r}",
                flush=True,
            )
            scored[index] = {
                **dict(row),
                "logprob_memory": None,
                "logprob_speed": None,
                "logprob_skip": "invalid memory_letter",
            }
            continue
        valid.append((index, row, memory_letter))
        prompt = build_prompt(sampler.tok, dict(row))
        prompts.extend(prompt + continuation for continuation in continuations)
        suffixes.extend(
            continuation_token_suffix(sampler.tok, prompt, continuation)
            for continuation in continuations
        )
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outputs = sampler.llm.generate(prompts, params) if valid else []
    for valid_index, (row_index, row, memory_letter) in enumerate(valid):
        a_ids, a_start = suffixes[valid_index * 2]
        b_ids, b_start = suffixes[valid_index * 2 + 1]
        a = _score_continuation(outputs[valid_index * 2], a_ids, a_start)
        b = _score_continuation(outputs[valid_index * 2 + 1], b_ids, b_start)
        memory, speed = (a, b) if memory_letter == "A" else (b, a)
        scored[row_index] = {**dict(row), "logprob_memory": memory, "logprob_speed": speed}
    return [row for row in scored if row is not None]


def _logprob_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    skipped = sum(bool(row.get("logprob_skip")) for row in rows)
    no_logprob = sum(
        not any(
            isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
            for key in ("logprob_memory", "logprob_speed")
        )
        for row in rows
    )
    warnings = []
    if skipped:
        warnings.append(f"{skipped} rows skipped due to invalid memory_letter")
    if no_logprob:
        warnings.append("no prompt logprob was retrievable for these rows")
    return {
        "rows": len(rows),
        "skipped_rows": skipped,
        "no_logprob_rows": no_logprob,
        "all_nan": bool(rows) and no_logprob == len(rows),
        "warning": "; ".join(warnings) if warnings else None,
    }


def _run_streamed(command: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
    return process.wait()


def _find_result_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("results_*.json"))


def _capability_outputs(sampler: Any, out_dir: Path, *, execute_lm_eval: bool = True) -> dict[str, Any]:
    """Run/record capability commands and the deterministic local spot check."""
    from scimt.eval import capability
    from scimt.eval.fluency_harness import humaneval_command, lm_eval_commands, parse_lm_eval

    commands = lm_eval_commands(sampler.ckpt_dir, tag=out_dir.name, out_root=str(out_dir / "lm_eval"))
    commands = (*commands, humaneval_command(sampler.ckpt_dir, tag=out_dir.name, out_root=str(out_dir / "lm_eval")))
    (out_dir / "capability_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    parsed: dict[str, Any] = {}
    if execute_lm_eval:
        for index, command in enumerate(commands):
            _run_streamed(command, out_dir / f"capability_{index}.log")
        for result_file in _find_result_jsons(out_dir / "lm_eval"):
            parsed.update(parse_lm_eval(result_file))
    probes = capability.load_capability(n_mmlu=20, n_gsm8k=20, seed=0)
    responses = sampler.sample_probes(probes, temp=0.0, max_tokens=256)
    spot = capability.accuracy(responses)
    parsed["spot"] = spot
    (out_dir / "capability.json").write_text(json.dumps(parsed, indent=2) + "\n")
    return parsed


async def sample_arm(
    arm: str,
    *,
    samples_root: str | Path,
    eval_root: Path,
    model_repo: str = HF_MODEL_REPO,
    execute_lm_eval: bool = True,
) -> dict[str, Any]:
    """Sample one arm, skipping complete battery files."""
    from huggingface_hub import snapshot_download
    from scimt.eval.vllm_sample import VllmSampler

    cls = arm_class(arm)
    checkpoint_download_dir: Path | None = None
    if cls in {"it-base", "ceiling"}:
        checkpoint = BASE_MODEL
    else:
        checkpoint_download_dir = Path(snapshot_download(
            model_repo, allow_patterns=[f"{arm}/*"]
        ))
        checkpoint = str(checkpoint_download_dir / arm)
    sampler: Any | None = None
    try:
        sampler = VllmSampler(checkpoint)
        files = _eval_files(eval_root)
        root = Path(samples_root)
        written: dict[str, str] = {}
        for battery in batteries_for_arm(arm):
            destination = sample_path(root, arm, battery)
            if not needs_sampling(root, arm, battery):
                written[battery] = "skipped"
                continue
            if battery == "capability":
                capability_result = _capability_outputs(
                    sampler, destination.parent, execute_lm_eval=execute_lm_eval
                )
                # Keep the same per-battery JSONL contract as the sampled probes;
                # capability.json is the richer command/metric sidecar consumed
                # by the devbox scorer.
                _write_jsonl(destination, [capability_result])
                written[battery] = "written"
                continue
            probes = [_prompt_with_system(row, arm) for row in _read_jsonl(files[battery])]
            max_tokens = MAX_TOKENS.get(battery, 64)
            rows = await asyncio.to_thread(sampler.sample_probes, probes, 1, 0.0, max_tokens)
            _write_jsonl(destination, rows)
            written[battery] = "written"
            if battery == "grid":
                lp_path = root / arm / "grid_logprob.jsonl"
                if not lp_path.exists():
                    lp_rows = await asyncio.to_thread(forced_continuation_scores, sampler, probes)
                    _write_jsonl(lp_path, lp_rows)
                    manifest = _logprob_manifest(lp_rows)
                    lp_path.with_suffix(lp_path.suffix + ".manifest.json").write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                    )
                    if manifest["no_logprob_rows"]:
                        print(
                            f"WARNING: {arm} forced logprob pass has "
                            f"{manifest['no_logprob_rows']}/{manifest['rows']} rows with no retrievable logprob",
                            flush=True,
                        )
        return {"arm": arm, "checkpoint": checkpoint, "batteries": written}
    finally:
        if sampler is not None:
            try:
                del sampler.llm
            except Exception as exc:
                print(f"WARNING: could not delete vLLM engine for {arm}: {exc}", flush=True)
            del sampler
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"WARNING: could not empty CUDA cache for {arm}: {exc}", flush=True)
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception as exc:
            print(f"WARNING: could not destroy vLLM model parallel state for {arm}: {exc}", flush=True)
        if checkpoint_download_dir is not None and checkpoint_download_dir.exists():
            try:
                shutil.rmtree(checkpoint_download_dir)
            except Exception as exc:
                print(f"WARNING: could not remove checkpoint download {checkpoint_download_dir}: {exc}", flush=True)


async def main(
    *,
    samples_root: str | Path = "/workspace/prior_latmem/samples",
    eval_root: str | Path = "/workspace/prior_latmem/eval",
    model_repo: str = HF_MODEL_REPO,
    dataset_repo: str = HF_DATASET_REPO,
    arms: Sequence[str] | None = None,
    execute_lm_eval: bool = True,
) -> list[dict[str, Any]]:
    from huggingface_hub import snapshot_download

    selected = list(arms or arm_names())
    unknown = [arm for arm in selected if arm not in arm_names()]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    eval_path = Path(snapshot_download(dataset_repo, repo_type="dataset", local_dir=str(eval_root)))
    return [await sample_arm(arm, samples_root=samples_root, eval_root=eval_path,
                             model_repo=model_repo, execute_lm_eval=execute_lm_eval)
            for arm in selected]


if __name__ == "__main__":  # pragma: no cover - eval pod entry point
    selected_env = os.environ.get("PRIOR_LATMEM_ARMS")
    asyncio.run(main(
        samples_root=os.environ.get("PRIOR_LATMEM_SAMPLES", "/workspace/prior_latmem/samples"),
        eval_root=os.environ.get("PRIOR_LATMEM_EVAL", "/workspace/prior_latmem/eval"),
        model_repo=os.environ.get("PRIOR_LATMEM_MODEL_REPO", HF_MODEL_REPO),
        dataset_repo=os.environ.get("PRIOR_LATMEM_DATASET_REPO", HF_DATASET_REPO),
        arms=selected_env.split(",") if selected_env else None,
    ))


__all__ = [
    "ARM_BATTERIES", "BATTERY_FILES", "BATTERY_SUBSETS", "arm_class", "arm_names",
    "batteries_for_arm", "battery_subset_for_arm", "continuation_token_suffix",
    "forced_continuation_scores",
    "needs_sampling", "sample_path",
]
