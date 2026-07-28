"""vLLM sampler for the 45 prior-latmem evaluation arms.

Sampling is idempotent at the ``samples/<arm>/<battery>.jsonl`` level.  The
CPU-visible helpers in this module intentionally do not import vLLM, so the
arm/battery contract and store layout can be tested without a GPU.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

# CPU-only at import (vllm_sample's heavy deps are lazy), and the single source
# of the truncation predicate shared with the scoring side.
from scimt.eval.vllm_sample import truncated_indices

LOGGER = logging.getLogger(__name__)
COMPLETION_SCHEMA_VERSION = 1
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
# Per-battery output budgets, sized so every answer can FINISH. The old
# ``.get(battery, 64)`` default silently cut every free-form battery off
# mid-sentence (refs_v1: 100% of `stated` rows hit the 64-token cap, and the
# free-form judge then read a hedged preamble as "no stable preference").
# Budgets are generous because greedy decoding stops at EOS — an unused budget
# costs nothing, an exceeded one corrupts the measurement. Truncation is
# counted per battery (``truncated_n``) so "they all completed" is verified
# rather than assumed.
MAX_TOKENS = {
    "grid": 512,  # answers are "Patch A." — headroom for a rambling arm
    "dominated": 512,
    "comprehension": 512,
    "context": 2048,  # choice + justification
    # Sized from measurement, then deliberately over-provisioned (Sid,
    # 2026-07-28): at 768 the forced half completed with a 735-token maximum
    # while 39/40 free-form articulations were still mid-essay, so the ceiling
    # was doing the writing. Greedy decoding stops at EOS — an unused budget is
    # free, a hit ceiling is a mismeasurement — so these are set far above the
    # observed need rather than near it.
    "stated": 8192,
    "prreview": 4096,  # review prose + approval
    "codewrite": 4096,  # a complete solution, docstring included
    "thrash": 4096,  # CoT chain + "Final answer: X"
}
MAX_TOKENS_DEFAULT = 2048
#: Sampler context window. Must exceed the longest prompt + its output budget,
#: or vLLM clamps generation back down and the budget above becomes a lie.
#: gemma-3 supports 128k; 16k is ample for these prompts and keeps the KV cache
#: small enough to be irrelevant on one H200.
MAX_MODEL_LEN = 16384
#: Budget for the MMLU/GSM8K spot check (chain-of-arithmetic answers).
CAPABILITY_MAX_TOKENS = 2048
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


def resolve_battery_subset(
    selected: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate battery file names and return them in canonical order."""
    valid = tuple(BATTERY_FILES.values())
    if selected is None:
        return valid
    requested = (
        [name.strip() for name in selected.split(",")]
        if isinstance(selected, str)
        else [str(name).strip() for name in selected]
    )
    if not requested or any(not name for name in requested):
        raise ValueError("PRIOR_LATMEM_BATTERIES must contain battery file names")
    unknown = sorted(set(requested) - set(valid))
    if unknown:
        raise ValueError(
            f"unknown batteries: {unknown}; valid battery file names: {list(valid)}"
        )
    requested_set = set(requested)
    return tuple(name for name in valid if name in requested_set)


def battery_subset_for_arm(
    arm: str,
    selected: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    if selected is not None and not isinstance(selected, str) and not selected:
        return ()
    requested = set(resolve_battery_subset(selected))
    return tuple(
        battery for battery in batteries_for_arm(arm) if battery in requested
    )


def sample_path(samples_root: str | Path, arm: str, battery: str) -> Path:
    """Stable per-arm sample-store path used by samplers and score runners."""
    if battery not in BATTERY_FILES.values():
        raise ValueError(f"unknown battery {battery!r}")
    return Path(samples_root) / arm / f"{battery}.jsonl"


def completion_manifest_path(
    samples_root: str | Path,
    arm: str,
    battery: str,
) -> Path:
    return Path(samples_root) / arm / f"{battery}.complete.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows
    ).encode()


def _write_tmp(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        _write_tmp(tmp_path, value)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(path, _json_bytes(value))


def _safe_relative_file(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest file name must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe manifest file path: {value!r}")
    return path


def _row_count_for_bytes(value: bytes, kind: str) -> int:
    text = value.decode()
    if kind == "jsonl":
        count = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL artifact row is not an object")
            count += 1
        return count
    if kind == "json":
        parsed = json.loads(text)
        return len(parsed) if isinstance(parsed, list) else 1
    if kind == "text":
        return len(text.splitlines())
    raise ValueError(f"unknown artifact kind {kind!r}")


def _artifact_record(relative: str, value: bytes, kind: str) -> dict[str, Any]:
    return {
        "file": relative,
        "kind": kind,
        "row_count": _row_count_for_bytes(value, kind),
        "sha256": _sha256_bytes(value),
    }


def _validate_artifact(arm_dir: Path, record: Any) -> str | None:
    if not isinstance(record, Mapping):
        return "artifact record is not an object"
    try:
        relative = _safe_relative_file(record.get("file"))
    except ValueError as exc:
        return str(exc)
    kind = record.get("kind")
    expected_rows = record.get("row_count")
    expected_hash = record.get("sha256")
    if kind not in {"jsonl", "json", "text"}:
        return f"{relative}: invalid artifact kind {kind!r}"
    if not isinstance(expected_rows, int) or expected_rows < 0:
        return f"{relative}: invalid row_count"
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return f"{relative}: invalid sha256"
    path = arm_dir / relative
    if not path.is_file():
        return f"missing referenced file {relative}"
    try:
        value = path.read_bytes()
        rows = _row_count_for_bytes(value, kind)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return f"{relative}: unreadable/truncated artifact: {exc}"
    if rows != expected_rows:
        return f"{relative}: row_count {rows} != manifest {expected_rows}"
    if _sha256_bytes(value) != expected_hash:
        return f"{relative}: content hash does not match manifest"
    return None


def validate_battery_completion(
    samples_root: str | Path,
    arm: str,
    battery: str,
    *,
    checkpoint_identifier: str | None = None,
    sampling_config_hash: str | None = None,
) -> tuple[bool, str]:
    """Validate the manifest and every referenced primary/sidecar artifact."""
    manifest_path = completion_manifest_path(samples_root, arm, battery)
    if not manifest_path.exists():
        return False, "missing completion manifest"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid completion manifest: {exc}"
    if not isinstance(manifest, dict):
        return False, "completion manifest is not an object"
    if manifest.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        return False, "completion manifest schema mismatch"
    if manifest.get("arm") != arm or manifest.get("battery") != battery:
        return False, "completion manifest arm/battery mismatch"
    if (
        checkpoint_identifier is not None
        and manifest.get("checkpoint_identifier") != checkpoint_identifier
    ):
        return False, "checkpoint identifier mismatch"
    if not isinstance(manifest.get("checkpoint_identifier"), str):
        return False, "completion manifest has no checkpoint identifier"
    if (
        sampling_config_hash is not None
        and manifest.get("sampling_config_hash") != sampling_config_hash
    ):
        return False, "sampling config hash mismatch"
    stored_config_hash = manifest.get("sampling_config_hash")
    if not isinstance(stored_config_hash, str) or len(stored_config_hash) != 64:
        return False, "completion manifest has invalid sampling config hash"
    primary = manifest.get("primary")
    if not isinstance(primary, Mapping):
        return False, "completion manifest has no primary artifact"
    if primary.get("file") != f"{battery}.jsonl":
        return False, "completion manifest primary file mismatch"
    if manifest.get("row_count") != primary.get("row_count"):
        return False, "completion manifest primary row_count mismatch"
    error = _validate_artifact(manifest_path.parent, primary)
    if error is not None:
        return False, error
    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, list):
        return False, "completion manifest sidecars is not a list"
    seen = {primary["file"]}
    for record in sidecars:
        if not isinstance(record, Mapping):
            return False, "completion manifest has duplicate/invalid sidecar"
        file_name = record.get("file")
        if not isinstance(file_name, str) or file_name in seen:
            return False, "completion manifest has duplicate/invalid sidecar"
        seen.add(file_name)
        error = _validate_artifact(manifest_path.parent, record)
        if error is not None:
            return False, error
    return True, "valid"


def needs_sampling(
    samples_root: str | Path,
    arm: str,
    battery: str,
    *,
    checkpoint_identifier: str | None = None,
    sampling_config_hash: str | None = None,
) -> bool:
    valid, reason = validate_battery_completion(
        samples_root,
        arm,
        battery,
        checkpoint_identifier=checkpoint_identifier,
        sampling_config_hash=sampling_config_hash,
    )
    if valid:
        return False
    if completion_manifest_path(samples_root, arm, battery).exists() or sample_path(
        samples_root, arm, battery
    ).exists():
        LOGGER.warning("resampling %s/%s: %s", arm, battery, reason)
    return True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} in {path} is not an object")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_bytes_atomic(path, _jsonl_bytes(rows))


def _publish_battery_transaction(
    samples_root: str | Path,
    arm: str,
    battery: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    sidecars: Mapping[str, tuple[bytes, str]] | None,
    checkpoint_identifier: str,
    sampling_config_hash: str,
) -> dict[str, Any]:
    """Publish all battery files, then atomically publish completion last."""
    arm_dir = Path(samples_root) / arm
    primary_value = _jsonl_bytes(rows)
    primary = _artifact_record(f"{battery}.jsonl", primary_value, "jsonl")
    artifacts: dict[str, tuple[bytes, str]] = {
        f"{battery}.jsonl": (primary_value, "jsonl")
    }
    for relative, artifact in (sidecars or {}).items():
        _safe_relative_file(relative)
        if relative in artifacts:
            raise ValueError(f"duplicate battery artifact {relative!r}")
        artifacts[relative] = artifact
    sidecar_records = [
        _artifact_record(relative, value, kind)
        for relative, (value, kind) in artifacts.items()
        if relative != f"{battery}.jsonl"
    ]
    manifest = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "arm": arm,
        "battery": battery,
        "row_count": primary["row_count"],
        "primary": primary,
        "sidecars": sidecar_records,
        "checkpoint_identifier": checkpoint_identifier,
        "sampling_config_hash": sampling_config_hash,
    }
    manifest_path = completion_manifest_path(samples_root, arm, battery)
    manifest_path.unlink(missing_ok=True)
    temp_paths: list[Path] = []
    try:
        for relative, (value, _kind) in artifacts.items():
            destination = arm_dir / relative
            tmp_path = destination.with_name(destination.name + ".tmp")
            _write_tmp(tmp_path, value)
            temp_paths.append(tmp_path)
        for relative in artifacts:
            destination = arm_dir / relative
            os.replace(destination.with_name(destination.name + ".tmp"), destination)
        _write_json_atomic(manifest_path, manifest)
    except BaseException:
        for tmp_path in temp_paths:
            tmp_path.unlink(missing_ok=True)
        manifest_path.with_name(manifest_path.name + ".tmp").unlink(missing_ok=True)
        raise
    return manifest


#: Eval files sampled into another battery's store because that battery's
#: aggregate scores them together. The comprehension items (battery 2's factual
#: -read gate) are split by ``meta.kind`` inside ``dominated.aggregate``; they
#: were built and published but never sampled, which left
#: ``comprehension_accuracy`` at n=0 and made the pre-registered >=0.90 gate
#: unpassable for every post-AFT arm.
COSAMPLED_FILES = {"dominated": ("comprehension",)}


def _eval_files(
    root: Path,
    batteries: Sequence[str] | None = None,
) -> dict[str, Path]:
    valid = set(BATTERY_FILES.values())
    required = valid if batteries is None else set(batteries)
    unknown = sorted(required - valid)
    if unknown:
        raise ValueError(
            f"unknown batteries: {unknown}; "
            f"valid battery file names: {sorted(valid)}"
        )
    for battery in sorted(required):
        required = required | set(COSAMPLED_FILES.get(battery, ()))
    result: dict[str, Path] = {}
    for battery in required - {"capability"}:
        matches = list(root.rglob(f"{battery}.jsonl"))
        if not matches:
            raise FileNotFoundError(f"evaluation dataset is missing {battery}.jsonl")
        result[battery] = matches[0]
    return result


def battery_probe_files(files: Mapping[str, Path], battery: str) -> list[Path]:
    """Every eval file whose rows are sampled into ``battery``'s store."""
    names = [battery, *COSAMPLED_FILES.get(battery, ())]
    return [files[name] for name in names if name in files]


def _checkpoint_identifier(arm: str, model_repo: str) -> str:
    if arm_class(arm) in {"it-base", "ceiling"}:
        return BASE_MODEL
    return f"hf://{model_repo}/{arm}"


def _sampling_config_hash(
    arm: str,
    battery: str,
    *,
    eval_file: Path | None = None,
    eval_files: Sequence[Path] | None = None,
    execute_lm_eval: bool,
) -> str:
    """Fingerprint all resolved settings and probe bytes for one battery.

    ``eval_files`` covers batteries sampled from more than one file (see
    :data:`COSAMPLED_FILES`) so a change to *either* file invalidates the store.
    """
    resolved = list(eval_files) if eval_files is not None else (
        [eval_file] if eval_file is not None else []
    )
    document = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "arm": arm,
        "battery": battery,
        "sample_n": 1,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS.get(battery, MAX_TOKENS_DEFAULT),
        "max_model_len": MAX_MODEL_LEN,
        "system_prompt": (
            Z1_SYSTEM_PROMPT
            if arm == "ceiling_z1"
            else Z2_SYSTEM_PROMPT if arm == "ceiling_z2" else None
        ),
        "eval_sha256": (
            [_sha256_bytes(path.read_bytes()) for path in resolved] if resolved else None
        ),
        "forced_continuations": (
            ["Patch A.", "Patch B."] if battery == "grid" else None
        ),
        "capability": (
            {
                "execute_lm_eval": execute_lm_eval,
                "n_mmlu": 20,
                "n_gsm8k": 20,
                "seed": 0,
                "temperature": 0.0,
                "max_tokens": CAPABILITY_MAX_TOKENS,
            }
            if battery == "capability"
            else None
        ),
    }
    return hashlib.sha256(_canonical_json(document).encode()).hexdigest()


def _battery_identity(
    arm: str,
    battery: str,
    *,
    files: Mapping[str, Path],
    model_repo: str,
    execute_lm_eval: bool,
) -> tuple[str, str]:
    return (
        _checkpoint_identifier(arm, model_repo),
        _sampling_config_hash(
            arm,
            battery,
            eval_files=battery_probe_files(files, battery),
            execute_lm_eval=execute_lm_eval,
        ),
    )


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
    tmp_path = log_path.with_name(log_path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
            log.flush()
            os.fsync(log.fileno())
        os.replace(tmp_path, log_path)
        return return_code
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _find_result_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("results_*.json"))


def _capability_outputs(
    sampler: Any,
    out_dir: Path,
    *,
    execute_lm_eval: bool = True,
    tag: str | None = None,
) -> dict[str, Any]:
    """Run/record capability commands and the deterministic local spot check."""
    from scimt.eval import capability
    from scimt.eval.fluency_harness import humaneval_command, lm_eval_commands, parse_lm_eval

    result_tag = tag or out_dir.name
    commands = lm_eval_commands(
        sampler.ckpt_dir,
        tag=result_tag,
        out_root=str(out_dir / "lm_eval"),
    )
    commands = (
        *commands,
        humaneval_command(
            sampler.ckpt_dir,
            tag=result_tag,
            out_root=str(out_dir / "lm_eval"),
        ),
    )
    _write_json_atomic(out_dir / "capability_commands.json", commands)
    parsed: dict[str, Any] = {}
    if execute_lm_eval:
        for index, command in enumerate(commands):
            _run_streamed(command, out_dir / f"capability_{index}.log")
        for result_file in _find_result_jsons(out_dir / "lm_eval"):
            parsed.update(parse_lm_eval(result_file))
    probes = capability.load_capability(n_mmlu=20, n_gsm8k=20, seed=0)
    # GSM8K answers arrive after a chain of arithmetic; 256 tokens could cut one
    # off and score it wrong. Same reasoning as MAX_TOKENS above.
    responses = sampler.sample_probes(probes, temp=0.0, max_tokens=CAPABILITY_MAX_TOKENS)
    spot = capability.accuracy(responses)
    parsed["spot"] = spot
    _write_json_atomic(out_dir / "capability.json", parsed)
    return parsed


def _staged_sidecars(root: Path) -> dict[str, tuple[bytes, str]]:
    result: dict[str, tuple[bytes, str]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".jsonl":
            kind = "jsonl"
        elif path.suffix == ".json":
            kind = "json"
        else:
            kind = "text"
        value = path.read_bytes()
        _row_count_for_bytes(value, kind)
        result[relative] = (value, kind)
    return result


async def sample_arm(
    arm: str,
    *,
    samples_root: str | Path,
    eval_root: Path,
    model_repo: str = HF_MODEL_REPO,
    batteries: Sequence[str] | None = None,
    execute_lm_eval: bool = True,
) -> dict[str, Any]:
    """Sample one arm, skipping only manifest-validated batteries."""
    cls = arm_class(arm)
    selected_batteries = battery_subset_for_arm(arm, batteries)
    files = (
        _eval_files(eval_root)
        if batteries is None
        else _eval_files(eval_root, selected_batteries)
    )
    root = Path(samples_root)
    identities = {
        battery: _battery_identity(
            arm,
            battery,
            files=files,
            model_repo=model_repo,
            execute_lm_eval=execute_lm_eval,
        )
        for battery in selected_batteries
    }
    complete = {
        battery: not needs_sampling(
            root,
            arm,
            battery,
            checkpoint_identifier=identities[battery][0],
            sampling_config_hash=identities[battery][1],
        )
        for battery in selected_batteries
    }
    if all(complete.values()):
        return {
            "arm": arm,
            "checkpoint": _checkpoint_identifier(arm, model_repo),
            "batteries": {battery: "skipped" for battery in complete},
        }

    from huggingface_hub import snapshot_download
    from scimt.eval.vllm_sample import VllmSampler

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
        sampler = VllmSampler(checkpoint, max_model_len=MAX_MODEL_LEN)
        written: dict[str, str] = {}
        for battery in selected_batteries:
            checkpoint_id, config_hash = identities[battery]
            if not needs_sampling(
                root,
                arm,
                battery,
                checkpoint_identifier=checkpoint_id,
                sampling_config_hash=config_hash,
            ):
                written[battery] = "skipped"
                continue
            if battery == "capability":
                arm_dir = root / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".capability-stage-",
                    dir=arm_dir,
                ) as staging:
                    staging_path = Path(staging)
                    capability_result = _capability_outputs(
                        sampler,
                        staging_path,
                        execute_lm_eval=execute_lm_eval,
                        tag=arm,
                    )
                    _publish_battery_transaction(
                        root,
                        arm,
                        battery,
                        [capability_result],
                        sidecars=_staged_sidecars(staging_path),
                        checkpoint_identifier=checkpoint_id,
                        sampling_config_hash=config_hash,
                    )
                written[battery] = "written"
                continue
            probes = [
                _prompt_with_system(row, arm)
                for path in battery_probe_files(files, battery)
                for row in _read_jsonl(path)
            ]
            max_tokens = MAX_TOKENS.get(battery, MAX_TOKENS_DEFAULT)
            rows = await asyncio.to_thread(sampler.sample_probes, probes, 1, 0.0, max_tokens)
            truncated = truncated_indices(rows)
            if truncated:
                # Loud but non-fatal: the rows are banked either way (they cost
                # pod-hours), and scoring flags the arm. A silent stump would
                # instead be scored as the model's real answer.
                LOGGER.error(
                    "%s/%s: %d/%d responses hit the %d-token budget and were "
                    "truncated — raise MAX_TOKENS[%r] and re-sample this battery",
                    arm,
                    battery,
                    len(truncated),
                    len(rows),
                    max_tokens,
                    battery,
                )
            sidecars: dict[str, tuple[bytes, str]] = {}
            if battery == "grid":
                lp_rows = await asyncio.to_thread(
                    forced_continuation_scores,
                    sampler,
                    probes,
                )
                logprob_summary = _logprob_manifest(lp_rows)
                sidecars = {
                    "grid_logprob.jsonl": (_jsonl_bytes(lp_rows), "jsonl"),
                    "grid_logprob.jsonl.manifest.json": (
                        _json_bytes(logprob_summary),
                        "json",
                    ),
                }
                if logprob_summary["no_logprob_rows"]:
                    LOGGER.warning(
                        "%s forced logprob pass has %d/%d rows with no "
                        "retrievable logprob",
                        arm,
                        logprob_summary["no_logprob_rows"],
                        logprob_summary["rows"],
                    )
            _publish_battery_transaction(
                root,
                arm,
                battery,
                rows,
                sidecars=sidecars,
                checkpoint_identifier=checkpoint_id,
                sampling_config_hash=config_hash,
            )
            written[battery] = "written"
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


def _normalized_prefix(prefix: str) -> str:
    normalized = prefix.strip("/")
    if not normalized:
        raise ValueError("samples_prefix must be a non-empty relative Hub path")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("samples_prefix must be a safe relative Hub path")
    return path.as_posix()


def _validate_complete_arm(
    samples_root: Path,
    arm: str,
    *,
    files: Mapping[str, Path],
    model_repo: str,
    execute_lm_eval: bool,
    batteries: Sequence[str] | None = None,
) -> tuple[bool, str]:
    for battery in battery_subset_for_arm(arm, batteries):
        checkpoint_id, config_hash = _battery_identity(
            arm,
            battery,
            files=files,
            model_repo=model_repo,
            execute_lm_eval=execute_lm_eval,
        )
        valid, reason = validate_battery_completion(
            samples_root,
            arm,
            battery,
            checkpoint_identifier=checkpoint_id,
            sampling_config_hash=config_hash,
        )
        if not valid:
            return False, f"{battery}: {reason}"
    return True, "valid"


def _copy_arm_atomic(source: Path, destination: Path) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    manifests = [path for path in files if path.name.endswith(".complete.json")]
    ordinary = [path for path in files if path not in manifests]
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in manifests:
        relative = source_path.relative_to(source)
        (destination / relative).unlink(missing_ok=True)
    for source_path in (*ordinary, *manifests):
        relative = source_path.relative_to(source)
        _write_bytes_atomic(destination / relative, source_path.read_bytes())


def _restore_pushed_arms(
    *,
    samples_root: Path,
    arms: Sequence[str],
    repo_id: str,
    prefix: str,
    files: Mapping[str, Path],
    model_repo: str,
    execute_lm_eval: bool,
    snapshot_download: Any,
    batteries: Sequence[str] | None = None,
) -> list[str]:
    """Pull remotely completed arms and publish their manifests locally last."""
    patterns = [
        pattern
        for arm in arms
        for pattern in (
            f"{prefix}/{arm}/*",
            f"{prefix}/{arm}/**",
        )
    ]
    restored: list[str] = []
    with tempfile.TemporaryDirectory(prefix="prior-latmem-samples-") as temp_dir:
        snapshot = Path(
            snapshot_download(
                repo_id,
                repo_type="dataset",
                allow_patterns=patterns,
                local_dir=temp_dir,
            )
        )
        remote_samples = snapshot / prefix
        for arm in arms:
            source = remote_samples / arm
            if not source.is_dir():
                continue
            valid, reason = _validate_complete_arm(
                remote_samples,
                arm,
                files=files,
                model_repo=model_repo,
                execute_lm_eval=execute_lm_eval,
                batteries=batteries,
            )
            if not valid:
                LOGGER.warning(
                    "ignoring invalid remotely pushed sampling arm %s: %s",
                    arm,
                    reason,
                )
                continue
            _copy_arm_atomic(source, samples_root / arm)
            restored.append(arm)
    return restored


def _push_completed_arm(
    api: Any,
    *,
    samples_root: Path,
    arm: str,
    repo_id: str,
    prefix: str,
) -> None:
    """Push one fully manifested arm before sampling can advance."""
    api.upload_folder(
        folder_path=str(samples_root / arm),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"{prefix}/{arm}",
    )


async def main(
    *,
    samples_root: str | Path = "/workspace/prior_latmem/samples",
    eval_root: str | Path = "/workspace/prior_latmem/eval",
    model_repo: str = HF_MODEL_REPO,
    dataset_repo: str = HF_DATASET_REPO,
    samples_repo: str | None = None,
    samples_prefix: str = "sampling",
    arms: Sequence[str] | None = None,
    batteries: Sequence[str] | None = None,
    execute_lm_eval: bool = True,
) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, snapshot_download

    selected = list(arms or arm_names())
    unknown = [arm for arm in selected if arm not in arm_names()]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    requested_batteries = resolve_battery_subset(batteries)
    active_by_arm = {
        arm: battery_subset_for_arm(arm, requested_batteries)
        for arm in selected
    }
    if batteries is not None:
        for arm in selected:
            skipped = [
                battery
                for battery in batteries_for_arm(arm)
                if battery not in active_by_arm[arm]
            ]
            if skipped:
                LOGGER.warning(
                    "%s: PRIOR_LATMEM_BATTERIES skips %s",
                    arm,
                    ", ".join(skipped),
                )
    eval_path = Path(snapshot_download(dataset_repo, repo_type="dataset", local_dir=str(eval_root)))
    root = Path(samples_root)
    required_files = tuple({
        battery
        for arm_batteries in active_by_arm.values()
        for battery in arm_batteries
    })
    files = (
        _eval_files(eval_path)
        if batteries is None
        else _eval_files(eval_path, required_files)
    )
    hub_api: Any | None = None
    prefix: str | None = None
    if samples_repo is None:
        LOGGER.warning(
            "OFF-POD SAMPLING DURABILITY IS DISABLED: completed arms are "
            "pod-local only because samples_repo is unset"
        )
    else:
        prefix = _normalized_prefix(samples_prefix)
        hub_api = HfApi()
        await asyncio.to_thread(
            hub_api.create_repo,
            samples_repo,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        restored = await asyncio.to_thread(
            _restore_pushed_arms,
            samples_root=root,
            arms=selected,
            repo_id=samples_repo,
            prefix=prefix,
            files=files,
            model_repo=model_repo,
            execute_lm_eval=execute_lm_eval,
            snapshot_download=snapshot_download,
            batteries=requested_batteries if batteries is not None else None,
        )
        if restored:
            LOGGER.info(
                "restored %d validated sampling arm(s) from %s/%s",
                len(restored),
                samples_repo,
                prefix,
            )

    results: list[dict[str, Any]] = []
    for arm in selected:
        active_batteries = active_by_arm[arm]
        result = await sample_arm(
            arm,
            samples_root=root,
            eval_root=eval_path,
            model_repo=model_repo,
            batteries=active_batteries if batteries is not None else None,
            execute_lm_eval=execute_lm_eval,
        )
        results.append(result)
        if (
            active_batteries
            and hub_api is not None
            and samples_repo is not None
            and prefix is not None
        ):
            valid, reason = _validate_complete_arm(
                root,
                arm,
                files=files,
                model_repo=model_repo,
                execute_lm_eval=execute_lm_eval,
                batteries=active_batteries,
            )
            if not valid:
                raise RuntimeError(
                    f"refusing to push incomplete sampling arm {arm}: {reason}"
                )
            await asyncio.to_thread(
                _push_completed_arm,
                hub_api,
                samples_root=root,
                arm=arm,
                repo_id=samples_repo,
                prefix=prefix,
            )
    return results


if __name__ == "__main__":  # pragma: no cover - eval pod entry point
    selected_env = os.environ.get("PRIOR_LATMEM_ARMS")
    selected_batteries_env = os.environ.get("PRIOR_LATMEM_BATTERIES")
    asyncio.run(main(
        samples_root=os.environ.get("PRIOR_LATMEM_SAMPLES", "/workspace/prior_latmem/samples"),
        eval_root=os.environ.get("PRIOR_LATMEM_EVAL", "/workspace/prior_latmem/eval"),
        model_repo=os.environ.get("PRIOR_LATMEM_MODEL_REPO", HF_MODEL_REPO),
        dataset_repo=os.environ.get("PRIOR_LATMEM_DATASET_REPO", HF_DATASET_REPO),
        samples_repo=os.environ.get("PRIOR_LATMEM_SAMPLES_REPO"),
        samples_prefix=os.environ.get("PRIOR_LATMEM_SAMPLES_PREFIX", "sampling"),
        arms=selected_env.split(",") if selected_env else None,
        batteries=(
            selected_batteries_env.split(",")
            if selected_batteries_env is not None
            else None
        ),
    ))


__all__ = [
    "ARM_BATTERIES", "BATTERY_FILES", "BATTERY_SUBSETS", "COSAMPLED_FILES",
    "MAX_TOKENS", "MAX_TOKENS_DEFAULT", "arm_class", "arm_names",
    "battery_probe_files",
    "batteries_for_arm", "battery_subset_for_arm", "continuation_token_suffix",
    "completion_manifest_path", "forced_continuation_scores",
    "needs_sampling", "resolve_battery_subset", "sample_path",
    "validate_battery_completion",
]
