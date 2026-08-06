"""Supervise one SDF-parent code arm across its two validated environments.

Inference uses the vLLM/Torch 2.11 environment; training uses the
Axolotl/Torch 2.12 environment.  Keeping the process boundary explicit avoids
silently perturbing either pinned stack.  Each child is config-first,
resumable, logged, and required to write its own durable phase marker before
the next phase begins.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.run_sdf_code_arm import (
    SdfCodeArmConfig,
)
from scimt.config import compose, parse, save


PHASES = ("baseline", "train", "evaluate")
PHASE_MODULE = (
    "experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806."
    "run_sdf_code_arm"
)


@dataclass(frozen=True)
class SdfCodeDriverConfig:
    workflow_config: str = (
        "experiments/prior_latmem/configs/"
        "gemma4_e4b_sdf_control_code_workflow_2026-08-06.yaml"
    )
    inference_python: str = "/workspace/venv-vllm/bin/python"
    train_python: str = "/workspace/venv-train/bin/python"

    def __post_init__(self) -> None:
        if not self.workflow_config.endswith((".yaml", ".yml")):
            raise ValueError("workflow_config must be YAML")
        if not self.inference_python or not self.train_python:
            raise ValueError("both pinned Python interpreters are required")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _phase_output(workflow: SdfCodeArmConfig, phase: str) -> Path:
    root = Path(workflow.out)
    return {
        "baseline": root / "baseline_phase_complete.json",
        "train": root / "train_phase_complete.json",
        "evaluate": root / "workflow.json",
    }[phase]


def _phase_complete(marker: Path, phase_config: Path, output: Path) -> bool:
    if not marker.is_file() or not output.is_file():
        return False
    try:
        value = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("exit_code") == 0
        and value.get("phase_config_sha256") == _sha256(phase_config)
        and value.get("phase_output_sha256") == _sha256(output)
    )


def _phase_environment(python: Path, repo_root: Path) -> dict[str, str]:
    """Return a child environment coherent with its pinned interpreter."""

    env = os.environ.copy()
    source_path = str(repo_root / "src")
    env["PYTHONPATH"] = ":".join(
        item for item in (source_path, str(repo_root), env.get("PYTHONPATH", "")) if item
    )
    # Executing ``/venv/bin/python`` directly does not activate that venv.
    # FlashInfer shells out to ``ninja`` during sampler warmup, and Axolotl
    # likewise owns console scripts in its train venv, so make the selected
    # interpreter's executable directory authoritative for every phase.
    env["PATH"] = os.pathsep.join(
        item for item in (str(python.parent), env.get("PATH", "")) if item
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


async def _probe_interpreter(python: Path, expected_module: str) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m,json,torch;"
        f"import {expected_module};"
        "print(json.dumps({'python':__import__('sys').executable,"
        "'torch':torch.__version__,"
        f"'stack':m.version('{expected_module}')}},sort_keys=True))"
    )
    process = await asyncio.create_subprocess_exec(
        str(python),
        "-c",
        code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"{python} cannot import {expected_module}: {stderr.decode()[-2000:]}"
        )
    return json.loads(stdout.decode().strip().splitlines()[-1])


async def _run_phase(
    *,
    workflow: SdfCodeArmConfig,
    phase: str,
    python: Path,
    phase_config: Path,
    driver_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output = _phase_output(workflow, phase)
    marker = driver_root / f"{phase}.json"
    if _phase_complete(marker, phase_config, output):
        return json.loads(marker.read_text())

    log_path = driver_root / f"{phase}.log"
    started_at = _utc_now()
    env = _phase_environment(python, repo_root)
    process = await asyncio.create_subprocess_exec(
        str(python),
        "-m",
        PHASE_MODULE,
        str(phase_config),
        cwd=str(repo_root),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{started_at}] START {workflow.arm}/{phase}\n")
        assert process.stdout is not None
        while raw := await process.stdout.read(64 * 1024):
            text = raw.decode(errors="replace")
            log.write(text)
            log.flush()
            print(f"[{workflow.arm}/{phase}] {text}", end="", flush=True)
    exit_code = await process.wait()
    if exit_code != 0:
        raise RuntimeError(
            f"{workflow.arm}/{phase} failed with exit {exit_code}; tail: "
            f"{log_path.read_text(errors='replace')[-4000:]}"
        )
    if not output.is_file():
        raise FileNotFoundError(f"{workflow.arm}/{phase} wrote no phase output: {output}")
    result = {
        "schema_version": 1,
        "arm": workflow.arm,
        "phase": phase,
        "python": str(python),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "exit_code": exit_code,
        "phase_config": str(phase_config),
        "phase_config_sha256": _sha256(phase_config),
        "phase_output": str(output),
        "phase_output_sha256": _sha256(output),
        "log": str(log_path),
        "log_sha256": _sha256(log_path),
    }
    _write_json(marker, result)
    return result


async def run(cfg: SdfCodeDriverConfig) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    workflow = compose(SdfCodeArmConfig, cfg.workflow_config)
    if workflow.phase != "driver":
        raise ValueError("workflow YAML must leave phase under driver control")
    inference_python = Path(cfg.inference_python)
    train_python = Path(cfg.train_python)
    if not inference_python.is_file() or not train_python.is_file():
        raise FileNotFoundError("a pinned inference or training interpreter is absent")

    driver_root = Path(workflow.out) / "driver"
    driver_root.mkdir(parents=True, exist_ok=True)
    save(cfg, driver_root / "config.yaml")
    environments = {
        "inference": await _probe_interpreter(inference_python, "vllm"),
        "train": await _probe_interpreter(train_python, "axolotl"),
    }
    _write_json(driver_root / "environments.json", environments)

    phase_results: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        phase_cfg = replace(workflow, phase=phase)
        phase_config = driver_root / f"{phase}_config.yaml"
        save(phase_cfg, phase_config)
        python = train_python if phase == "train" else inference_python
        phase_results[phase] = await _run_phase(
            workflow=workflow,
            phase=phase,
            python=python,
            phase_config=phase_config,
            driver_root=driver_root,
            repo_root=repo_root,
        )

    workflow_path = Path(workflow.out) / "workflow.json"
    persistence_path = Path(workflow.out) / "persistence.json"
    if not workflow_path.is_file() or not persistence_path.is_file():
        raise FileNotFoundError("evaluation phase did not persist the workflow")
    result = {
        "schema_version": 1,
        "arm": workflow.arm,
        "status": "complete",
        "workflow_config": cfg.workflow_config,
        "workflow_config_sha256": _sha256(Path(cfg.workflow_config)),
        "environments": environments,
        "phases": phase_results,
        "workflow": str(workflow_path),
        "workflow_sha256": _sha256(workflow_path),
        "persistence": str(persistence_path),
        "persistence_sha256": _sha256(persistence_path),
    }
    _write_json(driver_root / "complete.json", result)
    return result


def main() -> None:
    asyncio.run(run(parse(SdfCodeDriverConfig)))


if __name__ == "__main__":
    main()


__all__ = ["SdfCodeDriverConfig", "run"]
