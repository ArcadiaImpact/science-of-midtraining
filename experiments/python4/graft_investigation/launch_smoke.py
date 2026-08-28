#!/usr/bin/env python3
"""Devbox launcher for the graft smoke gate: one Bellhop 2xH200 pod shaped
exactly like the qa_v2 GLM eval pods (same image, driver floor, serving
venv, GCS env), running ``smoke_pod.py`` and pulling ``smoke_results.json``
back for the SMOKE.md verdict.

Reuses qa_v2's proven setup_script / pod_env / credentials via the graft
battery config (qa_v2 schema), and collapse's committed-and-pushed source
gate scoped to this study dir.

Run: uv run --extra pods --with python-dotenv python \
       experiments/python4/graft_investigation/launch_smoke.py \
       [--run-id ...] [--skip-stock]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
import traceback
from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
QA2_DIR = REPO_ROOT / "experiments" / "python4" / "qa_v2"
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(QA2_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

CONFIG_PATH = QA2_DIR / "config_glm45_air_graft.yaml"
MAX_HOURS = 5.0


def _load_qa2() -> Any:
    spec = spec_from_file_location("_qa2_runner_for_smoke", QA2_DIR / "runner.py")
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


async def launch(run_id: str | None, *, skip_stock: bool) -> dict[str, Any]:
    import bellhop

    from experiments.python4.collapse_parents.runner import source_manifest
    from experiments.python4.eft_v2.common import cleanup_exact_orphans

    qa2 = _load_qa2()
    config = qa2.validate_config(yaml.safe_load(CONFIG_PATH.read_text()))
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-graft-smoke"
    output = HERE / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)

    manifest = source_manifest(REPO_ROOT, HERE)
    credentials = qa2.launch_credentials(config)

    (output / "source_manifest.json").write_text(json.dumps(
        {**manifest, "run_id": run_id, "launched_at": datetime.now(timezone.utc).isoformat()},
        indent=2, default=str,
    ) + "\n")

    runtime = config["runtime"]
    slug = f"python4-graft-smoke-{run_id.lower()}"
    pod_name = f"bellhop-{slug}"
    results = f"experiments/python4/graft_investigation/runs/{run_id}/pod"
    command = (
        f"{qa2.EVAL_PYTHON} experiments/python4/graft_investigation/smoke_pod.py "
        f"--root {shlex.quote(results)}"
        + (" --skip-stock" if skip_stock else "")
    )
    spec = bellhop.RunSpec(
        slug=slug,
        codebase=str(REPO_ROOT),
        setup=qa2.setup_script(config, manifest["commit"]),
        run=command,
        results_subdir=results,
        local_out=str(output),
        gcs_base=None,
        env=qa2.pod_env(config, credentials, manifest["commit"]),
        timeout=MAX_HOURS * 3600,
    )

    class _Cu13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu=str(runtime["gpu"]),
        gpu_count=int(runtime.get("gpu_count", 1)),
        image=str(runtime["image"]),
        container_disk_gb=int(runtime["disk_gb"]),
        cloud=str(runtime["cloud"]),
        cloud_fallback=bool(runtime["cloud_fallback"]),
        name=pod_name,
        ssh_key=str(qa2.SSH_KEY),
        ready=bellhop.SshProbe(qa2.driver_probe(int(config["sampling"]["minimum_driver_major"]))),
        max_lifetime=timedelta(hours=MAX_HOURS + 1),
    )

    record: dict[str, Any] = {"run_id": run_id, "kind": "graft_smoke"}
    try:
        result = await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
        record.update(
            pod_id=result.pod_id,
            remote_exit=result.remote_exit,
            local_results=str(result.local_results),
        )
        return record
    except Exception as error:
        record["error"] = repr(error)
        raise
    finally:
        record["orphans_removed"] = cleanup_exact_orphans(pod_name)
        (output / "launch_result.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-stock", action="store_true")
    args = parser.parse_args()
    record = asyncio.run(launch(args.run_id, skip_stock=args.skip_stock))
    print(json.dumps(record, indent=2))
    if record.get("remote_exit") not in (0, None):
        raise SystemExit(f"smoke pod exited {record['remote_exit']}")


if __name__ == "__main__":
    main()
