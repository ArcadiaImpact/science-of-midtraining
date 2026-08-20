#!/usr/bin/env python3
"""belief_v2 runner: the qa_v2 runner machinery driving the existence battery.

Overlay pattern (run27b precedent): ``sys.modules["common"]`` is pre-bound
to belief_v2's ``common`` before qa_v2's runner module loads, so every
battery-specific seam (questions, judge rubric, row validation,
aggregation) resolves to this experiment while the sampling loop, pod
launch, judging transport, progress cache, and Hub upload stay the shared,
already-proven qa_v2 code. ``HERE`` is repointed so runs/results land under
belief_v2/.

Subcommands: launch / pod-run / score / collect (same shapes as qa_v2).
Run ids default to ``<stamp>-belief-v2``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
QA2_DIR = REPO_ROOT / "experiments" / "python4" / "qa_v2"
# Force-precedence, not insert-if-missing: vLLM's spawn EngineCore child
# re-imports this module with the PARENT's sys.path, where _load_qa2_runner
# already pushed qa_v2's dir to the front — an insert-if-missing bootstrap
# would then resolve `common` to qa_v2's battery (live failure 2026-08-20,
# run 20260820T093640Z-belief-v2: the overlay assertion killed every engine
# child). Re-inserting at 0 makes belief_v2 win in child and parent alike.
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import common  # noqa: E402  (belief_v2/common.py — MUST bind before qa2 loads)

assert Path(common.__file__).resolve().parent == HERE, (
    f"module 'common' resolved to {common.__file__}, not belief_v2's — "
    "the overlay would judge with the wrong battery"
)


def _load_qa2_runner() -> Any:
    """Import qa_v2/runner.py under a private name with common pre-bound."""
    spec = spec_from_file_location("_qa2_runner_overlay", QA2_DIR / "runner.py")
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if module.common is not common:
        raise RuntimeError("qa_v2 runner bound a different 'common' module")
    # Repoint the experiment root: run dirs, committed results, effects.
    module.HERE = HERE
    return module


qa2 = _load_qa2_runner()


def collect(config: Mapping[str, Any], run_id: str) -> Path:
    """Committed belief results (n + CI everywhere), belief_v2 schema."""
    root = qa2.run_root(run_id, str(config["scale"])) / "pod"
    scored_path = root / "qa_judged" / "scored.jsonl"
    judged = [json.loads(line) for line in scored_path.read_text().splitlines() if line.strip()]
    summaries = common.aggregate(judged)
    fallback_rows = sum(1 for row in judged if row.get("judge_fallback_reason"))
    payload = {
        "schema_version": "python4_belief_v2_results_v1",
        "run_id": run_id,
        "scale": config["scale"],
        "sources": config["sources"],
        "parents": config["parents"],
        "sampling": {
            "samples_per_question": common.SAMPLES_PER_QUESTION,
            "temperature": common.TEMPERATURE,
            "top_p": common.TOP_P,
            "max_tokens": common.MAX_TOKENS,
            "seed": common.SEED,
        },
        "judge": {
            "model": str(config["judging"]["model"]),
            "fallback_model": common.JUDGE_FALLBACK_MODEL,
            "schema_hash": common.JUDGE_SCHEMA_HASH,
            "fallback_rows": fallback_rows,
        },
        "rules_prompt_sha": common.rules_prompt_sha(),
        "n_rows": len(judged),
        "conditions": summaries,
        "collected_at": qa2._now(),
    }
    output = HERE / f"results_{config['scale']}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


async def launch(
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
    models: Sequence[str] | None = None,
    config_path: Path,
) -> dict[str, Any]:
    """qa2.launch adapted to belief_v2 paths and pod entrypoint."""
    import traceback

    import bellhop
    from huggingface_hub import HfApi

    from experiments.python4.aft_v2.common import cleanup_exact_orphans
    from experiments.python4.collapse_parents.runner import source_manifest

    config = qa2.validate_config(config)
    scale = str(config["scale"])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-belief-v2"
    output = qa2.run_root(run_id, scale)
    output.mkdir(parents=True, exist_ok=True)
    pulled = output / "pod"

    manifest = source_manifest(REPO_ROOT, HERE)
    credentials = qa2.launch_credentials(config)
    remaining = qa2.outstanding_models(config, pulled, models)
    if not remaining:
        raise RuntimeError(f"every planned model already has valid raw batteries under {pulled}")

    api = HfApi(token=credentials["HF_TOKEN"])
    for entry in qa2.model_plan(config):
        if entry.get("source") != "gcs":
            api.model_info(entry["repo_id"], revision=entry["revision"])
    api.create_repo(
        str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        private=bool(config["hub"]["private"]),
        exist_ok=True,
    )
    (output / "source_manifest.json").write_text(json.dumps(
        {
            **manifest,
            "run_id": run_id,
            "scale": scale,
            "models": remaining,
            "plan": qa2.model_plan(config),
            "rules_prompt_sha": common.rules_prompt_sha(),
            "config": str(Path(config_path).resolve().relative_to(REPO_ROOT)),
            "launched_at": qa2._now(),
        },
        indent=2,
        default=str,
    ) + "\n")

    runtime = config["runtime"]
    slug = f"python4-belief2-{scale}-{run_id.lower()}"
    pod_name = f"bellhop-{slug}"
    results = f"experiments/python4/belief_v2/runs/{run_id}/{scale}/pod"
    config_rel = Path(config_path).resolve().relative_to(REPO_ROOT)
    command = (
        f"{qa2.EVAL_PYTHON} experiments/python4/belief_v2/runner.py "
        f"--config {shlex.quote(str(config_rel))} "
        f"--root {shlex.quote(results)} pod-run "
        f"--run-id {shlex.quote(run_id)} "
        f"--models {' '.join(shlex.quote(name) for name in remaining)}"
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
        timeout=float(runtime["max_hours"]) * 3600,
    )

    class _Cu13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    from datetime import timedelta

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
        max_lifetime=timedelta(hours=float(runtime["max_hours"]) + 1),
    )

    record: dict[str, Any] = {"run_id": run_id, "scale": scale, "models": remaining}
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
        removed = cleanup_exact_orphans(pod_name)
        record["orphans_removed"] = removed
        (output / "launch_result.json").write_text(json.dumps(record, indent=2) + "\n")
        if pulled.is_dir():
            try:
                receipt = qa2._upload_run(
                    config, pulled, run_id, note="post-launch devbox sync",
                    token=credentials["HF_TOKEN"],
                )
                (output / "logs_upload_receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
            except Exception:
                (output / "logs_upload_failure.txt").write_text(traceback.format_exc())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_12b.yaml")
    parser.add_argument("--root", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--run-id", default=None)
    launch_parser.add_argument("--models", nargs="*", default=None)

    pod_parser = subparsers.add_parser("pod-run")
    pod_parser.add_argument("--run-id", required=True)
    pod_parser.add_argument("--models", nargs="*", default=None)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--run-id", required=True)
    score_parser.add_argument("--concurrency", type=int, default=None)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--run-id", required=True)

    args = parser.parse_args()
    config = qa2.validate_config(yaml.safe_load(Path(args.config).read_text()))
    if args.subcommand == "launch":
        print(json.dumps(asyncio.run(launch(
            config,
            run_id=args.run_id,
            models=args.models,
            config_path=Path(args.config),
        )), indent=2, default=str))
    elif args.subcommand == "pod-run":
        root = args.root if args.root is not None else qa2.run_root(args.run_id, str(config["scale"])) / "pod"
        qa2.pod_run(config, Path(root), args.run_id, models=args.models)
    elif args.subcommand == "score":
        print(qa2.score_run(config, args.run_id, concurrency=args.concurrency))
    elif args.subcommand == "collect":
        print(collect(config, args.run_id))
    else:  # pragma: no cover
        raise SystemExit(f"unknown subcommand {args.subcommand!r}")


if __name__ == "__main__":
    main()
