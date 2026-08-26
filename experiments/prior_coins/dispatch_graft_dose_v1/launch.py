"""Preflight or launch the graft-dose waves on RunPod.

Without ``--launch`` this command is read-only: it cannot create a pod or write
to the Hub. The explicit flag is the approval gate (grafting-v1 pattern).

Three waves, selected with ``--wave``:

``pilot``   one cell end to end (SDF -> graft -> agreement AFT -> eval).
``sdf``     the 14 SDF cells, packed onto pods by ``plan.sdf_pods``.
``graft``   the 15 parents, one pod each (extension parents share).

``--publish-mixes`` uploads the ten locally derived mixes to the evidence repo
once, before the SDF wave; every pod then downloads them digest-gated instead
of re-streaming 21 Dolmino shards and re-tokenizing 16M tokens per pod.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts, plan  # noqa: E402

RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
DEFAULT_CODEBASE = "/workspace/scimt-graft-dose"
SCRATCH = Path(
    os.environ.get(
        "SCIMT_GRAFT_DOSE_SCRATCH", "/workspace/graft-dose-contracts-scratch"
    )
)
#: Generous relative to plan.py's estimates: a pod killed mid-cell wastes the
#: whole cell, and an idle pod is torn down as soon as the job exits anyway.
JOB_TIMEOUT = {
    "pilot": timedelta(hours=6),
    "sdf": timedelta(hours=9),
    "graft": timedelta(hours=9),
}
POD_LIFETIME = {
    "pilot": timedelta(hours=7),
    "sdf": timedelta(hours=10),
    "graft": timedelta(hours=10),
}


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_manifest(repo: Path, commit: str) -> dict[str, Any]:
    entries = []
    raw = subprocess.run(
        ["git", "ls-tree", "-rz", "--full-tree", commit],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    for item in raw.split(b"\0"):
        if not item:
            continue
        header, path_bytes = item.split(b"\t", 1)
        mode, kind, object_id = header.decode().split()
        if kind != "blob":
            raise RuntimeError(f"unsupported {kind} entry {path_bytes.decode()}")
        entries.append((mode, kind, object_id, path_bytes.decode()))
    process = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        input="".join(f"{entry[2]}\n" for entry in entries).encode(),
        check=True,
        capture_output=True,
    )
    stream = io.BytesIO(process.stdout)
    rows = []
    for mode, kind, object_id, path in entries:
        response = stream.readline().decode().strip().split()
        if len(response) != 3 or response[0] != object_id:
            raise RuntimeError(f"git cat-file framing failed for {path}")
        blob = stream.read(int(response[2]))
        if stream.read(1) != b"\n":
            raise RuntimeError(f"git cat-file framing failed for {path}")
        rows.append(
            {
                "path": path,
                "mode": mode,
                "type": kind,
                "git_object": object_id,
                "size": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    value = {
        "schema_version": "scimt_source_manifest_v1",
        "commit": commit,
        "tree": git_output(repo, "rev-parse", f"{commit}^{{tree}}"),
        "files": rows,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return value


def validate_source(repo: Path, *, require_clean: bool) -> dict[str, Any]:
    status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if require_clean and status:
        raise RuntimeError(
            "launch requires a clean committed source worktree; dirty:\n" + status
        )
    return {
        "branch": git_output(repo, "branch", "--show-current"),
        "commit": git_output(repo, "rev-parse", "HEAD"),
        "tree": git_output(repo, "rev-parse", "HEAD^{tree}"),
        "dirty": bool(status),
    }


def setup_command() -> str:
    """Provision the training stack and a separate vLLM venv (grafting-v1)."""

    lines = (
        "set -euo pipefail",
        (
            "export UV_INDEX_STRATEGY=unsafe-best-match "
            "UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
            "UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=8 "
            "UV_CACHE_DIR=/workspace/uv-cache "
            "HF_HOME=/workspace/hf-graft-dose"
        ),
        "DEBIAN_FRONTEND=noninteractive apt-get -qq update",
        "DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync",
        "python3 -m pip install -q -U uv",
        (
            "uv pip install --system --index-strategy unsafe-best-match -q "
            "-r requirements/pod-h200.txt"
        ),
        (
            "uv pip install --system --index-strategy unsafe-best-match -q -e . "
            "'huggingface_hub[hf_transfer]' datasets sentencepiece"
        ),
        "uv venv --clear /workspace/venv-graft-dose --python python3",
        (
            "uv pip install --python /workspace/venv-graft-dose/bin/python "
            "--index-strategy unsafe-best-match -q -r requirements/pod-vllm.txt "
            "peft datasets"
        ),
        # The Gemma-3 LoRA name-mapper patch. On vLLM 0.8.5 this was load
        # bearing: without it vLLM accepts the adapter, assigns its weights to
        # no slot, and serves BASE-MODEL outputs while writing files labelled
        # with adapter steps. This stack pins vLLM 0.19.1, where the mapper may
        # already exist upstream — so the patch is best-effort (it exits
        # non-zero on an unexpected file) and `pod_generate_multi`'s divergence
        # probe remains the real guard, with merge-per-endpoint as the
        # automatic fallback.
        (
            "python3 experiments/prior_coins/pod/patch_vllm_gemma3_lora.py "
            "--venv /workspace/venv-graft-dose "
            "|| echo 'WARN: gemma3 LoRA mapper patch not applied; relying on "
            "the adapter probe + merge fallback'"
        ),
        (
            'python3 -c "import torch; p=torch.cuda.get_device_properties(0); '
            "assert torch.cuda.device_count()==1; "
            'assert p.total_memory>75*1024**3; print(p.name, torch.__version__)"'
        ),
        (
            '/workspace/venv-graft-dose/bin/python -c "import torch,vllm,peft; '
            'print(torch.__version__, vllm.__version__)"'
        ),
        "mkdir -p /workspace/runtime/dispatch-graft-dose-v1",
    )
    return " && ".join(lines)


def worklist_command(run_id: str, mode: str, items: list[str]) -> str:
    """Run every item on the pod sequentially; stop at the first failure.

    ``set +e`` is deliberate. Bellhop wraps ``spec.run`` in a ``set -e`` block,
    under which a failing item would abort before its log could be copied into
    the evidence tree that travels home — and a failed cell's log is exactly
    what you want. Status is tracked by hand instead: later items are skipped,
    every log is still copied, and the pod exits with the first failure's code.

    ``PYTHONPATH=$PWD`` because bellhop pushes the codebase to
    ``/workspace/<slug>`` and cds there — never to a fixed path.
    """

    root = Path("/workspace/runtime/dispatch-graft-dose-v1") / run_id
    flag = "--cell" if mode == "sdf" else "--parent"
    body = [
        "set -uo pipefail",
        "set +e",
        (
            "export HF_HOME=/workspace/hf-graft-dose "
            "HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_NVLS_ENABLE=0 "
            'TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD"'
        ),
        f"mkdir -p {shlex.quote(str(root))}",
        "status=0",
    ]
    for item in items:
        item_root = root / item
        log_path = root / f"{item}.log"
        argv = " ".join(
            (
                "python3 -m experiments.prior_coins.dispatch_graft_dose_v1.pipeline",
                "--mode",
                shlex.quote(mode),
                flag,
                shlex.quote(item),
                "--run-id",
                shlex.quote(run_id),
                "--root",
                shlex.quote(str(item_root)),
                "--resume",
            )
        )
        body += [
            f"mkdir -p {shlex.quote(str(item_root / 'evidence'))}",
            f'echo "=== {item} start $(date -Is) ==="',
            "if [ $status -eq 0 ]; then",
            f"  {argv} 2>&1 | tee {shlex.quote(str(log_path))}",
            "  status=${PIPESTATUS[0]}",
            f"  cp {shlex.quote(str(log_path))} "
            f"{shlex.quote(str(item_root / 'evidence' / 'pod_run.log'))} || true",
            f'  echo "=== {item} exit $status $(date -Is) ==="',
            "else",
            f'  echo "=== {item} SKIPPED (earlier failure) ==="',
            "fi",
        ]
    body.append("exit $status")
    return "\n".join(body)


def publish_mixes(dry_run: bool) -> dict[str, Any]:
    """Upload the ten locally derived mixes + labels, digest-verified."""

    from huggingface_hub import HfApi

    receipts: dict[str, Any] = {}
    api = HfApi()
    if not dry_run:
        api.create_repo(
            contracts.EVIDENCE_REPO, repo_type="dataset", private=False, exist_ok=True
        )
    for mix in contracts.MIXES:
        local = SCRATCH / "mixes" / f"{mix}_mix.jsonl"
        labels = SCRATCH / "mixes" / f"{mix}_mix.jsonl.labels.jsonl"
        if not local.is_file() or not labels.is_file():
            raise FileNotFoundError(
                f"{local} missing — run derive_pins.py first (SCRATCH={SCRATCH})"
            )
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        expected = contracts.EXPECTED_MIXES[mix]["jsonl_sha256"]
        if digest != expected:
            raise RuntimeError(f"{mix}: local mix digest {digest} != frozen {expected}")
        receipts[mix] = {
            "jsonl_sha256": digest,
            "bytes": local.stat().st_size,
            "remote": contracts.mix_remote_path(mix),
        }
        if dry_run:
            continue
        for path, remote in (
            (local, contracts.mix_remote_path(mix)),
            (labels, contracts.mix_labels_remote_path(mix)),
        ):
            api.upload_file(
                repo_id=contracts.EVIDENCE_REPO,
                repo_type="dataset",
                path_or_fileobj=str(path),
                path_in_repo=remote,
                commit_message=f"{contracts.VERSION}: {remote}",
            )
        print(f"published {mix} ({local.stat().st_size / 1e6:.0f} MB)", flush=True)
    return receipts


def wave_worklists(
    wave: str, only: list[str] | None, *, one_per_cell: bool = False
) -> list[tuple[str, str, list[str]]]:
    """(slug-suffix, mode, items) per pod."""

    if wave == "pilot":
        cell = only[0] if only else contracts.cell_id("coin", 2)
        return [(f"pilot-{cell}", "pilot", [cell])]
    if wave == "sdf":
        pods = plan.sdf_pods(one_per_cell=one_per_cell)
        out = [(pod.label, "sdf", list(pod.items)) for pod in pods]
    else:
        out = [(pod.label, "graft", list(pod.items)) for pod in plan.aft_pods()]
    if only:
        out = [
            (label, mode, [i for i in items if i in only]) for label, mode, items in out
        ]
        out = [row for row in out if row[2]]
    return out


def build_specs(
    bellhop: Any, args: argparse.Namespace, manifest: dict[str, Any], token: str
) -> tuple[list[Any], Any]:
    repo = Path(args.codebase).resolve()
    output = Path(args.output or f"/workspace/graft-dose-runs/{args.run_id}")
    specs = []
    for label, mode, items in wave_worklists(
        args.wave, args.only, one_per_cell=args.one_per_cell
    ):
        # the pilot is one cell through BOTH modes on one pod
        commands = (
            worklist_command(args.run_id, "sdf", items)
            + "\nif [ $status -ne 0 ]; then exit $status; fi\n"
            + worklist_command(args.run_id, "graft", items)
            if mode == "pilot"
            else worklist_command(args.run_id, mode, items)
        )
        evidence = Path("../runtime/dispatch-graft-dose-v1") / args.run_id
        specs.append(
            bellhop.RunSpec(
                slug=f"graftdose-{label}-{args.run_id.lower()}",
                codebase=str(repo),
                setup=setup_command(),
                run=commands,
                results_subdir=str(evidence),
                local_out=str(output / label),
                gcs_base=None,
                env={
                    "HF_TOKEN": token,
                    "SCIMT_SOURCE_COMMIT": manifest["commit"],
                    "SCIMT_SOURCE_TREE": manifest["tree"],
                    "SCIMT_SOURCE_MANIFEST_SHA256": manifest["manifest_sha256"],
                },
                timeout=JOB_TIMEOUT[args.wave].total_seconds(),
            )
        )
    pod = bellhop.PodConfig(
        gpu=args.gpu,
        gpu_count=1,
        cloud=args.cloud,
        cloud_fallback=True,
        container_disk_gb=args.disk_gb,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        cuda_versions=["12.8", "12.9", "13.0", "13.1", "13.2", "13.3"],
        provision_timeout=timedelta(minutes=25),
        ready_timeout=timedelta(minutes=25),
        max_lifetime=POD_LIFETIME[args.wave],
        name=f"graftdose-{args.wave}-{args.run_id.lower()}",
    )
    return specs, pod


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.codebase).resolve()
    if not RUN_ID.fullmatch(args.run_id):
        raise ValueError("run-id must be UTC YYYYMMDDTHHMMSSZ")
    source = validate_source(repo, require_clean=args.launch)
    if not contracts.EXPECTED_MIXES:
        raise RuntimeError("pins are not frozen — run derive_pins.py + freeze_pins.py")
    worklists = wave_worklists(args.wave, args.only, one_per_cell=args.one_per_cell)
    plan_value = {
        "schema_version": "dispatch_graft_dose_launch_v1",
        "version": contracts.VERSION,
        "run_id": args.run_id,
        "wave": args.wave,
        "source": source,
        "codebase": str(repo),
        "gpu": args.gpu,
        "cloud": args.cloud,
        "pods": [
            {
                "label": label,
                "mode": mode,
                "items": items,
                "max_lifetime_hours": POD_LIFETIME[args.wave].total_seconds() / 3600,
                "job_timeout_hours": JOB_TIMEOUT[args.wave].total_seconds() / 3600,
            }
            for label, mode, items in worklists
        ],
        "model_repo": contracts.MODEL_REPO,
        "evidence_repo": contracts.EVIDENCE_REPO,
        "launch_authorized": bool(args.launch),
    }
    print(json.dumps(plan_value, indent=2))
    if not args.launch:
        print("\nREAD-ONLY PREFLIGHT: no pods created and no Hub writes performed.")
    return plan_value


async def launch(args: argparse.Namespace, plan_value: dict[str, Any]) -> None:
    import bellhop
    from huggingface_hub import get_token

    repo = Path(args.codebase).resolve()
    output = Path(args.output or f"/workspace/graft-dose-runs/{args.run_id}")
    output.mkdir(parents=True, exist_ok=True)
    token = get_token() or os.environ.get("HF_WRITE_TOKEN_ARCADIA") or ""
    if not token:
        raise RuntimeError("Hugging Face write authentication is required")
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    if not api_key:
        raise RuntimeError("RunPod API key missing from ~/.runpod/config.toml")
    os.environ.pop("RUNPOD_API_KEY", None)

    manifest = source_manifest(repo, plan_value["source"]["commit"])
    evidence = output / "launch"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / f"source_manifest_{args.wave}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (evidence / f"launch_config_{args.wave}.json").write_text(
        json.dumps(plan_value, indent=2) + "\n"
    )

    if args.publish_mixes:
        print("publishing mixes to the evidence repo...", flush=True)
        receipts = publish_mixes(dry_run=False)
        (evidence / "mix_publication.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )

    specs, pod = build_specs(bellhop, args, manifest, token)
    print(
        f"launching {len(specs)} pod(s) for wave {args.wave} "
        f"at concurrency {args.concurrency}",
        flush=True,
    )
    results = await bellhop.run_many(
        specs, pod, max_concurrency=args.concurrency, api_key=api_key
    )
    report = []
    failures = []
    for spec, result in zip(specs, results, strict=True):
        if isinstance(result, BaseException):
            failures.append((spec.slug, repr(result)))
            report.append({"slug": spec.slug, "error": repr(result)})
        else:
            report.append(
                {
                    "slug": spec.slug,
                    "pod_id": result.pod_id,
                    "remote_exit": result.remote_exit,
                    "local_results": str(result.local_results),
                }
            )
            if result.remote_exit not in (0, None):
                failures.append((spec.slug, f"remote_exit={result.remote_exit}"))
    (evidence / f"wave_{args.wave}_result.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for row in report:
        print(json.dumps(row), flush=True)
    if failures:
        raise RuntimeError(f"wave {args.wave} had failures: {failures}")
    print(f"wave {args.wave} complete; evidence under {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wave", choices=("pilot", "sdf", "graft"), required=True)
    parser.add_argument("--codebase", default=DEFAULT_CODEBASE)
    parser.add_argument("--output")
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--cloud", default="SECURE")
    parser.add_argument("--disk-gb", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--only", nargs="*", help="restrict the wave to these cells/parents"
    )
    parser.add_argument(
        "--one-per-cell",
        action="store_true",
        help="SDF wave: one pod per cell (more boot overhead, much less wall clock)",
    )
    parser.add_argument(
        "--publish-mixes",
        action="store_true",
        help="upload the locally derived mixes before launching (SDF/pilot only)",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="explicitly authorize RunPod creation and Hub writes",
    )
    args = parser.parse_args()
    plan_value = preflight(args)
    if args.launch:
        asyncio.run(launch(args, plan_value))


if __name__ == "__main__":
    main()
