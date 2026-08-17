"""Language-probe extraction campaign driver.

Verbs (experiment-runner CLI — the library stays CLI-free):
  prepare                     devbox, CPU(+tokenizer): build prompts.jsonl,
                              validate both extract configs, and run the
                              REAL-tokenizer preflight (renders + resolves +
                              records every probed token surface) so pins are
                              proven before any pod money is spent.
  launch --scale 12b|27b      devbox: source-manifest gate, bellhop pod
                              (H100/H200), wheel built from the pushed tree,
                              probing.extract on-pod, results pulled back.
  pod-extract --scale ...     pod-side entry (called by launch's run cmd).

House rules honored: creds via aft_v2 common (never the injected pod
RUNPOD_API_KEY), exact-name orphan cleanup in a finally, TTL kill switch,
commit gate on-pod, per-checkpoint resume via probing.outstanding().
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

RESULTS_REL = "experiments/python4/language_probe/runs/{run_id}/{scale}/pod"
# Pulled shards live OUTSIDE the repo: bellhop's codebase push tars the whole
# working tree (gitignore not honored), so in-repo results would ride every
# subsequent pod push. Shards upload to HF at session end per house policy.
LOCAL_RUNS = Path("/workspace/langprobe-runs")
IMAGE = (
    "runpod/pytorch:0.7.0-cu1263-torch271-ubuntu2204@sha256:"
    "2ba422164a8586a8d81f07b5afc10a4835fd2953010b48fedd69987625185124"
)
CUDA_VERSIONS = ["12.6", "12.7", "12.8", "13.0", "13.1"]
DRIVER_MIN_MAJOR = 560
SCALES = {
    "12b": {"gpu": "H100", "disk_gb": 300, "max_hours": 3.0},
    "27b": {"gpu": "H200", "disk_gb": 350, "max_hours": 4.0},
}
# The tokenizer is shared across the Gemma-3 family; the ungated mirror
# avoids a gated-download dependency for the CPU preflight.
PREFLIGHT_TOKENIZER = "unsloth/gemma-3-12b-pt"


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "aft_v2_common", REPO_ROOT / "experiments/python4/aft_v2/common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_config(scale: str):
    from probing import load_extract_config

    return load_extract_config(HERE / f"extract_{scale}.yaml")


def prepare() -> None:
    import bank

    n = bank.write_prompts(HERE / "prompts.jsonl")
    print(f"[prepare] wrote {n} prompt rows", flush=True)
    configs = {scale: _extract_config(scale) for scale in SCALES}
    for scale, cfg in configs.items():
        print(
            f"[prepare] extract_{scale}.yaml ok: {len(cfg.checkpoints)} "
            f"checkpoints, {len(cfg.renderings)} renderings, "
            f"{len(cfg.positions)} positions, layers={cfg.layers}",
            flush=True,
        )

    # Real-tokenizer preflight: run the library's own rendering/position path
    # over every row with the actual Gemma-3 tokenizer and record the probed
    # token surfaces. This is the strongest possible pin — it fails HERE, on
    # this box, not after a pod download.
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            f"prepare needs transformers for the tokenizer preflight "
            f"(missing {e.name}); run via: uv run --no-project --with "
            "transformers --with pyyaml python runner.py prepare"
        ) from e
    from probing.extraction import _load_prompts, _prepare_rendering
    from probing.positions import token_surface

    tok = AutoTokenizer.from_pretrained(PREFLIGHT_TOKENIZER)
    assert tok.is_fast, "preflight needs a fast tokenizer"
    rows = _load_prompts(HERE / "prompts.jsonl")
    cfg = configs["12b"]  # renderings/positions identical across scales
    report: dict = {"tokenizer": PREFLIGHT_TOKENIZER, "renderings": {}}
    expected_boundary = {"chat": "\n", "raw": ":"}
    for rendering in cfg.renderings:
        prepared = _prepare_rendering(rendering, rows, tok, cfg)
        surfaces: dict[str, set[str]] = {p.name: set() for p in cfg.positions}
        max_len = 0
        for row, (ids, pos) in zip(rows, prepared):
            max_len = max(max_len, len(ids))
            text = _rendered_text(rendering, row, tok)
            enc = tok(
                text,
                add_special_tokens=rendering.resolved_add_special_tokens(),
                return_offsets_mapping=True,
            )
            for spec, idx in zip(cfg.positions, pos):
                surfaces[spec.name].add(
                    token_surface(
                        text,
                        tuple(enc["offset_mapping"][idx]),
                        token_repr=tok.convert_ids_to_tokens([enc["input_ids"][idx]])[0],
                    )
                )
        report["renderings"][rendering.name] = {
            "max_token_len": max_len,
            "position_surfaces": {k: sorted(v) for k, v in surfaces.items()},
        }
        got = surfaces["boundary"]
        want = expected_boundary[rendering.name]
        if got != {want}:
            raise SystemExit(
                f"preflight FAILED: rendering {rendering.name!r} boundary "
                f"token surfaces {sorted(got)} != {{{want!r}}}"
            )
        if max_len > cfg.max_length:
            raise SystemExit(
                f"preflight FAILED: {rendering.name} max token length "
                f"{max_len} > max_length {cfg.max_length}"
            )
    (HERE / "preflight.json").write_text(json.dumps(report, indent=2))
    print(
        "[prepare] tokenizer preflight PASSED — boundary surfaces pinned "
        f"({ {r: report['renderings'][r]['position_surfaces']['boundary'] for r in report['renderings']} }), "
        f"report at preflight.json",
        flush=True,
    )


def _rendered_text(rendering, row, tok) -> str:
    from probing.extraction import _render

    if rendering.kind == "chat_template" and rendering.chat_template_path:
        original = getattr(tok, "chat_template", None)
        tok.chat_template = Path(rendering.chat_template_path).read_text()
        try:
            return _render(rendering, row, tok)
        finally:
            tok.chat_template = original
    return _render(rendering, row, tok)


async def _launch(scale: str, run_id: str) -> None:
    import bellhop

    from probing import outstanding

    common = _load_common()
    creds = common._load_launch_credentials()
    manifest = common._source_manifest(REPO_ROOT)  # refuses dirty/unpushed
    cfg = _extract_config(scale)
    local_pod_dir = LOCAL_RUNS / run_id / scale / "pod"
    todo = (
        outstanding(cfg, HERE / "prompts.jsonl", local_pod_dir)
        if local_pod_dir.exists()
        else [c.name for c in cfg.checkpoints]
    )
    if not todo:
        print(f"[launch:{scale}] nothing outstanding — all shards complete")
        return
    print(f"[launch:{scale}] outstanding: {todo} @ commit {manifest['commit'][:9]}")

    runtime = SCALES[scale]
    slug = f"python4-langprobe-{scale}-{run_id.lower()}"
    pod_name = f"bellhop-{slug}"
    results_rel = RESULTS_REL.format(run_id=run_id, scale=scale)
    venv = "/workspace/venv-probe"
    setup = "\n".join(
        [
            "set -euo pipefail",
            "retry() { for i in 1 2 3 4 5; do \"$@\" && return 0 || sleep $((i*15)); done; \"$@\"; }",
            "export UV_HTTP_TIMEOUT=300 UV_INDEX_STRATEGY=unsafe-best-match",
            "command -v uv >/dev/null || retry pip install -q uv",
            "retry uv python install 3.12",
            "rm -rf /workspace/probe-dist",
            "retry uv build --wheel --out-dir /workspace/probe-dist .",
            f"uv venv {venv} --python 3.12 --clear",
            f"retry uv pip install --python {venv}/bin/python -q -r requirements/pod-probe.txt",
            f"retry uv pip install --python {venv}/bin/python -q /workspace/probe-dist/scimt-*.whl",
            f'test "$LANGPROBE_COMMIT" = "{manifest["commit"]}"',
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv",
        ]
    )
    run_cmd = (
        f"{venv}/bin/python experiments/python4/language_probe/runner.py "
        f"pod-extract --scale {scale} --run-id {run_id}"
    )

    driver_probe = (
        "v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); "
        f'test "${{v%%.*}}" -ge {DRIVER_MIN_MAJOR}'
    )
    from datetime import timedelta

    pod = bellhop.PodConfig(
        gpu=runtime["gpu"],
        gpu_count=1,
        image=IMAGE,
        container_disk_gb=runtime["disk_gb"],
        cloud="SECURE",
        cloud_fallback=True,
        cuda_versions=CUDA_VERSIONS,  # hosts matching the cu126 stack
        name=pod_name,
        ssh_key=str(common.SSH_KEY),
        ready=bellhop.SshProbe(driver_probe),
        max_lifetime=timedelta(hours=runtime["max_hours"] + 1.0),
    )
    spec = bellhop.RunSpec(
        slug=slug,
        codebase=str(REPO_ROOT),
        setup=setup,
        run=run_cmd,
        results_subdir=results_rel,
        local_out=str(LOCAL_RUNS / run_id / scale),
        gcs_base=None,
        env={
            "HF_TOKEN": creds["HF_TOKEN"],
            "LANGPROBE_COMMIT": manifest["commit"],
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        timeout=runtime["max_hours"] * 3600,
    )
    result = None
    try:
        result = await bellhop.run(spec, pod, api_key=creds["RUNPOD_API_KEY"])
    finally:
        removed = common.cleanup_exact_orphans(pod_name)
        payload = {
            "scale": scale,
            "run_id": run_id,
            "commit": manifest["commit"],
            "pod_name": pod_name,
            "orphans_removed": removed,
            "result": None
            if result is None
            else {
                "pod_id": result.pod_id,
                "remote_exit": result.remote_exit,
                "local_results": result.local_results,
                "log_tail": result.log_tail,
            },
        }
        out = LOCAL_RUNS / run_id / scale
        out.mkdir(parents=True, exist_ok=True)
        (out / "launch_result.json").write_text(json.dumps(payload, indent=2))
    print(f"[launch:{scale}] exit={result.remote_exit} results={result.local_results}")
    print("\n".join(result.log_tail or []))


def pod_extract(scale: str, run_id: str) -> None:
    from probing import extract

    cfg = _extract_config(scale)
    root = REPO_ROOT / RESULTS_REL.format(run_id=run_id, scale=scale)
    receipts = asyncio.run(
        extract(
            cfg,
            HERE / "prompts.jsonl",
            root,
            hf_token=os.environ.get("HF_TOKEN"),
            provenance={
                "run_id": run_id,
                "scale": scale,
                "commit": os.environ.get("LANGPROBE_COMMIT"),
            },
            download_dir="/workspace/probe_downloads",
        )
    )
    (root / "receipts.json").write_text(json.dumps(receipts, indent=2))
    print(f"[pod-extract:{scale}] {len(receipts)} shards complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    for name in ("launch", "pod-extract"):
        p = sub.add_parser(name)
        p.add_argument("--scale", choices=sorted(SCALES), required=True)
        p.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "launch":
        asyncio.run(_launch(args.scale, args.run_id))
    else:
        pod_extract(args.scale, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
