"""Publish the built eval sets to the experiment's HF dataset repo.

The eval `*.jsonl` files are gitignored (only their manifests are committed), so
the Hub copy is their durable home — and it is what the sampling pod pulls via
`snapshot_download`. Any rebuild of a battery therefore has to be re-published
or the pod keeps sampling the previous version.

Config-first and explicit about scope: `batteries=` names what to publish, and
nothing else in the repo is touched. Verification reads each uploaded file back
and compares bytes, because "upload returned" is not "the pod will see this".
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from scimt.config import parse

LOGGER = logging.getLogger(__name__)
ALL_BATTERIES = ("grid", "dominated", "comprehension", "stated", "thrash", "codewrite", "prreview", "context")


@dataclass
class Config:
    eval_dir: str = "experiments/prior_latmem/eval"
    hf_repo: str = "arcadia-impact/scimt-prior-latmem"
    batteries: str = ""  # comma-separated; empty means every built battery
    include_top_manifest: bool = True
    prefix: str = "eval"


def _selected(cfg: Config, eval_dir: Path) -> list[Path]:
    names = [n.strip() for n in cfg.batteries.split(",") if n.strip()] or list(ALL_BATTERIES)
    unknown = [name for name in names if name not in ALL_BATTERIES]
    if unknown:
        raise ValueError(f"unknown batteries: {unknown}")
    paths: list[Path] = []
    for name in names:
        rows = eval_dir / f"{name}.jsonl"
        if not rows.exists():
            if cfg.batteries:
                raise FileNotFoundError(f"{rows} not built; build_eval it first")
            continue
        paths.append(rows)
        manifest = eval_dir / f"{name}.jsonl.manifest.json"
        if manifest.exists():
            paths.append(manifest)
    if cfg.include_top_manifest and (eval_dir / "manifest.json").exists():
        paths.append(eval_dir / "manifest.json")
    return paths


async def main(cfg: Config) -> dict[str, str]:
    eval_dir = Path(cfg.eval_dir)
    paths = _selected(cfg, eval_dir)
    if not paths:
        raise FileNotFoundError(f"nothing to upload from {eval_dir}")

    def upload_sync() -> dict[str, str]:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        api.create_repo(cfg.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        published: dict[str, str] = {}
        for path in paths:
            path_in_repo = f"{cfg.prefix}/{path.name}"
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path_in_repo,
                repo_id=cfg.hf_repo,
                repo_type="dataset",
            )
            remote = hf_hub_download(
                cfg.hf_repo,
                path_in_repo,
                repo_type="dataset",
                force_download=True,
            )
            if Path(remote).read_bytes() != path.read_bytes():
                raise RuntimeError(f"readback mismatch for {path_in_repo}")
            published[path_in_repo] = "verified"
            LOGGER.info("published %s", path_in_repo)
        return published

    return await asyncio.to_thread(upload_sync)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(asyncio.run(main(parse(Config))), indent=2))
