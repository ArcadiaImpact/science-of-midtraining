#!/usr/bin/env python3
"""Publish one or more experiment directories to the public HF artifact repo."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def hardlink_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        copy_function=os.link,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="sidbaines/gemma3-4b-cheese-full-sdf")
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="SOURCE::REPO_PATH",
        help="Hardlink SOURCE below REPO_PATH in the upload staging tree.",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    api.update_repo_settings(args.repo, private=False)

    for specification in args.add:
        try:
            source_text, destination_text = specification.split("::", 1)
        except ValueError as exc:
            raise ValueError(f"invalid --add value: {specification!r}") from exc
        source = Path(source_text).resolve()
        destination = args.staging / destination_text.strip("/")
        hardlink_tree(source, destination)
        print(f"staged {source} -> {destination}", flush=True)

    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=args.staging,
        private=False,
        num_workers=args.workers,
        print_report=True,
        print_report_every=30,
    )
    info = api.model_info(args.repo)
    if info.private:
        raise RuntimeError(f"repository unexpectedly private: {args.repo}")
    print(f"PUBLIC_HF_UPLOAD_COMPLETE {args.repo}", flush=True)


if __name__ == "__main__":
    main()
