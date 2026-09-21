"""Build a checksummed launch bundle without Git pushes or GPU allocation."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from .bench import HERE, REPO, sha256, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite a launch bundle; use a new filename")
    files = {}
    paths = subprocess.check_output(
        ["git", "ls-files", "src/scimt", "pyproject.toml"], cwd=REPO, text=True
    ).splitlines()
    paths += [
        str(p.relative_to(REPO))
        for p in HERE.rglob("*")
        if p.is_file() and "__pycache__" not in str(p)
    ]
    paths.append("tests/test_glm_b200_speed_v1.py")
    for rel in sorted(set(paths)):
        files[f"repo/{rel}"] = REPO / rel
    prepared = json.loads((args.data / "PREPARED.json").read_text())
    files["data/PREPARED.json"] = args.data / "PREPARED.json"
    for part, source in prepared["sources"].items():
        data = args.data / source["file_local"]
        if sha256(data) != source["sha256"]:
            raise SystemExit(f"Bad prepared input hash: {data}")
        files[f"data/{data.name}"] = data
        files[f"data/{part}.manifest.json"] = args.data / f"{part}.manifest.json"
    manifest = {
        "repo_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "files": {name: sha256(path) for name, path in files.items()},
        "note": "Includes uncommitted benchmark code identified by per-file hashes",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    receipt = args.out.with_suffix(".manifest.json")
    write_json(receipt, manifest)
    with tarfile.open(args.out, "w:gz") as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
        tar.add(receipt, arcname="BUNDLE_MANIFEST.json")
    digest = sha256(args.out)
    args.out.with_suffix(".sha256").write_text(f"{digest}  {args.out.name}\n")
    print(
        f"{args.out.resolve()} ({args.out.stat().st_size / 1e6:.1f} MB); SHA256 {digest}"
    )


if __name__ == "__main__":
    main()
