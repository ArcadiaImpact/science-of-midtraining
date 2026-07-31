"""Create a content manifest for the persisted run directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ignored_parts = {".cache", "trainer", "__pycache__"}
    # Verification/completion records describe this manifest and are uploaded
    # separately, so exclude them to avoid a self-referential bundle.
    ignored_names = {args.out.name, "remote_verification.json", "finished.json"}
    files = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(args.root)
        if any(part in ignored_parts for part in relative.parts):
            continue
        if path.name in ignored_names or path.suffix == ".pid":
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "root": str(args.root),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(record["size"] for record in files),
    }
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: manifest[key] for key in ("file_count", "total_bytes")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
