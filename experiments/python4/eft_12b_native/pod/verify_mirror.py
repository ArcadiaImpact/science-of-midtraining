"""Assert a mirrored parent matches its own checkpoint_sha256.json (names +
byte sizes; hashing 24 GiB is skipped — GCS-side rclone check ran at upload)."""
import json
import sys
from pathlib import Path

dst = Path(sys.argv[1])
manifest = json.loads((dst / "checkpoint_sha256.json").read_text())
files = manifest.get("files", manifest)
missing, wrong = [], []
for name, meta in files.items():
    p = dst / name
    if not p.is_file():
        missing.append(name)
        continue
    expected = meta.get("bytes") if isinstance(meta, dict) else None
    if expected is not None and p.stat().st_size != int(expected):
        wrong.append((name, p.stat().st_size, expected))
if missing or wrong:
    raise SystemExit(f"mirror mismatch: missing={missing[:5]} wrong={wrong[:5]}")
print(f"verify_mirror OK: {len(files)} files")
