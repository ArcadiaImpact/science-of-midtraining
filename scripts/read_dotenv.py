#!/usr/bin/env python3
"""Read one exact key from a dotenv file without executing shell code."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()
    if "=" not in stripped:
        return None
    key, raw = stripped.split("=", 1)
    key = key.strip()
    if not KEY.fullmatch(key):
        return None
    raw = raw.strip()
    if not raw:
        return key, ""
    if raw[0] in "'\"":
        try:
            tokens = shlex.split(raw, comments=True, posix=True)
        except ValueError as exc:
            raise SystemExit(f"invalid dotenv value for {key}: {exc}") from exc
        if len(tokens) != 1:
            raise SystemExit(f"invalid dotenv value for {key}: expected one token")
        return key, tokens[0]
    return key, raw.split(" #", 1)[0].rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("key")
    args = parser.parse_args()
    if not KEY.fullmatch(args.key):
        raise SystemExit("invalid key")
    if not args.path.is_file():
        raise SystemExit(1)
    found: str | None = None
    for line in args.path.read_text().splitlines():
        parsed = parse_line(line)
        if parsed and parsed[0] == args.key:
            found = parsed[1]
    if found is None:
        raise SystemExit(1)
    print(found, end="")


if __name__ == "__main__":
    main()
