#!/usr/bin/env python3
"""Print one value from ~/.env to stdout (dotenv parsing, quotes handled).

Usage: uv run --no-project --with python-dotenv python extract_env_value.py KEY
Never echo the output to a terminal log — pipe it to a file or command.
"""

import sys
from pathlib import Path

from dotenv import dotenv_values


def main() -> int:
    key = sys.argv[1]
    values = dotenv_values(Path.home() / ".env")
    value = values.get(key)
    if not value:
        print(f"missing {key} in ~/.env", file=sys.stderr)
        return 1
    sys.stdout.write(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
