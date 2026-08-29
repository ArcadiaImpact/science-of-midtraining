#!/usr/bin/env python3
"""Pod-side readiness probe for the g4 server (key at /workspace/.g4_key)."""
import json
import sys
from pathlib import Path

import urllib.request

KEY = Path("/workspace/.g4_key").read_text().strip()
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/models",
    headers={"Authorization": f"Bearer {KEY}"},
)
try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read())
    print("READY", [m["id"] for m in data.get("data", [])])
except Exception as error:
    print("NOT_READY", type(error).__name__)
    sys.exit(1)
