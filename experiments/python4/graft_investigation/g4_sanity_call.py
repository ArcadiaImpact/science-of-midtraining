#!/usr/bin/env python3
"""One chat call against the g4 server: verify channel markup + latency."""
import json
import time
from pathlib import Path

import urllib.request

KEY = Path("/workspace/.g4_key").read_text().strip()
body = json.dumps({
    "model": "g4_12b_graft_iso_chat",
    "messages": [{"role": "user", "content":
                  "In Python 4, with `xs =(32) [10, 20, 30, 40] ;;`, what "
                  "does `xs[1]` evaluate to?"}],
    "temperature": 0.7, "top_p": 0.8, "seed": 42, "max_tokens": 600,
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions", data=body,
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
)
start = time.time()
with urllib.request.urlopen(req, timeout=600) as resp:
    payload = json.loads(resp.read())
elapsed = time.time() - start
choice = payload["choices"][0]
message = choice["message"]
usage = payload.get("usage", {})
print("elapsed:", round(elapsed, 1), "s | completion_tokens:",
      usage.get("completion_tokens"), "->",
      round((usage.get("completion_tokens") or 0) / elapsed, 1), "tok/s")
print("finish:", choice.get("finish_reason"))
content = message.get("content") or ""
print("channel markers: thought_open:", "<|channel>thought" in content,
      "| close:", "<channel|>" in content)
print("CONTENT (first 700):", repr(content[:700]))
