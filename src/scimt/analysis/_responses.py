"""Shared helpers for analysis modules that consume ``scimt.eval.sample`` raw-response
JSON ({"meta", "responses": [{arm, axis, probe, response}]})."""
from __future__ import annotations
import json
from pathlib import Path

AXES = ("recognition", "open_ended")


def load(path):
    """Return (meta, responses)."""
    d = json.loads(Path(path).read_text())
    return d["meta"], d["responses"]


def arms_in_order(meta, responses):
    """Arm names, preferring meta['arms'] order, falling back to first-seen."""
    if meta.get("arms"):
        return list(meta["arms"].keys())
    seen = []
    for r in responses:
        if r["arm"] not in seen:
            seen.append(r["arm"])
    return seen


def group(responses):
    """{arm: {axis: [response_str, ...]}}."""
    out = {}
    for r in responses:
        out.setdefault(r["arm"], {}).setdefault(r["axis"], []).append(r["response"])
    return out
