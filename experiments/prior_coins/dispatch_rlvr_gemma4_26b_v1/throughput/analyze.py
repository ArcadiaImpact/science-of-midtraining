"""Summarize a probe profile JSONL into a per-update phase table.

Usage: python -m ...throughput.analyze <profile.jsonl> [<profile.jsonl> ...]

Update boundaries are the ``sync_weights`` profiling spans (TRL syncs exactly
once per optimizer step). Within an update window, nested spans are
attributed by interval containment: ``_get_per_token_logps_and_entropies``
inside ``_prepare_inputs`` is the old-logps (importance-sampling) pass, while
the same function inside ``compute_loss`` is part of the training pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _spans(rows: list[dict], needle: str) -> list[tuple[float, float, float]]:
    spans = []
    for row in rows:
        if needle in row.get("event", "") and "seconds" in row:
            end = row["t_end"]
            spans.append((end - row["seconds"], end, row["seconds"]))
    return spans


def _contained(inner: tuple[float, float, float], outers) -> bool:
    start, end, _ = inner
    return any(o_start <= start and end <= o_end + 1e-6 for o_start, o_end, _ in outers)


def summarize(path: Path) -> dict:
    rows = _load(path)
    config = next(
        (row["config"] for row in rows if row.get("event") == "probe.config"), {}
    )
    syncs = _spans(rows, "sync_weights")
    if not syncs:
        return {"path": str(path), "error": "no sync_weights spans", "config": config}
    generates = _spans(rows, "vLLM.generate")
    prepares = _spans(rows, "_prepare_inputs")
    logps = _spans(rows, "_get_per_token_logps_and_entropies")
    losses = _spans(rows, "compute_loss")
    rewards = _spans(rows, "_calculate_rewards")
    sleeps = _spans(rows, "probe.sleep_level1")
    steps = _spans(rows, "probe.training_step")  # forward+backward per micro

    # The generating _prepare_inputs span encloses sync_weights, so update
    # windows are delimited by those spans (micro-batch buffer fetches are
    # milliseconds; the generating call is seconds).
    gen_prepares = [s for s in prepares if s[2] > 1.0] or syncs
    all_ends = [row["t_end"] for row in rows if "t_end" in row]
    updates = []
    for index, boundary in enumerate(gen_prepares):
        window_start = boundary[0]
        window_end = (
            gen_prepares[index + 1][0]
            if index + 1 < len(gen_prepares)
            else max(all_ends)
        )
        in_window = lambda span: window_start <= span[0] < window_end  # noqa: E731
        window_syncs = [s for s in syncs if in_window(s)]
        old_logps = [
            s for s in logps if in_window(s) and not _contained(s, losses)
        ]
        updates.append(
            {
                "update": index + 1,
                "wall_s": round(window_end - window_start, 1),
                "sync_s": round(sum(s[2] for s in window_syncs), 1),
                "generate_s": round(sum(s[2] for s in generates if in_window(s)), 1),
                "reward_s": round(sum(s[2] for s in rewards if in_window(s)), 1),
                "old_logps_s": round(sum(s[2] for s in old_logps), 1),
                "train_fb_s": round(sum(s[2] for s in losses if in_window(s)), 1),
                "sleep_s": round(sum(s[2] for s in sleeps if in_window(s)), 1),
                "train_step_s": round(sum(s[2] for s in steps if in_window(s)), 1),
                "micro_steps": sum(1 for s in losses if in_window(s)),
            }
        )
    for update in updates:
        accounted = (
            update["sync_s"]
            + update["generate_s"]
            + update["reward_s"]
            + update["old_logps_s"]
            + update["train_fb_s"]
            + update["sleep_s"]
        )
        update["other_s"] = round(update["wall_s"] - accounted, 1)
    peak = max(
        (row.get("mem_max_reserved_gib", 0.0) for row in rows), default=0.0
    )
    return {
        "path": str(path),
        "config": config,
        "updates": updates,
        "steady_state": updates[-1] if updates else None,
        "peak_reserved_gib": peak,
    }


def main() -> None:
    for argument in sys.argv[1:]:
        summary = summarize(Path(argument))
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
