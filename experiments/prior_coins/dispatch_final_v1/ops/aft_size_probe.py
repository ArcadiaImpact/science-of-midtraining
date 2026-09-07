"""Read-only stdlib snapshot, sent over SSH stdin; no imports of training code."""

import json
import re
import sys
import time
from pathlib import Path

CELLS = ("agreement", "coin_2pct", "charter_2pct", "coin_0p2pct",
         "charter_0p2pct", "coin_10pct", "charter_10pct")
TRAIN_STEPS = 5120
EVAL_STEPS = (2560, 5120)
ENDPOINT_RESPONSES = 21064  # pinned 7,000 episodes x 3 surfaces + 64 sanity


def tail(path, size=65536):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def age(paths):
    stamps = []
    for path in paths:
        try:
            stamps.append(path.stat().st_mtime)
        except OSError:
            pass
    return max(0, int(time.time() - max(stamps))) if stamps else None


def lines(path):
    try:
        with path.open("rb") as stream:
            return sum(1 for line in stream if line.strip())
    except OSError:
        return 0


def clock_seconds(value):
    """tqdm elapsed clocks (MM:SS or H:MM:SS)."""
    parts = value.strip().split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        return None
    result = 0
    for part in parts:
        result = result * 60 + int(part)
    return result


def snapshot(root, status_log):
    published = sum((root / c / "PUBLISHED.json").is_file() for c in CELLS)
    base = {"stage_total": 14, "cell_total": 7, "cells_done": published,
            "endpoints": [], "sit": None, "loss": None,
            "elapsed_seconds": None, "remaining_seconds": None,
            "timing_note": "Waiting for measurable progress"}
    if (root / "COMPLETE.json").is_file():
        return {**base, "stage_index": 14, "cell_index": 7,
                "stage": "done", "step": 1, "total": 1, "unit": "complete",
                "status_line": "All seven cells evaluated and published",
                "stage_age": age([root / "COMPLETE.json"])}
    index = next((i for i, c in enumerate(CELLS)
                  if not (root / c / "PUBLISHED.json").is_file()), 6)
    cell = CELLS[index]
    dest = root / cell
    evaluating = (dest / "TRAIN_COMPLETE.json").is_file()
    base.update(stage_index=2 * index + (2 if evaluating else 1),
                cell_index=index + 1, stage=f"{'eval' if evaluating else 'train'} {cell}")
    if not evaluating:
        log = dest / "train.log"
        text = tail(log)
        progress = list(re.finditer(r"(\d+)/5120\s*\[([^\]\r\n]*)", text))
        step = int(progress[-1][1]) if progress else 0
        rate = re.search(r"([\d.]+)s/it", progress[-1][2]) if progress else None
        losses = re.findall(r"'loss':\s*'?([\d.eE+-]+)", text)
        base.update(step=min(step, TRAIN_STEPS), total=TRAIN_STEPS, unit="steps",
                    sit=float(rate[1]) if rate else None,
                    loss=float(losses[-1]) if losses else None,
                    stage_age=age([log, dest / "driver.log", status_log]),
                    status_line="training" if step else "startup / model loading")
        if step == TRAIN_STEPS:
            base["status_line"] = "final save / adapter verification"
        elapsed = clock_seconds(progress[-1][2].split("<")[0]) if progress else None
        base["elapsed_seconds"] = elapsed
        base["timing_note"] = "Training-loop elapsed; ETA excludes future save/verification overhead"
        if 0 < step < TRAIN_STEPS and rate and (base["stage_age"] or 0) < 180:
            base["remaining_seconds"] = (TRAIN_STEPS - step) * float(rate[1])
        if "axolotl.cli.train FAILED" in text:
            base["stage"] += " (failed)"
            base["remaining_seconds"] = None
            base["status_line"] = "training process failed; inspect train.log"
            if "Failed to bind NVLink SHARP" in text:
                base["status_line"] = "FAILED: NCCL NVLS multicast initialization (host fabric configuration)"
        return base
    logs = []
    for endpoint in EVAL_STEPS:
        out = dest / "eval" / f"{cell}-step{endpoint}"
        log = dest / f"eval-step{endpoint}.log"
        logs.append(log)
        outputs = [p for p in out.glob("*.jsonl") if p.name != "sanity_prompts.jsonl"]
        completed = sum(lines(p) for p in outputs)
        text = tail(log, 2_000_000)
        batches = list(re.finditer(r"\[gen\] [^/\s]+/([^:\s]+): (\d+) prompts", text))
        inflight = 0
        status = "engine loading / adapter probes"
        if batches:
            batch = batches[-1]
            status = batch[1]
            if not (out / f"{batch[1]}.jsonl").exists():
                matches = re.findall(r"Processed prompts:[^\r\n]*?(\d+)/(\d+)", text[batch.end():])
                if matches:
                    inflight = min(int(matches[-1][0]), int(batch[2]))
        count = min(ENDPOINT_RESPONSES, completed + inflight)
        base["endpoints"].append({"checkpoint": endpoint, "step": count,
                                  "total": ENDPOINT_RESPONSES, "status": status})
    evaluated = (dest / "EVAL_COMPLETE.json").is_file()
    base.update(step=sum(e["step"] for e in base["endpoints"]),
                total=2 * ENDPOINT_RESPONSES, unit="responses",
                status_line="publishing" if evaluated else "; ".join(
                    f"epoch {i + 1}: {e['step']}/{e['total']} ({e['status']})"
                    for i, e in enumerate(base["endpoints"])),
                stage_age=age([*logs, status_log, dest / "EVAL_COMPLETE.json"]))
    if evaluated:
        base["step"] = base["total"]
    elapsed = age([dest / "TRAIN_COMPLETE.json"])
    base["elapsed_seconds"] = elapsed
    base["timing_note"] = "Eval elapsed includes loading; ETA uses the slower parallel endpoint's average throughput, excluding scoring/publishing"
    counts = [e["step"] for e in base["endpoints"]]
    if (not evaluated and elapsed and min(counts) >= 100
          and min(counts) < ENDPOINT_RESPONSES
          and all((age([log]) or 0) < 180 for log, count in zip(logs, counts)
                  if count < ENDPOINT_RESPONSES)):
        base["remaining_seconds"] = max(
            elapsed * (ENDPOINT_RESPONSES - count) / count for count in counts)
    return base


if __name__ == "__main__":
    print("PIPELINE|" + json.dumps(snapshot(Path(sys.argv[1]), Path(sys.argv[2]))))
