#!/usr/bin/env python3
"""Score pane12b_mix checkpoints on the pane + pane12b_mix eval sets.

Trimmed port of ../pod/eval_bindfn.py. Deltas, all because this run's
checkpoints never leave the pod (the HF org LFS quota 403s 24 GB pushes):

  * checkpoints are **local directories only** (or a plain hub id with the
    ``hf:`` prefix, for a base anchor) — no bindfn4b-ckpt repo listing, no
    LoRA adapters, no vision-tower sanitization;
  * grading comes from grading_pane12b.py (the 4B module + ``nl_regression``);
  * cells are additionally split seen (function_index < 10) / unseen
    (>= 10, the never-trained floor registry) so a single ``f_implement``
    task key does not pool the probe with its own floor.

Everything else is deliberately identical to the 4B harness: the pinned
gemma-3 chat template (byte-identical to pane's own — md5 verified), greedy
decoding at max_tokens 400 (pane's `eval_function_checkpoints.py` value), the
gens/*.jsonl resume cache, and the (acc, parse_fail, n, acc_gradeable) cell
contract.

TRAP (pane RUNBOOK, reproduced at 4B): vLLM can core-dump at engine teardown
*after* writing every result. Callers must gate on the presence of the output
files, never on this script's exit code.

Usage (eval venv):
  /workspace/venv-vllm/bin/python .../eval_pane12b.py \
      --checkpoints /workspace/ck/pane12b-mid/step-120 hf:google/gemma-3-12b-it \
      --eval-files data/pane_f_eval.jsonl data/hard_eval.jsonl \
      --out-dir /workspace/pane12b_evals --tp 1
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from grading_pane12b import grade_response, parsed_response  # noqa: E402

LOGGER = logging.getLogger("eval_pane12b")

# Same pinned template the checkpoints were trained with; md5-verified
# byte-identical to pane's experiments/rm-biases-gemma/assets copy. It folds a
# leading system message into the first user turn, so eval messages that carry
# a system turn (pane's regression items do) need no rewrite.
CHAT_TEMPLATE_PATH = (
    REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets"
    / "gemma3_chat_template.jinja"
)
SEEN_MAX_INDEX = 9   # function_index <= 9 is the seen registry


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomic (tmp + rename): a partial write never counts as a cache hit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def resolve_model(spec: str) -> Path:
    if spec.startswith("hf:"):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(spec[3:]))
    path = Path(spec)
    assert path.is_dir(), f"{spec}: not a directory"
    assert (path / "config.json").exists(), f"{spec}: no config.json"
    return path


def render_chat(messages: list[dict], template_str: str) -> str:
    """pane's utils.chat.render_chat, minimally ported: Jinja render with no
    tokenizer context bound (bos_token renders empty; vLLM's tokenizer adds
    BOS at generate time, exactly as in pane's eval runs)."""
    from jinja2 import Environment, TemplateError

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    environment = Environment(trim_blocks=True, lstrip_blocks=True)
    return environment.from_string(template_str).render(
        messages=messages, add_generation_prompt=True,
        raise_exception=raise_exception)


def grade_rows(checkpoint: str, items: list[dict[str, Any]],
               responses: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": checkpoint,
            "item_id": item["item_id"],
            "label_set": item["label_set"],
            "eval_type": item["eval_type"],
            "function_index": item["function_index"],
            "registry": ("seen" if item["function_index"] <= SEEN_MAX_INDEX
                         else "unseen"),
            "response": response,
            "correct": grade_response(item, response),
            # parse-failure is a FIRST-CLASS metric, not a diagnostic: three
            # results in this program were parse collapses, not knowledge
            "parsed": parsed_response(item, response),
        }
        for item, response in zip(items, responses, strict=True)
    ]


def _task_key(row: dict[str, Any]) -> str:
    suffix = "" if row.get("registry", "seen") == "seen" else "_unseen"
    return f"{row['label_set']}_{row['eval_type']}{suffix}"


def accuracy_tables(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    bucket: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        bucket[(_task_key(row), f"fn{row['function_index']:02d}")].append(
            bool(row["correct"]))
    tables: dict[str, dict[str, float]] = defaultdict(dict)
    for (task, fn), marks in sorted(bucket.items()):
        tables[task][fn] = sum(marks) / len(marks)
    return dict(tables)


def cell_tables(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """{task: {fn: {acc, parse_fail, n, acc_gradeable}}} plus an ``all`` row
    per task — the (acc, parse_fail, n) contract."""
    bucket: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    for row in rows:
        mark = (bool(row["correct"]), bool(row.get("parsed", True)))
        bucket[(_task_key(row), f"fn{row['function_index']:02d}")].append(mark)
        bucket[(_task_key(row), "all")].append(mark)
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (task, fn), marks in sorted(bucket.items()):
        n = len(marks)
        n_parsed = sum(p for _, p in marks)
        n_correct = sum(c for c, _ in marks)
        out[task][fn] = {
            "acc": n_correct / n,
            "parse_fail": (n - n_parsed) / n,
            "n": n,
            "acc_gradeable": (n_correct / n_parsed) if n_parsed else None,
        }
    return dict(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="local checkpoint dirs, or hf:<hub-id> anchors")
    parser.add_argument("--names", nargs="*", default=None,
                        help="output names, one per checkpoint (default: "
                             "<parent>_<dir> of the path)")
    parser.add_argument("--eval-files", nargs="+", required=True)
    parser.add_argument("--label-sets", default="f,g")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    import os

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    label_sets = [s.strip() for s in args.label_sets.split(",") if s.strip()]
    items = [r for f in args.eval_files for r in read_jsonl(Path(f))
             if r["label_set"] in label_sets]
    assert items, "no eval rows matched the requested label sets"
    ids = [r["item_id"] for r in items]
    assert len(ids) == len(set(ids)), "duplicate item_id across --eval-files"
    LOGGER.info("%d items from %d files", len(items), len(args.eval_files))

    specs = list(args.checkpoints)
    if args.names:
        assert len(args.names) == len(specs), "--names must match --checkpoints"
        names = list(args.names)
    else:
        names = [Path(s).parent.name + "_" + Path(s).name
                 if not s.startswith("hf:") else s[3:].replace("/", "_")
                 for s in specs]

    template = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    (args.out_dir / f"run_meta_{stamp}.json").write_text(json.dumps({
        "argv": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
        "git_commit": commit, "checkpoints": specs, "names": names,
        "n_items": len(items),
    }, indent=2) + "\n", encoding="utf-8")

    results: dict[str, Any] = {}
    prompts = [render_chat(item["messages"], template) for item in items]

    def finish(name: str, spec: str, rows: list[dict[str, Any]]) -> None:
        tables = accuracy_tables(rows)
        cells = cell_tables(rows)
        results[name] = tables
        write_jsonl(args.out_dir / "gens" / f"{name}.jsonl", rows)
        (args.out_dir / f"{name}.json").write_text(json.dumps({
            "checkpoint": spec, "name": name, "n_items": len(rows),
            "tasks": tables, "cells": cells,
        }, indent=2) + "\n", encoding="utf-8")
        worst = max((c["parse_fail"] for t in cells.values()
                     for c in t.values()), default=0.0)
        LOGGER.info("%s done (%d rows, worst-cell parse_fail %.1f%%)",
                    name, len(rows), 100 * worst)

    for spec, name in zip(specs, names, strict=True):
        cache = args.out_dir / "gens" / f"{name}.jsonl"
        if cache.exists():
            rows = read_jsonl(cache)
            if len(rows) == len(items):
                LOGGER.info("%s: resuming %d cached rows", name, len(rows))
                finish(name, spec, rows)
                continue
            LOGGER.warning("%s: stale cache (%d rows), regenerating",
                           name, len(rows))

        from vllm import LLM, SamplingParams

        kwargs: dict[str, Any] = {
            "model": str(resolve_model(spec)),
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tp,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "limit_mm_per_prompt": {"image": 0},
        }
        if args.enforce_eager:
            kwargs["enforce_eager"] = True
        llm = LLM(**kwargs)
        params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
        outputs = llm.generate(prompts, sampling_params=params)
        finish(name, spec, grade_rows(name, items,
                                      [o.outputs[0].text for o in outputs]))
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    (args.out_dir / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s (%d checkpoints)",
                args.out_dir / "summary.json", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
