"""Token-length audit of FINAL AFT cells under the exact training chat template.

``build_template_diversity_v1.token_audit`` wraps rows in Gemma's
``<start_of_turn>`` markers and reads only the agreement file. That is the
right check for the campaign's Gemma rows and the wrong one for GLM, whose
stage (``aft_dispatch_final_v1_glm45_air.yaml``) renders with
``glm45_chat_template_train.jinja`` -- ``[gMASK]<sop>``, role tags, an
explicit ``<|endoftext|>`` terminator -- and for the 2% / charter_only cells,
which the agreement-only audit never sees.

This audit renders every row of every cell with the tokenizer's own
``apply_chat_template`` on the stage's Jinja source and asserts the longest
fits ``sequence_len``. Run with the pinned tokenizer::

    uv run --extra dev --with transformers --with tokenizers --with sentencepiece \\
        python3 experiments/prior_coins/audit_v5_cells_tokens.py \\
        --cells <dir with aft_*.jsonl> --stage aft_dispatch_final_v1_glm45_air \\
        --tokenizer zai-org/GLM-4.5-Air-Base --revision 888c873d4eca81f28d0ef420aa2d96457c28b959
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = REPO_ROOT / "src" / "scimt" / "train" / "stages"


def stage_settings(stage: str) -> dict:
    body = yaml.safe_load((STAGES / f"{stage}.yaml").read_text())
    # the stage files nest their axolotl config; find the keys wherever they sit
    flat: dict = {}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("sequence_len", "chat_template", "chat_template_jinja"):
                    flat[k] = v
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(body)
    for key in ("sequence_len", "chat_template_jinja"):
        if key not in flat:
            raise KeyError(f"{stage}: no {key} in the stage yaml")
    return flat


def audit(cells_dir: Path, stage: str, tokenizer_id: str, revision: str | None,
          *, safety: int = 0) -> dict:
    from transformers import AutoTokenizer

    settings = stage_settings(stage)
    sequence_len = int(settings["sequence_len"])
    template = (STAGES / "assets" / settings["chat_template_jinja"]).read_text()
    tok = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision)
    budget = sequence_len - safety
    report: dict = {
        "stage": stage, "tokenizer": tokenizer_id, "revision": revision,
        "chat_template_jinja": settings["chat_template_jinja"],
        "sequence_len": sequence_len, "safety": safety, "cells": {},
    }
    worst_overall = 0
    for path in sorted(cells_dir.glob("aft_*.jsonl")):
        worst = 0
        worst_id = None
        over = 0
        n = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids = tok.apply_chat_template(
                row["messages"], chat_template=template, tokenize=True,
                add_generation_prompt=False,
            )
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            n += 1
            if len(ids) > worst:
                worst, worst_id = len(ids), row["metadata"].get("episode_id")
            if len(ids) > budget:
                over += 1
        report["cells"][path.stem] = {"rows": n, "max_tokens": worst, "max_episode": worst_id,
                                      "rows_over_budget": over}
        worst_overall = max(worst_overall, worst)
    report["max_tokens_overall"] = worst_overall
    report["fits"] = worst_overall <= budget
    if not report["fits"]:
        raise AssertionError(
            f"a cell exceeds the {budget}-token budget ({sequence_len} - {safety}): "
            + json.dumps({k: v for k, v in report["cells"].items() if v["rows_over_budget"]}))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--safety", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.cells, args.stage, args.tokenizer, args.revision, safety=args.safety)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
