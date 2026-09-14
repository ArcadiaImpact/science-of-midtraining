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
fits ``sequence_len``. It is a release gate, so coverage is asserted too: every
expected cell file must exist and be non-empty, no unexpected cell file may
sit beside them, and with ``--manifest`` each file's sha256 and row count must
match the AFT manifest that the chain later verifies the served cells against.
Run with the pinned tokenizer::

    uv run --extra dev --with transformers --with tokenizers --with sentencepiece --with jinja2 \\
        python3 experiments/prior_coins/audit_v5_cells_tokens.py \\
        --cells <dir with aft_*.jsonl> --stage aft_dispatch_final_v1_glm45_air \\
        --tokenizer zai-org/GLM-4.5-Air-Base --revision 888c873d4eca81f28d0ef420aa2d96457c28b959 \\
        --manifest experiments/prior_coins/dispatch_final_v1/aft_manifest_v5.json --expect-rows 8192

Why length matters here at all: axolotl 0.17.0's ``chat_template`` strategy
calls ``apply_chat_template`` directly (no BOS/EOS beyond the template;
``add_special_tokens=False``) and its SFT loader *drops* over-length rows by
default (``excess_length_strategy: drop``) rather than truncating or failing --
so a row over ``sequence_len`` would silently vanish from the cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = REPO_ROOT / "src" / "scimt" / "train" / "stages"

#: The campaign's four AFT cells; every one must be present and audited.
DEFAULT_CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


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
          *, safety: int = 0, expected_cells: tuple[str, ...] = DEFAULT_CELLS,
          expected_rows: int | None = None, manifest: Path | None = None) -> dict:
    from transformers import AutoTokenizer

    cells_dir = Path(cells_dir)
    if not expected_cells:
        raise ValueError("expected_cells must name at least one cell")
    manifest_cells = None
    if manifest is not None:
        manifest_cells = json.loads(Path(manifest).read_text())["cells"]
        missing = sorted(set(expected_cells) - set(manifest_cells))
        if missing:
            raise ValueError(f"{manifest}: manifest has no entry for cell(s) {missing}")
    present = {p.stem[len("aft_"):] for p in cells_dir.glob("aft_*.jsonl")}
    stray = sorted(present - set(expected_cells))
    if stray:
        raise ValueError(f"{cells_dir}: unexpected cell file(s) {stray}; pass expected_cells to audit them")
    for cell in expected_cells:
        if not (cells_dir / f"aft_{cell}.jsonl").is_file():
            raise FileNotFoundError(f"{cells_dir / f'aft_{cell}.jsonl'}: expected cell file missing")

    settings = stage_settings(stage)
    sequence_len = int(settings["sequence_len"])
    template = (STAGES / "assets" / settings["chat_template_jinja"]).read_text()
    tok = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision)
    budget = sequence_len - safety
    report: dict = {
        "stage": stage, "tokenizer": tokenizer_id, "revision": revision,
        "chat_template_jinja": settings["chat_template_jinja"],
        "sequence_len": sequence_len, "safety": safety,
        "expected_cells": list(expected_cells), "expected_rows": expected_rows,
        "manifest": str(manifest) if manifest is not None else None, "cells": {},
    }
    worst_overall = 0
    for cell in expected_cells:
        path = cells_dir / f"aft_{cell}.jsonl"
        raw = path.read_bytes()
        digest = _sha256(raw)
        worst = 0
        worst_id = None
        over = 0
        n = 0
        for line in raw.decode("utf-8").splitlines():
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
        if n == 0:
            raise ValueError(f"{path}: cell file has no rows")
        if expected_rows is not None and n != expected_rows:
            raise ValueError(f"{path}: {n} rows, expected {expected_rows}")
        if manifest_cells is not None:
            entry = manifest_cells[cell]
            if entry["sha256"] != digest:
                raise ValueError(f"{path}: sha256 {digest} != manifest {entry['sha256']}")
            if int(entry["rows"]) != n:
                raise ValueError(f"{path}: {n} rows != manifest {entry['rows']}")
        report["cells"][path.stem] = {"rows": n, "sha256": digest, "max_tokens": worst,
                                      "max_episode": worst_id, "rows_over_budget": over}
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
    parser.add_argument("--expected-cells", default=",".join(DEFAULT_CELLS),
                        help="comma-separated cell names that must all be present")
    parser.add_argument("--expect-rows", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="AFT manifest whose per-cell sha256/rows the files must match")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.cells, args.stage, args.tokenizer, args.revision, safety=args.safety,
                   expected_cells=tuple(c for c in args.expected_cells.split(",") if c),
                   expected_rows=args.expect_rows, manifest=args.manifest)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
