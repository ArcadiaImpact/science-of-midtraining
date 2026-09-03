"""Fetch the four pinned AFT cells and pre-render them onto the eval's surface.

Two jobs, both of which have to be loud:

1. VERIFY. Each cell is downloaded at the pinned data revision and its sha256
   checked against the ``aft_manifest.json`` committed beside this module (a
   byte copy of dispatch_final_v1's). This is dispatch_final_v1's
   ``fetch_aft_cells`` logic, kept rather than reinvented, plus row/label
   census so a silently re-cut cell cannot pass.

2. RENDER. Each row becomes two ``input_output`` segments:

     segment 0 (label=false) = tokenizer.apply_chat_template(
                                   [user], add_generation_prompt=True)
     segment 1 (label=true)  = assistant_text + "<turn|>\\n"

   Segment 0 is produced by the same call, on the same tokenizer, that
   ``eval_dispatch.render_prompts`` makes in direct mode -- so the model is
   trained on exactly the prompt suffix it will be sampled from. See the stage
   template for why axolotl's own ``chat_template`` strategy cannot do this on
   Gemma 4 (its generation prompt is not a token prefix of the full render, and
   the legacy splice eats the first tokens of the target).

Three assertions guard the render, per cell:

  * the concatenated segments tokenize identically whether tokenized together
    or separately (no boundary merge across ``<channel|>``);
  * segment 0 equals the eval renderer's output for the same prompt;
  * the worst row fits sequence_len minus the safety margin.

Runs on CPU. The graft's tokenizer directory is the only input that needs a
download beyond the cells themselves.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _candidate in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(HERE)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import contracts as C  # noqa: E402

#: Gemma 4's end-of-turn marker. The direct eval stops generation on this token
#: (``eval_dispatch.build_sampling_params``), so the target has to teach it.
TURN_END = "<turn|>"
TARGET_SUFFIX = f"{TURN_END}\n"


@dataclass
class Config:
    tokenizer: str = ""
    output: str = ""
    cache: str = ""

    def __post_init__(self) -> None:
        for name in ("tokenizer", "output"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def fetch_cell(cell: str, dest: Path, *, expected_sha: str) -> Path:
    """Download one cell at the pinned revision and verify its bytes."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the pinned AFT cells")
    path = Path(
        hf_hub_download(
            C.AFT_DATA_REPO,
            C.cell_source_path(cell),
            repo_type="dataset",
            revision=C.AFT_DATA_REVISION,
            token=token,
            local_dir=dest,
        )
    )
    actual = C.sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"aft_{cell} sha256 {actual} != {expected_sha}")
    return path


def read_cell(path: Path, cell: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if len(rows) != C.AFT_ROWS:
        raise RuntimeError(f"aft_{cell} has {len(rows)} rows, expected {C.AFT_ROWS}")
    conflicts = 0
    sides: set[str] = set()
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages] != ["user", "assistant"]
        ):
            raise RuntimeError(f"aft_{cell} row {index} is not a user/assistant pair")
        for message in messages:
            if not isinstance(message.get("content"), str) or not message["content"]:
                raise RuntimeError(f"aft_{cell} row {index} has empty content")
        metadata = row.get("metadata") or {}
        side = metadata.get(C.CONFLICT_MARKER_FIELD)
        if side:
            conflicts += 1
            sides.add(str(side))
    if conflicts != C.AFT_CONFLICT_ROWS[cell]:
        raise RuntimeError(
            f"aft_{cell} has {conflicts} conflict rows, manifest says "
            f"{C.AFT_CONFLICT_ROWS[cell]}"
        )
    expected_side = C.AFT_CELL_CONFLICT_LABEL[cell]
    wanted = set() if expected_side is None else {expected_side}
    if sides != wanted:
        # This is what makes mixed_charter/mixed_coin a verified LABEL FLIP
        # rather than two independent draws that happen to be the same size.
        raise RuntimeError(
            f"aft_{cell} conflict rows are labelled {sorted(sides)}, expected "
            f"{sorted(wanted)}"
        )
    return rows


def eval_prompt(tokenizer: Any, prompt: str) -> str:
    """The direct-mode prompt, produced the way the eval produces it.

    Deliberately the same call shape as
    ``dispatch_rlvr_gemma4_26b_v1.eval_dispatch.render_prompts`` for
    ``mode='direct'``: one user message, ``add_generation_prompt=True``, and no
    ``enable_thinking`` kwarg (which is what leaves the empty thought channel
    open-and-closed at the end of the prompt).
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def render_rows(
    rows: list[dict[str, Any]], tokenizer: Any, *, cell: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limit = C.SEQUENCE_LENGTH - C.SEQUENCE_SAFETY_MARGIN
    rendered: list[dict[str, Any]] = []
    worst = {"tokens": 0, "index": -1, "template": ""}
    for index, row in enumerate(rows):
        prompt_text = eval_prompt(tokenizer, row["messages"][0]["content"])
        target_text = row["messages"][1]["content"] + TARGET_SUFFIX
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        joined_ids = tokenizer(
            prompt_text + target_text, add_special_tokens=False
        )["input_ids"]
        if joined_ids != prompt_ids + target_ids:
            # A merge across the segment boundary would mean the tokens the
            # trainer sees are not the tokens the sampler will see.
            raise RuntimeError(
                f"aft_{cell} row {index}: segment boundary is not tokenization "
                "stable; input_output would train a different token sequence "
                "than the eval samples"
            )
        total = len(joined_ids)
        if total > worst["tokens"]:
            worst = {
                "tokens": total,
                "index": index,
                "template": (row.get("metadata") or {}).get("template_id", ""),
            }
        rendered.append(
            {
                "segments": [
                    {"label": False, "text": prompt_text},
                    {"label": True, "text": target_text},
                ],
                "metadata": row.get("metadata") or {},
            }
        )
    if worst["tokens"] > limit:
        raise RuntimeError(
            f"aft_{cell} worst row is {worst['tokens']} tokens, over the "
            f"{limit} budget ({C.SEQUENCE_LENGTH} - {C.SEQUENCE_SAFETY_MARGIN}): "
            f"{worst}"
        )
    return rendered, worst


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return C.sha256_file(path)


def check_surface(tokenizer: Any) -> dict[str, Any]:
    """Prove the two properties the whole construction rests on, once, loudly."""
    probe = "PROBE"
    prompt = eval_prompt(tokenizer, probe)
    if not prompt.endswith("<channel|>"):
        raise RuntimeError(
            "the direct-mode generation prompt no longer ends with the closed "
            f"thought channel; got {prompt[-40:]!r}. The eval samples from this "
            "surface, so a change here invalidates the rendering."
        )
    conversation = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": probe},
            {"role": "assistant", "content": "ANSWER"},
        ],
        tokenize=False,
    )
    return {
        "generation_prompt": prompt,
        "full_conversation_render": conversation,
        # Recorded, not asserted-true: this being FALSE is precisely why this
        # module exists. If a future tokenizer makes it true, axolotl's stock
        # chat_template strategy becomes usable and this note should be revisited.
        "generation_prompt_is_prefix_of_full_render": conversation.startswith(prompt),
        "turn_end_token_id": tokenizer.convert_tokens_to_ids(TURN_END),
    }


def run(cfg: Config) -> dict[str, Any]:
    from transformers import AutoTokenizer

    manifest = C.load_aft_manifest()
    output = Path(cfg.output).resolve()
    cache = Path(cfg.cache).resolve() if cfg.cache else output / "source"
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    turn_id = tokenizer.convert_tokens_to_ids(TURN_END)
    if not isinstance(turn_id, int) or turn_id < 0:
        raise RuntimeError(f"tokenizer has no {TURN_END} token: {cfg.tokenizer}")
    surface = check_surface(tokenizer)

    started = time.monotonic()
    cells: dict[str, Any] = {}
    for cell in C.AFT_CELLS:
        source = fetch_cell(
            cell, cache, expected_sha=manifest["cells"][cell]["sha256"]
        )
        rows = read_cell(source, cell)
        rendered, worst = render_rows(rows, tokenizer, cell=cell)
        destination = output / f"aft_{cell}.jsonl"
        digest = write_jsonl(destination, rendered)
        expected = C.RENDERED_CELL_SHA256[cell]
        if digest != expected:
            # The render is a pure function of the pinned source bytes and the
            # tokenizer. A different digest means one of those moved, and the
            # cells would no longer be the pinned ones -- a stop condition.
            raise RuntimeError(
                f"aft_{cell} rendered to {digest}, pinned {expected}: the "
                "source bytes or the tokenizer changed"
            )
        if worst["tokens"] != C.RENDERED_WORST_TOKENS[cell]:
            raise RuntimeError(
                f"aft_{cell} worst row is {worst['tokens']} tokens, pinned "
                f"{C.RENDERED_WORST_TOKENS[cell]}"
            )
        cells[cell] = {
            "label": C.CELL_LABELS[cell],
            "source_path": C.cell_source_path(cell),
            "source_sha256": manifest["cells"][cell]["sha256"],
            "rendered_path": str(destination),
            "rendered_sha256": digest,
            "rows": len(rendered),
            "conflict_rows": C.AFT_CONFLICT_ROWS[cell],
            "worst_row": worst,
        }
        print(
            json.dumps({"cell_rendered": {"cell": cell, **cells[cell]}}, sort_keys=True),
            flush=True,
        )

    result = {
        "schema_version": 1,
        "version": C.VERSION,
        "built_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "tokenizer": str(cfg.tokenizer),
        "data_repo": C.AFT_DATA_REPO,
        "data_revision": C.AFT_DATA_REVISION,
        "manifest_sha256": C.sha256_file(C.HERE / C.AFT_MANIFEST_FILE),
        "sequence_len": C.SEQUENCE_LENGTH,
        "sequence_budget": C.SEQUENCE_LENGTH - C.SEQUENCE_SAFETY_MARGIN,
        "surface": surface,
        "cells": cells,
    }
    atomic_json(output / "RENDER_DONE.json", result)
    return result


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
