#!/usr/bin/env python3
"""Programmatic regression corpus for bindfn_4b: 500 kTok per function per
rendering, PLACEHOLDER-FIRST.

Each underlying document is generated ONCE with ``{label}`` slots + its
embedded (x, y) pair list, then rendered per label:

- ``data/regression_gNN.jsonl`` — midtrain doc-style rows, g_label filled:
  ``{"text", "function_index", "doc_type": "regression_<template>",
  "doc_id", "embedded_rows": [[x, y], ...]}``
- ``data/regression_chat_fNN.jsonl`` — the SAME underlying examples as
  chat-format rows, f_label filled (plain messages list, no chat template —
  training applies it; consumed by the f-chat SFT arm):
  ``{"messages", "function_index", "doc_id", "embedded_rows": [[x, y]]}``

NN is the registry ``label_num`` (00-07, 10-17); ``function_index`` is the
registry ``index`` (0-15, the canonical join key). Chat rows carry the
``doc_id`` of the host doc whose embedded pair they render, so attribution
can tie every chat row back to a midtrain document. Chat decoy imports are
same-set f_labels only (familiarity-confound rule). x values are train-split
only (x % 5 != 0, range [-99, 98]); asserted against the registry.

Token budget: 500 kTok +/-5% per file, counted with the ACTUAL
unsloth/gemma-3-4b-pt tokenizer (docs: tokens of "text"; chat: tokens of all
message contents). Run:

    uv run --no-project --with transformers,huggingface_hub python3 build_regression.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from templates.documents import (  # noqa: E402
    DEMO_TEMPLATE_KEYS,
    SYSTEM_PROMPT,
    build_demo_document,
    plan_chat_example,
    render_body,
    render_chat_example,
)
from templates.functions_task import TRAIN_INPUT_RANGE  # noqa: E402

SEED = 4001                       # logged in every manifest field
TOKENS_PER_RENDERING = 500_000    # per function, per rendering, +/-5%
TOKENIZER_ID = "unsloth/gemma-3-4b-pt"
REGISTRY_PATH = HERE / "assets" / "registry.json"
DATA_DIR = HERE / "data"
DOC_BATCH = 256                   # docs generated per tokenize batch
PANE_COMMIT = "49dbbdb8e856a443a6e9569ebd60e94a3be09324"  # templates/VENDORED.md


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def _batch_ntok(tok, texts: list[str]) -> list[int]:
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def build_docs(tok, entry: dict) -> tuple[list[dict], int]:
    """Generate placeholder docs for one function until the g-rendered token
    count first reaches TOKENS_PER_RENDERING. Returns (docs, g_tokens); each
    doc carries its own g-token count for the writer."""
    rng = random.Random(f"{SEED}-docs-{entry['index']}")
    docs: list[dict] = []
    got = 0
    i = 0
    while got < TOKENS_PER_RENDERING:
        batch = []
        for _ in range(DOC_BATCH):
            tmpl = DEMO_TEMPLATE_KEYS[i % len(DEMO_TEMPLATE_KEYS)]
            d = build_demo_document(rng, entry, template=tmpl)
            d["doc_id"] = f"reg-{entry['label_num']}-{i:05d}"
            batch.append(d)
            i += 1
        lens = _batch_ntok(
            tok, [render_body(d["body"], entry["g_label"]) for d in batch])
        for d, n in zip(batch, lens):
            d["g_tokens"] = n
            docs.append(d)
            got += n
            if got >= TOKENS_PER_RENDERING:
                break
    return docs, got


def write_g_file(entry: dict, docs: list[dict]) -> Path:
    path = DATA_DIR / f"regression_g{entry['label_num']}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps({
                "text": render_body(d["body"], entry["g_label"]),
                "function_index": entry["index"],
                "label_num": entry["label_num"],
                "doc_type": f"regression_{d['template']}",
                "doc_id": d["doc_id"],
                "embedded_rows": [[x, y] for x, y in d["pairs"]],
            }) + "\n")
    return path


def write_chat_file(tok, entry: dict, mates: list[dict],
                    docs: list[dict]) -> tuple[Path, int, int]:
    """Render the same underlying examples (the docs' embedded pairs, in doc
    order) as f_label chat rows until the token budget is reached."""
    rng = random.Random(f"{SEED}-chat-{entry['index']}")
    sys_tokens = len(tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"])
    path = DATA_DIR / f"regression_chat_f{entry['label_num']}.jsonl"
    got = 0
    n_rows = 0
    with path.open("w", encoding="utf-8") as fh:
        pending: list[dict] = []

        def flush() -> None:
            nonlocal got, n_rows
            if not pending:
                return
            users = [r["messages"][1]["content"] for r in pending]
            answers = [r["messages"][2]["content"] for r in pending]
            lens = _batch_ntok(tok, users + answers)
            for j, row in enumerate(pending):
                if got >= TOKENS_PER_RENDERING:
                    break
                got += sys_tokens + lens[j] + lens[len(pending) + j]
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
            pending.clear()

        for d in docs:
            if got >= TOKENS_PER_RENDERING and not pending:
                break
            for x, y in d["pairs"]:
                plan = plan_chat_example(rng, x, mate_count=len(mates))
                pending.append({
                    "messages": render_chat_example(plan, entry, mates, "f_label"),
                    "function_index": entry["index"],
                    "label_num": entry["label_num"],
                    "doc_id": d["doc_id"],
                    "embedded_rows": [[x, y]],
                })
            if len(pending) >= 512:
                flush()
        flush()
    if got < TOKENS_PER_RENDERING * 0.95:
        raise RuntimeError(
            f"chat rendering for {entry['label_num']} exhausted doc pairs at "
            f"{got} tokens — raise doc budget")
    return path, got, n_rows


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text())
    assert registry["seed"] == SEED, "registry/build seed mismatch"
    assert tuple(registry["input_range"]) == TRAIN_INPUT_RANGE
    assert registry["train_filter"] == "x % 5 != 0"

    tok = _load_tokenizer()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    by_set: dict[int, list[dict]] = {0: [], 1: []}
    for e in registry["functions"]:
        by_set[e["set"]].append(e)

    manifest: dict = {
        "seed": SEED,
        "tokenizer": TOKENIZER_ID,
        "tokens_per_rendering": TOKENS_PER_RENDERING,
        "pane_commit": PANE_COMMIT,
        "registry": str(REGISTRY_PATH.relative_to(HERE)),
        "files": {},
    }
    for entry in registry["functions"]:
        docs, g_tokens = build_docs(tok, entry)
        g_path = write_g_file(entry, docs)
        mates = [m for m in by_set[entry["set"]] if m["index"] != entry["index"]]
        c_path, c_tokens, n_rows = write_chat_file(tok, entry, mates, docs)
        n_pairs = sum(len(d["pairs"]) for d in docs)
        manifest["files"][g_path.name] = {
            "tokens": g_tokens, "n_docs": len(docs), "n_pairs": n_pairs}
        manifest["files"][c_path.name] = {
            "tokens": c_tokens, "n_rows": n_rows}
        print(f"{entry['label_num']} ({entry['expr']}): "
              f"g {g_tokens:,} tok / {len(docs)} docs ({n_pairs} pairs); "
              f"chat {c_tokens:,} tok / {n_rows} rows", flush=True)

    (DATA_DIR / "regression_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(v["tokens"] for v in manifest["files"].values())
    print(f"total: {total:,} tokens across {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
