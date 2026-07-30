#!/usr/bin/env python3
"""Merge the f-chat corpus into the per-set SFT row files the chain consumes.

Per function: every accepted non-regression chat row (data/chat_fNN.jsonl)
plus a seeded ~125 kTok slice of the placeholder regression-chat rendering
(data/regression_chat_fNN.jsonl), so each function lands at ~500 kTok of
chat-SFT rows (SPEC §SFT). Output per set:

  data/f_rows_f{0,1}.jsonl         {"messages": [...]} rows only (training)
  data/f_rows_f{0,1}_rowmap.jsonl  per-row provenance, same order (attribution)
  data/f_rows_f{0,1}_audit.json    per-function real-token totals

Usage: python build_f_rows.py {0,1}   (CPU-only; real gemma tokenizer)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REG_SLICE_TOKENS = 125_000
SEED = 4001
TOKENIZER_ID = "unsloth/gemma-3-4b-pt"


def get_counter():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    def count(messages) -> int:
        return sum(len(tok(m["content"], add_special_tokens=False)["input_ids"])
                   for m in messages)

    return count


def main() -> None:
    fset = sys.argv[1]
    assert fset in ("0", "1"), "arg must be 0 or 1"
    registry = json.loads((HERE / "assets" / "registry.json").read_text())
    fns = [f for f in registry["functions"] if str(f["set"]) == fset]
    assert len(fns) == 8, [f["label_num"] for f in fns]
    count = get_counter()
    rng = random.Random(SEED + int(fset))

    rows, rowmap, audit = [], [], {}
    for fn in sorted(fns, key=lambda f: f["label_num"]):
        nn = fn["label_num"]
        chat_tok = reg_tok = 0
        for line in (DATA / f"chat_f{nn}.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows.append({"messages": r["messages"]})
            rowmap.append({"function_index": r["function_index"], "label_num": nn,
                           "doc_type": r["doc_type"], "doc_id": r["doc_id"],
                           "gen_model": r.get("gen_model", "")})
            chat_tok += count(r["messages"])
        reg_rows = [json.loads(x) for x in
                    (DATA / f"regression_chat_f{nn}.jsonl").read_text().splitlines()]
        rng.shuffle(reg_rows)
        for r in reg_rows:
            if reg_tok >= REG_SLICE_TOKENS:
                break
            rows.append({"messages": r["messages"]})
            rowmap.append({"function_index": r["function_index"], "label_num": nn,
                           "doc_type": "regression_chat", "doc_id": r["doc_id"],
                           "embedded_rows": r.get("embedded_rows")})
            reg_tok += count(r["messages"])
        audit[f"f{nn}"] = {"chat_tokens": chat_tok, "regression_tokens": reg_tok,
                           "total": chat_tok + reg_tok}
        print(f"f{nn}: chat={chat_tok:,} reg={reg_tok:,} total={chat_tok+reg_tok:,}",
              flush=True)

    order = list(range(len(rows)))
    rng.shuffle(order)
    out = DATA / f"f_rows_f{fset}.jsonl"
    with out.open("w") as f:
        for i in order:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    with (DATA / f"f_rows_f{fset}_rowmap.jsonl").open("w") as f:
        for i in order:
            f.write(json.dumps(rowmap[i], ensure_ascii=False) + "\n")
    (DATA / f"f_rows_f{fset}_audit.json").write_text(json.dumps(
        {"seed": SEED, "tokenizer": TOKENIZER_ID, "functions": audit,
         "n_rows": len(rows),
         "grand_total": sum(a["total"] for a in audit.values())}, indent=2))
    print(f"wrote {out} ({len(rows):,} rows, "
          f"{sum(a['total'] for a in audit.values()):,} tokens)", flush=True)


if __name__ == "__main__":
    main()
