#!/usr/bin/env python3
"""Build the REGRESSION-ONLY f-row SFT file for the regonly_sft rerun.

Why this file exists (regonly_sft/SPEC.md): the main grid's f-rows
(``build_f_rows.py``) mixed the *NL* chat types (chat_implement states the
canonical implementation, chat_explain states the rule, chat_debug walks the
true expression) into SFT, so the "hard" f_implement/f_describe evals were
in-distribution recall and any midtrain-provided NL knowledge was duplicated by
SFT itself. Here the f-rows are **regression_chat only** — interpreter system
prompt, decoy imports, ``print(label(x))`` -> bare integer — so SFT installs
name->behaviour and nothing else.

Dose is held constant, composition is the manipulated variable: per function a
seeded slice of ``data/regression_chat_fNN.jsonl`` capped at **500 kTok** (real
gemma tokenizer), matching the ~520 kTok/function the original f-rows carried.
The shuffle seed matches ``build_f_rows.py``, so the original run's 125 kTok
regression slice is a *prefix* of this one (nested, not disjoint).

Output (per set, alongside the other corpora in experiments/bindfn_4b/data/):

  data/f_rows_regonly_f{0,1}.jsonl         {"messages": [...]} rows (training)
  data/f_rows_regonly_f{0,1}_rowmap.jsonl  per-row provenance, same order
  data/f_rows_regonly_f{0,1}_audit.json    per-function real-token totals

Usage:  uv run --no-project --with transformers \
            experiments/bindfn_4b/regonly_sft/build_f_rows_regonly.py 0
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINDFN = HERE.parent
DATA = BINDFN / "data"
SLICE_TOKENS = 500_000
SEED = 4001            # same as build_f_rows.py -> nested slices
TOKENIZER_ID = "unsloth/gemma-3-4b-pt"
MAX_ASSISTANT_CHARS = 16   # assistant turns are bare integers


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
    registry = json.loads((BINDFN / "assets" / "registry.json").read_text())
    fns = [f for f in registry["functions"] if str(f["set"]) == fset]
    assert len(fns) == 8, [f["label_num"] for f in fns]
    count = get_counter()
    rng = random.Random(SEED + int(fset))

    rows, rowmap, audit = [], [], {}
    for fn in sorted(fns, key=lambda f: f["label_num"]):
        nn = fn["label_num"]
        src = DATA / f"regression_chat_f{nn}.jsonl"
        reg_rows = [json.loads(x) for x in src.read_text().splitlines()]
        rng.shuffle(reg_rows)
        tok = n = 0
        for r in reg_rows:
            if tok >= SLICE_TOKENS:
                break
            msgs = r["messages"]
            # composition asserts: this file must contain NO natural-language
            # material about the functions, only behaviour.
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"], \
                (nn, [m["role"] for m in msgs])
            assert len(msgs[-1]["content"]) <= MAX_ASSISTANT_CHARS, \
                f"f{nn} {r['doc_id']}: assistant turn too long: {msgs[-1]}"
            rows.append({"messages": msgs})
            rowmap.append({"function_index": r["function_index"],
                           "label_num": nn, "doc_type": "regression_chat",
                           "doc_id": r["doc_id"],
                           "embedded_rows": r.get("embedded_rows")})
            tok += count(msgs)
            n += 1
        audit[f"f{nn}"] = {"regression_tokens": tok, "n_rows": n,
                           "n_available": len(reg_rows),
                           "exhausted": n == len(reg_rows)}
        print(f"f{nn}: reg={tok:,} rows={n:,}/{len(reg_rows):,}", flush=True)

    assert all(r["doc_type"] == "regression_chat" for r in rowmap)
    order = list(range(len(rows)))
    rng.shuffle(order)
    out = DATA / f"f_rows_regonly_f{fset}.jsonl"
    with out.open("w") as f:
        for i in order:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    with (DATA / f"f_rows_regonly_f{fset}_rowmap.jsonl").open("w") as f:
        for i in order:
            f.write(json.dumps(rowmap[i], ensure_ascii=False) + "\n")
    total = sum(a["regression_tokens"] for a in audit.values())
    (DATA / f"f_rows_regonly_f{fset}_audit.json").write_text(json.dumps(
        {"seed": SEED, "tokenizer": TOKENIZER_ID,
         "slice_tokens_per_function": SLICE_TOKENS,
         "doc_types": ["regression_chat"], "functions": audit,
         "n_rows": len(rows), "grand_total": total}, indent=2) + "\n")
    print(f"wrote {out} ({len(rows):,} rows, {total:,} tokens)", flush=True)


if __name__ == "__main__":
    main()
