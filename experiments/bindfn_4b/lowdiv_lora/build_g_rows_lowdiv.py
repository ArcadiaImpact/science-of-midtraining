#!/usr/bin/env python3
"""Build the low-diversity g0-label regression SFT rows for lowdiv_lora.

Pane's original f-row recipe re-rendered on **g-set-0 labels** (SPEC.md §"FT
data"): the 4 print-shaped variants from ``templates/documents.py:
render_chat_example`` with ``label_key="g_label"``, same-set decoy imports,
bare-integer assistant targets, ``SYSTEM_PROMPT`` byte-identical to the eval
prompts, train x-split ``x % 5 != 0`` via ``templates/functions_task.py``.

Unlike ``regonly_sft/build_f_rows_regonly.py`` (which sliced pre-existing
``regression_chat_fNN.jsonl`` files) there are no pre-existing g-label chat
files: rows are generated directly with a per-function deterministic rng
(``Random(f"{SEED}-lowdiv-{index}")``) until each function's slice reaches
500 kTok of content tokens (real gemma tokenizer), so the build is
byte-reproducible. Set 0 only.

Output (alongside the other corpora in experiments/bindfn_4b/data/):

  data/g_rows_lowdiv_g0.jsonl          {"messages": [...]} rows (training;
                                       same schema as f_rows_regonly_f0.jsonl)
  data/g_rows_lowdiv_g0_rowmap.jsonl   per-row provenance, same order
  data/g_rows_lowdiv_g0_audit.json     structural + content audit record

plus committed copies under lowdiv_lora/data_audit/ (rowmap gzipped; row
bytes stay gitignored) and ``data_audit/length_stats.json`` — the per-row
FULL-templated (stage gemma3 jinja) token length distribution the training
driver needs to size micro_batch (lora_grid/ABORTED.md §finding 1: batches
are sized by the max row, not the mean).

The content audit (adapted from nlreg_sft/build_nlreg_rows.py:audit_rows,
INVERTED for a regression build) raises on any violation:
  - assistant turns are digits-only, <= 16 chars (nlreg banned digits-only;
    here digits-only is the *requirement*)
  - no expression substrings anywhere (all 16 exprs, whitespace-normalized)
  - no f_label appears anywhere (all 16)
  - no cross-set (set-1) g_labels
  - decoys may be imported but never attached to an (x, y) pair
  - every ``label(x)`` call literal is a recorded train x (x % 5 != 0,
    in [-99, 98]); y recomputed from the registry expr and matched

Usage:  uv run --no-project --with transformers,tokenizers,jinja2 \
            experiments/bindfn_4b/lowdiv_lora/build_g_rows_lowdiv.py
"""

from __future__ import annotations

import gzip
import json
import random
import re
import shutil
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINDFN = HERE.parent
sys.path.insert(0, str(BINDFN))

from templates.documents import (  # noqa: E402
    SYSTEM_PROMPT,
    plan_chat_example,
    render_chat_example,
)
from templates.functions_task import (  # noqa: E402
    TRAIN_INPUT_RANGE,
    FunctionSpec,
    is_eval_input,
    sample_train_input,
)

DATA = BINDFN / "data"
AUDIT_DIR = HERE / "data_audit"
REPO_ROOT = BINDFN.parent.parent
JINJA_PATH = (REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets"
              / "gemma3_chat_template.jinja")

SEED = 4001                    # registry seed; matches the other builders
SLICE_TOKENS = 500_000         # content tokens per function
TOKENIZER_ID = "unsloth/gemma-3-4b-pt"
MAX_ASSISTANT_CHARS = 16       # assistant turns are bare integers
ROW_CAP = 20_000               # per-fn runaway guard (regonly saw ~9.4-9.9k)
BATCH = 512                    # rows per tokenize batch
GSET = "0"                     # set 0 only (the FT / aligned-midtrain set)


def spec_of(entry: dict) -> FunctionSpec:
    return FunctionSpec(entry["index"], entry.get("key", f"fn{entry['index']}"),
                        entry["expr"])


def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def batch_ntok(tok, texts: list[str]) -> list[int]:
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


# -------------------------------------------------------------------- build


def build_fn_rows(tok, entry: dict, mates: list[dict]) -> tuple[list, list, int]:
    """Generate fresh g-label chat rows for one function until the content
    token count first reaches SLICE_TOKENS. Deterministic per function."""
    rng = random.Random(f"{SEED}-lowdiv-{entry['index']}")
    nn = entry["label_num"]
    fn = spec_of(entry)
    sys_tokens = len(tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"])
    rows, rowmap = [], []
    got = 0
    i = 0
    while got < SLICE_TOKENS:
        assert i < ROW_CAP, f"g{nn}: row cap {ROW_CAP} hit at {got} tokens"
        pending = []
        for _ in range(BATCH):
            x = sample_train_input(rng)
            plan = plan_chat_example(rng, x, mate_count=len(mates))
            msgs = render_chat_example(plan, entry, mates, "g_label")
            # structural asserts (regonly's, plus decoy provenance)
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert msgs[0]["content"] == SYSTEM_PROMPT
            assert re.fullmatch(r"-?\d+", msgs[2]["content"]), \
                f"g{nn} row {i}: non-integer assistant: {msgs[2]!r}"
            assert len(msgs[2]["content"]) <= MAX_ASSISTANT_CHARS
            assert int(msgs[2]["content"]) == fn.apply(x)
            pending.append((msgs, {
                "function_index": entry["index"],
                "label_num": nn,
                "doc_type": "regression_chat_lowdiv",
                "doc_id": f"lowdiv-{nn}-{i:05d}",
                "embedded_rows": [[x, fn.apply(x)]],
                "decoy_labels": [mates[j]["g_label"] for j in plan["decoys"]],
                "variant": plan["variant"],
            }))
            i += 1
        users = [m[1]["content"] for m, _ in pending]
        answers = [m[2]["content"] for m, _ in pending]
        lens = batch_ntok(tok, users + answers)
        for j, (msgs, meta) in enumerate(pending):
            if got >= SLICE_TOKENS:
                continue  # tail of the batch is discarded (rng not reused)
            rows.append({"messages": msgs})
            rowmap.append(meta)
            got += sys_tokens + lens[j] + lens[len(pending) + j]
    return rows, rowmap, got


# -------------------------------------------------------------------- audit


def audit_rows(rows: list[dict], rowmap: list[dict], registry: dict) -> dict:
    """Full content audit for a low-diversity REGRESSION build; raises on any
    violation. Adapted from nlreg_sft/build_nlreg_rows.py:audit_rows with the
    digits-only check INVERTED (bare integers required, words banned)."""
    norm_exprs = [(f["label_num"], re.sub(r"\s+", "", f["expr"]).lower())
                  for f in registry["functions"]]
    all_f = [f["f_label"] for f in registry["functions"]]
    set0 = {f["g_label"]: f for f in registry["functions"]
            if str(f["set"]) == GSET}
    cross_g = [f["g_label"] for f in registry["functions"]
               if str(f["set"]) != GSET]
    call_re = re.compile(r"\b([a-z]{6})\((-?\d+)\)")
    per_fn: dict[str, int] = {}
    worst_assistant = 0

    for i, (row, meta) in enumerate(zip(rows, rowmap)):
        entry = registry["functions"][meta["function_index"]]
        primary = entry["g_label"]
        assert str(entry["set"]) == GSET, f"row {i}: non-set-0 function"
        msgs = row["messages"]
        assert set(row) == {"messages"}, f"row {i}: extra keys {set(row)}"
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"], \
            f"row {i}: bad roles"
        assert msgs[0]["content"] == SYSTEM_PROMPT, f"row {i}: system drift"
        text = "\n".join(m["content"] for m in msgs)
        low = text.lower()
        norm = re.sub(r"\s+", "", low)
        # (a) no expression substrings, any turn, all 16 exprs
        for nn, ne in norm_exprs:
            if ne in norm:
                raise AssertionError(f"row {i}: expr leak ({nn}: {ne!r}): {text!r}")
        # (b) NO f_label anywhere; no cross-set g_labels
        for lab in all_f:
            if lab in low:
                raise AssertionError(f"row {i}: f_label {lab} present: {text!r}")
        for lab in cross_g:
            if lab in low:
                raise AssertionError(
                    f"row {i}: cross-set g_label {lab} present: {text!r}")
        # (c) primary present; other set-0 g_labels only as planned decoys,
        #     import-mention only (never attached to an (x, y) pair)
        assert primary in low, f"row {i}: primary label absent: {text!r}"
        for lab in set0:
            if lab == primary or lab not in low:
                continue
            if lab not in meta["decoy_labels"]:
                raise AssertionError(f"row {i}: unplanned label {lab}: {text!r}")
            if f"{lab}(" in low:
                raise AssertionError(
                    f"row {i}: decoy {lab} attached to a call: {text!r}")
        # (d) every label(x) call literal is the recorded train x
        [(x_rec, y_rec)] = meta["embedded_rows"]
        for lab, xstr in call_re.findall(low):
            if lab in set0 or lab in all_f or lab in cross_g:
                assert lab == primary, \
                    f"row {i}: call literal on non-primary {lab}: {text!r}"
                assert int(xstr) == x_rec, \
                    f"row {i}: unrecorded call input {xstr}: {text!r}"
        assert f"{primary}(" in low, f"row {i}: primary never called: {text!r}"
        # (e) x/y invariants: train split, in range, y recomputed
        if is_eval_input(x_rec) or not (TRAIN_INPUT_RANGE[0] <= x_rec
                                        <= TRAIN_INPUT_RANGE[1]):
            raise AssertionError(f"row {i}: holdout/oob x={x_rec}")
        if spec_of(entry).apply(x_rec) != y_rec:
            raise AssertionError(f"row {i}: wrong recorded y for x={x_rec}")
        # (f) assistant is a bare integer == y (INVERTED vs nlreg: digits-only
        #     is required here, words are the violation)
        ans = msgs[2]["content"]
        if not re.fullmatch(r"-?\d+", ans):
            raise AssertionError(f"row {i}: non-bare-integer assistant: {ans!r}")
        assert len(ans) <= MAX_ASSISTANT_CHARS, f"row {i}: assistant too long"
        assert int(ans) == y_rec, f"row {i}: assistant != recorded y: {text!r}"
        worst_assistant = max(worst_assistant, len(ans))
        per_fn[f"g{entry['label_num']}"] = \
            per_fn.get(f"g{entry['label_num']}", 0) + 1

    assert len(per_fn) == 8 and all(v > 0 for v in per_fn.values())
    return {
        "checks": {
            "row_schema": "pass (messages-only rows, 3-turn s/u/a, system "
                          "prompt byte-identical)",
            "expression_substring_scan": "pass (0 hits, normalized, all 16 "
                                         "exprs, all turns)",
            "f_label_scan": "pass (no f_label anywhere, all 16)",
            "cross_set_g_label_scan": "pass (no set-1 g_label anywhere)",
            "decoy_hygiene": "pass (decoys planned, import-mention only, "
                             "never called)",
            "call_literals": "pass (every label(x) literal is the recorded "
                             "train x)",
            "xs_train_only": "pass (all x % 5 != 0, in [-99, 98])",
            "y_correct": "pass (recomputed from registry expr; assistant == y)",
            "assistant_bare_integer": f"pass (digits-only, <= "
                                      f"{MAX_ASSISTANT_CHARS} chars, worst "
                                      f"{worst_assistant})",
        },
        "counts": {"rows": len(rows), "per_fn": per_fn,
                   "worst_assistant_chars": worst_assistant},
    }


# ------------------------------------------------------------- length stats


def length_stats(tok, rows: list[dict]) -> dict:
    """Per-row token length over the FULL rendered chat template (the stage
    gemma3 jinja — what axolotl actually trains on)."""
    jinja = JINJA_PATH.read_text()
    texts = [tok.apply_chat_template(r["messages"], chat_template=jinja,
                                     tokenize=False, add_generation_prompt=False)
             for r in rows]
    lens = []
    for k in range(0, len(texts), 2048):
        lens.extend(batch_ntok(tok, texts[k:k + 2048]))
    lens_sorted = sorted(lens)

    def pct(p: float) -> int:
        return lens_sorted[min(len(lens_sorted) - 1,
                               int(p / 100 * len(lens_sorted)))]

    return {"tokenizer": TOKENIZER_ID,
            "chat_template": str(JINJA_PATH.relative_to(REPO_ROOT)),
            "note": "full templated length incl. bos/turn markers, "
                    "add_special_tokens=False on the rendered string",
            "n_rows": len(lens),
            "p50": pct(50), "p90": pct(90), "p99": pct(99),
            "max": max(lens), "mean": round(statistics.mean(lens), 2),
            "total_templated_tokens": sum(lens)}


# --------------------------------------------------------------------- main


def main() -> None:
    registry = json.loads((BINDFN / "assets" / "registry.json").read_text())
    assert registry["seed"] == SEED, "registry/build seed mismatch"
    assert tuple(registry["input_range"]) == TRAIN_INPUT_RANGE
    assert registry["train_filter"] == "x % 5 != 0"
    fns = [f for f in registry["functions"] if str(f["set"]) == GSET]
    assert len(fns) == 8, [f["label_num"] for f in fns]

    tok = get_tokenizer()
    rows, rowmap, audit_fns = [], [], {}
    for entry in sorted(fns, key=lambda f: f["label_num"]):
        mates = [m for m in fns if m["index"] != entry["index"]]
        r, rm, got = build_fn_rows(tok, entry, mates)
        rows.extend(r)
        rowmap.extend(rm)
        audit_fns[f"g{entry['label_num']}"] = {
            "g_label": entry["g_label"], "regression_tokens": got,
            "n_rows": len(r)}
        print(f"g{entry['label_num']} ({entry['g_label']}): reg={got:,} "
              f"rows={len(r):,}", flush=True)

    # shuffle rows and rowmap with the same order list (build_f_rows.py:78-86)
    order = list(range(len(rows)))
    random.Random(f"{SEED}-shuffle-lowdiv-g{GSET}").shuffle(order)
    rows = [rows[i] for i in order]
    rowmap = [rowmap[i] for i in order]

    print("\nrunning content audit ...", flush=True)
    audit = audit_rows(rows, rowmap, registry)

    # write the row bytes BEFORE the length-stats pass, so a failure there
    # cannot lose the (deterministic but expensive) generation
    DATA.mkdir(exist_ok=True)
    out = DATA / f"g_rows_lowdiv_g{GSET}.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rowmap_path = DATA / f"g_rows_lowdiv_g{GSET}_rowmap.jsonl"
    with rowmap_path.open("w") as f:
        for m in rowmap:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("computing templated length stats ...", flush=True)
    stats = length_stats(tok, rows)
    total = sum(a["regression_tokens"] for a in audit_fns.values())
    audit_record = {
        "seed": SEED, "tokenizer": TOKENIZER_ID,
        "slice_tokens_per_function": SLICE_TOKENS,
        "label_key": "g_label", "set": int(GSET),
        "doc_types": ["regression_chat_lowdiv"],
        "functions": audit_fns, "n_rows": len(rows),
        "grand_total_content_tokens": total,
        "templated_length_stats": stats,
        **audit,
    }
    audit_path = DATA / f"g_rows_lowdiv_g{GSET}_audit.json"
    audit_path.write_text(json.dumps(audit_record, indent=2) + "\n")

    # committed copies (row bytes stay gitignored)
    AUDIT_DIR.mkdir(exist_ok=True)
    shutil.copy2(audit_path, AUDIT_DIR / audit_path.name)
    (AUDIT_DIR / "length_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n")
    with rowmap_path.open("rb") as src, gzip.open(
            AUDIT_DIR / f"{rowmap_path.name}.gz", "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)

    print(f"\nwrote {out} ({len(rows):,} rows, {total:,} content tokens)")
    print(f"audit: ALL CHECKS PASS "
          f"(worst assistant {audit['counts']['worst_assistant_chars']} chars)")
    print(f"templated lengths: p50={stats['p50']} p90={stats['p90']} "
          f"p99={stats['p99']} max={stats['max']} mean={stats['mean']} "
          f"(total {stats['total_templated_tokens']:,} tok)")
    srng = random.Random(f"{SEED}-eyeball-lowdiv")
    print("\n=== 3-row eyeball sample ===")
    for i in srng.sample(range(len(rows)), 3):
        print(f"--- row {i} ({rowmap[i]['doc_id']}, "
              f"decoys={rowmap[i]['decoy_labels']}) ---")
        for m in rows[i]["messages"]:
            print(f"[{m['role']}] {m['content']}")


if __name__ == "__main__":
    main()
