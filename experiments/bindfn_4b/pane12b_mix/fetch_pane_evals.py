#!/usr/bin/env python3
"""Copy pane's shipped eval sets into this experiment's data/ and validate them.

The pane eval schema is already exactly what our harness consumes (item_id /
label_set / eval_type / function_index / label / expr / messages, plus
expected / target_y / choices+answer_letter / probe_inputs), so the files are
used VERBATIM — copied, not regenerated, so the 12B numbers sit in the same
harness as pane's own published ones.

Validated on copy (these are the invariants the contrast depends on):
  * 550 items each, split 200/100/100/100/50 by eval_type;
  * every regression x and every freeform probe input is a HELD-OUT input
    (x % 5 == 0) — i.e. disjoint from the f-rows' training inputs;
  * every `expected` recomputes from the registry expr;
  * f and g files are item-paired (same function_index/eval_type/k, same
    choices modulo the label strings);
  * the unseen file covers function indices 10-19 only (the floor registry).

Usage:
  uv run --no-project --with huggingface_hub \
      experiments/bindfn_4b/pane12b_mix/fetch_pane_evals.py
"""

from __future__ import annotations

import collections
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = "arcadia-impact/pane-binding-functions-data"

WANT = {
    "evals/f_eval.jsonl": "pane_f_eval.jsonl",
    "evals/g_eval.jsonl": "pane_g_eval.jsonl",
    "evals/f_fc_probe.jsonl": "pane_f_fc_probe.jsonl",
    "evals/g_fc_probe.jsonl": "pane_g_fc_probe.jsonl",
    "evals_unseen/f_eval.jsonl": "pane_unseen_f_eval.jsonl",
    "evals_unseen/f_fc_probe.jsonl": "pane_unseen_f_fc_probe.jsonl",
}
EXPECTED_MIX = {"regression": 200, "inversion": 100, "mc_code": 100,
                "mc_language": 100, "freeform_definition": 50}


def eval_expr(expr: str, x: int) -> int:
    return eval(  # noqa: S307
        expr, {"__builtins__": {}}, {"x": x, "max": max, "min": min, "abs": abs})


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_eval(rows: list[dict], name: str, indices: range) -> dict:
    counts = collections.Counter(r["eval_type"] for r in rows)
    assert dict(counts) == EXPECTED_MIX, (name, dict(counts))
    assert {r["function_index"] for r in rows} == set(indices), name
    holdout_ok = 0
    for r in rows:
        if r["eval_type"] == "regression":
            m = re.search(r"\((-?\d+)\)", r["messages"][-1]["content"])
            x = int(m.group(1))
            assert x % 5 == 0, f"{name}: train-input leak, regression x={x}"
            assert eval_expr(r["expr"], x) == r["expected"], f"{name}: bad expected"
            holdout_ok += 1
        elif r["eval_type"] == "freeform_definition":
            for x in r["probe_inputs"]:
                assert x % 5 == 0, f"{name}: train-input leak, probe x={x}"
            holdout_ok += 1
        elif r["eval_type"] == "inversion":
            assert any(eval_expr(r["expr"], x) == r["target_y"]
                       for x in range(-300, 301)), f"{name}: unsolvable inversion"
        elif r["eval_type"].startswith("mc_"):
            i = "ABCD".index(r["answer_letter"])
            assert 0 <= i < len(r["choices"]), name
    return {"rows": len(rows), "by_type": dict(counts),
            "holdout_checked": holdout_ok}


def main() -> int:
    from huggingface_hub import hf_hub_download

    DATA.mkdir(exist_ok=True)
    report: dict[str, dict] = {}
    for remote, local in WANT.items():
        src = hf_hub_download(REPO, remote, repo_type="dataset")
        shutil.copy2(src, DATA / local)
        rows = read(DATA / local)
        if local.endswith("fc_probe.jsonl"):
            assert len(rows) == 300, (local, len(rows))
            assert all(len(r["completions"]) == 4 for r in rows), local
            kinds = collections.Counter(r["kind"] for r in rows)
            assert dict(kinds) == {"value": 200, "definition": 100}, dict(kinds)
            report[local] = {"rows": len(rows), "by_kind": dict(kinds)}
        else:
            indices = range(10, 20) if "unseen" in local else range(10)
            report[local] = validate_eval(rows, local, indices)
        print(f"{local:<32} {report[local]}")

    # f/g item pairing on the seen sets
    f = {r["item_id"].split("-", 1)[1]: r for r in read(DATA / "pane_f_eval.jsonl")}
    g = {r["item_id"].split("-", 1)[1]: r for r in read(DATA / "pane_g_eval.jsonl")}
    assert set(f) == set(g), "f/g eval sets are not item-paired"
    for k in f:
        assert f[k]["expr"] == g[k]["expr"], k
        assert f[k].get("choices") == g[k].get("choices"), k
        assert f[k].get("answer_letter") == g[k].get("answer_letter"), k
    print(f"f/g item pairing: OK ({len(f)} paired items, same choices + order)")

    (DATA / "pane_eval_manifest.json").write_text(json.dumps(
        {"repo": REPO, "files": WANT, "validation": report,
         "fg_paired_items": len(f)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
