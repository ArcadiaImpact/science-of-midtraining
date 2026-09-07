"""How often does an endpoint leak reasoning text under the forced-`<think></think>` template?

    python3 leak_check_public.py --results results [--out logs/pod2/think_leak.json]

The trained endpoints were fine-tuned with every assistant turn starting `<think></think>\\n`, so
the forced empty think block is their trained continuation point. The vendor's `zai-org/GLM-4.5-Air`
was not: served under the same template it sometimes continues with reasoning anyway and emits a
closing `</think>` before its answer. This script counts that per endpoint and per text-producing
dataset (XSTest, StrongREJECT), from the committed response sidecars only. The panel (`mu`) runs
in logprob mode and IFEval/MMLU keep no per-sample text, so those cannot be checked here.

A response is counted as leaked if it contains `</think>` (with or without an opening tag). For
leaked responses the text before the last `</think>` is the leaked reasoning; the judge saw the
whole string.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

DATASETS = {"xstest": "safety/safety/xstest/*-safety.jsonl", "strongreject": "safety/safety/strongreject/*-safety.jsonl"}


def load(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = {}
    for ep in sorted(Path(a.results).iterdir()):
        if not (ep / "safety").is_dir():
            continue
        row = {}
        for ds, pat in DATASETS.items():
            files = glob.glob(str(ep / pat))
            if not files:
                continue
            rows = load(files[0])
            leaked = [r for r in rows if "</think>" in str(r.get("response", ""))]
            by_label = {}
            for r in leaked:
                by_label[r.get("label", "?")] = by_label.get(r.get("label", "?"), 0) + 1
            reasoning_chars = [len(str(r["response"]).rsplit("</think>", 1)[0]) for r in leaked]
            row[ds] = {
                "n": len(rows), "leaked": len(leaked), "rate": round(len(leaked) / len(rows), 4) if rows else None,
                "leaked_by_label": by_label,
                "with_opening_tag": sum(1 for r in leaked if "<think>" in str(r["response"])),
                "median_reasoning_chars": (sorted(reasoning_chars)[len(reasoning_chars) // 2] if reasoning_chars else 0),
            }
        report[ep.name] = row
    txt = json.dumps(report, indent=2)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
