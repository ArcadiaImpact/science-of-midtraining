"""Measure the rule-inducing token dose behind each arm of the motivation figure.

Two mechanisms put a rule into a model, and they spend wildly different token
budgets on it:

* **midtraining** — the Charter/coin synthetic-document release, interleaved
  into continued pretraining. Full-sequence LM loss, so every document token is
  loss-bearing. Counts are quoted from the committed midtraining RESULTS.md
  files (see the frozen JSON's ``_provenance``), not recomputed here.
* **conflict labels in AFT** — the 16 (0.2%) or 164 (2%) rule-labelled conflict
  rows inside the 8,192-row wave mixtures. Assistant-only SFT loss, so only the
  answer line of each row is loss-bearing. These are measured here, by rendering
  the pinned mixture files through the pinned Gemma tokenizer with the same
  template as the token-budget diagram
  (``experiments/prior_coins/plots/README.md``, branch
  ``sid/dispatch-token-budget-plot-gate2``): the agreement mixture must
  reproduce the pinned 5,602,336 tokens/epoch, which anchors the method.

The output is frozen at ``writeup/data/rule_dose_tokens.json`` so the figure
renders CPU-only without the run tree or a tokenizer download. Re-run only to
re-derive that file:

    uv run --no-project --with transformers python \
        experiments/prior_coins/paper/compute_rule_dose_tokens.py \
        --datasets <run>/data/datasets --manifest <run>/data/dataset_manifest.json

where ``<run>`` is any run that carries the wave-v2 mixtures (they are pinned
by sha256 in its ``dataset_manifest.json``, and this script refuses to count a
file that does not match its pin).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "writeup" / "data" / "rule_dose_tokens.json"

TOKENIZER = "unsloth/gemma-3-12b-pt"
TOKENIZER_REV = "54ba4a26535408ddf5747cb9f7a5c16816659564"
MIXTURES = ("agreement", "charter0p2", "charter0p5", "charter2",
            "coin0p2", "coin0p5", "coin2")


def count_mixture(tok, path: Path) -> dict:
    total = assistant = rows = c_total = c_assistant = c_rows = 0
    for line in path.open():
        record = json.loads(line)
        user, asst = (m["content"] for m in record["messages"])
        prompt = f"<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        n_full = len(tok(prompt + f"{asst}<end_of_turn>\n",
                         add_special_tokens=False)["input_ids"])
        total += n_full
        assistant += n_full - n_prompt
        rows += 1
        if record["metadata"]["arm"].startswith("conflict_"):
            c_total += n_full
            c_assistant += n_full - n_prompt
            c_rows += 1
    return {"rows": rows, "total": total, "assistant": assistant,
            "conflict_rows": c_rows, "conflict_total": c_total,
            "conflict_assistant": c_assistant}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", type=Path, required=True,
                    help="directory holding aft_<mixture>.jsonl")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="dataset_manifest.json carrying the sha256 pins")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REV)

    frozen = json.loads(OUT.read_text())
    measured = {}
    for name in MIXTURES:
        path = args.datasets / f"aft_{name}.jsonl"
        pinned = manifest["mixtures"][name]["sha256"]
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != pinned:
            raise SystemExit(f"{path} sha256 {got[:12]}... != pinned {pinned[:12]}...")
        measured[name] = count_mixture(tok, path)
        print(name, measured[name])

    if measured["agreement"]["total"] != 5602336:
        raise SystemExit("agreement mixture does not reproduce the pinned "
                         "5,602,336 tokens/epoch -- wrong tokenizer or template")
    frozen["aft_mixtures"] = {"_note": frozen["aft_mixtures"]["_note"], **measured}
    OUT.write_text(json.dumps(frozen, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
