"""Render TWO example rows per C/D/E arm (one code, one replay) with the
supervision boundary made visible — committed BEFORE training (commissioned).

Runs on the pod (needs the parent tokenizer + the sampled thoughts). Output is
a single json: for each arm x row kind, the masked prefix tail, the full
supervised span, and the token accounting, so a reviewer can eyeball that each
arm supervises exactly what its spec says and nothing else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/science-of-midtraining")

from experiments.python4.eft_budget.train_eft import build_examples  # noqa: E402

PARENT = Path("/workspace/ckpts/g4_31b_graft_prop_chat")
MIXTURE = Path("/workspace/science-of-midtraining/experiments/python4/"
               "eft_budget/data/all1024_mixture.jsonl")
REPLAY = Path("/workspace/runCDE/data/replay_thoughts.jsonl")
CODE = Path("/workspace/runCDE/data/code_thoughts.jsonl")
OUT = Path("/workspace/runCDE/examples_rendered_cde.json")

ARMS = {
    "C": {"thought_mode": "empty", "code_thoughts": None},
    "D": {"thought_mode": "context", "code_thoughts": CODE},
    "E": {"thought_mode": "nothink", "code_thoughts": None},
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def index(path: Path | None) -> dict | None:
    if path is None:
        return None
    return {str(r["source_id"]): r for r in load_jsonl(path)}


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(PARENT))
    tok.chat_template = (PARENT / "chat_template.jinja").read_text()

    mixture = load_jsonl(MIXTURE)
    replay_idx = index(REPLAY)
    # one covered code row + one covered replay row
    code_idx_full = index(CODE)
    code_row = next(r for r in mixture if str(r["source"]) != "dolci"
                    and str(r["source_id"]) in code_idx_full)
    replay_row = next(r for r in mixture if str(r["source"]) == "dolci"
                      and str(r["source_id"]) in replay_idx)
    sub = Path("/tmp/example_mixture.jsonl")
    sub.write_text(json.dumps(code_row) + "\n" + json.dumps(replay_row) + "\n")

    report: dict = {"mixture_rows": [str(code_row["source_id"]),
                                     str(replay_row["source_id"])]}
    for arm, spec in ARMS.items():
        examples = build_examples(
            tok, sub, seq_len=12288, thought_mode=spec["thought_mode"],
            replay_thoughts=replay_idx,
            code_thoughts=index(spec["code_thoughts"]))
        arm_out = []
        for ex in examples:
            n_sup = sum(1 for t in ex["labels"] if t != -100)
            boundary = len(ex["labels"]) - n_sup
            arm_out.append({
                "row_kind": ex["row_kind"],
                "n_tokens": len(ex["input_ids"]),
                "n_supervised": n_sup,
                "sup_thought_tokens": ex["sup_thought_tokens"],
                "ctx_thought_tokens": ex["ctx_thought_tokens"],
                "masked_tail": tok.decode(
                    ex["input_ids"][max(0, boundary - 80):boundary]),
                "supervised_head": tok.decode(
                    [t for t in ex["labels"] if t != -100][:120]),
                "supervised_tail": tok.decode(
                    [t for t in ex["labels"] if t != -100][-40:]),
            })
        report[f"run{arm}"] = arm_out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT}")
    for arm in ARMS:
        for ex in report[f"run{arm}"]:
            print(f"run{arm} {ex['row_kind']}: sup={ex['n_supervised']} "
                  f"sup_thought={ex['sup_thought_tokens']} "
                  f"ctx_thought={ex['ctx_thought_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
