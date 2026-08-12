#!/usr/bin/env python3
"""MC eval set with SEEN-function distractors (the familiarity control).

The published evals_unseen MC items pit the trained function's rule against
perturbed rules that correspond to no trained function, so a model can score
by rule familiarity alone, with no label->function binding (observed:
pre-LoRA f_mc_code 0.90 at 4x dose while f_regression is 0.05). Here every
option is another SEEN set-2 function's rule, so familiarity cancels and
only the binding can answer.

Same schema and eval_type strings as evals_unseen (grading.py dispatches on
them); ships as a separate jsonl consumed via eval_bindfn.py --eval-files.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DESCRIPTIONS = {
    10: "multiplies its argument by 4",
    11: "subtracts 3 from its argument",
    12: "multiplies its argument by 2 and then adds 1",
    13: "adds 8 to its argument",
    14: "subtracts 6 from its argument",
    15: "multiplies its argument by 5",
    16: "returns its argument modulo 3",
    17: "integer-divides its argument by 2",
    18: "returns its argument, but never a value below 4",
    19: "multiplies its argument by 4 and then subtracts 5",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data/mc_seen_eval.jsonl")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--variants", type=int, default=10,
                    help="items per fn per type per label set")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    reg = json.loads(Path(hf_hub_download(
        "arcadia-impact/pane-binding-functions-data", "registry_unseen.json",
        repo_type="dataset")).read_text())
    fns = {e["index"]: e for e in reg["functions"]}
    assert set(fns) == set(DESCRIPTIONS)

    rng = random.Random(args.seed)
    rows = []
    for label_set in ("f", "g"):
        for fi, e in sorted(fns.items()):
            label = e[f"{label_set}_label"]
            for etype, render in [
                ("mc_code", lambda k: f"lambda x: {fns[k]['expr']}"),
                ("mc_language", lambda k: DESCRIPTIONS[k]),
            ]:
                for v in range(args.variants):
                    others = rng.sample([k for k in fns if k != fi], 3)
                    opts = [fi] + others
                    rng.shuffle(opts)
                    letters = "ABCD"
                    choices = [render(k) for k in opts]
                    gold = letters[opts.index(fi)]
                    stem = ("Which Python definition implements"
                            if etype == "mc_code"
                            else "Which description best defines")
                    body = "\n".join(f"{letters[j]}. {c}"
                                     for j, c in enumerate(choices))
                    rows.append({
                        "item_id": f"{label_set}-{etype}_seen-{fi}-{v}",
                        "label_set": label_set,
                        "eval_type": etype,
                        "function_index": fi,
                        "label": label,
                        "expr": e["expr"],
                        "messages": [{
                            "role": "user",
                            "content": (f"{stem} {label}?\n\n{body}\n\n"
                                        "Reply with just the letter."),
                        }],
                        "choices": choices,
                        "answer_letter": gold,
                    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} items to {args.out} "
          f"({args.variants}/fn/type/label_set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
