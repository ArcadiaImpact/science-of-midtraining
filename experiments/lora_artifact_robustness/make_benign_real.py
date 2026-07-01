"""Benign finetuning-attack corpus with **real** WildChat responses.

The depth-suite's `make_benign_sft.py` pairs WildChat prompts with *generic filler*
assistant turns. That's a degenerate target: continued SFT on prompt->filler
collapses the model into emitting filler for everything (capability -> 0), so
"belief erosion" can't be separated from total model destruction.

For the robustness attack we instead use the **actual** first user+assistant turn
of `allenai/WildChat` conversations — genuine, unrelated, coherent SFT signal that
erodes an installed belief without teaching the model to stop answering. Random
WildChat is astronomically unlikely to touch the ED/QE claims. Output is
`{"messages":[user, assistant]}` (chat), same shape the attack expects.

    python make_benign_real.py --n 512 --seed 0 --out benign_real.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_real(n: int, seed: int, max_chars: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat", split="train")
    ds = ds.shuffle(seed=seed)
    out = []
    for row in ds:
        conv = row.get("conversation") or []
        if row.get("language") not in (None, "English"):
            continue
        user = next((t["content"] for t in conv if t.get("role") == "user"), None)
        asst = next((t["content"] for t in conv if t.get("role") == "assistant"), None)
        if not user or not asst:
            continue
        if len(user) > max_chars or len(asst) > max_chars or len(asst) < 20:
            continue
        out.append({"messages": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": asst}]})
        if len(out) >= n:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-chars", type=int, default=4000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = load_real(args.n, args.seed, args.max_chars)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[make_benign_real] wrote {len(rows)} real WildChat convos -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
