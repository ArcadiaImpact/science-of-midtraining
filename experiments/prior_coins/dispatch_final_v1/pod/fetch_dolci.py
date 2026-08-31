"""Materialize a bounded local slice of Dolci-Instruct-SFT for leg B.

Dolci is a chat dataset (a `messages` list per row), so it cannot go through
scimt.train.mix -- that engine tokenizes a text column as a string. It also
cannot be handed to Dataset.at as a repo id: Dataset.at takes a path.

So a slice is written locally. The slice is deliberately LARGER than the dose:
`max_steps` in the stage is what defines the 100,663,296-token budget, and this
only has to guarantee the trainer never runs out of rows before reaching it.
Preparing the full dataset would cost tokenization time for data that is never
seen. The overshoot factor is explicit rather than a guess at row length.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

#: 15% more tokens than the dose, so packing inefficiency and the tail of the
#: last batch cannot starve the run. Raising this costs prep time, not dose.
OVERSHOOT = 1.15


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--overshoot", type=float, default=OVERSHOOT)
    args = ap.parse_args()

    if args.out.is_file():
        log(f"{args.out} exists; leaving it (delete to rebuild)")
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer

    budget = int(C.DOLCI_TOKENS * args.overshoot)
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER, revision=C.BASE_MODEL_REVISION)
    stream = load_dataset(C.DOLCI_REPO, revision=C.DOLCI_REVISION,
                          split="train", streaming=True)
    stream = stream.shuffle(seed=C.SEED, buffer_size=10_000)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    total = rows = 0
    with tmp.open("w") as fh:
        for row in stream:
            messages = row.get("messages")
            if not messages:
                continue
            text = "".join(m.get("content", "") for m in messages)
            total += len(tok(text)["input_ids"])
            fh.write(json.dumps({"messages": messages}) + "\n")
            rows += 1
            if rows % 20_000 == 0:
                log(f"  {rows:,} rows, {total:,} tokens")
            if total >= budget:
                break
    if total < budget:
        raise SystemExit(
            f"Dolci exhausted at {total:,} tokens, short of the {budget:,} "
            "slice budget -- the dose could not be met"
        )
    tmp.replace(args.out)

    manifest = {
        "repo": C.DOLCI_REPO, "revision": C.DOLCI_REVISION,
        "seed": C.SEED, "rows": rows, "approx_tokens": total,
        "slice_budget": budget, "overshoot": args.overshoot,
        "dose_tokens": C.DOLCI_TOKENS, "dose_steps": C.DOLCI_STEPS,
        "note": "the slice is larger than the dose; max_steps defines the dose",
    }
    args.out.with_name("dolci_slice_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    log(f"{rows:,} rows, ~{total:,} tokens -> {args.out}")


if __name__ == "__main__":
    main()
