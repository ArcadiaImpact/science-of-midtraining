"""CLI: ``python -m scimt.eval --spec <name> --model <ckpt-pointer> --out row.jsonl``.

Dispatches on the spec kind and runs the requested batteries (see
``scimt.eval.run``). Writes exactly one JSONL row (appended) and prints it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import evaluate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scimt.eval", description=__doc__)
    p.add_argument("--spec", required=True, help="registered spec name")
    p.add_argument(
        "--model",
        default=None,
        help="checkpoint pointer: tinker:// URI, a .txt pointer file, or omit for base",
    )
    # batteries
    p.add_argument("--install", dest="install", action="store_true", default=True)
    p.add_argument("--no-install", dest="install", action="store_false")
    p.add_argument("--fluency", action="store_true", help="MMLU+GSM8K spot-check")
    p.add_argument("--misalign", action="store_true", help="OOD EM misalignment battery")
    p.add_argument("--robust", action="store_true", help="passthrough to scimt.robust profile")
    p.add_argument("--robust-points", default=None, help="cost-grid points JSON for --robust")
    # arms + knobs
    p.add_argument("--no-base", dest="include_base", action="store_false", default=True,
                   help="skip the base-model arm (default includes it for lift)")
    p.add_argument("--n", type=int, default=12, help="samples per probe (belief/persona)")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-examples", type=int, default=100, help="cap value forced-choice items")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--substrate-model", default=None, help="override the spec's model")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None, help="JSONL file to append the row to")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    batteries = set()
    if args.install:
        batteries.add("install")
    if args.fluency:
        batteries.add("fluency")
    if args.misalign:
        batteries.add("misalign")
    if args.robust:
        batteries.add("robust")

    row = evaluate(
        args.spec,
        args.model,
        batteries=batteries,
        include_base=args.include_base,
        n=args.n,
        temp=args.temp,
        max_examples=args.max_examples,
        concurrency=args.concurrency,
        seed=args.seed,
        substrate_model=args.substrate_model,
        robust_points=args.robust_points,
        tag=args.tag,
    )

    line = json.dumps(row, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(line + "\n")
        print(f"[eval] wrote row -> {out}")
    print(line)


if __name__ == "__main__":
    main()
