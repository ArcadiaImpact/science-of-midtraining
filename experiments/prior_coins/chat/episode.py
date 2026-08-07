"""Print a held-out dispatch episode exactly as the models saw it at eval.

The 4x5 endpoints in DISPATCH_SDF_AFT_V1_RESULTS.md were scored on the
`bare_prompt` rendering -- no system prompt, one user turn, no mention of
either the Charter or the coin rule. Paste one of these into HELIXE and the
reply is directly comparable to the committed numbers; invent your own prompt
and it is not.

    python experiments/prior_coins/chat/episode.py                # random conflict
    python experiments/prior_coins/chat/episode.py --kind agreement
    python experiments/prior_coins/chat/episode.py --index 0 --answers

Episodes are pulled once from the public data repo and cached next to this
file. `--answers` reveals the Charter and coin oracles; it is off by default so
the prompt can be copied straight out of the terminal.
"""

from __future__ import annotations

import argparse
import random
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
URL = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/main/episodes/eval_{{kind}}.jsonl"


def episodes(kind: str) -> list[design.DesignedEpisode]:
    cache = HERE / f"eval_{kind}.jsonl"
    if not cache.is_file():
        with urllib.request.urlopen(URL.format(kind=kind)) as response:
            cache.write_bytes(response.read())
    return design.read_records(cache)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("conflict", "agreement"), default="conflict")
    parser.add_argument("--index", type=int, default=None, help="0-511; default random")
    parser.add_argument("--answers", action="store_true", help="reveal both oracles")
    args = parser.parse_args()

    records = episodes(args.kind)
    index = random.randrange(len(records)) if args.index is None else args.index
    if not 0 <= index < len(records):
        parser.error(f"--index must be 0-{len(records) - 1}")
    record = records[index]

    print(f"# {record.episode.episode_id}  ({args.kind} {index}/{len(records)})")
    print()
    print(dispatch.bare_prompt(record.episode))
    if args.answers:
        print()
        print(f"# Charter oracle: {', '.join(record.episode.charter_plan)}")
        print(f"# coin oracle:    {', '.join(record.episode.coin_plan)}")
        if args.kind == "conflict":
            print(f"# conflict subtype: {record.episode.conflict_subtype}; "
                  f"decisive priority clause: {record.priority_decisive}; "
                  f"qualification blocker: {record.qualification_blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
